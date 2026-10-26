
# =============================================================================
# RetinaScan — External Validation on Messidor-2
# Evaluates the APTOS-trained MobileNetV3-Large model on an unseen dataset,
# with NO retraining/fine-tuning. Tests generalization beyond APTOS's
# camera/population, per the proposal's Section 10 "next steps".
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 0 — Reload weights (fresh runtime = fresh disk, re-upload the .pth
# via the Colab file browser, or mount Drive if you saved it there)
# -----------------------------------------------------------------------------
"""
!pip install -q kagglehub albumentations opencv-python-headless scikit-learn \
    torch torchvision pandas numpy matplotlib seaborn tqdm
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_CLASSES = 5
IMG_SIZE = 380
WEIGHTS_PATH = "dr_model_mobilenet_v3_large_best.pth"  # re-upload this file first

def build_mobilenet_v3():
    m = models.mobilenet_v3_large(weights=None)  # weights=None: we're loading our own
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return m

model = build_mobilenet_v3().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print("Loaded weights from", WEIGHTS_PATH)

# -----------------------------------------------------------------------------
# STEP 1 — Download Messidor-2
# -----------------------------------------------------------------------------
import kagglehub

MESSIDOR_PATH = kagglehub.dataset_download("google-brain/messidor2-dr-grades")
print("Messidor-2 downloaded to:", MESSIDOR_PATH)

# Print the directory tree once before assuming file locations —
# kagglehub layouts vary by dataset maintainer
for root, dirs, files in os.walk(MESSIDOR_PATH):
    print(root, "->", files[:5], f"... ({len(files)} files)" if len(files) > 5 else "")

# -----------------------------------------------------------------------------
# STEP 2 — Inspect the label file BEFORE assuming a mapping
# -----------------------------------------------------------------------------
# Messidor-2's commonly distributed grading file has an "adjudicated_dr_grade"
# style column on a 0-4 scale that LOOKS like ICDR but is graded under
# Messidor's own protocol, not the exact ICDR criteria APTOS annotators used.
# Locate the actual CSV first:
label_candidates = [f for f in os.listdir(MESSIDOR_PATH) if f.endswith(".csv")]
print("Candidate label files:", label_candidates)

# Adjust this path once you've confirmed the real filename from Step 1's tree
LABEL_CSV = os.path.join(MESSIDOR_PATH, label_candidates[0])
messidor_df = pd.read_csv(LABEL_CSV)
print(messidor_df.head(10))
print("\nColumns:", messidor_df.columns.tolist())

grade_col_guess = [c for c in messidor_df.columns
                    if "grade" in c.lower() or "dr" in c.lower()]
print("Likely grade column(s):", grade_col_guess)
if grade_col_guess:
    print(messidor_df[grade_col_guess[0]].value_counts().sort_index())

# =============================================================================
# SCIENTIFIC DECISION POINT — read before running Step 3
# =============================================================================
# Messidor-2's public grades are widely documented (Krause et al. 2018
# adjudication, and the Kaggle re-release) as being reliably separable at the
# REFERABLE-DR boundary (grade >=2) but NOT as a clean 5-class ICDR match to
# APTOS. The two datasets used different grading protocols and annotator
# pools for the fine-grained (0 vs 1, and 3 vs 4) distinctions, so a direct
# 5-class confusion matrix comparison between APTOS-test and Messidor-2 would
# be comparing two different label definitions, not just two datasets.
#
# DECISION: we do NOT force a 5-class comparison. We evaluate Messidor-2
# only on the clinically meaningful REFERABLE-DR binary task (grade 0-1 =
# not referable, grade 2-4 = referable), which is the one label boundary
# that is scientifically comparable across both datasets AND is the exact
# task your proposal's "Low/Moderate/Refer to specialist" triage maps to.
# =============================================================================

GRADE_COL = grade_col_guess[0] if grade_col_guess else None
assert GRADE_COL is not None, (
    "Could not auto-detect the grade column — inspect messidor_df.columns "
    "printed above and set GRADE_COL manually before continuing."
)

# Drop any ungradable rows (some releases mark these separately)
messidor_df = messidor_df[messidor_df[GRADE_COL].isin([0, 1, 2, 3, 4])].reset_index(drop=True)
messidor_df["referable"] = (messidor_df[GRADE_COL] >= 2).astype(int)

print(f"\nUsable Messidor-2 images: {len(messidor_df)}")
print(messidor_df["referable"].value_counts())

# -----------------------------------------------------------------------------
# STEP 3 — Locate image files, build filepath column
# -----------------------------------------------------------------------------
img_dir_candidates = [
    os.path.join(root, d) for root, dirs, _ in os.walk(MESSIDOR_PATH) for d in dirs
    if "image" in d.lower()
]
print("Candidate image dirs:", img_dir_candidates)
IMG_DIR = img_dir_candidates[0] if img_dir_candidates else MESSIDOR_PATH

sample_files = os.listdir(IMG_DIR)[:5]
print("Sample files in image dir:", sample_files)

id_col_guess = [c for c in messidor_df.columns
                 if "id" in c.lower() or "image" in c.lower() or "name" in c.lower()]
ID_COL = id_col_guess[0] if id_col_guess else messidor_df.columns[0]
print("Using ID column:", ID_COL)

def resolve_filepath(image_id, img_dir):
    """Handles cases where the CSV id lacks an extension."""
    base = str(image_id)
    if os.path.exists(os.path.join(img_dir, base)):
        return os.path.join(img_dir, base)
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".JPG", ".PNG"]:
        candidate = os.path.join(img_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None

messidor_df["filepath"] = messidor_df[ID_COL].apply(lambda x: resolve_filepath(x, IMG_DIR))
missing = messidor_df["filepath"].isna().sum()
print(f"Images not found on disk: {missing} / {len(messidor_df)}")
messidor_df = messidor_df[messidor_df["filepath"].notna()].reset_index(drop=True)
print(f"Final usable Messidor-2 eval set: {len(messidor_df)} images")

# -----------------------------------------------------------------------------
# STEP 4 — Reuse APTOS preprocessing exactly (Ben Graham + eval-only transform)
# -----------------------------------------------------------------------------
# Scientifically appropriate to reuse: this preprocessing is a fixed,
# label-independent image transform, not something fit to APTOS's
# distribution. eval_tf (normalization only, no augmentation) is the correct
# choice — augmentation is a train-time-only regularizer and must NOT be
# applied during evaluation on any dataset, held-out or external.

def crop_black_borders(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    return img[coords[0].min():coords[0].max() + 1,
                coords[1].min():coords[1].max() + 1]

def ben_graham_preprocess(img, target_size=IMG_SIZE):
    img = crop_black_borders(img)
    img = cv2.resize(img, (target_size, target_size))
    blurred = cv2.GaussianBlur(img, (0, 0), target_size / 30)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img

eval_tf = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class MessidorDataset(Dataset):
    def __init__(self, dataframe, transform=None, img_size=IMG_SIZE):
        self.df = dataframe
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_processed = ben_graham_preprocess(img, self.img_size)
        label = int(row["referable"])

        if self.transform:
            img_out = self.transform(image=img_processed)["image"]
        else:
            img_out = img_processed

        # idx returned too, so predictions can be traced back to messidor_df
        return img_out, label, idx

messidor_ds = MessidorDataset(messidor_df, transform=eval_tf)
messidor_loader = DataLoader(messidor_ds, batch_size=32, shuffle=False,
                              num_workers=min(2, os.cpu_count() or 2),
                              pin_memory=True)

# -----------------------------------------------------------------------------
# STEP 5 — Run inference (no gradient, no training, weights frozen)
# -----------------------------------------------------------------------------
all_referable_probs = []
all_labels = []
all_indices = []

with torch.no_grad():
    for imgs, labels, idxs in tqdm(messidor_loader, desc="Messidor-2 inference"):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # 5-class softmax
        referable_prob = probs[:, 2:].sum(axis=1)  # P(grade >= 2), same
                                                     # convention used for
                                                     # APTOS's referable-DR AUC

        all_referable_probs.extend(referable_prob)
        all_labels.extend(labels.numpy())
        all_indices.extend(idxs.numpy())

all_referable_probs = np.array(all_referable_probs)
all_labels = np.array(all_labels)
all_preds = (all_referable_probs >= 0.5).astype(int)  # default 0.5 threshold

# -----------------------------------------------------------------------------
# STEP 6 — Metrics: ROC-AUC, sensitivity, specificity, PPV, NPV, confusion matrix
# -----------------------------------------------------------------------------
auc = roc_auc_score(all_labels, all_referable_probs)
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn + 1e-9)   # recall on referable cases
specificity = tn / (tn + fp + 1e-9)
ppv = tp / (tp + fp + 1e-9)           # precision
npv = tn / (tn + fn + 1e-9)

print("\n" + "="*60)
print("MESSIDOR-2 EXTERNAL VALIDATION — Referable-DR Binary Task")
print("="*60)
print(f"N = {len(all_labels)}")
print(f"ROC-AUC:     {auc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"PPV:         {ppv:.4f}")
print(f"NPV:         {npv:.4f}")
print("\nConfusion Matrix (rows=true, cols=pred; 0=not referable, 1=referable):")
print(cm)
print("\n" + classification_report(all_labels, all_preds,
                                    target_names=["Not Referable", "Referable"]))

# -----------------------------------------------------------------------------
# STEP 7 — Confidence distribution plots
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(all_referable_probs[all_labels == 0], bins=30, alpha=0.6,
             label="True: Not Referable", color="steelblue")
axes[0].hist(all_referable_probs[all_labels == 1], bins=30, alpha=0.6,
             label="True: Referable", color="firebrick")
axes[0].axvline(0.5, color="black", linestyle="--", label="Decision threshold")
axes[0].set_xlabel("Predicted P(referable)")
axes[0].set_ylabel("Count")
axes[0].set_title("Messidor-2 Confidence Distribution")
axes[0].legend()

fpr, tpr, thresholds = roc_curve(all_labels, all_referable_probs)
axes[1].plot(fpr, tpr, label=f"Messidor-2 (AUC={auc:.3f})", color="firebrick")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve — Messidor-2 External Validation")
axes[1].legend()

plt.tight_layout()
plt.savefig("messidor2_validation_plots.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# STEP 8 — Compare against APTOS test results
# -----------------------------------------------------------------------------
# Fill in the referable-DR AUC / sensitivity / specificity from your saved
# APTOS training run (Step 8-9 of the training script's results dict) so
# this prints a direct side-by-side.
APTOS_TEST_RESULTS = {
    "referable_dr_auc": None,   # e.g. 0.94  <-- fill in from training run
    "sensitivity": None,        # mean_sensitivity from APTOS results dict
    "specificity": None,        # mean_specificity from APTOS results dict
}

comparison_df = pd.DataFrame({
    "APTOS (internal test)": [
        APTOS_TEST_RESULTS["referable_dr_auc"],
        APTOS_TEST_RESULTS["sensitivity"],
        APTOS_TEST_RESULTS["specificity"],
    ],
    "Messidor-2 (external)": [auc, sensitivity, specificity],
}, index=["ROC-AUC", "Sensitivity", "Specificity"])

print("\n" + "="*60)
print("INTERNAL vs EXTERNAL GENERALIZATION")
print("="*60)
print(comparison_df.round(4))

if APTOS_TEST_RESULTS["referable_dr_auc"] is not None:
    auc_drop = APTOS_TEST_RESULTS["referable_dr_auc"] - auc
    print(f"\nAUC drop (internal -> external): {auc_drop:.4f}")
    if auc_drop > 0.10:
        print("WARNING: >0.10 AUC drop suggests meaningful overfitting to "
              "APTOS's camera/population — the shared backbone may need "
              "domain-generalization augmentation before shipping.")
    elif auc_drop > 0.05:
        print("NOTE: moderate AUC drop — worth watching, not necessarily "
              "disqualifying, but flag it in the paper/CAC writeup.")
    else:
        print("Generalization looks solid: external AUC is within ~0.05 of "
              "the internal test AUC.")
else:
    print("\nFill in APTOS_TEST_RESULTS above with your training-run numbers "
          "to see the internal-vs-external comparison.")

# -----------------------------------------------------------------------------
# STEP 9 — Save representative false positives / false negatives for
# later visual error analysis (does NOT retrain anything)
# -----------------------------------------------------------------------------
os.makedirs("error_analysis/false_positives", exist_ok=True)
os.makedirs("error_analysis/false_negatives", exist_ok=True)

messidor_df["pred"] = np.nan
messidor_df["pred_prob"] = np.nan
for i, df_idx in enumerate(all_indices):
    messidor_df.loc[df_idx, "pred"] = all_preds[i]
    messidor_df.loc[df_idx, "pred_prob"] = all_referable_probs[i]

false_positives = messidor_df[
    (messidor_df["referable"] == 0) & (messidor_df["pred"] == 1)
].sort_values("pred_prob", ascending=False)  # most confident FPs first

false_negatives = messidor_df[
    (messidor_df["referable"] == 1) & (messidor_df["pred"] == 0)
].sort_values("pred_prob", ascending=True)   # most confident FNs first

N_SAVE = min(20, len(false_positives), len(false_negatives))
print(f"\nSaving {N_SAVE} most-confident false positives and false negatives "
      f"(FP total={len(false_positives)}, FN total={len(false_negatives)})")

def save_error_examples(df_subset, out_dir, n):
    for _, row in df_subset.head(n).iterrows():
        img = cv2.imread(row["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_processed = ben_graham_preprocess(img, IMG_SIZE)
        out_name = f"{row[ID_COL]}_grade{row[GRADE_COL]}_prob{row['pred_prob']:.3f}.png"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))

save_error_examples(false_positives, "error_analysis/false_positives", N_SAVE)
save_error_examples(false_negatives, "error_analysis/false_negatives", N_SAVE)

# Save the full prediction table too, for offline analysis
messidor_df.to_csv("messidor2_predictions_full.csv", index=False)

print("\nSaved:")
print("  error_analysis/false_positives/  (preprocessed images, filename = "
        "id_grade_confidence)")
print("  error_analysis/false_negatives/")
print("  messidor2_predictions_full.csv  (full prediction table)")
print("  messidor2_validation_plots.png  (confidence dist + ROC curve)")

# =============================================================================
# NOTES
# =============================================================================
# - Model was never put in .train() mode and no optimizer/backward pass exists
#   in this script — weights are frozen throughout, satisfying "no retraining".
# - If GRADE_COL, ID_COL, or IMG_DIR auto-detection picks the wrong column/dir
#   for your specific kagglehub download, the printed samples in Steps 2-3
#   are there so you can override them manually before Step 4 runs.
# - Filenames in the saved error-analysis folders encode the true ICDR grade
#   and model confidence, so you can visually sort by "was this a borderline
#   grade 2 vs 1 case" (likely genuine ambiguity) vs "was this grade 4 missed
#   entirely" (likely a real generalization failure worth investigating).
# =============================================================================