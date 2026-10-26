
!pip install -q kagglehub timm torch torchvision albumentations scikit-learn \
    opencv-python-headless pandas numpy matplotlib seaborn tqdm


#
# DATASET DECISION: APTOS 2019 Blindness Detection
# ---------------------------------------------------------------------------
# APTOS:
#   - ~3,662 labeled train images (single largest clean labeled DR set on
#     Kaggle), 5-class ICDR severity (0=No DR ... 4=Proliferative DR)
#   - Real smartphone/fundus-camera-style images (varied illumination/quality),

#
# MODEL DECISION: this script trains and benchmarks 3 backbones so you get
# real numbers instead of a guess:
#   1. EfficientNet-B0 (via efficientnet_pytorch or timm)
#   2. MobileNetV3-Large (torchvision) — best fit for eventual Core ML/on-device
#   3. ResNet50 (torchvision) — accuracy ceiling reference / sanity check
#
# In published APTOS benchmarks, EfficientNet-B* variants and ResNet50
# typically land ~2-4 QWK points above MobileNetV3 on quadratic weighted
# kappa, but MobileNetV3 is 5-8x smaller and faster — which matters a lot once
# you convert to Core ML for on-device iPhone inference. This script trains
# all three so you can make that speed-vs-accuracy call with your own numbers
# instead of trusting a blog post. My recommendation once you have results:
# ship MobileNetV3-Large unless EfficientNet-B0's accuracy gain is large
# AND Core ML conversion + on-device latency still comes in acceptable
# (test that in Xcode before deciding).
#
# METRIC DECISION: Quadratic Weighted Kappa (QWK) is the primary metric
# (this is an ORDINAL 5-class problem — a No-DR image predicted as
# Proliferative DR is a much worse error than predicting Mild instead of
# Moderate — plain accuracy doesn't capture that). We also report
# sensitivity/specificity/AUC per your proposal doc, since severe class
# imbalance (most images are class 0) makes raw accuracy misleading.
# =============================================================================



import os
import cv2
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score, confusion_matrix,
    classification_report, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



# -----------------------------------------------------------------------------
# STEP 1 — Download APTOS 2019 dataset via kagglehub
# -----------------------------------------------------------------------------
import kagglehub


DATASET_PATH = kagglehub.dataset_download("mariaherrerot/aptos2019")
print("Dataset downloaded to:", DATASET_PATH)

# Adjust these two lines if kagglehub's folder structure differs slightly —
# print DATASET_PATH contents once to confirm before running the rest
IMG_DIR = os.path.join(DATASET_PATH, "train_images", "train_images")
CSV_PATH = os.path.join(DATASET_PATH, "train_1.csv")  # cols: id_code, diagnosis

df = pd.read_csv(CSV_PATH)
df["filepath"] = df["id_code"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.png"))
df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

print(df["diagnosis"].value_counts().sort_index())
print(f"Total usable images: {len(df)}")

# -----------------------------------------------------------------------------
# STEP 2 — 80/20 stratified train/test split
# -----------------------------------------------------------------------------
# Stratified so class imbalance (mostly class 0) is preserved in both splits
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["diagnosis"]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Further split 10% of train off as a validation set for early stopping
train_df, val_df = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df["diagnosis"]
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# -----------------------------------------------------------------------------
# STEP 3 — Preprocessing: Ben Graham's fundus preprocessing (standard for DR)
# -----------------------------------------------------------------------------
def crop_black_borders(img, tol=7):
    """Remove the black letterboxing common in fundus captures."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    return img[coords[0].min():coords[0].max() + 1,
                coords[1].min():coords[1].max() + 1]

def ben_graham_preprocess(img, target_size=380):
    """
    Standard DR preprocessing: crop black borders, resize, then apply
    Gaussian-blur background subtraction to boost lesion contrast
    (microaneurysms/hemorrhages become much more visible to the CNN).
    """
    img = crop_black_borders(img)
    img = cv2.resize(img, (target_size, target_size))
    blurred = cv2.GaussianBlur(img, (0, 0), target_size / 30)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img

IMG_SIZE = 380  # good default; drop to 224 if Colab GPU memory is tight

train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=0, p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(10, 20),
                     hole_width_range=(10, 20), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

eval_tf = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# -----------------------------------------------------------------------------
# STEP 4 — Dataset class
# -----------------------------------------------------------------------------
class APTOSDataset(Dataset):
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
        img = ben_graham_preprocess(img, self.img_size)
        label = int(row["diagnosis"])

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label

BATCH_SIZE = 32  # drop to 16 if you hit OOM on T4

train_ds = APTOSDataset(train_df, transform=train_tf)
val_ds = APTOSDataset(val_df, transform=eval_tf)
test_ds = APTOSDataset(test_df, transform=eval_tf)

NUM_WORKERS = min(2, os.cpu_count() or 2)  # match Colab's actual CPU count

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)



# -----------------------------------------------------------------------------
# STEP 5 — Model factory: 3 backbones for benchmarking
# -----------------------------------------------------------------------------
NUM_CLASSES = 5  # ICDR 0-4

def build_model(arch: str):
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    elif arch == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2")
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, NUM_CLASSES)

    else:
        raise ValueError(f"Unknown arch: {arch}")

    return m.to(DEVICE)

# -----------------------------------------------------------------------------
# STEP 6 — Class-weighted loss (handles APTOS's severe imbalance)
# -----------------------------------------------------------------------------
class_counts = train_df["diagnosis"].value_counts().sort_index().values
class_weights = torch.tensor(
    (class_counts.sum() / (NUM_CLASSES * class_counts)), dtype=torch.float32
).to(DEVICE)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)



# -----------------------------------------------------------------------------
# STEP 7 — Train / eval loop functions
# -----------------------------------------------------------------------------
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, qwk, acc, all_preds, all_labels


def train_model(arch, epochs=15, lr=1e-4, patience=4):
    print(f"\n{'='*60}\nTraining: {arch}\n{'='*60}")
    model = build_model(arch)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_qwk = -1
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_qwk": []}

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_qwk, train_acc, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_qwk, val_acc, _, _ = run_epoch(model, val_loader)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_qwk"].append(val_qwk)

        print(f"Epoch {epoch+1}/{epochs} ({time.time()-t0:.0f}s) | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_QWK={val_qwk:.4f} val_acc={val_acc:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history, best_qwk



# -----------------------------------------------------------------------------
# STEP 8 — Run the 3-way benchmark
# -----------------------------------------------------------------------------
ARCHS = ["efficientnet_b0", "mobilenet_v3_large", "resnet50"]
results = {}
trained_models = {}

for arch in ARCHS:
    model, history, best_val_qwk = train_model(arch, epochs=15)
    trained_models[arch] = model

    test_loss, test_qwk, test_acc, test_preds, test_labels = run_epoch(model, test_loader)

    # Per-class sensitivity/specificity via one-vs-rest confusion
    cm = confusion_matrix(test_labels, test_preds)
    sensitivities, specificities = [], []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn + 1e-9))
        specificities.append(tn / (tn + fp + 1e-9))

    # Referable-DR AUC (binary: class 0-1 = no referral, 2-4 = refer) —
    # the metric that actually matters clinically, per your proposal's
    # "Low/Moderate/Refer to specialist" triage framing
    binary_labels = [1 if l >= 2 else 0 for l in test_labels]
    model.eval()
    all_probs = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            all_probs.extend(probs)
    all_probs = np.array(all_probs)
    referable_probs = all_probs[:, 2:].sum(axis=1)
    referable_auc = roc_auc_score(binary_labels, referable_probs)

    results[arch] = {
        "test_qwk": test_qwk,
        "test_acc": test_acc,
        "referable_dr_auc": referable_auc,
        "mean_sensitivity": np.mean(sensitivities),
        "mean_specificity": np.mean(specificities),
        "per_class_sensitivity": sensitivities,
        "per_class_specificity": specificities,
        "confusion_matrix": cm,
        "num_params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    print(f"\n{arch} TEST RESULTS:")
    print(f"  QWK: {test_qwk:.4f} | Acc: {test_acc:.4f} | "
          f"Referable-DR AUC: {referable_auc:.4f}")
    print(f"  Params: {results[arch]['num_params_millions']:.1f}M")
    print(classification_report(test_labels, test_preds, digits=3))



# -----------------------------------------------------------------------------
# STEP 9 — Comparison summary + plots
# -----------------------------------------------------------------------------
summary_df = pd.DataFrame({
    arch: {
        "Test QWK": results[arch]["test_qwk"],
        "Test Accuracy": results[arch]["test_acc"],
        "Referable-DR AUC": results[arch]["referable_dr_auc"],
        "Mean Sensitivity": results[arch]["mean_sensitivity"],
        "Mean Specificity": results[arch]["mean_specificity"],
        "Params (M)": results[arch]["num_params_millions"],
    }
    for arch in ARCHS
}).T

print("\n" + "="*70)
print("FINAL BENCHMARK COMPARISON")
print("="*70)
print(summary_df.round(4))

best_arch = summary_df["Test QWK"].idxmax()
print(f"\nBest QWK: {best_arch}")
print("Reminder: cross-check this against on-device Core ML latency in Xcode "
      "before finalizing — the QWK winner isn't automatically the ship choice "
      "if it's too slow/large for on-device inference on the target iPhone.")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for arch in ARCHS:
    cm = results[arch]["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ax = axes[ARCHS.index(arch)]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=range(5), yticklabels=range(5))
    ax.set_title(f"{arch}\nQWK={results[arch]['test_qwk']:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

summary_df[["Test QWK", "Referable-DR AUC", "Mean Sensitivity", "Mean Specificity"]].plot(
    kind="bar", figsize=(10, 6), rot=0
)
plt.title("Model Comparison — DR Classification")
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# STEP 10 — Save best model + export for Core ML conversion later
# -----------------------------------------------------------------------------
best_model = trained_models[best_arch]
torch.save(best_model.state_dict(), f"dr_model_{best_arch}_best.pth")
torch.save(best_model, f"dr_model_{best_arch}_full.pth")

# Export a TorchScript trace — this is what you'll hand to coremltools
# in a later step when you move to the Xcode app build
best_model.eval()
example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
traced = torch.jit.trace(best_model, example_input)
traced.save(f"dr_model_{best_arch}_traced.pt")

print(f"\nSaved: dr_model_{best_arch}_best.pth (weights)")
print(f"Saved: dr_model_{best_arch}_traced.pt (TorchScript, for Core ML conversion)")
print(f"Saved: confusion_matrices.png, model_comparison.png")

# =============================================================================
# NEXT STEPS (per your proposal doc's section 10)
# =============================================================================
# 1. Once you have QWK/AUC numbers, decide MobileNetV3 vs EfficientNet-B0
#    for shipping — MobileNetV3 is the safer default for the shared backbone
#    since the other 3 disease heads (glaucoma, AMD, hypertensive) will all
#    hang off the same backbone and need to run on-device together.
# 2. Cross-dataset generalization check: re-run test_loader eval (Step 8-9
#    logic) on Messidor-2 images with your trained model, no retraining —
#    this tells you if you're overfit to APTOS's specific camera/population.
# 3. Image quality gate (Section 6 of your proposal): train a small
#    Laplacian-variance-based blur classifier separately — this is a ~30 line
#    addition, not a CNN, and should run BEFORE this pipeline in the real app.
# 4. Once backbone is picked, this exact script structure (Dataset class +
#    train_model function) is directly reusable for the glaucoma segmentation
#    head — swap in ORIGA/REFUGE and change the head to U-Net.
# 5. Core ML conversion: `pip install coremltools`, then
#    `coremltools.convert(traced_model, ...)` — happens once you're ready to
#    move into Xcode; the .pt file saved above is the handoff point.
# =============================================================================
