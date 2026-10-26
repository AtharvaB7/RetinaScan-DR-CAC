# =============================================================================
# RetinaScan — Phase 1: Setup, exact split recreation, internal validation,
# operating threshold chosen on the VALIDATION set only.
# Google Colab (T4 GPU). Run cells in order. Artifacts saved to Google Drive.
# Results obtained: test QWK 0.922, referable-DR AUC 0.992 [0.988,0.996],
# sens 0.958 / spec 0.963 at threshold 0.5059 (chosen on val).
# =============================================================================

# %% Cell 1: Mount Drive, create project folders, check GPU
from google.colab import drive
drive.mount('/content/drive')

import os, sys, torch
ROOT = "/content/drive/MyDrive/RetinaScan"
DIRS = {k: os.path.join(ROOT, k) for k in
        ["weights", "splits", "preds", "results", "figures", "thresholds", "handoff"]}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

print("Python:", sys.version.split()[0], "| torch:", torch.__version__,
      "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("Weights folder contents:", os.listdir(DIRS["weights"]))
WEIGHTS_PATH = os.path.join(DIRS["weights"], "dr_model_mobilenet_v3_large_best.pth")
assert os.path.exists(WEIGHTS_PATH), "Upload the _best.pth into MyDrive/RetinaScan/weights/ first"

# %% Cell 2: Installs, imports, seeds, constants
# !pip install -q kagglehub albumentations
import cv2, json, random, numpy as np, pandas as pd, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES, IMG_SIZE, BATCH_SIZE = 5, 380, 32
NUM_WORKERS = min(2, os.cpu_count() or 2)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
print("cv2", cv2.__version__, "| albumentations", A.__version__)

# %% Cell 3: Download APTOS and inspect structure
import kagglehub
DATASET_PATH = kagglehub.dataset_download("mariaherrerot/aptos2019")
print("Path:", DATASET_PATH)
for root, dirs, files in os.walk(DATASET_PATH):
    depth = root.replace(DATASET_PATH, "").count(os.sep)
    if depth <= 2:
        print("  " * depth + os.path.basename(root) + "/", f"[{len(files)} files]", files[:3])
for f in sorted(os.listdir(DATASET_PATH)):
    if f.endswith(".csv"):
        c = pd.read_csv(os.path.join(DATASET_PATH, f))
        print(f"\n{f}: {len(c)} rows, cols={c.columns.tolist()}")
        print(c.iloc[:, -1].value_counts().sort_index().to_dict())

# %% Cell 4: Recreate the exact training split and save it
CSV_PATH = os.path.join(DATASET_PATH, "train_1.csv")
IMG_DIR = os.path.join(DATASET_PATH, "train_images", "train_images")
assert os.path.exists(CSV_PATH) and os.path.isdir(IMG_DIR), "Paths differ — check Cell 3 tree"

df = pd.read_csv(CSV_PATH)
df["filepath"] = df["id_code"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.png"))
n0 = len(df)
df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
print(f"CSV rows: {n0} | with image on disk: {len(df)}")

train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["diagnosis"])
train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
train_df, val_df = train_test_split(train_df, test_size=0.10, random_state=SEED, stratify=train_df["diagnosis"])
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name:5s} n={len(d):4d}  classes={d['diagnosis'].value_counts().sort_index().to_dict()}  "
          f"referable={int((d['diagnosis']>=2).sum())}")

split_path = os.path.join(DIRS["splits"], "aptos_split_seed42.csv")
split_tbl = pd.concat([d.assign(split=s)[["id_code", "diagnosis", "split"]]
                       for s, d in [("train", train_df), ("val", val_df), ("test", test_df)]])
if os.path.exists(split_path):
    same = pd.read_csv(split_path).sort_values("id_code").reset_index(drop=True).equals(
           split_tbl.sort_values("id_code").reset_index(drop=True))
    print("Matches split already saved on Drive:", same)
else:
    split_tbl.to_csv(split_path, index=False); print("Saved", split_path)

# %% Cell 5: Preprocessing, dataset, frozen model (identical to training)
def crop_black_borders(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    return img[coords[0].min():coords[0].max()+1, coords[1].min():coords[1].max()+1]

def ben_graham_preprocess(img, target_size=IMG_SIZE):
    img = crop_black_borders(img)
    img = cv2.resize(img, (target_size, target_size))
    blurred = cv2.GaussianBlur(img, (0, 0), target_size / 30)
    return cv2.addWeighted(img, 4, blurred, -4, 128)

eval_tf = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

class FundusDataset(Dataset):
    """Needs 'filepath'; optional per-image degrade_fn(raw_rgb)->raw_rgb."""
    def __init__(self, dataframe, transform=eval_tf, degrade_fn=None):
        self.df, self.transform, self.degrade_fn = dataframe, transform, degrade_fn
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.df.iloc[idx]["filepath"]), cv2.COLOR_BGR2RGB)
        if self.degrade_fn is not None:
            img = self.degrade_fn(img)
        img = ben_graham_preprocess(img)
        return self.transform(image=img)["image"], idx

def build_mobilenet_v3():
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    return m

model = build_mobilenet_v3()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True), strict=True)
model = model.to(DEVICE).eval()
print("Weights loaded (strict). Params: %.2fM" % (sum(p.numel() for p in model.parameters())/1e6))

@torch.no_grad()
def predict(dataframe, degrade_fn=None):
    loader = DataLoader(FundusDataset(dataframe, degrade_fn=degrade_fn), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    logits = [model(x.to(DEVICE)).float().cpu() for x, _ in tqdm(loader, leave=False)]
    logits = torch.cat(logits)
    return logits.numpy(), torch.softmax(logits, 1).numpy()

# %% Cell 6: Inference on val and test, save per-image predictions
pred_tables = {}
for name, d in [("val", val_df), ("test", test_df)]:
    logits, probs = predict(d)
    t = d[["id_code", "diagnosis"]].copy()
    for c in range(5):
        t[f"logit{c}"], t[f"p{c}"] = logits[:, c], probs[:, c]
    t["p_referable"] = probs[:, 2:].sum(1)
    t["referable"] = (t["diagnosis"] >= 2).astype(int)
    t["pred_grade"] = probs.argmax(1)
    t.to_csv(os.path.join(DIRS["preds"], f"aptos_{name}_preds.csv"), index=False)
    pred_tables[name] = t
    print(f"{name}: {len(t)} predictions saved")

# %% Cell 7: Sanity check vs reported numbers + binary referable metrics with bootstrap CIs
def binary_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b > 0 else np.nan
    return {"auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "sens": sd(tp, tp+fn), "spec": sd(tn, tn+fp), "ppv": sd(tp, tp+fp), "npv": sd(tn, tn+fn),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def bootstrap_ci(y, p, thr, n_boot=2000, seed=SEED):
    rng, n = np.random.default_rng(seed), len(y)
    keys = ["auc", "sens", "spec", "ppv", "npv"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2: continue
        m = binary_metrics(y[i], p[i], thr)
        for k in keys: samples[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in samples.items()}

def fmt(m, ci):
    return "  ".join(f"{k.upper()}={m[k]:.3f} [{ci[k][0]:.3f},{ci[k][1]:.3f}]" for k in ["auc","sens","spec","ppv","npv"])

t = pred_tables["test"]
y5, yh5 = t["diagnosis"].values, t["pred_grade"].values
qwk = cohen_kappa_score(y5, yh5, weights="quadratic")
cm5 = confusion_matrix(y5, yh5, labels=range(5))
per_sens = [cm5[c, c] / cm5[c].sum() for c in range(5)]
yb, pb = t["referable"].values, t["p_referable"].values
auc = roc_auc_score(yb, pb)

REPORTED = {"qwk": 0.92, "auc": 0.99, "mean_sens5": 0.76}
print("=== SANITY vs reported (tolerance ±0.015) ===")
for k, v in [("qwk", qwk), ("auc", auc), ("mean_sens5", np.mean(per_sens))]:
    flag = "OK" if abs(v - REPORTED[k]) <= 0.015 else "MISMATCH"
    print(f"{k:11s} now={v:.4f}  reported≈{REPORTED[k]}  {flag}")
print("5-class confusion (rows=true):\n", cm5)

m05 = binary_metrics(yb, pb, 0.5); ci05 = bootstrap_ci(yb, pb, 0.5)
print("\n=== APTOS TEST, binary referable (grade≥2), thr=0.50 ===")
print(f"n={len(yb)} (referable={yb.sum()})  ", fmt(m05, ci05))
print(f"TP={m05['tp']} FP={m05['fp']} FN={m05['fn']} TN={m05['tn']}")

# %% Cell 8: Choose operating threshold on VAL (sens >= 0.90), evaluate once on TEST
v = pred_tables["val"]
yv, pv = v["referable"].values, v["p_referable"].values
fpr, tpr, thr = roc_curve(yv, pv)
ok = tpr >= 0.90
THR = float(np.max(thr[ok][np.isfinite(thr[ok])]))
mv = binary_metrics(yv, pv, THR)
mt = binary_metrics(yb, pb, THR); cit = bootstrap_ci(yb, pb, THR)

print(f"=== Operating point chosen on VAL (n={len(yv)}, referable={yv.sum()}) ===")
print(f"THR={THR:.4f}  val sens={mv['sens']:.3f} spec={mv['spec']:.3f}  val AUC={roc_auc_score(yv, pv):.3f}")
print("\n=== Applied unchanged to APTOS TEST ===")
print(fmt(mt, cit)); print(f"TP={mt['tp']} FP={mt['fp']} FN={mt['fn']} TN={mt['tn']}")
print(f"H2 (internal part): test sens ≥0.90? {'YES' if mt['sens'] >= 0.90 else 'NO'}")

op = {"referable_threshold": THR, "chosen_on": "APTOS val", "rule": "max threshold with val sens>=0.90",
      "val": mv, "n_val": int(len(yv)), "n_val_referable": int(yv.sum())}
json.dump(op, open(os.path.join(DIRS["thresholds"], "operating_point.json"), "w"), indent=2)
res = {"qwk_5class": qwk, "per_class_sens": per_sens, "test_thr0.5": {**m05, "ci": ci05},
       "test_thrOP": {**mt, "ci": cit}, "threshold": THR}
json.dump(res, open(os.path.join(DIRS["results"], "phase1_aptos.json"), "w"), indent=2, default=float)
print("\nSaved: thresholds/operating_point.json, results/phase1_aptos.json")

# %% Cell 18b: Restore the operating threshold after a runtime restart
THR = json.load(open(os.path.join(DIRS["thresholds"], "operating_point.json")))["referable_threshold"]
print(f"THR = {THR:.4f} | GPU: {torch.cuda.is_available()}")
