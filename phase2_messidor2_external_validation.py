# =============================================================================
# RetinaScan — Phase 2: External validation on Messidor-2 + diagnostics.
# Requires Phase 1 cells 1-5 and 18b in the same runtime.
# Results obtained: Messidor-2 (n=1,744 gradable) AUC 0.844 [0.820,0.865];
# sens 0.414 / spec 0.971 at THR; AUC drop 0.149 -> H1 not supported.
# Diagnosis: grade 3-4 flagged 86-91%, grade 2 only 26%; severe-only AUC 0.980;
# downscaling APTOS to 512px does not reproduce the drop.
# Bonus: never-touched APTOS hold-out (n=732) AUC 0.985.
# =============================================================================

# %% Cell 9: Download Messidor-2 (preprocessed Kaggle copy) and inspect
MESS_SLUG = "mariaherrerot/messidor2preprocess"
MESS_PATH = kagglehub.dataset_download(MESS_SLUG)
print("Path:", MESS_PATH)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
all_imgs, tables = [], []
for root, dirs, files in os.walk(MESS_PATH):
    depth = root.replace(MESS_PATH, "").count(os.sep)
    if depth <= 3:
        print("  " * depth + os.path.basename(root) + "/", f"[{len(files)} files]", files[:3])
    all_imgs += [os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXTS)]
    tables += [os.path.join(root, f) for f in files if f.lower().endswith((".csv", ".xls", ".xlsx"))]

from collections import Counter
print("\nTotal images:", len(all_imgs), "| extensions:", Counter(os.path.splitext(p)[1].lower() for p in all_imgs))
for tpath in tables:
    t = pd.read_csv(tpath) if tpath.endswith(".csv") else pd.read_excel(tpath)
    print(f"\n{os.path.relpath(tpath, MESS_PATH)}: {len(t)} rows, cols={t.columns.tolist()}"); print(t.head(3))

rng = np.random.default_rng(SEED)
for p in rng.choice(all_imgs, 8, replace=False):
    im = cv2.imread(p); h, w = im.shape[:2]
    print(f"{os.path.basename(p):40s} {w}x{h} aspect={w/h:.2f} corner_mean={im[:15, :15].mean():.1f}")

# %% Cell 10: Join official adjudicated labels by filename, keep gradable only
label_csv = next((t for t in tables if "adjudicated_dr_grade" in pd.read_csv(t).columns), None) \
            if any(t.endswith(".csv") for t in tables) else None
if label_csv is None:
    print("No grade CSV in this dataset -> pulling google-brain/messidor2-dr-grades labels")
    gb = kagglehub.dataset_download("google-brain/messidor2-dr-grades")
    label_csv = [os.path.join(r, f) for r, _, fs in os.walk(gb) for f in fs if f.endswith(".csv")][0]
print("Label file:", label_csv)
lab = pd.read_csv(label_csv)
need = ["image_id", "adjudicated_dr_grade", "adjudicated_gradable"]
assert all(c in lab.columns for c in need), lab.columns.tolist()
print(f"rows={len(lab)} | gradable={lab['adjudicated_gradable'].value_counts().to_dict()}")

stem = lambda s: os.path.splitext(os.path.basename(str(s)))[0].lower()
stems = Counter(stem(p) for p in all_imgs)
assert max(stems.values()) == 1, "duplicate image stems — inspect"
img_index = {stem(p): p for p in all_imgs}
lab["filepath"] = lab["image_id"].map(lambda s: img_index.get(stem(s)))
unmatched_lab = lab[lab["filepath"].isna()]["image_id"].tolist()
unmatched_img = sorted(set(img_index) - set(lab["image_id"].map(stem)))
print(f"labels w/o image: {len(unmatched_lab)} {unmatched_lab[:5]} | images w/o label: {len(unmatched_img)} {unmatched_img[:5]}")

mess = lab[(lab["adjudicated_gradable"] == 1) & lab["filepath"].notna()].copy().reset_index(drop=True)
mess["adjudicated_dr_grade"] = mess["adjudicated_dr_grade"].astype(int)
mess["referable"] = (mess["adjudicated_dr_grade"] >= 2).astype(int)
print(f"\nEval set: {len(mess)} gradable images | grades={mess['adjudicated_dr_grade'].value_counts().sort_index().to_dict()}"
      f" | referable={mess['referable'].sum()} ({mess['referable'].mean():.1%})")
assert len(mess) > 1500

# %% Cell 11: Visual check — preprocessing parity Messidor-2 vs APTOS
import matplotlib.pyplot as plt
picks = [("Messidor-2", p) for p in mess.sample(3, random_state=SEED)["filepath"]] + \
        [("APTOS", p) for p in test_df.sample(3, random_state=SEED)["filepath"]]
fig, ax = plt.subplots(2, 6, figsize=(20, 7))
for j, (src, p) in enumerate(picks):
    raw = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    ax[0, j].imshow(raw); ax[0, j].set_title(f"{src} raw {raw.shape[1]}x{raw.shape[0]}", fontsize=9)
    ax[1, j].imshow(ben_graham_preprocess(raw)); ax[1, j].set_title("Ben Graham 380²", fontsize=9)
    for a in ax[:, j]: a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "preproc_check_messidor_vs_aptos.png"), dpi=120); plt.show()

# %% Cell 12: Frozen inference on Messidor-2
logits, probs = predict(mess)
for c in range(5):
    mess[f"logit{c}"], mess[f"p{c}"] = logits[:, c], probs[:, c]
mess["p_referable"] = probs[:, 2:].sum(1)
mess.to_csv(os.path.join(DIRS["preds"], "messidor2_preds.csv"), index=False)
print("Saved messidor2_preds.csv |", len(mess), "rows")

# %% Cell 13: Fair side-by-side comparison (binary referable, same thresholds) + H1/H2
THR = json.load(open(os.path.join(DIRS["thresholds"], "operating_point.json")))["referable_threshold"]
at = pd.read_csv(os.path.join(DIRS["preds"], "aptos_test_preds.csv"))
sets = {"APTOS test (internal)": (at["referable"].values, at["p_referable"].values),
        "Messidor-2 (external)": (mess["referable"].values, mess["p_referable"].values)}

def auc_boot(y, p, n=2000, seed=SEED):
    r, out = np.random.default_rng(seed), []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2: out.append(roc_auc_score(y[i], p[i]))
    return np.array(out)

rows, res2 = [], {}
for name, (y, p) in sets.items():
    for tname, t in [("0.50", 0.5), (f"OP {THR:.3f}", THR)]:
        m, ci = binary_metrics(y, p, t), bootstrap_ci(y, p, t)
        rows.append({"dataset": name, "thr": tname, "n": len(y), "prev": y.mean(),
                     **{k: f"{m[k]:.3f} [{ci[k][0]:.3f},{ci[k][1]:.3f}]" for k in ["auc","sens","spec","ppv","npv"]}})
        res2[f"{name}|{tname}"] = {**m, "ci": ci}
print(pd.DataFrame(rows).to_string(index=False))

ba, bm = auc_boot(*sets["APTOS test (internal)"]), auc_boot(*sets["Messidor-2 (external)"])
k = min(len(ba), len(bm)); d = ba[:k] - bm[:k]
auc_a, auc_m = roc_auc_score(*sets["APTOS test (internal)"]), roc_auc_score(*sets["Messidor-2 (external)"])
drop, dci = auc_a - auc_m, (np.percentile(d, 2.5), np.percentile(d, 97.5))
print(f"\nAUC drop internal→external: {drop:.3f}  95% CI [{dci[0]:.3f},{dci[1]:.3f}]")
print("H1 (drop ≤0.05):", "SUPPORTED" if drop <= 0.05 else ("PARTIAL (0.05–0.10)" if drop <= 0.10 else "NOT SUPPORTED (>0.10)"))
mm = res2[f"Messidor-2 (external)|OP {THR:.3f}"]; ma = res2[f"APTOS test (internal)|OP {THR:.3f}"]
print(f"H2 transfer @THR: sens {ma['sens']:.3f}→{mm['sens']:.3f} (Δ{mm['sens']-ma['sens']:+.3f}), "
      f"spec {ma['spec']:.3f}→{mm['spec']:.3f} (Δ{mm['spec']-ma['spec']:+.3f}) | Messidor sens≥0.90: {'YES' if mm['sens']>=0.90 else 'NO'}")
json.dump({"rows": res2, "auc_drop": drop, "auc_drop_ci": dci, "thr": THR},
          open(os.path.join(DIRS["results"], "phase2_messidor.json"), "w"), indent=2, default=float)

# %% Cell 14: Figures — ROC, confidence histogram, calibration (ECE)
def ece_curve(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(bins):
        s = idx == b
        if s.sum(): conf.append(p[s].mean()); acc.append(y[s].mean()); cnt.append(s.sum())
    conf, acc, cnt = map(np.array, (conf, acc, cnt))
    return float(np.sum(cnt / cnt.sum() * np.abs(conf - acc))), conf, acc

fig, ax = plt.subplots(1, 3, figsize=(19, 5.5)); ece_out = {}
for (name, (y, p)), col in zip(sets.items(), ["steelblue", "firebrick"]):
    f, t_, _ = roc_curve(y, p); ax[0].plot(f, t_, color=col, label=f"{name} AUC={roc_auc_score(y,p):.3f}")
    e, c, a = ece_curve(y, p); ece_out[name] = e
    ax[2].plot(c, a, "o-", color=col, label=f"{name} ECE={e:.3f}")
ax[0].plot([0,1],[0,1],"--",color="gray"); ax[0].set(xlabel="FPR", ylabel="TPR", title="Referable-DR ROC"); ax[0].legend(loc="lower right")
y, p = sets["Messidor-2 (external)"]
ax[1].hist(p[y==0], bins=30, alpha=.6, label="Not referable"); ax[1].hist(p[y==1], bins=30, alpha=.6, label="Referable", color="firebrick")
ax[1].axvline(THR, color="k", ls="--", label=f"THR={THR:.3f}"); ax[1].set(xlabel="P(referable)", title="Messidor-2 confidence"); ax[1].legend()
ax[2].plot([0,1],[0,1],"--",color="gray"); ax[2].set(xlabel="Mean predicted P(referable)", ylabel="Observed fraction referable", title="Calibration"); ax[2].legend()
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "phase2_roc_hist_calibration.png"), dpi=150); plt.show()
print("ECE:", {k: round(v, 4) for k, v in ece_out.items()})

# %% Cell 15: Error analysis — 20 most confident FPs / FNs
ea = os.path.join(DIRS["results"], "error_analysis")
for sub in ["false_positives", "false_negatives"]: os.makedirs(os.path.join(ea, sub), exist_ok=True)
mess["pred"] = (mess["p_referable"] >= THR).astype(int)
fps = mess[(mess.referable == 0) & (mess.pred == 1)].sort_values("p_referable", ascending=False)
fns = mess[(mess.referable == 1) & (mess.pred == 0)].sort_values("p_referable")
for d, sub in [(fps, "false_positives"), (fns, "false_negatives")]:
    for _, r in d.head(20).iterrows():
        img = ben_graham_preprocess(cv2.cvtColor(cv2.imread(r.filepath), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(ea, sub, f"{stem(r.image_id)}_grade{r.adjudicated_dr_grade}_p{r.p_referable:.3f}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"FP={len(fps)} (true grade dist {fps.adjudicated_dr_grade.value_counts().sort_index().to_dict()})")
print(f"FN={len(fns)} (true grade dist {fns.adjudicated_dr_grade.value_counts().sort_index().to_dict()})")
mess.to_csv(os.path.join(DIRS["preds"], "messidor2_preds.csv"), index=False)

# %% Cell 16: Bonus internal check on the 732 never-touched APTOS images
ho = pd.concat([
    pd.read_csv(os.path.join(DATASET_PATH, "valid.csv")).assign(filepath=lambda d: d.id_code.map(
        lambda x: os.path.join(DATASET_PATH, "val_images", "val_images", f"{x}.png"))),
    pd.read_csv(os.path.join(DATASET_PATH, "test.csv")).assign(filepath=lambda d: d.id_code.map(
        lambda x: os.path.join(DATASET_PATH, "test_images", "test_images", f"{x}.png")))], ignore_index=True)
assert ho.filepath.map(os.path.exists).all() and not set(ho.id_code) & set(df.id_code), "overlap or missing files"
lg, pr = predict(ho)
ho["p_referable"], ho["pred_grade"], ho["referable"] = pr[:, 2:].sum(1), pr.argmax(1), (ho.diagnosis >= 2).astype(int)
ho.to_csv(os.path.join(DIRS["preds"], "aptos_holdout732_preds.csv"), index=False)
m, ci = binary_metrics(ho.referable.values, ho.p_referable.values, THR), bootstrap_ci(ho.referable.values, ho.p_referable.values, THR)
print(f"APTOS hold-out n={len(ho)} referable={ho.referable.sum()} | QWK={cohen_kappa_score(ho.diagnosis, ho.pred_grade, weights='quadratic'):.3f}")
print(fmt(m, ci))

# %% Cell 17: Diagnose where misses happen (source, true grade, severe-only AUC)
if "mess" not in globals(): mess = pd.read_csv(os.path.join(DIRS["preds"], "messidor2_preds.csv"))
at = pd.read_csv(os.path.join(DIRS["preds"], "aptos_test_preds.csv"))
mess["source"] = np.where(mess.image_id.str.upper().str.startswith("IM"), "Brest IM* (jpg)", "Messidor-orig 2005-06 (png)")

print("=== AUC by Messidor-2 source ===")
for s, d in mess.groupby("source"):
    print(f"{s:30s} n={len(d):4d} prev={d.referable.mean():.2f} AUC={roc_auc_score(d.referable, d.p_referable):.3f} "
          f"sens@THR={(d[d.referable==1].p_referable>=THR).mean():.3f}")

print("\n=== Per true grade: median P(referable) and fraction flagged @THR ===")
for name, d, gcol in [("APTOS test", at, "diagnosis"), ("Messidor-2", mess, "adjudicated_dr_grade")]:
    print(name)
    for g in range(5):
        s = d[d[gcol] == g]
        print(f"  grade {g}: n={len(s):4d} median_p={s.p_referable.median():.3f} flagged={(s.p_referable>=THR).mean():.3f}")

print("\n=== Severe-only AUC (grades 0-1 vs 3-4) ===")
for name, d, gcol in [("APTOS test", at, "diagnosis"), ("Messidor-2", mess, "adjudicated_dr_grade")]:
    s = d[d[gcol].isin([0, 1, 3, 4])]
    print(f"{name:11s} AUC={roc_auc_score((s[gcol]>=3).astype(int), s.p_referable):.3f}  n={len(s)}")

f, t_, th = roc_curve(mess.referable, mess.p_referable); i = np.argmax(t_ >= 0.90)
print(f"\nDIAGNOSTIC ONLY (tuned on test, not deployable): Messidor thr for sens 0.90 = {th[i]:.4f}, spec there = {1-f[i]:.3f}")

# %% Cell 18: Controlled test — does Messidor-like 512px format reproduce the drop?
def messidor_like(jpeg_q=None):
    def f(img):
        img = cv2.resize(crop_black_borders(img), (512, 512), interpolation=cv2.INTER_AREA)
        if jpeg_q:
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            img = cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
        return img
    return f

ws = [cv2.imread(p).shape[1] for p in test_df.filepath.sample(50, random_state=SEED)]
print(f"APTOS raw width: median={np.median(ws):.0f} min={min(ws)} max={max(ws)}  (Messidor files: 512)")
yb = (test_df.diagnosis >= 2).astype(int).values
ctrl = {}
for name, fn in [("APTOS as-is", None), ("APTOS→512 PNG", messidor_like()), ("APTOS→512 JPEG q90", messidor_like(90))]:
    _, pr = predict(test_df, degrade_fn=fn); p = pr[:, 2:].sum(1)
    ctrl[name] = {"auc": roc_auc_score(yb, p), "sens": (p[yb == 1] >= THR).mean(), "spec": (p[yb == 0] < THR).mean()}
    print(f"{name:20s} AUC={ctrl[name]['auc']:.3f} sens={ctrl[name]['sens']:.3f} spec={ctrl[name]['spec']:.3f}")

def bg_stats(paths):
    a = np.stack([ben_graham_preprocess(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).reshape(-1, 3).astype(float) for p in paths])
    return a.mean((0, 1)).round(1), a.std((0, 1)).round(1)
print("\nPost-Ben-Graham RGB mean/std  APTOS:", bg_stats(test_df.filepath.sample(60, random_state=SEED)),
      "| Messidor:", bg_stats(mess.filepath.sample(60, random_state=SEED)))
json.dump(ctrl, open(os.path.join(DIRS["results"], "phase2_controlled_resolution.json"), "w"), indent=2, default=float)
