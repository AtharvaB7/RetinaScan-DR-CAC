# =============================================================================
# RetinaScan — Phase 3: Robustness to simulated phone-camera degradations.
# 7 degradations x 3 severities on the APTOS test set (n=586), applied to the RAW
# image before Ben Graham preprocessing. Also records image-quality features
# for every degraded image (used by Phase 4's quality gate).
# Requires Phase 1 cells 1-5 and 18b.
# Worst AUC at max severity: motion blur 0.908, JPEG q10 0.936, low light 0.941
# (clean 0.992). Sensitivity is hit harder than AUC (JPEG q10 sens 0.406).
# =============================================================================

# %% Cell 19: Degradation functions + quality features + preview
import matplotlib.pyplot as plt
FOV_TOL, QG_SIZE = 7, 512
QKEYS = ["lap_var", "glare_frac", "mean_lum", "dark_frac", "fov_frac"]

def fov_mask(img):
    return (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > FOV_TOL).astype(np.uint8)

def d_gauss_blur(img, s, rng):            # s = sigma in px at the final 380px scale
    return cv2.GaussianBlur(img, (0, 0), s * img.shape[1] / IMG_SIZE)

def d_motion_blur(img, s, rng):           # s = streak length in px at 380 scale, random angle
    L = max(3, int(round(s * img.shape[1] / IMG_SIZE)) | 1)
    ker = np.zeros((L, L), np.float32); ker[L // 2, :] = 1
    ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D(((L-1)/2, (L-1)/2), rng.uniform(0, 180), 1.0), (L, L))
    return cv2.filter2D(img, -1, ker / ker.sum())

def d_glare(img, s, rng):                 # s = blob radius as fraction of FOV width; 1-2 saturated blobs
    m = fov_mask(img); ys, xs = np.nonzero(m); h, w = m.shape
    R = s * (xs.max() - xs.min() + 1); yy, xx = np.mgrid[0:h, 0:w]
    out = img.astype(np.float32)
    for _ in range(rng.integers(1, 3)):
        j = rng.integers(len(xs))
        out += 255 * 1.5 * np.exp(-((yy - ys[j])**2 + (xx - xs[j])**2) / (2 * (R / 2)**2))[..., None]
    return np.where(m[..., None] == 1, np.clip(out, 0, 255), img).astype(np.uint8)

def d_vignette(img, s, rng):              # s = falloff strength
    h, w = img.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - h/2) / (h/2))**2 + ((xx - w/2) / (w/2))**2
    return (img * np.clip(1 - s * r2, 0, 1)[..., None]).astype(np.uint8)

def d_low_light(img, prm, rng):           # prm = (gain, sensor-noise sd) inside FOV
    g, sd = prm; m = fov_mask(img)
    out = img.astype(np.float32) * g + rng.normal(0, sd, img.shape)
    return np.where(m[..., None] == 1, np.clip(out, 0, 255), 0).astype(np.uint8)

def d_jpeg(img, q, rng):
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)

def d_downscale(img, tw, rng):            # tw = target width, then upscale back
    h, w = img.shape[:2]
    small = cv2.resize(img, (tw, int(round(h * tw / w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

DEGS = {
    "gaussian_blur": (d_gauss_blur, [0.75, 1.5, 3.0]),
    "motion_blur":   (d_motion_blur, [3, 7, 15]),
    "glare":         (d_glare, [0.05, 0.10, 0.18]),
    "vignetting":    (d_vignette, [0.3, 0.5, 0.75]),
    "low_light":     (d_low_light, [(0.6, 2), (0.4, 4), (0.25, 6)]),
    "jpeg":          (d_jpeg, [50, 25, 10]),
    "downscale":     (d_downscale, [300, 200, 128]),
}

def quality_features(img):
    """crop -> 512x512 (INTER_AREA) -> gray. The iOS app (QualityGate.swift) replicates this."""
    c = cv2.resize(crop_black_borders(img), (QG_SIZE, QG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    fov = cv2.erode((gray > FOV_TOL).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if fov.sum() < 1000:
        return dict(lap_var=0.0, glare_frac=0.0, mean_lum=0.0, dark_frac=1.0, fov_frac=float(fov.mean()))
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return dict(lap_var=float(lap[fov].var()), glare_frac=float((c.max(2)[fov] >= 250).mean()),
                mean_lum=float(gray[fov].mean()), dark_frac=float((gray[fov] < 25).mean()), fov_frac=float(fov.mean()))

p0 = test_df[test_df.diagnosis == 2].iloc[0].filepath
raw0 = cv2.cvtColor(cv2.imread(p0), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(len(DEGS), 4, figsize=(12, 3 * len(DEGS)))
for r, (name, (fn, prms)) in enumerate(DEGS.items()):
    for c_, prm in enumerate([None] + prms):
        im = raw0 if prm is None else fn(raw0, prm, np.random.default_rng(SEED))
        q = quality_features(im)
        ax[r, c_].imshow(cv2.resize(crop_black_borders(im), (256, 256)))
        ax[r, c_].set_title(f"{name} {prm}\nlap={q['lap_var']:.0f} glare={q['glare_frac']:.3f} lum={q['mean_lum']:.0f}", fontsize=7)
        ax[r, c_].axis("off")
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "phase3_degradation_preview.png"), dpi=110); plt.show()

# %% Cell 20: Sweep (baseline + 7 degradations x 3 severities), resumable, saved per condition
class RobustDS(Dataset):
    def __init__(self, d, fn=None, prm=None): self.d, self.fn, self.prm = d, fn, prm
    def __len__(self): return len(self.d)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.d.iloc[idx].filepath), cv2.COLOR_BGR2RGB)
        if self.fn is not None:
            img = self.fn(img, self.prm, np.random.default_rng(SEED * 100003 + idx))
        q = quality_features(img)
        x = eval_tf(image=ben_graham_preprocess(img))["image"]
        return x, torch.tensor([q[k] for k in QKEYS], dtype=torch.float64), idx

ROB_CSV = os.path.join(DIRS["preds"], "robustness_long.csv")
done = set(pd.read_csv(ROB_CSV)[["degradation", "severity"]].itertuples(index=False, name=None)) if os.path.exists(ROB_CSV) else set()
conds = [("none", 0, None, None)] + [(n, i + 1, fn, prm) for n, (fn, ps) in DEGS.items() for i, prm in enumerate(ps)]

for name, sev, fn, prm in conds:
    if (name, sev) in done:
        print("skip", name, sev); continue
    loader = DataLoader(RobustDS(test_df, fn, prm), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    P, Q, I = [], [], []
    with torch.no_grad():
        for x, q, idx in tqdm(loader, desc=f"{name}/{sev}", leave=False):
            P.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy()); Q.append(q.numpy()); I.append(idx.numpy())
    P, Q, I = np.concatenate(P), np.concatenate(Q), np.concatenate(I)
    out = pd.DataFrame({"id_code": test_df.id_code.values[I], "diagnosis": test_df.diagnosis.values[I],
                        "degradation": name, "severity": sev, "param": str(prm),
                        "p_referable": P[:, 2:].sum(1), "pred_grade": P.argmax(1)})
    for j, k in enumerate(QKEYS): out[k] = Q[:, j]
    out["referable"] = (out.diagnosis >= 2).astype(int)
    out.to_csv(ROB_CSV, mode="a", header=not os.path.exists(ROB_CSV), index=False)
    print(f"{name:14s} sev{sev} AUC={roc_auc_score(out.referable, out.p_referable):.3f} "
          f"sens={(out[out.referable==1].p_referable>=THR).mean():.3f} spec={(out[out.referable==0].p_referable<THR).mean():.3f}")

# %% Cell 21R: Light restore for analysis-only cells (no dataset download needed)
# THR = json.load(open(os.path.join(DIRS["thresholds"], "operating_point.json")))["referable_threshold"]
# ROB_CSV = os.path.join(DIRS["preds"], "robustness_long.csv")
# DEGS = {k: None for k in ["gaussian_blur", "motion_blur", "glare", "vignetting", "low_light", "jpeg", "downscale"]}
# test_df = pd.read_csv(os.path.join(DIRS["splits"], "aptos_split_seed42.csv")).query("split == 'test'")

# %% Cell 21: Summary table and robustness curves (fixed: keep the clean baseline row)
rob = pd.read_csv(ROB_CSV)
rob["param"] = rob["param"].fillna("clean").astype(str)

def summ(d):
    y, p = d.referable.values, d.p_referable.values
    return pd.Series({"AUC": roc_auc_score(y, p), "sens": (p[y==1] >= THR).mean(), "spec": (p[y==0] < THR).mean(),
                      "lap_var_med": d.lap_var.median(), "glare_med": d.glare_frac.median(), "lum_med": d.mean_lum.median()})

tbl = (rob.groupby(["degradation", "severity", "param"], dropna=False)[["referable", "p_referable", "lap_var", "glare_frac", "mean_lum"]]
          .apply(summ).reset_index())
print(tbl.round(3).to_string(index=False))
tbl.to_csv(os.path.join(DIRS["results"], "phase3_robustness_summary.csv"), index=False)

base = tbl[tbl.degradation == "none"].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for name in DEGS:
    t = tbl[tbl.degradation == name].sort_values("severity")
    for a, m in zip(ax, ["AUC", "sens"]):
        a.plot([0] + t.severity.tolist(), [base[m]] + t[m].tolist(), "o-", label=name)
for a, m, ttl in zip(ax, ["AUC", "sens"], ["Referable-DR AUC", f"Sensitivity @ THR={THR:.3f}"]):
    a.set(xlabel="Severity (0 = clean)", ylabel=m, title=f"{ttl} vs phone-style degradation (APTOS test, n={len(test_df)})")
    a.set_xticks([0, 1, 2, 3]); a.grid(alpha=.3)
ax[1].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "phase3_robustness_curves.png"), dpi=150); plt.show()

worst = tbl[tbl.severity == 3].sort_values("AUC")[["degradation", "AUC", "sens", "spec"]]
print(f"\nClean baseline: AUC={base.AUC:.3f} sens={base.sens:.3f} spec={base.spec:.3f}")
print("H3 (part 1): worst AUC drops at max severity:\n", worst.assign(dAUC=worst.AUC - base.AUC).round(3).to_string(index=False))
