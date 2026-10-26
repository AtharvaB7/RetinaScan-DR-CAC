# =============================================================================
# RetinaScan — Phase 4: Image quality gate (analysis only, no new inference).
# Uses robustness_long.csv from Phase 3. Test images split 50/50 by image:
# half A fits thresholds (each check rejects <=2% of clean images), half B evaluates.
# Results (half B, pooled, excluding JPEG): no gate AUC 0.972 / sens 0.896 / spec 0.928
# -> gated AUC 0.991 / sens 0.943 / spec 0.964 (recovers 88% of AUC gap).
# Thresholds: lap_var_min 52.90, glare_frac_max 0.0364, mean_lum_min 38.03.
# =============================================================================

# %% Cell 22: Fit gate thresholds on half A
rob = pd.read_csv(ROB_CSV)
rob["param"] = rob["param"].fillna("clean").astype(str)
rob = rob.drop_duplicates(["id_code", "degradation", "severity"]).reset_index(drop=True)
ids = np.sort(rob.id_code.unique())
A_ids = set(np.random.default_rng(SEED).choice(ids, len(ids) // 2, replace=False))
rob["half"] = np.where(rob.id_code.isin(A_ids), "A_fit", "B_eval")

clean_A = rob[(rob.degradation == "none") & (rob.half == "A_fit")]
print(f"Clean half-A n={len(clean_A)} | feature percentiles:")
print(clean_A[["lap_var", "glare_frac", "mean_lum", "dark_frac"]].quantile([.01, .02, .05, .5, .95, .98, .99]).round(4))

CLEAN_REJECT = 0.02
T = {"lap_var_min":    float(np.quantile(clean_A.lap_var, CLEAN_REJECT)),
     "glare_frac_max": float(np.quantile(clean_A.glare_frac, 1 - CLEAN_REJECT)),
     "mean_lum_min":   float(np.quantile(clean_A.mean_lum, CLEAN_REJECT))}
print("\nGate thresholds:", {k: round(v, 4) for k, v in T.items()})

def gate(d, T):
    """Priority if several fail: blurry > too_dark > glare."""
    r = pd.Series("accept", index=d.index)
    r[d.glare_frac > T["glare_frac_max"]] = "glare"
    r[d.mean_lum < T["mean_lum_min"]] = "too_dark"
    r[d.lap_var < T["lap_var_min"]] = "blurry"
    return r

rob["gate"] = gate(rob, T); rob["accepted"] = rob.gate == "accept"
print("\nClean-image reject rate, half A:", f"{1 - rob[(rob.degradation=='none')&(rob.half=='A_fit')].accepted.mean():.3f}")

# %% Cell 23: Evaluate the gate on half B (H3 part 2)
def mets(d):
    y, p = d.referable.values, d.p_referable.values
    if len(d) == 0 or len(np.unique(y)) < 2: return dict(AUC=np.nan, sens=np.nan, spec=np.nan, n=len(d))
    return dict(AUC=roc_auc_score(y, p), sens=(p[y==1] >= THR).mean(), spec=(p[y==0] < THR).mean(), n=len(d))

B = rob[rob.half == "B_eval"]
rows = []
for (deg, sev, prm), d in B.groupby(["degradation", "severity", "param"]):
    a, g = mets(d), mets(d[d.accepted])
    rows.append({"degradation": deg, "sev": sev, "param": prm, "accept_rate": d.accepted.mean(),
                 "top_reason": d[~d.accepted].gate.mode().iat[0] if (~d.accepted).any() else "-",
                 "AUC_all": a["AUC"], "sens_all": a["sens"], "spec_all": a["spec"],
                 "AUC_acc": g["AUC"], "sens_acc": g["sens"], "spec_acc": g["spec"], "n_acc": g["n"]})
gt = pd.DataFrame(rows).sort_values(["degradation", "sev"])
print(gt.round(3).to_string(index=False))
gt.to_csv(os.path.join(DIRS["results"], "phase4_gate_by_condition.csv"), index=False)

def boot_pair(d, n=1000, seed=SEED):
    """Image-clustered bootstrap of sens/spec before vs after the gate."""
    r, ids_ = np.random.default_rng(seed), d.id_code.unique()
    grp = {i: g for i, g in d.groupby("id_code")}; out = []
    for _ in range(n):
        s = pd.concat([grp[i] for i in r.choice(ids_, len(ids_))])
        out.append([mets(s)["sens"], mets(s)["spec"], mets(s[s.accepted])["sens"], mets(s[s.accepted])["spec"]])
    return np.nanpercentile(np.array(out), [2.5, 97.5], axis=0)

clean = mets(B[B.degradation == "none"])
print(f"\nClean half-B reference: AUC={clean['AUC']:.3f} sens={clean['sens']:.3f} spec={clean['spec']:.3f}")
summary = {}
for label, d in [("All conditions pooled", B), ("Pooled, excl. JPEG (prevented by design)", B[B.degradation != "jpeg"])]:
    a, g = mets(d), mets(d[d.accepted]); ci = boot_pair(d)
    rec = lambda m: (g[m] - a[m]) / (clean[m] - a[m]) if clean[m] != a[m] else np.nan
    print(f"\n{label}: accept={d.accepted.mean():.3f}")
    print(f"  no gate : AUC={a['AUC']:.3f} sens={a['sens']:.3f} [{ci[0,0]:.3f},{ci[1,0]:.3f}] spec={a['spec']:.3f} [{ci[0,1]:.3f},{ci[1,1]:.3f}]")
    print(f"  gated   : AUC={g['AUC']:.3f} sens={g['sens']:.3f} [{ci[0,2]:.3f},{ci[1,2]:.3f}] spec={g['spec']:.3f} [{ci[0,3]:.3f},{ci[1,3]:.3f}]")
    print(f"  fraction of gap to clean recovered: AUC {rec('AUC'):.2f}, sens {rec('sens'):.2f}, spec {rec('spec'):.2f}")
    summary[label] = {"accept_rate": d.accepted.mean(), "no_gate": a, "gated": g, "ci_95": ci.tolist()}
json.dump({"thresholds": T, "clean_ref_halfB": clean, "pooled": summary},
          open(os.path.join(DIRS["results"], "phase4_gate.json"), "w"), indent=2, default=float)

# %% Cell 24: Gate-effect figure + quality_gate.json for the app
d = gt[gt.degradation != "none"].copy(); d["cond"] = d.degradation + "\n" + d.param
fig, ax = plt.subplots(2, 1, figsize=(17, 9), sharex=True)
x = np.arange(len(d))
ax[0].bar(x - .2, d.sens_all, .4, label="All captures"); ax[0].bar(x + .2, d.sens_acc, .4, label="Accepted by gate")
ax[0].axhline(clean["sens"], color="k", ls="--", lw=1, label="Clean reference"); ax[0].set(ylabel=f"Sensitivity @ {THR:.3f}", ylim=(0, 1.05)); ax[0].legend(fontsize=8)
ax[1].bar(x, d.accept_rate, color="gray"); ax[1].set(ylabel="Accept rate", ylim=(0, 1.05))
ax[1].set_xticks(x); ax[1].set_xticklabels(d.cond, rotation=60, fontsize=7)
fig.suptitle("Quality gate effect on APTOS test half B (thresholds fit on half A)")
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "phase4_gate_effect.png"), dpi=150); plt.show()

qg = {
  "version": 1, "status": "PROVISIONAL — fit on simulated degradations of clinical-camera images; recalibrate on real 20D-lens phone captures",
  "thresholds": T, "clean_reject_target_per_check": CLEAN_REJECT,
  "feature_spec": {
    "input": "RGB uint8 frame, BEFORE Ben Graham preprocessing, from the uncompressed camera buffer (never a saved JPEG/HEIC)",
    "step1_crop": "gray = OpenCV RGB2GRAY; keep bounding box of pixels with gray > 7",
    "step2_resize": "512x512, INTER_AREA (area averaging)",
    "step3_fov_mask": "gray(512) > 7, then erode with 15x15 rectangular kernel; if mask has < 1000 px -> reject as 'no_fundus'",
    "lap_var": "variance over FOV pixels of cv2.Laplacian(gray, CV_64F, ksize=3) (kernel [[2,0,2],[0,-8,0],[2,0,2]]), border = reflect-101",
    "glare_frac": "fraction of FOV pixels where max(R,G,B) >= 250",
    "mean_lum": "mean gray over FOV pixels"},
  "rule_order": ["no_fundus", "blurry", "too_dark", "glare"],
  "user_messages": {
    "no_fundus": "Can't see the retina. Center the lens over the pupil and move closer.",
    "blurry": "Image is blurry. Hold steady and let the camera refocus.",
    "too_dark": "Image is too dark. Make sure the flashlight is on and the lens is aligned.",
    "glare": "Too much glare. Tilt the lens slightly to move the reflection off the retina."},
  "known_gap": "JPEG compression degrades sensitivity but is invisible to these checks; prevented by running inference on the raw camera buffer."
}
json.dump(qg, open(os.path.join(DIRS["thresholds"], "quality_gate.json"), "w"), indent=2)
print("Saved thresholds/quality_gate.json")
