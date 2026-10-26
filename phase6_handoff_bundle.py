# =============================================================================
# RetinaScan — Phase 6: Triage cutoffs (chosen on VAL only), specs, handoff zip.
# Results: Moderate cutoff 0.0805 (val sens >= 0.98), Refer cutoff 0.5059.
# Moderate-or-Refer sensitivity: APTOS test 0.992, hold-out 0.990, Messidor-2 0.608.
# =============================================================================

# %% Cell 30: Triage cutoffs + thresholds.json + preprocessing spec + README + zip
import shutil
P = DIRS["preds"]
v  = pd.read_csv(os.path.join(P, "aptos_val_preds.csv"))
te = pd.read_csv(os.path.join(P, "aptos_test_preds.csv"))
ho = pd.read_csv(os.path.join(P, "aptos_holdout732_preds.csv"))
me = pd.read_csv(os.path.join(P, "messidor2_preds.csv"))

fpr, tpr, th = roc_curve(v.referable, v.p_referable)
ok = (tpr >= 0.98) & np.isfinite(th)
T_LOW = float(np.max(th[ok]))
print(f"T_LOW (val sens>=0.98) = {T_LOW:.4f} | THR (Refer) = {THR:.4f}")

def bands(d):
    y, p = d.referable.values, d.p_referable.values
    low, mod, ref = p < T_LOW, (p >= T_LOW) & (p < THR), p >= THR
    return {"n": len(d), "pct_low": low.mean(), "pct_moderate": mod.mean(), "pct_refer": ref.mean(),
            "sens_refer_only": ref[y == 1].mean(), "sens_moderate_or_refer": (~low)[y == 1].mean(),
            "nonreferable_in_low": low[y == 0].mean()}
band_res = {k: bands(d) for k, d in [("APTOS_val", v), ("APTOS_test", te), ("APTOS_holdout732", ho), ("Messidor2_external", me)]}
print(pd.DataFrame(band_res).T.round(3).to_string())

thresholds = {
  "version": 1,
  "score": "P(referable) = probs[2] + probs[3] + probs[4]",
  "cutoffs": {"moderate_min": T_LOW, "refer_min": THR},
  "cutoff_selection": "Both chosen on APTOS validation set only (refer: max threshold with val sens>=0.90; moderate: val sens>=0.98). Never tuned on test or Messidor-2.",
  "triage": {
    "low": {"label": "Low", "color": "green",
            "message": "No signs that need a specialist referral were found in this image. Keep your routine yearly eye exam."},
    "moderate": {"label": "Moderate", "color": "amber",
            "message": "Borderline result. Possible early changes the screen can't rule out. Schedule an eye exam within the next few months."},
    "refer": {"label": "Refer", "color": "red",
            "message": "Signs of diabetic retinopathy that should be checked by an eye doctor soon. Please make an appointment."}},
  "score_display_note": "Show as 'model score', not as a calibrated probability: calibration degraded on external data (Messidor-2 ECE 0.132).",
  "disclaimer": "RetinaScan is a screening aid, not a diagnosis. It has not been validated on smartphone-lens images. A Low result does not rule out eye disease.",
  "validation_summary": band_res,
}
json.dump(thresholds, open(os.path.join(DIRS["handoff"], "thresholds.json"), "w"), indent=2, default=float)
shutil.copy(os.path.join(DIRS["thresholds"], "quality_gate.json"), os.path.join(DIRS["handoff"], "quality_gate.json"))

# preprocessing_spec.md and README.md were also written here (see docs/ in this repo).
zip_path = shutil.make_archive(os.path.join(ROOT, "RetinaScan_handoff"), "zip", DIRS["handoff"])
print(f"Zipped -> {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
