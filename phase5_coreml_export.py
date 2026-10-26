# =============================================================================
# RetinaScan — Phase 5: Core ML export, precision check, golden vectors.
# Requires Phase 1 cells 1-5 and 18b.
# ImageNet normalization is built INTO the exported model, so the iOS app feeds the
# Ben Graham output directly as a 380x380 RGB image (0-255).
# fp16 export: argmax agreement 100% but max prob diff 0.017 (> 0.01 target) -> shipped fp32.
# fp32 on-device parity (Xcode unit test, 60 images): argmax 100%, max |Δprob| = 0.00001.
# =============================================================================

# %% Cell 25: Install coremltools
# !pip install -q -U coremltools
import coremltools as ct
print("coremltools", ct.__version__, "| torch", torch.__version__, "| python", sys.version.split()[0])

# %% Cell 26: Wrapper with normalization inside; re-trace on CPU; verify vs original
class RetinaScanCore(nn.Module):
    """Input: (1,3,380,380) RGB float in [0,255] = Ben Graham output. Normalization done inside."""
    def __init__(self, m):
        super().__init__(); self.m = m
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255.0)
    def forward(self, x):
        logits = self.m((x - self.mean) / self.std)
        return logits, torch.softmax(logits, dim=1)

cpu_m = build_mobilenet_v3(); cpu_m.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)); cpu_m.eval()
core = RetinaScanCore(cpu_m).eval()
traced_core = torch.jit.trace(core, torch.rand(1, 3, IMG_SIZE, IMG_SIZE) * 255)

to255 = lambda img: torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
diffs = []
for p in test_df.filepath.sample(8, random_state=SEED):
    bg = ben_graham_preprocess(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        ref = cpu_m(eval_tf(image=bg)["image"].unsqueeze(0))
        lg, _ = traced_core(to255(bg))
    diffs.append((ref - lg).abs().max().item())
print(f"Traced wrapper vs original eager pipeline: max |Δlogit| = {max(diffs):.2e}")

# %% Cell 27: Convert to Core ML (fp16 ML Program) — first attempt
mlmodel = ct.convert(
    traced_core,
    inputs=[ct.ImageType(name="image", shape=(1, 3, IMG_SIZE, IMG_SIZE),
                         color_layout=ct.colorlayout.RGB, scale=1.0, bias=[0.0, 0.0, 0.0])],
    outputs=[ct.TensorType(name="logits"), ct.TensorType(name="probs")],
    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16)
mlmodel.author = "RetinaScan"
mlmodel.short_description = ("Diabetic retinopathy 5-class (ICDR 0-4) classifier, MobileNetV3-Large trained on APTOS 2019. "
                             "Input MUST be Ben Graham-preprocessed 380x380 RGB. Screening aid only, not a diagnosis.")
mlmodel.version = "1.0"
mlmodel.input_description["image"] = "380x380 RGB, Ben Graham preprocessed"
mlmodel.output_description["logits"] = "Raw logits, classes 0-4"
mlmodel.output_description["probs"] = "Softmax probabilities, classes 0-4; P(referable) = p2+p3+p4"
mlmodel.user_defined_metadata.update({"referable_threshold": f"{THR:.6f}", "classes": "0,1,2,3,4",
                                      "trained_on": "APTOS2019 train_1.csv (n=2109)"})
ML_PATH = os.path.join(DIRS["handoff"], "RetinaScanDR.mlpackage")
mlmodel.save(ML_PATH)
print("Saved", ML_PATH)

# %% Cell 27b: FINAL export — fp32 (fp16 exceeded the 0.01 probability tolerance)
import shutil
m = build_mobilenet_v3(); m.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
tr = torch.jit.trace(RetinaScanCore(m.eval()).eval(), torch.rand(1, 3, IMG_SIZE, IMG_SIZE) * 255)
ml = ct.convert(tr, inputs=[ct.ImageType(name="image", shape=(1, 3, IMG_SIZE, IMG_SIZE), color_layout=ct.colorlayout.RGB,
                                         scale=1.0, bias=[0, 0, 0])],
                outputs=[ct.TensorType(name="logits"), ct.TensorType(name="probs")], convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS16)
ml.user_defined_metadata["referable_threshold"] = "0.505898"
shutil.rmtree(ML_PATH, ignore_errors=True); ml.save(ML_PATH); print("Saved fp32", ML_PATH)

# %% Cell 28: fp16 proxy check (PyTorch half vs float on all 586 test images)
class To255:
    def __call__(self, image): return {"image": torch.from_numpy(image).permute(2, 0, 1).float()}

gpu_core = RetinaScanCore(build_mobilenet_v3()).to(DEVICE).eval()
gpu_core.m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
loader = DataLoader(FundusDataset(test_df, transform=To255()), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
P32, P16 = [], []
with torch.no_grad():
    for x, _ in tqdm(loader, leave=False):
        x = x.to(DEVICE)
        P32.append(gpu_core(x)[1].cpu()); P16.append(gpu_core.half()(x.half())[1].float().cpu()); gpu_core.float()
P32, P16 = torch.cat(P32).numpy(), torch.cat(P16).numpy()
r32, r16 = P32[:, 2:].sum(1), P16[:, 2:].sum(1)
h4 = {"n": len(P32), "argmax_agree": float((P32.argmax(1) == P16.argmax(1)).mean()),
      "referable_decision_agree": float(((r32 >= THR) == (r16 >= THR)).mean()),
      "max_abs_prob_diff": float(np.abs(P32 - P16).max()), "max_abs_pref_diff": float(np.abs(r32 - r16).max())}
print("fp16 PROXY:", h4)
json.dump(h4, open(os.path.join(DIRS["results"], "phase5_fp16_proxy.json"), "w"), indent=2)

# %% Cell 29: Golden vectors (60 model-parity PNGs + 10 raw end-to-end cases) + Mac parity script
GV = os.path.join(DIRS["handoff"], "golden_vectors"); MP, E2E = os.path.join(GV, "model_parity"), os.path.join(GV, "end_to_end")
for d in (MP, E2E): os.makedirs(d, exist_ok=True)

FOV_TOL, QG_SIZE = 7, 512
def quality_features(img):   # identical to Phase 3
    c = cv2.resize(crop_black_borders(img), (QG_SIZE, QG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    fov = cv2.erode((gray > FOV_TOL).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if fov.sum() < 1000:
        return dict(lap_var=0.0, glare_frac=0.0, mean_lum=0.0, dark_frac=1.0, fov_frac=float(fov.mean()))
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return dict(lap_var=float(lap[fov].var()), glare_frac=float((c.max(2)[fov] >= 250).mean()),
                mean_lum=float(gray[fov].mean()), dark_frac=float((gray[fov] < 25).mean()), fov_frac=float(fov.mean()))

gv = test_df.groupby("diagnosis", group_keys=False).apply(lambda g: g.sample(12, random_state=SEED)).reset_index(drop=True)
rows = []
for i, r in gv.iterrows():
    raw = cv2.cvtColor(cv2.imread(r.filepath), cv2.COLOR_BGR2RGB)
    bg = ben_graham_preprocess(raw)
    fn = f"gv{i:02d}_{r.id_code}_g{r.diagnosis}.png"
    cv2.imwrite(os.path.join(MP, fn), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
    reread = cv2.cvtColor(cv2.imread(os.path.join(MP, fn)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(reread, bg)
    with torch.no_grad(): lg, pr = traced_core(to255(reread))
    lg, pr = lg[0].numpy(), pr[0].numpy()
    row = {"file": fn, "id_code": r.id_code, "grade": int(r.diagnosis),
           **{f"logit{c}": float(lg[c]) for c in range(5)}, **{f"p{c}": float(pr[c]) for c in range(5)},
           "p_referable": float(pr[2:].sum()), "end_to_end": False}
    if i % 12 < 2:
        shutil.copy(r.filepath, os.path.join(E2E, f"raw_{fn}"))
        cr = crop_black_borders(raw); q = quality_features(raw)
        row.update({"end_to_end": True, "raw_w": raw.shape[1], "raw_h": raw.shape[0],
                    "crop_w": cr.shape[1], "crop_h": cr.shape[0], **{f"q_{k}": v for k, v in q.items()}})
    rows.append(row)
gvdf = pd.DataFrame(rows); gvdf.to_csv(os.path.join(GV, "golden_vectors.csv"), index=False)
print(f"Saved {len(gvdf)} model-parity PNGs, {gvdf.end_to_end.sum()} end-to-end raw cases")
# (parity_check_mac.py is saved separately in this repo under evaluation_colab/)
