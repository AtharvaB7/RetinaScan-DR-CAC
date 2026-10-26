# Run on a Mac (Python 3.10-3.12): pip install coremltools pillow pandas numpy
# python parity_check_mac.py   (from inside the handoff folder)
# Checks Core ML outputs against the PyTorch "golden vectors" (hypothesis H4).
import numpy as np, pandas as pd, coremltools as ct
from PIL import Image
gv = pd.read_csv("golden_vectors/golden_vectors.csv")
for cu in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.ALL]:
    m = ct.models.MLModel("RetinaScanDR.mlpackage", compute_units=cu)
    P = np.stack([m.predict({"image": Image.open(f"golden_vectors/model_parity/{f}").convert("RGB")})["probs"].reshape(-1)
                  for f in gv.file])
    R = gv[[f"p{c}" for c in range(5)]].values
    thr = float(m.user_defined_metadata["referable_threshold"])
    print(f"{cu}: n={len(gv)} argmax_agree={(P.argmax(1)==R.argmax(1)).mean():.4f} "
          f"referable_agree={((P[:,2:].sum(1)>=thr)==(R[:,2:].sum(1)>=thr)).mean():.4f} "
          f"max_abs_prob_diff={np.abs(P-R).max():.5f}  -> H4 {'PASS' if (P.argmax(1)==R.argmax(1)).mean()>=.99 and np.abs(P-R).max()<.01 else 'FAIL'}")
