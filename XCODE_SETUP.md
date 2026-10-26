# RetinaScan — Xcode setup (about 30 minutes)

You need: a Mac with a recent Xcode, an iPhone on iOS 17+, a USB cable, your Apple ID.
From Google Drive download `RetinaScan/RetinaScan_handoff.zip` and unzip it next to this folder.

## 1. Create the project
1. Xcode → File → New → Project → **iOS → App**.
2. Product Name **RetinaScan** (exact — the tests use `@testable import RetinaScan`), Interface **SwiftUI**,
   Language **Swift**, Testing System **XCTest**, Storage **None**. Tick "Include Tests".
3. Select the project → target **RetinaScan** → General → Minimum Deployments: **iOS 17.0**.
4. Build Settings (search each name, target RetinaScan AND RetinaScanTests):
   - **Swift Language Version** → Swift 5
   - **Default Actor Isolation** → nonisolated (only present in newer Xcode; skip if missing)
   - **Approachable Concurrency** → No (skip if missing)

## 2. Add files (drag from Finder into the Xcode navigator)
| Drag this | Into group | Options in the dialog |
|---|---|---|
| everything in `App/` | RetinaScan | Copy items ✔, target **RetinaScan** only. Replace the template's `ContentView.swift` (delete the old one first). Keep `RetinaScanApp.swift`. |
| `RetinaScanDR.mlpackage` (from the handoff zip) | RetinaScan | Copy ✔, target **RetinaScan** |
| `thresholds.json`, `quality_gate.json` (handoff zip) | RetinaScan | Copy ✔, target **RetinaScan** |
| `Tests/GoldenVectorTests.swift` | RetinaScanTests | Copy ✔, target **RetinaScanTests** only. Delete the template test file. |
| `golden_vectors/` folder (handoff zip) | RetinaScanTests | Copy ✔, **Create folder references** (blue folder), target **RetinaScanTests** only |

Click `RetinaScanDR.mlpackage` in Xcode: you should see input `image` (Image 380×380) and outputs `logits`, `probs`.

## 3. Camera permission
Target RetinaScan → Info → add key **Privacy - Camera Usage Description** =
"RetinaScan uses the camera to capture images of the retina through a condensing lens."

## 4. Run the tests FIRST (Cmd+U, any iPhone simulator)
- `testConfigLoaded` — JSONs are in the app target.
- `testEndToEnd` — Swift preprocessing matches Python. The console prints a line per image
  (`E2E ... px mean ...`). Expect pixel mean ≈ 0.1 and score diff < 0.01.
- `testModelParity` — the real H4 test (Core ML fp16 vs PyTorch). Console prints `H4 [...] argmax_agree=...`.
  Copy those two H4 lines into chat; they are your H4 result.

If `testModelParity` fails only on `max_abs_prob_diff` (fp16 rounding), run Colab Cell 27b (fp32 model),
replace the .mlpackage, and rerun. If `testEndToEnd` fails, paste the E2E lines into chat.

## 5. Run on your iPhone
1. Plug in the phone, pick it as the run destination.
2. Target → Signing & Capabilities → Team = your Apple ID (Personal Team). Change the bundle ID if Xcode complains.
3. On the phone: Settings → Privacy & Security → **Developer Mode** on (restarts). After first install:
   Settings → General → VPN & Device Management → trust your developer certificate.
4. Run (Cmd+R). For a Release-speed build: Product → Scheme → Edit Scheme → Run → Build Configuration: Release.

## 6. Demo tips
- "Analyze a photo" works on the Simulator too. AirDrop the raw PNGs in `golden_vectors/end_to_end/` to your phone
  and save them to Photos: they cover grades 0–4, so you can show Low / Moderate / Refer on real fundus images.
- "Scan with lens": align the retina inside the circle, wait for the green hint, tap Scan, hold 3 s.
  Lower the flashlight slider if you see glare.
- Only the circle region is analyzed (everything outside is blacked out), so the face/room around the lens is ignored.

## Notes / known deviations
- `quality_gate.json` describes the Laplacian as [[0,1,0],[1,-4,1],[0,1,0]]; the Python code actually used OpenCV
  `ksize=3`, i.e. [[2,0,2],[0,-8,0],[2,0,2]]. The thresholds are correct; the Swift code uses the real kernel.
- `Preprocessing.swift` → `opencvMajorVersion`: set to 5 if Colab Cell 2 printed `cv2 5.x` (4 by default).
- Swift preprocessing was checked against OpenCV numerically: gray, crop, resize and area-resize are bit-exact;
  the Gaussian blur differs by ±1 on ~2% of pixels, which becomes ±4 after Ben Graham's ×4 (mean diff ≈ 0.1).
- Live capture is not validated clinically; library photos (JPEG/HEIC) are compressed, which Phase 3 showed can lower
  sensitivity. The lens path uses uncompressed camera frames.
