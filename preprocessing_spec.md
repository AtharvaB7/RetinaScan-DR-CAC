# RetinaScan preprocessing spec (Python ⇄ Swift)

Order: camera RGB uint8 (uncompressed, never JPEG/HEIC) → quality gate → Ben Graham → Core ML.

## Ben Graham (model input)
1. Gray: OpenCV RGB2GRAY 8-bit fixed point (4.x: 14-bit coefficients 4899/9617/1868; 5.x: 15-bit 9798/19235/3735).
2. Crop: keep the bounding box of pixels with gray > 7 (keep the full image if none).
3. Resize to 380×380, OpenCV INTER_LINEAR: half-pixel centers, 11-bit fixed-point weights, NO antialiasing.
   (vImage / Core Image scaling antialias and do NOT match.)
4. Gaussian blur: sigma = 380/30, kernel 77 taps, separable, border REFLECT_101.
5. out = clamp(4·img − 4·blur + 128, 0, 255).
6. Feed the 380×380 RGB result to the Core ML input `image` (0–255). ImageNet normalization is inside the model.

## Quality gate features
crop → resize 512×512 INTER_AREA → gray → FOV = gray>7 eroded by 15×15 →
lap_var = variance of cv2.Laplacian(ksize=3) = kernel [[2,0,2],[0,-8,0],[2,0,2]] over FOV;
glare_frac = fraction of FOV px with max(R,G,B) ≥ 250; mean_lum = mean gray over FOV.

## Verified parity
- `tools/swift_port_vs_opencv_check.py`: Python port of the Swift math vs OpenCV → gray, crop, resize, area-resize bit-exact; blur ±1 on ~2% of pixels.
- Xcode `GoldenVectorTests`: end-to-end mean pixel diff 0.03–0.06, quality features identical, Core ML (fp32) vs PyTorch max |Δprob| 0.00001.
