// Preprocessing.swift — RetinaScan
// Ben Graham preprocessing, written to reproduce the Python/OpenCV pipeline:
//   crop_black_borders(gray > 7) -> cv2.resize(380x380, INTER_LINEAR)
//   -> cv2.GaussianBlur(sigma = 380/30) -> cv2.addWeighted(img, 4, blur, -4, 128)
// Every OpenCV detail that changes pixel values is reproduced on purpose:
//   * gray uses OpenCV's fixed-point RGB2GRAY coefficients
//   * resize uses OpenCV's pixel-center mapping, NO antialiasing, 11-bit fixed-point weights
//   * blur uses a 77-tap kernel with BORDER_REFLECT_101 and rounds back to uint8
//   * addWeighted is then exact integer math (4*img - 4*blur + 128, clamped)
// Do NOT replace any step with vImage/Core Image scaling: they antialias and will not match.

import Foundation
import Accelerate

enum Preproc {
    static let modelSize = 380
    static let cropTolerance: UInt8 = 7
    static var blurSigma: Double { Double(modelSize) / 30.0 }   // 12.6667

    // MARK: - Gray (OpenCV COLOR_RGB2GRAY, 8-bit fixed point)

    /// OpenCV 4.x uses 14-bit coefficients; OpenCV 5.x switched to 15-bit.
    /// Set this to match the `cv2` version printed by Colab Cell 2 (it only moves ~0.3% of
    /// gray values by 1 level, so the effect on the crop and the quality gate is tiny).
    static let opencvMajorVersion = 4

    @inline(__always)
    static func gray(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> UInt8 {
        if opencvMajorVersion >= 5 {
            return UInt8((Int(r) * 9798 + Int(g) * 19235 + Int(b) * 3735 + 16384) >> 15)
        }
        return UInt8((Int(r) * 4899 + Int(g) * 9617 + Int(b) * 1868 + 8192) >> 14)
    }

    // MARK: - Crop black borders (bounding box of gray > tol)

    static func cropBlackBorders(_ img: RGBImage, tol: UInt8 = cropTolerance) -> RGBImage {
        let w = img.width, h = img.height
        var minX = w, maxX = -1, minY = h, maxY = -1
        img.data.withUnsafeBufferPointer { d in
            for y in 0..<h {
                var rowHit = false
                let base = y * w * 3
                for x in 0..<w {
                    let i = base + x * 3
                    if gray(d[i], d[i + 1], d[i + 2]) > tol {
                        rowHit = true
                        if x < minX { minX = x }
                        if x > maxX { maxX = x }
                    }
                }
                if rowHit {
                    if y < minY { minY = y }
                    maxY = y
                }
            }
        }
        if maxX < 0 { return img }   // nothing above tolerance -> keep full image (same as Python)
        let cw = maxX - minX + 1, ch = maxY - minY + 1
        if cw == w && ch == h { return img }
        var out = [UInt8](repeating: 0, count: cw * ch * 3)
        img.data.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<ch {
                    let srcRow = ((y + minY) * w + minX) * 3
                    let dstRow = y * cw * 3
                    for k in 0..<(cw * 3) { o[dstRow + k] = s[srcRow + k] }
                }
            }
        }
        return RGBImage(width: cw, height: ch, data: out)
    }

    // MARK: - Resize (OpenCV INTER_LINEAR, and INTER_AREA's upscale variant)

    /// `areaMode` = true reproduces OpenCV INTER_AREA when it is NOT a pure downscale.
    static func resizeLinear(_ src: RGBImage, width dw: Int, height dh: Int, areaMode: Bool = false) -> RGBImage {
        let sw = src.width, sh = src.height
        if sw == dw && sh == dh { return src }

        func table(_ dn: Int, _ sn: Int) -> (idx0: [Int], idx1: [Int], c0: [Int], c1: [Int]) {
            let scale = Double(sn) / Double(dn)
            let invScale = Double(dn) / Double(sn)
            var i0 = [Int](repeating: 0, count: dn), i1 = i0, c0 = i0, c1 = i0
            for d in 0..<dn {
                var s: Int
                var f: Float
                if areaMode {
                    s = Int(floor(Double(d) * scale))
                    f = Float(Double(d + 1) - Double(s + 1) * invScale)
                    f = f <= 0 ? 0 : f - floor(f)
                } else {
                    let fx = Float((Double(d) + 0.5) * scale - 0.5)
                    s = Int(floor(fx))
                    f = fx - Float(s)
                }
                if s < 0 { f = 0; s = 0 }
                if s >= sn - 1 { f = 0; s = sn - 1 }
                let w0 = Int(((1 - f) * 2048).rounded(.toNearestOrEven))
                i0[d] = s
                i1[d] = min(s + 1, sn - 1)
                c0[d] = w0
                c1[d] = 2048 - w0
            }
            return (i0, i1, c0, c1)
        }

        let tx = table(dw, sw), ty = table(dh, sh)
        var out = [UInt8](repeating: 0, count: dw * dh * 3)
        var row0 = [Int](repeating: 0, count: dw * 3)
        var row1 = [Int](repeating: 0, count: dw * 3)

        src.data.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                func hpass(_ sy: Int, _ dst: inout [Int]) {
                    let base = sy * sw * 3
                    for dx in 0..<dw {
                        let a = base + tx.idx0[dx] * 3, b = base + tx.idx1[dx] * 3
                        let w0 = tx.c0[dx], w1 = tx.c1[dx]
                        for c in 0..<3 { dst[dx * 3 + c] = Int(s[a + c]) * w0 + Int(s[b + c]) * w1 }
                    }
                }
                for dy in 0..<dh {
                    hpass(ty.idx0[dy], &row0)
                    hpass(ty.idx1[dy], &row1)
                    let b0 = ty.c0[dy], b1 = ty.c1[dy]
                    let ob = dy * dw * 3
                    for k in 0..<(dw * 3) {
                        // OpenCV's vectorised 8-bit path (VResizeLinearVec_32s8u); matches cv2 exactly
                        let v = (((b0 * (row0[k] >> 4)) >> 16) + ((b1 * (row1[k] >> 4)) >> 16) + 2) >> 2
                        o[ob + k] = UInt8(clamping: v)
                    }
                }
            }
        }
        return RGBImage(width: dw, height: dh, data: out)
    }

    // MARK: - Gaussian blur (OpenCV GaussianBlur, ksize from sigma, BORDER_REFLECT_101)

    @inline(__always)
    static func reflect101(_ i: Int, _ n: Int) -> Int {
        if n == 1 { return 0 }
        var i = i
        while i < 0 || i >= n {
            if i < 0 { i = -i }
            if i >= n { i = 2 * n - 2 - i }
        }
        return i
    }

    static func gaussianKernel(sigma: Double) -> [Float] {
        let n = Int((sigma * 6 + 1).rounded(.toNearestOrEven)) | 1        // 77 for sigma 12.667
        let half = Double(n - 1) / 2
        var k = (0..<n).map { i -> Double in
            let x = Double(i) - half
            return exp(-(x * x) / (2 * sigma * sigma))
        }
        let sum = k.reduce(0, +)
        k = k.map { $0 / sum }
        return k.map { Float($0) }
    }

    private static func convolveRows(_ src: [Float], width w: Int, height h: Int, kernel: [Float]) -> [Float] {
        let r = kernel.count / 2
        let padN = w + 2 * r
        let idx = (0..<padN).map { reflect101($0 - r, w) }
        var out = [Float](repeating: 0, count: w * h)
        var padded = [Float](repeating: 0, count: padN)
        src.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                kernel.withUnsafeBufferPointer { k in
                    padded.withUnsafeMutableBufferPointer { p in
                        for y in 0..<h {
                            let row = y * w
                            for i in 0..<padN { p[i] = s[row + idx[i]] }
                            vDSP_conv(p.baseAddress!, 1, k.baseAddress!, 1,
                                      o.baseAddress! + row, 1, vDSP_Length(w), vDSP_Length(kernel.count))
                        }
                    }
                }
            }
        }
        return out
    }

    private static func transpose(_ a: [Float], rows h: Int, cols w: Int) -> [Float] {
        var t = [Float](repeating: 0, count: w * h)
        a.withUnsafeBufferPointer { s in
            t.withUnsafeMutableBufferPointer { d in
                vDSP_mtrans(s.baseAddress!, 1, d.baseAddress!, 1, vDSP_Length(w), vDSP_Length(h))
            }
        }
        return t
    }

    static func gaussianBlur(_ img: RGBImage, sigma: Double) -> RGBImage {
        let w = img.width, h = img.height
        let kernel = gaussianKernel(sigma: sigma)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for c in 0..<3 {
            var plane = [Float](repeating: 0, count: w * h)
            img.data.withUnsafeBufferPointer { d in
                for i in 0..<(w * h) { plane[i] = Float(d[i * 3 + c]) }
            }
            let horiz = convolveRows(plane, width: w, height: h, kernel: kernel)
            let t = transpose(horiz, rows: h, cols: w)                 // now w rows x h cols
            let vert = convolveRows(t, width: h, height: w, kernel: kernel)
            let back = transpose(vert, rows: w, cols: h)
            for i in 0..<(w * h) {
                let v = back[i].rounded(.toNearestOrEven)
                out[i * 3 + c] = UInt8(max(0, min(255, v)))
            }
        }
        return RGBImage(width: w, height: h, data: out)
    }

    // MARK: - Ben Graham

    /// Full model preprocessing. Input: raw RGB (uncompressed). Output: 380x380 RGB for Core ML.
    static func benGraham(_ raw: RGBImage) -> RGBImage {
        let cropped = cropBlackBorders(raw)
        let resized = resizeLinear(cropped, width: modelSize, height: modelSize)
        let blurred = gaussianBlur(resized, sigma: blurSigma)
        var out = [UInt8](repeating: 0, count: resized.data.count)
        resized.data.withUnsafeBufferPointer { a in
            blurred.data.withUnsafeBufferPointer { b in
                for i in 0..<out.count {
                    out[i] = UInt8(clamping: 4 * Int(a[i]) - 4 * Int(b[i]) + 128)
                }
            }
        }
        return RGBImage(width: modelSize, height: modelSize, data: out)
    }
}
