// QualityGate.swift — RetinaScan
// Reproduces Python quality_features() from the Colab notebook (Cell 19):
//   crop_black_borders -> cv2.resize(512x512, INTER_AREA) -> gray
//   FOV = erode(gray > 7, 15x15 rect)
//   lap_var   = variance over FOV of cv2.Laplacian(gray, CV_64F, ksize=3)
//               NOTE: OpenCV ksize=3 kernel is [[2,0,2],[0,-8,0],[2,0,2]]
//               (the [[0,1,0],[1,-4,1],[0,1,0]] text in quality_gate.json is wrong; thresholds are right)
//   glare_frac = fraction of FOV px with max(R,G,B) >= 250
//   mean_lum   = mean gray over FOV
// Then applies the thresholds from quality_gate.json in rule order:
//   no_fundus -> blurry -> too_dark -> glare.

import Foundation

struct QualityFeatures: Hashable {
    var lapVar: Double
    var glareFrac: Double
    var meanLum: Double
    var darkFrac: Double
    var fovFrac: Double
    var fovPixels: Int
}

enum QualityVerdict: String {
    case accept
    case noFundus = "no_fundus"
    case blurry
    case tooDark = "too_dark"
    case glare
}

enum QualityGate {
    static let size = 512
    static let erodeRadius = 7          // 15x15 kernel
    static let minFovPixels = 1000

    // MARK: - INTER_AREA resize

    private struct Tap { let src: Int; let w: Float }

    private static func areaTable(dst dn: Int, src sn: Int) -> [[Tap]] {
        let scale = Double(sn) / Double(dn)
        var tabs: [[Tap]] = []
        tabs.reserveCapacity(dn)
        for d in 0..<dn {
            let fs1 = Double(d) * scale
            let fs2 = fs1 + scale
            let cell = min(scale, Double(sn) - fs1)
            var s1 = Int(ceil(fs1))
            var s2 = Int(floor(fs2))
            s2 = min(s2, sn - 1)
            s1 = min(s1, s2)
            var t: [Tap] = []
            if Double(s1) - fs1 > 1e-3 { t.append(Tap(src: s1 - 1, w: Float((Double(s1) - fs1) / cell))) }
            if s1 < s2 { for s in s1..<s2 { t.append(Tap(src: s, w: Float(1.0 / cell))) } }
            if fs2 - Double(s2) > 1e-3 {
                t.append(Tap(src: s2, w: Float(min(min(fs2 - Double(s2), 1.0), cell) / cell)))
            }
            tabs.append(t)
        }
        return tabs
    }

    static func resizeArea(_ src: RGBImage, width dw: Int, height dh: Int) -> RGBImage {
        let sw = src.width, sh = src.height
        if sw == dw && sh == dh { return src }
        // OpenCV uses true area averaging only when shrinking in both directions.
        guard sw >= dw && sh >= dh else {
            return Preproc.resizeLinear(src, width: dw, height: dh, areaMode: true)
        }
        let tx = areaTable(dst: dw, src: sw), ty = areaTable(dst: dh, src: sh)
        var horiz = [Float](repeating: 0, count: sh * dw * 3)
        src.data.withUnsafeBufferPointer { s in
            horiz.withUnsafeMutableBufferPointer { hb in
                for y in 0..<sh {
                    let sb = y * sw * 3, hbRow = y * dw * 3
                    for dx in 0..<dw {
                        var r: Float = 0, g: Float = 0, b: Float = 0
                        for t in tx[dx] {
                            let i = sb + t.src * 3
                            r += Float(s[i]) * t.w; g += Float(s[i + 1]) * t.w; b += Float(s[i + 2]) * t.w
                        }
                        hb[hbRow + dx * 3] = r; hb[hbRow + dx * 3 + 1] = g; hb[hbRow + dx * 3 + 2] = b
                    }
                }
            }
        }
        var out = [UInt8](repeating: 0, count: dw * dh * 3)
        horiz.withUnsafeBufferPointer { hb in
            out.withUnsafeMutableBufferPointer { o in
                for dy in 0..<dh {
                    let ob = dy * dw * 3
                    for k in 0..<(dw * 3) {
                        var acc: Float = 0
                        for t in ty[dy] { acc += hb[t.src * dw * 3 + k] * t.w }
                        o[ob + k] = UInt8(max(0, min(255, acc.rounded(.toNearestOrEven))))
                    }
                }
            }
        }
        return RGBImage(width: dw, height: dh, data: out)
    }

    // MARK: - Features

    /// Returns the features plus the 512x512 cropped color image (reused for display / PDF).
    static func features(_ raw: RGBImage) -> (QualityFeatures, RGBImage) {
        let c = resizeArea(Preproc.cropBlackBorders(raw), width: size, height: size)
        let n = size * size
        var gray = [UInt8](repeating: 0, count: n)
        var maxc = [UInt8](repeating: 0, count: n)
        c.data.withUnsafeBufferPointer { d in
            for i in 0..<n {
                let r = d[i * 3], g = d[i * 3 + 1], b = d[i * 3 + 2]
                gray[i] = Preproc.gray(r, g, b)
                maxc[i] = max(r, max(g, b))
            }
        }
        let fov = erode(gray.map { $0 > Preproc.cropTolerance }, w: size, h: size, r: erodeRadius)
        let fovCount = fov.reduce(0) { $0 + ($1 ? 1 : 0) }
        let fovFrac = Double(fovCount) / Double(n)
        if fovCount < minFovPixels {
            return (QualityFeatures(lapVar: 0, glareFrac: 0, meanLum: 0, darkFrac: 1, fovFrac: fovFrac, fovPixels: fovCount), c)
        }

        // Laplacian ksize=3 (OpenCV): 2*(4 diagonal neighbours) - 8*center, border REFLECT_101
        var lapSum = 0.0, lumSum = 0.0
        var glare = 0, dark = 0
        var laps = [Double](); laps.reserveCapacity(fovCount)
        gray.withUnsafeBufferPointer { g in
            for y in 0..<size {
                let ym = Preproc.reflect101(y - 1, size), yp = Preproc.reflect101(y + 1, size)
                for x in 0..<size where fov[y * size + x] {
                    let xm = Preproc.reflect101(x - 1, size), xp = Preproc.reflect101(x + 1, size)
                    let diag = Int(g[ym * size + xm]) + Int(g[ym * size + xp]) + Int(g[yp * size + xm]) + Int(g[yp * size + xp])
                    let lap = Double(2 * diag - 8 * Int(g[y * size + x]))
                    laps.append(lap); lapSum += lap
                    let v = g[y * size + x]
                    lumSum += Double(v)
                    if v < 25 { dark += 1 }
                    if maxc[y * size + x] >= 250 { glare += 1 }
                }
            }
        }
        let mean = lapSum / Double(fovCount)
        var ss = 0.0
        for l in laps { ss += (l - mean) * (l - mean) }
        let f = QualityFeatures(lapVar: ss / Double(fovCount),
                                glareFrac: Double(glare) / Double(fovCount),
                                meanLum: lumSum / Double(fovCount),
                                darkFrac: Double(dark) / Double(fovCount),
                                fovFrac: fovFrac, fovPixels: fovCount)
        return (f, c)
    }

    /// Binary erosion with a (2r+1)x(2r+1) rectangle; pixels outside the image count as "on"
    /// (OpenCV default border for erode), so the image edge itself does not erode the mask.
    static func erode(_ m: [Bool], w: Int, h: Int, r: Int) -> [Bool] {
        func pass(_ src: [Bool], len: Int, count: Int, at: (Int, Int) -> Int) -> [Bool] {
            var dst = src
            var prefix = [Int](repeating: 0, count: len + 1)
            for line in 0..<count {
                for i in 0..<len { prefix[i + 1] = prefix[i] + (src[at(line, i)] ? 0 : 1) }
                for i in 0..<len {
                    let lo = max(0, i - r), hi = min(len - 1, i + r)
                    dst[at(line, i)] = (prefix[hi + 1] - prefix[lo]) == 0
                }
            }
            return dst
        }
        let horiz = pass(m, len: w, count: h) { y, x in y * w + x }
        return pass(horiz, len: h, count: w) { x, y in y * w + x }
    }

    // MARK: - Decision

    static func evaluate(_ f: QualityFeatures, _ t: QualityGateConfig.Thresholds) -> QualityVerdict {
        if f.fovPixels < minFovPixels { return .noFundus }
        if f.lapVar < t.lap_var_min { return .blurry }
        if f.meanLum < t.mean_lum_min { return .tooDark }
        if f.glareFrac > t.glare_frac_max { return .glare }
        return .accept
    }
}

// MARK: - Cheap live metrics for the camera preview (approximate; final decision uses QualityGate)

enum QuickMetrics {
    struct Snapshot { let fovFrac: Double; let meanLum: Double; let glareFrac: Double; let sharpness: Double }

    static func snapshot(_ img: RGBImage) -> Snapshot {
        let w = img.width, h = img.height
        var gray = [UInt8](repeating: 0, count: w * h)
        var fovN = 0, glareN = 0, lum = 0
        img.data.withUnsafeBufferPointer { d in
            for i in 0..<(w * h) {
                let r = d[i * 3], g = d[i * 3 + 1], b = d[i * 3 + 2]
                let v = Preproc.gray(r, g, b)
                gray[i] = v
                if v > Preproc.cropTolerance {
                    fovN += 1; lum += Int(v)
                    if max(r, max(g, b)) >= 250 { glareN += 1 }
                }
            }
        }
        var s = 0.0, s2 = 0.0, n = 0.0
        if w > 2 && h > 2 {
            for y in 1..<(h - 1) {
                for x in 1..<(w - 1) {
                    let i = y * w + x
                    if gray[i] <= Preproc.cropTolerance { continue }
                    let l = Double(Int(gray[i - 1]) + Int(gray[i + 1]) + Int(gray[i - w]) + Int(gray[i + w]) - 4 * Int(gray[i]))
                    s += l; s2 += l * l; n += 1
                }
            }
        }
        let sharp = n > 0 ? (s2 / n - (s / n) * (s / n)) : 0
        let fov = Double(fovN)
        return Snapshot(fovFrac: fov / Double(max(1, w * h)),
                        meanLum: fovN > 0 ? Double(lum) / fov : 0,
                        glareFrac: fovN > 0 ? Double(glareN) / fov : 0,
                        sharpness: sharp)
    }
}

// MARK: - Capture region of interest

/// Through a 20D lens the retina is a bright disc inside a non-black scene (face, iris, room).
/// The capture screen shows a circle guide; we keep only that circle and set everything else to
/// black, so the frame looks like a clinical fundus photo to crop_black_borders and the FOV mask.
enum CaptureROI {
    static var fraction: Double = 0.70   // circle diameter as a fraction of the frame's short side

    static func apply(_ img: RGBImage) -> RGBImage {
        let side = max(8, Int(Double(min(img.width, img.height)) * fraction))
        let x0 = (img.width - side) / 2, y0 = (img.height - side) / 2
        let c = Double(side - 1) / 2, r2 = Double(side) * Double(side) / 4
        var out = [UInt8](repeating: 0, count: side * side * 3)
        img.data.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<side {
                    let dy = Double(y) - c
                    for x in 0..<side {
                        let dx = Double(x) - c
                        if dx * dx + dy * dy > r2 { continue }
                        let si = ((y + y0) * img.width + (x + x0)) * 3, di = (y * side + x) * 3
                        o[di] = s[si]; o[di + 1] = s[si + 1]; o[di + 2] = s[si + 2]
                    }
                }
            }
        }
        return RGBImage(width: side, height: side, data: out)
    }
}
