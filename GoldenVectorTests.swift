// GoldenVectorTests.swift — RetinaScan (add to the RetinaScanTests target ONLY)
// Proves the iPhone pipeline reproduces the Python pipeline:
//   1. testModelParity     — H4: Core ML fp16 vs PyTorch on 60 preprocessed PNGs
//   2. testEndToEnd        — Swift Ben Graham + quality features vs Python on 10 raw images
//   3. testConfigLoaded    — thresholds.json / quality_gate.json are in the app bundle
// Requires the golden_vectors files to be in the RetinaScanTests target (folder reference or plain files).

import XCTest
import CoreML
import UIKit
@testable import RetinaScan

final class GoldenVectorTests: XCTestCase {

    // MARK: - Helpers

    /// Every file in the test bundle, by file name. Works whether Xcode kept the
    /// golden_vectors folder structure or flattened the files into the bundle root.
    private static let bundleIndex: [String: URL] = {
        var map = [String: URL]()
        let root = Bundle(for: GoldenVectorTests.self).bundleURL
        if let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let url as URL in e { map[url.lastPathComponent] = url }
        }
        return map
    }()

    private func resource(_ name: String) throws -> URL {
        guard let url = Self.bundleIndex[name] else {
            XCTFail("\(name) is not in the RetinaScanTests bundle. Select it in the left sidebar → right panel → "
                    + "Target Membership → tick RetinaScanTests. (\(Self.bundleIndex.count) files found in bundle)")
            throw XCTSkip("missing \(name)")
        }
        return url
    }

    private func loadCSV() throws -> [[String: String]] {
        let text = try String(contentsOf: try resource("golden_vectors.csv"), encoding: .utf8)
        var lines = text.split(whereSeparator: \.isNewline).map(String.init)
        let header = lines.removeFirst().split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        return lines.map { line in
            let f = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            var row = [String: String]()
            for (i, h) in header.enumerated() where i < f.count { row[h] = f[i] }
            return row
        }
    }

    private func loadPNG(_ url: URL) throws -> RGBImage {
        guard let ui = UIImage(contentsOfFile: url.path), let cg = ui.cgImage, let img = RGBImage.from(cgImage: cg) else {
            throw XCTSkip("Could not load \(url.lastPathComponent)")
        }
        return img
    }

    private func d(_ row: [String: String], _ k: String) -> Double { Double(row[k] ?? "") ?? .nan }

    // MARK: - 1. Model parity (H4)

    func testModelParity() throws {
        try runParity(units: .all, label: "ALL compute units")
        try runParity(units: .cpuOnly, label: "CPU only")
    }

    private func runParity(units: MLComputeUnits, label: String) throws {
        let clf = try DRClassifier(computeUnits: units)
        let rows = try loadCSV()
        let thr = AppConfig.shared.thresholds.cutoffs.refer_min
        var agree = 0, refAgree = 0, maxDiff = 0.0, maxRefDiff = 0.0
        for r in rows {
            let img = try loadPNG(try resource(r["file"]!))
            let p = try clf.predict(preprocessed: img)
            let ref = (0..<5).map { d(r, "p\($0)") }
            let refArg = ref.indices.max { ref[$0] < ref[$1] }!
            if p.argmaxGrade == refArg { agree += 1 }
            let refPR = d(r, "p_referable")
            if (p.pReferable >= thr) == (refPR >= thr) { refAgree += 1 }
            maxDiff = max(maxDiff, zip(p.probs, ref).map { abs($0 - $1) }.max()!)
            maxRefDiff = max(maxRefDiff, abs(p.pReferable - refPR))
        }
        let n = Double(rows.count)
        let msg = String(format: "[%@] n=%d argmax_agree=%.4f referable_agree=%.4f max_abs_prob_diff=%.5f max_abs_pref_diff=%.5f",
                         label, rows.count, Double(agree) / n, Double(refAgree) / n, maxDiff, maxRefDiff)
        print("H4 " + msg)
        XCTAssertGreaterThanOrEqual(Double(agree) / n, 0.99, msg)
        XCTAssertLessThan(maxDiff, 0.01, msg)
    }

    // MARK: - 2. End-to-end preprocessing + quality features

    func testEndToEnd() throws {
        let clf = try DRClassifier()
        let cut = AppConfig.shared.thresholds.cutoffs
        let rows = try loadCSV().filter { $0["end_to_end"] == "True" }
        XCTAssertEqual(rows.count, 10, "expected 10 end-to-end cases")

        for r in rows {
            let file = r["file"]!
            let raw = try loadPNG(try resource("raw_" + file))
            let golden = try loadPNG(try resource(file))

            // crop must match exactly
            let crop = Preproc.cropBlackBorders(raw)
            XCTAssertEqual(crop.width, Int(d(r, "crop_w")), "\(file) crop width")
            XCTAssertEqual(crop.height, Int(d(r, "crop_h")), "\(file) crop height")

            // Ben Graham pixels
            let swiftBG = Preproc.benGraham(raw)
            var sum = 0, maxPx = 0, over4 = 0
            for i in 0..<swiftBG.data.count {
                let diff = abs(Int(swiftBG.data[i]) - Int(golden.data[i]))
                sum += diff; maxPx = max(maxPx, diff); if diff > 4 { over4 += 1 }
            }
            let meanPx = Double(sum) / Double(swiftBG.data.count)

            // model on Swift-preprocessed image vs Python end-to-end score
            let p = try clf.predict(preprocessed: swiftBG)
            let pyScore = d(r, "p_referable")
            let sameBand = Triage.from(score: p.pReferable, cut) == Triage.from(score: pyScore, cut)

            // quality features
            let (q, _) = QualityGate.features(raw)
            let lapRel = abs(q.lapVar - d(r, "q_lap_var")) / max(1, d(r, "q_lap_var"))
            let lumAbs = abs(q.meanLum - d(r, "q_mean_lum"))
            let glareAbs = abs(q.glareFrac - d(r, "q_glare_frac"))

            print(String(format: "E2E %@ | px mean %.3f max %d >4: %.4f%% | score swift %.4f py %.4f | lap rel %.3f lum %.2f glare %.4f",
                         file, meanPx, maxPx, 100 * Double(over4) / Double(swiftBG.data.count),
                         p.pReferable, pyScore, lapRel, lumAbs, glareAbs))

            XCTAssertLessThan(meanPx, 1.5, "\(file) mean pixel diff")
            XCTAssertLessThan(abs(p.pReferable - pyScore), 0.03, "\(file) referable score diff")
            let nearCutoff = abs(pyScore - cut.moderate_min) < 0.03 || abs(pyScore - cut.refer_min) < 0.03
            if nearCutoff {
                print("E2E \(file): borderline (Python score within 0.03 of a cutoff), band check skipped; same band: \(sameBand)")
            } else {
                XCTAssertTrue(sameBand, "\(file) triage band differs")
            }
            XCTAssertLessThan(lapRel, 0.10, "\(file) lap_var")
            XCTAssertLessThan(lumAbs, 1.5, "\(file) mean_lum")
            XCTAssertLessThan(glareAbs, 0.005, "\(file) glare_frac")
        }
    }

    // MARK: - 3. Config

    func testConfigLoaded() {
        XCTAssertTrue(AppConfig.shared.loadedFromBundle, "thresholds.json / quality_gate.json not found in the APP target")
        XCTAssertEqual(AppConfig.shared.thresholds.cutoffs.refer_min, 0.5059, accuracy: 0.001)
    }

    func testPipelineSpeed() throws {
        let rows = try loadCSV().filter { $0["end_to_end"] == "True" }
        let raw = try loadPNG(try resource("raw_" + rows[0]["file"]!))
        measure { _ = ScreeningEngine.shared.analyzeSync(raw: raw, source: .photo, overrideGate: true) }
    }
}
