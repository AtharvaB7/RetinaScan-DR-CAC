// ScreeningEngine.swift — RetinaScan
// Core ML wrapper + the full on-device pipeline:
//   raw RGB -> quality gate -> Ben Graham -> RetinaScanDR (Core ML) -> P(referable) -> triage band.
// `RetinaScanDR` is the class Xcode generates automatically from RetinaScanDR.mlpackage.

import Foundation
import CoreML
import UIKit

struct DRPrediction: Hashable {
    let probs: [Double]       // classes 0-4 (ICDR scale used by APTOS)
    let logits: [Double]
    var pReferable: Double { probs[2] + probs[3] + probs[4] }
    var argmaxGrade: Int { probs.indices.max { probs[$0] < probs[$1] } ?? 0 }
}

enum RetinaScanError: LocalizedError {
    case modelUnavailable(String)
    case badInput(String)
    var errorDescription: String? {
        switch self {
        case .modelUnavailable(let s): return "Model could not be loaded: \(s)"
        case .badInput(let s): return s
        }
    }
}

final class DRClassifier {
    private let model: RetinaScanDR

    init(computeUnits: MLComputeUnits = .all) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        model = try RetinaScanDR(configuration: cfg)
    }

    /// `img` must already be Ben Graham preprocessed, 380x380.
    func predict(preprocessed img: RGBImage) throws -> DRPrediction {
        guard img.width == Preproc.modelSize, img.height == Preproc.modelSize else {
            throw RetinaScanError.badInput("Model input must be 380x380, got \(img.width)x\(img.height)")
        }
        guard let pb = img.toPixelBuffer() else { throw RetinaScanError.badInput("Could not build pixel buffer") }
        let out = try model.prediction(image: pb)
        let probs = (0..<5).map { out.probs[[0, $0] as [NSNumber]].doubleValue }
        let logits = (0..<5).map { out.logits[[0, $0] as [NSNumber]].doubleValue }
        return DRPrediction(probs: probs, logits: logits)
    }
}

enum CaptureSource: String { case lens = "Lens capture", photo = "Photo library" }

struct ScreeningResult: Identifiable, Hashable {
    let id = UUID()
    let date: Date
    let source: CaptureSource
    let displayImage: UIImage        // cropped 512x512 color image
    let enhancedImage: UIImage       // Ben Graham 380x380 (what the model saw)
    let quality: QualityFeatures
    let qualityOverridden: Bool
    let prediction: DRPrediction
    let triage: Triage
    let inferenceMs: Double
    let totalMs: Double

    static func == (a: ScreeningResult, b: ScreeningResult) -> Bool { a.id == b.id }
    func hash(into h: inout Hasher) { h.combine(id) }
}

enum ScreeningOutcome {
    case result(ScreeningResult)
    case rejected(QualityVerdict, QualityFeatures, String)
    case failed(String)
}

final class ScreeningEngine {
    static let shared = ScreeningEngine()
    private let queue = DispatchQueue(label: "retinascan.engine", qos: .userInitiated)
    private var classifier: DRClassifier?

    private func loadClassifier() throws -> DRClassifier {
        if let c = classifier { return c }
        do {
            let c = try DRClassifier()
            classifier = c
            return c
        } catch {
            throw RetinaScanError.modelUnavailable(error.localizedDescription)
        }
    }

    /// Runs off the main thread; `completion` is called on the main thread.
    func analyze(raw: RGBImage, source: CaptureSource, overrideGate: Bool = false,
                 completion: @escaping (ScreeningOutcome) -> Void) {
        queue.async {
            let outcome = self.analyzeSync(raw: raw, source: source, overrideGate: overrideGate)
            DispatchQueue.main.async { completion(outcome) }
        }
    }

    func analyzeSync(raw: RGBImage, source: CaptureSource, overrideGate: Bool = false) -> ScreeningOutcome {
        let cfg = AppConfig.shared
        let t0 = CFAbsoluteTimeGetCurrent()
        let (q, crop512) = QualityGate.features(raw)
        let verdict = QualityGate.evaluate(q, cfg.qualityGate.thresholds)
        if verdict != .accept && !(overrideGate && verdict != .noFundus) {
            return .rejected(verdict, q, cfg.message(for: verdict))
        }
        do {
            let clf = try loadClassifier()
            let pre = Preproc.benGraham(raw)
            let t1 = CFAbsoluteTimeGetCurrent()
            let pred = try clf.predict(preprocessed: pre)
            let t2 = CFAbsoluteTimeGetCurrent()
            guard let disp = crop512.uiImage, let enh = pre.uiImage else { return .failed("Could not render images") }
            let r = ScreeningResult(
                date: Date(), source: source, displayImage: disp, enhancedImage: enh,
                quality: q, qualityOverridden: verdict != .accept, prediction: pred,
                triage: Triage.from(score: pred.pReferable, cfg.thresholds.cutoffs),
                inferenceMs: (t2 - t1) * 1000, totalMs: (t2 - t0) * 1000)
            return .result(r)
        } catch {
            return .failed(error.localizedDescription)
        }
    }
}

extension UIImage {
    /// Pixel data with EXIF orientation applied (photos from the library may be rotated).
    func uprightCGImage() -> CGImage? {
        if imageOrientation == .up, let cg = cgImage { return cg }
        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = 1
        return UIGraphicsImageRenderer(size: size, format: fmt).image { _ in draw(at: .zero) }.cgImage
    }
}
