// Config.swift — RetinaScan
// Loads thresholds.json and quality_gate.json from the app bundle (both produced in Colab).
// If a file is missing, falls back to the exact values computed in Colab, so the app still runs.

import Foundation
import SwiftUI

struct TriageBandInfo: Decodable {
    let label: String
    let color: String
    let message: String
}

struct ThresholdsConfig: Decodable {
    struct Cutoffs: Decodable {
        let moderate_min: Double
        let refer_min: Double
    }
    let cutoffs: Cutoffs
    let triage: [String: TriageBandInfo]
    let disclaimer: String
    let score_display_note: String?

    static let fallback = ThresholdsConfig(
        cutoffs: Cutoffs(moderate_min: 0.0805, refer_min: 0.505898),
        triage: [
            "low": TriageBandInfo(label: "Low", color: "green",
                message: "No signs that need a specialist referral were found in this image. Keep your routine yearly eye exam."),
            "moderate": TriageBandInfo(label: "Moderate", color: "amber",
                message: "Borderline result. Possible early changes the screen can't rule out. Schedule an eye exam within the next few months."),
            "refer": TriageBandInfo(label: "Refer", color: "red",
                message: "Signs of diabetic retinopathy that should be checked by an eye doctor soon. Please make an appointment."),
        ],
        disclaimer: "RetinaScan is a screening aid, not a diagnosis. It has not been validated on smartphone-lens images. A Low result does not rule out eye disease.",
        score_display_note: nil)
}

struct QualityGateConfig: Decodable {
    struct Thresholds: Decodable {
        let lap_var_min: Double
        let glare_frac_max: Double
        let mean_lum_min: Double
    }
    let thresholds: Thresholds
    let user_messages: [String: String]
    let status: String?

    static let fallback = QualityGateConfig(
        thresholds: Thresholds(lap_var_min: 52.9047, glare_frac_max: 0.0364, mean_lum_min: 38.0301),
        user_messages: [
            "no_fundus": "Can't see the retina. Center the lens over the pupil and move closer.",
            "blurry": "Image is blurry. Hold steady and let the camera refocus.",
            "too_dark": "Image is too dark. Make sure the flashlight is on and the lens is aligned.",
            "glare": "Too much glare. Tilt the lens slightly to move the reflection off the retina.",
        ],
        status: nil)
}

enum Triage: String, CaseIterable {
    case low, moderate, refer

    static func from(score p: Double, _ c: ThresholdsConfig.Cutoffs) -> Triage {
        if p >= c.refer_min { return .refer }
        if p >= c.moderate_min { return .moderate }
        return .low
    }

    var color: Color {
        switch self {
        case .low: return Color(red: 0.13, green: 0.55, blue: 0.33)
        case .moderate: return Color(red: 0.85, green: 0.55, blue: 0.05)
        case .refer: return Color(red: 0.78, green: 0.16, blue: 0.16)
        }
    }

    var symbol: String {
        switch self {
        case .low: return "checkmark.circle.fill"
        case .moderate: return "exclamationmark.circle.fill"
        case .refer: return "exclamationmark.triangle.fill"
        }
    }
}

final class AppConfig {
    static let shared = AppConfig()

    let thresholds: ThresholdsConfig
    let qualityGate: QualityGateConfig
    let loadedFromBundle: Bool

    private init() {
        let t: ThresholdsConfig? = AppConfig.load("thresholds")
        let q: QualityGateConfig? = AppConfig.load("quality_gate")
        thresholds = t ?? .fallback
        qualityGate = q ?? .fallback
        loadedFromBundle = (t != nil && q != nil)
        if !loadedFromBundle { print("⚠️ RetinaScan: config JSON missing from bundle, using built-in fallback values") }
    }

    private static func load<T: Decodable>(_ name: String) -> T? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { print("⚠️ RetinaScan: could not decode \(name).json: \(error)"); return nil }
    }

    func band(_ t: Triage) -> TriageBandInfo {
        thresholds.triage[t.rawValue] ?? ThresholdsConfig.fallback.triage[t.rawValue]!
    }

    func message(for verdict: QualityVerdict) -> String {
        qualityGate.user_messages[verdict.rawValue] ?? "Please retake the image."
    }
}
