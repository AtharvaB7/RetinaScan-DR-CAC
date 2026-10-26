// Persistence.swift — RetinaScan
// Local storage with SwiftData: up to 5 profiles, each with its own saved scans.
// Everything is stored only on this device.

import Foundation
import SwiftData
import UIKit

@Model
final class Profile {
    var id: UUID
    var name: String
    @Attribute(.externalStorage) var photo: Data?
    var age: Int?
    var sex: String              // "", "female", "male", "other"
    var createdAt: Date

    init(name: String, photo: Data? = nil, age: Int? = nil, sex: String = "") {
        self.id = UUID()
        self.name = name
        self.photo = photo
        self.age = age
        self.sex = sex
        self.createdAt = Date()
    }

    var sexLabel: String {
        switch sex {
        case "female": return L("Female")
        case "male": return L("Male")
        case "other": return L("Other")
        default: return L("Not set")
        }
    }
}

enum EyeSide: String, CaseIterable, Identifiable {
    case left, right, unknown
    var id: String { rawValue }
    var label: String {
        switch self {
        case .left: return L("Left eye")
        case .right: return L("Right eye")
        case .unknown: return L("Not specified")
        }
    }
}

@Model
final class ScanRecord {
    var id: UUID
    var profileID: UUID
    var date: Date
    var eye: String
    var source: String
    var triage: String
    var score: Double
    var probs: [Double]
    var logits: [Double]
    var lapVar: Double
    var glareFrac: Double
    var meanLum: Double
    var darkFrac: Double
    var fovFrac: Double
    var fovPixels: Int
    var qualityOverridden: Bool
    var inferenceMs: Double
    var totalMs: Double
    @Attribute(.externalStorage) var displayJPEG: Data
    @Attribute(.externalStorage) var enhancedPNG: Data

    init(result r: ScreeningResult, eye: EyeSide, profileID: UUID) {
        self.id = UUID()
        self.profileID = profileID
        self.date = r.date
        self.eye = eye.rawValue
        self.source = r.source.rawValue
        self.triage = r.triage.rawValue
        self.score = r.prediction.pReferable
        self.probs = r.prediction.probs
        self.logits = r.prediction.logits
        self.lapVar = r.quality.lapVar
        self.glareFrac = r.quality.glareFrac
        self.meanLum = r.quality.meanLum
        self.darkFrac = r.quality.darkFrac
        self.fovFrac = r.quality.fovFrac
        self.fovPixels = r.quality.fovPixels
        self.qualityOverridden = r.qualityOverridden
        self.inferenceMs = r.inferenceMs
        self.totalMs = r.totalMs
        self.displayJPEG = r.displayImage.jpegData(compressionQuality: 0.9) ?? Data()
        self.enhancedPNG = r.enhancedImage.pngData() ?? Data()
    }

    var eyeSide: EyeSide { EyeSide(rawValue: eye) ?? .unknown }
    var triageValue: Triage { Triage(rawValue: triage) ?? .low }
    var thumbnail: UIImage? { UIImage(data: displayJPEG) }
}

extension ScreeningResult {
    /// Rebuild a result from a saved record (for viewing history).
    init(record r: ScanRecord) {
        self.init(date: r.date,
                  source: CaptureSource(rawValue: r.source) ?? .photo,
                  displayImage: UIImage(data: r.displayJPEG) ?? UIImage(),
                  enhancedImage: UIImage(data: r.enhancedPNG) ?? UIImage(),
                  quality: QualityFeatures(lapVar: r.lapVar, glareFrac: r.glareFrac, meanLum: r.meanLum,
                                           darkFrac: r.darkFrac, fovFrac: r.fovFrac, fovPixels: r.fovPixels),
                  qualityOverridden: r.qualityOverridden,
                  prediction: DRPrediction(probs: r.probs, logits: r.logits),
                  triage: r.triageValue,
                  inferenceMs: r.inferenceMs,
                  totalMs: r.totalMs)
    }
}

enum Store {
    static func scans(for profileID: UUID, in context: ModelContext) -> [ScanRecord] {
        let d = FetchDescriptor<ScanRecord>(predicate: #Predicate { $0.profileID == profileID },
                                            sortBy: [SortDescriptor(\.date, order: .reverse)])
        return (try? context.fetch(d)) ?? []
    }

    /// Deletes a profile and all of its scans.
    static func delete(_ profile: Profile, in context: ModelContext) {
        for s in scans(for: profile.id, in: context) { context.delete(s) }
        context.delete(profile)
        try? context.save()
    }

    /// Shrinks a profile photo so the database stays small.
    static func thumbnailData(_ data: Data, maxSide: CGFloat = 400) -> Data? {
        guard let img = UIImage(data: data) else { return nil }
        let scale = min(1, maxSide / max(img.size.width, img.size.height))
        let size = CGSize(width: img.size.width * scale, height: img.size.height * scale)
        let fmt = UIGraphicsImageRendererFormat(); fmt.scale = 1
        let out = UIGraphicsImageRenderer(size: size, format: fmt).image { _ in img.draw(in: CGRect(origin: .zero, size: size)) }
        return out.jpegData(compressionQuality: 0.85)
    }
}
