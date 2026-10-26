// ResultView.swift — RetinaScan
// Triage card, score gauge, images, save to profile history (with eye side), read aloud,
// find an eye doctor, PDF export, and delete (for saved scans).

import SwiftUI
import SwiftData

struct ResultView: View {
    let result: ScreeningResult
    var savedRecord: ScanRecord? = nil

    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \Profile.createdAt) private var profiles: [Profile]
    @AppStorage(AppSettings.activeProfileKey) private var activeProfileID = ""
    @AppStorage(AppSettings.autoReadKey) private var autoRead = false

    @State private var pdfURL: URL?
    @State private var showDetails = false
    @State private var eye: EyeSide = .unknown
    @State private var justSaved = false
    @State private var confirmDelete = false

    private var cfg: AppConfig { .shared }
    private var band: TriageBandInfo { cfg.band(result.triage) }
    private var activeProfile: Profile? { profiles.first { $0.id.uuidString == activeProfileID } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                triageCard
                ReadAloudButton(text: spokenSummary).buttonStyle(.bordered)
                ScoreGauge(score: result.prediction.pReferable, cutoffs: cfg.thresholds.cutoffs)
                if result.triage != .low { findDoctor }
                images
                if result.qualityOverridden {
                    Label(L("The image quality check was overridden. This result is less reliable."),
                          systemImage: "exclamationmark.triangle")
                        .font(.footnote).foregroundStyle(.orange)
                }
                saveSection
                DisclosureGroup(L("Details"), isExpanded: $showDetails) { details.padding(.top, 8) }
                    .font(.headline)
                disclaimer
                actions
            }
            .padding(20)
        }
        .navigationTitle(savedRecord == nil ? L("Result") : L("Saved scan"))
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            if pdfURL == nil { pdfURL = try? PDFReport.make(for: result, profile: activeProfile, eye: savedRecord?.eyeSide ?? eye) }
            if autoRead { Speaker.shared.speak(spokenSummary()) }
        }
        .onDisappear { Speaker.shared.stop() }
        .confirmationDialog(L("Delete this saved scan?"), isPresented: $confirmDelete, titleVisibility: .visible) {
            Button(L("Delete"), role: .destructive) {
                if let r = savedRecord { context.delete(r); try? context.save() }
                dismiss()
            }
        }
    }

    // MARK: Pieces

    private func spokenSummary() -> String {
        String(format: L("Screening result: %@. %@ Model score: %.2f. %@"),
               L(band.label), L(band.message), result.prediction.pReferable, L("This is a screening aid, not a diagnosis."))
    }

    private var triageCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: result.triage.symbol).font(.largeTitle)
                VStack(alignment: .leading, spacing: 2) {
                    Text(L("Screening result")).font(.caption).textCase(.uppercase).opacity(0.85)
                    Text(L(band.label)).font(.largeTitle.bold())
                }
            }
            Text(L(band.message)).font(.body)
        }
        .foregroundStyle(.white)
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(result.triage.color, in: RoundedRectangle(cornerRadius: 18))
        .accessibilityElement(children: .combine)
    }

    private var findDoctor: some View {
        Link(destination: URL(string: AppSettings.isSpanish
                              ? "https://maps.apple.com/?q=oftalmologo"
                              : "https://maps.apple.com/?q=ophthalmologist")!) {
            Label(L("Find an eye doctor nearby"), systemImage: "mappin.and.ellipse")
                .frame(maxWidth: .infinity).padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .tint(result.triage.color)
    }

    private var images: some View {
        HStack(spacing: 12) {
            labeledImage(result.displayImage, L("Captured"))
            labeledImage(result.enhancedImage, L("Enhanced (model input)"))
        }
    }

    private func labeledImage(_ img: UIImage, _ title: String) -> some View {
        VStack(spacing: 6) {
            Image(uiImage: img).resizable().scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .accessibilityLabel(title)
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var saveSection: some View {
        if let rec = savedRecord {
            Label(String(format: L("Saved · %@"), rec.eyeSide.label), systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        } else if let p = activeProfile {
            VStack(alignment: .leading, spacing: 10) {
                if justSaved {
                    Label(String(format: L("Saved to %@'s history"), p.name), systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.headline)
                } else {
                    Text(L("Which eye is this?")).font(.subheadline.bold())
                    Picker(L("Eye"), selection: $eye) {
                        ForEach(EyeSide.allCases) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    Button {
                        context.insert(ScanRecord(result: result, eye: eye, profileID: p.id))
                        try? context.save()
                        withAnimation { justSaved = true }
                    } label: {
                        Label(String(format: L("Save to %@'s history"), p.name), systemImage: "tray.and.arrow.down")
                            .frame(maxWidth: .infinity).padding(.vertical, 6)
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }

    private var details: some View {
        let names = [L("No DR"), L("Mild"), L("Moderate"), L("Severe"), L("Proliferative")]
        let q = result.quality
        let t = cfg.qualityGate.thresholds
        return VStack(alignment: .leading, spacing: 12) {
            Text(L("Grade estimate (APTOS scale)")).font(.subheadline.bold())
            ForEach(0..<5, id: \.self) { i in
                HStack {
                    Text(names[i]).font(.caption).frame(width: 100, alignment: .leading)
                    GeometryReader { g in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color(.systemGray5))
                            Capsule().fill(Color.accentColor).frame(width: max(2, g.size.width * result.prediction.probs[i]))
                        }
                    }
                    .frame(height: 8)
                    Text(String(format: "%.2f", result.prediction.probs[i])).font(.caption.monospacedDigit()).frame(width: 44)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(names[i]) \(String(format: "%.2f", result.prediction.probs[i]))")
            }
            Text(L("Fine grades were only validated on APTOS images; the triage uses the referable score above."))
                .font(.caption2).foregroundStyle(.secondary)
            Divider()
            Text(L("Image quality")).font(.subheadline.bold())
            Text(String(format: L("Sharpness %.0f (min %.0f) · Brightness %.0f (min %.0f) · Glare %.1f%% (max %.1f%%)"),
                        q.lapVar, t.lap_var_min, q.meanLum, t.mean_lum_min, q.glareFrac * 100, t.glare_frac_max * 100))
                .font(.caption)
            Text(String(format: L("Analyzed on this device in %.0f ms (model %.0f ms). Source: %@."),
                        result.totalMs, result.inferenceMs, L(result.source.rawValue)))
                .font(.caption).foregroundStyle(.secondary)
        }
        .font(.body)
    }

    private var disclaimer: some View {
        Text(L(cfg.thresholds.disclaimer))
            .font(.footnote)
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private var actions: some View {
        VStack(spacing: 12) {
            if let url = pdfURL {
                ShareLink(item: url) {
                    Label(L("Export PDF report"), systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
            }
            if savedRecord != nil {
                Button(role: .destructive) { confirmDelete = true } label: {
                    Label(L("Delete this scan"), systemImage: "trash").frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.bordered)
            } else {
                Button { dismiss() } label: {
                    Label(L("Done"), systemImage: "arrow.uturn.backward").frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.bordered)
            }
        }
    }
}

/// Horizontal bar with Low / Moderate / Refer zones and a marker at the model score.
struct ScoreGauge: View {
    let score: Double
    let cutoffs: ThresholdsConfig.Cutoffs

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(L("Model score")).font(.subheadline.bold())
                Spacer()
                Text(String(format: "%.2f", score)).font(.subheadline.monospacedDigit().bold())
            }
            GeometryReader { g in
                let w = g.size.width
                ZStack(alignment: .leading) {
                    HStack(spacing: 0) {
                        Triage.low.color.frame(width: w * cutoffs.moderate_min)
                        Triage.moderate.color.frame(width: w * (cutoffs.refer_min - cutoffs.moderate_min))
                        Triage.refer.color
                    }
                    .opacity(0.85)
                    .clipShape(Capsule())
                    Circle().fill(.white).overlay(Circle().stroke(.black, lineWidth: 2))
                        .frame(width: 18, height: 18)
                        .offset(x: min(max(0, w * score - 9), w - 18))
                }
            }
            .frame(height: 18)
            HStack {
                Text(L("Low")).foregroundStyle(Triage.low.color)
                Spacer()
                Text(L("Moderate")).foregroundStyle(Triage.moderate.color)
                Spacer()
                Text(L("Refer")).foregroundStyle(Triage.refer.color)
            }
            .font(.caption.bold())
            Text(L("Ranks how likely the image needs referral. Not a calibrated probability."))
                .font(.caption2).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(String(format: L("Model score %.2f out of 1"), score))
    }
}
