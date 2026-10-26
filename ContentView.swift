// ContentView.swift — RetinaScan
// Home screen for the active profile: scan, analyze a photo, history, assistant, how it works, settings.

import SwiftUI
import SwiftData
import PhotosUI

struct ContentView: View {
    var onSwitchProfile: () -> Void

    @Environment(\.modelContext) private var context
    @Query(sort: \Profile.createdAt) private var profiles: [Profile]
    @AppStorage(AppSettings.activeProfileKey) private var activeProfileID = ""

    @State private var pickerItem: PhotosPickerItem?
    @State private var result: ScreeningResult?
    @State private var busy = false
    @State private var rejection: (message: String, raw: RGBImage)?
    @State private var errorMessage: String?

    private var profile: Profile? { profiles.first { $0.id.uuidString == activeProfileID } }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    NavigationLink { CaptureView() } label: {
                        HomeCard(title: L("Scan with lens"), subtitle: L("Attach the 20D lens, align the eye, and capture"),
                                 systemImage: "camera.aperture", prominent: true)
                    }
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        HomeCard(title: L("Analyze a photo"), subtitle: L("Use an existing fundus image from your library"),
                                 systemImage: "photo.on.rectangle", prominent: false)
                    }
                    if let p = profile {
                        NavigationLink { HistoryView(profile: p) } label: {
                            HomeCard(title: L("History"), subtitle: historySubtitle(p),
                                     systemImage: "clock.arrow.circlepath", prominent: false)
                        }
                        NavigationLink { ChatView(profile: p) } label: {
                            HomeCard(title: L("Ask RetinaScan"), subtitle: L("AI assistant that can explain your saved results"),
                                     systemImage: "bubble.left.and.text.bubble.right", prominent: false)
                        }
                    }
                    NavigationLink { AboutView() } label: {
                        HomeCard(title: L("How it works"), subtitle: L("What RetinaScan checks, how it was tested, and its limits"),
                                 systemImage: "info.circle", prominent: false)
                    }
                    Text(L(AppConfig.shared.thresholds.disclaimer))
                        .font(.footnote).foregroundStyle(.secondary).padding(.top, 6)
                }
                .padding(20)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink { SettingsView() } label: { Image(systemName: "gearshape") }
                        .accessibilityLabel(L("Settings"))
                }
            }
            .navigationDestination(item: $result) { ResultView(result: $0) }
            .overlay { if busy { BusyOverlay(text: L("Analyzing on device…")) } }
            .onChange(of: pickerItem) { _, item in if let item { load(item) } }
            .alert(L("Image check failed"), isPresented: Binding(get: { rejection != nil }, set: { if !$0 { rejection = nil } })) {
                Button(L("Choose another"), role: .cancel) { rejection = nil }
                Button(L("Analyze anyway")) { if let r = rejection { run(r.raw, override: true) }; rejection = nil }
            } message: {
                Text(L(rejection?.message ?? "") + "\n\n" + L("Results on low-quality images are less reliable."))
            }
            .alert(L("Something went wrong"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
                Button("OK", role: .cancel) {}
            } message: { Text(errorMessage ?? "") }
        }
    }

    private func historySubtitle(_ p: Profile) -> String {
        let n = Store.scans(for: p.id, in: context).count
        return n == 0 ? L("No saved scans yet") : String(format: L("%d saved scans"), n)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "eye.circle.fill").font(.largeTitle).foregroundStyle(Brand.blue)
                Text("RetinaScan").font(.largeTitle.bold())
            }
            if let p = profile {
                Button(action: onSwitchProfile) {
                    HStack(spacing: 12) {
                        AvatarView(profile: p, size: 48)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(p.name).font(.headline).foregroundStyle(.primary)
                            Text(L("Switch profile")).font(.subheadline).foregroundStyle(Brand.blue)
                        }
                        Spacer()
                        Image(systemName: "person.2.circle").font(.title2).foregroundStyle(Brand.blue)
                    }
                    .padding(12)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
                }
                .accessibilityLabel(String(format: L("Current profile %@. Switch profile"), p.name))
            }
            Text(L("Low-cost diabetic retinopathy screening with a phone and a lens. Runs fully on this device."))
                .font(.subheadline).foregroundStyle(.secondary)
        }
        .padding(.bottom, 4)
    }

    private func load(_ item: PhotosPickerItem) {
        busy = true
        Task {
            defer { pickerItem = nil }
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let ui = UIImage(data: data), let cg = ui.uprightCGImage(),
                  let raw = RGBImage.from(cgImage: cg) else {
                busy = false; errorMessage = L("Could not read that image."); return
            }
            run(raw, override: false)
        }
    }

    private func run(_ raw: RGBImage, override: Bool) {
        busy = true
        ScreeningEngine.shared.analyze(raw: raw, source: .photo, overrideGate: override) { outcome in
            busy = false
            switch outcome {
            case .result(let r): result = r
            case .rejected(_, _, let msg): rejection = (msg, raw)
            case .failed(let msg): errorMessage = msg
            }
        }
    }
}

struct HomeCard: View {
    let title: String, subtitle: String, systemImage: String, prominent: Bool
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: systemImage).font(.title2).frame(width: 36)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.headline)
                Text(subtitle).font(.subheadline).opacity(0.8).multilineTextAlignment(.leading)
            }
            Spacer()
            Image(systemName: "chevron.right").opacity(0.6)
        }
        .padding(18)
        .foregroundStyle(prominent ? Color.white : Color.primary)
        .background(prominent ? AnyShapeStyle(Brand.blue) : AnyShapeStyle(Color(.secondarySystemBackground)),
                    in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .combine)
    }
}

struct BusyOverlay: View {
    let text: String
    var body: some View {
        ZStack {
            Color.black.opacity(0.35).ignoresSafeArea()
            VStack(spacing: 12) { ProgressView().controlSize(.large); Text(text).font(.headline) }
                .padding(24).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}
