// Accessibility.swift — RetinaScan
// App settings (language, text size, read-aloud), the text-to-speech helper, and the Settings screen.

import SwiftUI
import AVFoundation

enum AppSettings {
    static let languageKey = "appLanguage"          // "en" or "es"
    static let textSizeKey = "textSize"
    static let autoReadKey = "autoReadResults"
    static let activeProfileKey = "activeProfileID"

    static var language: String { UserDefaults.standard.string(forKey: languageKey) ?? "en" }
    static var isSpanish: Bool { language == "es" }
}

enum TextSizeOption: String, CaseIterable, Identifiable {
    case standard, large, extraLarge, huge
    var id: String { rawValue }

    var size: DynamicTypeSize {
        switch self {
        case .standard: return .large
        case .large: return .xLarge
        case .extraLarge: return .xxxLarge
        case .huge: return .accessibility2
        }
    }

    var label: String {
        switch self {
        case .standard: return L("Standard")
        case .large: return L("Large")
        case .extraLarge: return L("Extra large")
        case .huge: return L("Huge")
        }
    }
}

/// Text-to-speech. Speaks in Spanish or English depending on the app language.
final class Speaker: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    static let shared = Speaker()
    @Published private(set) var isSpeaking = false
    private let synth = AVSpeechSynthesizer()

    private override init() {
        super.init()
        synth.delegate = self
    }

    func speak(_ text: String) {
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: AppSettings.isSpanish ? "es-MX" : "en-US")
        u.rate = AVSpeechUtteranceDefaultSpeechRate * 0.95
        synth.speak(u)
        isSpeaking = true
    }

    func stop() {
        synth.stopSpeaking(at: .immediate)
        isSpeaking = false
    }

    func toggle(_ text: String) { isSpeaking ? stop() : speak(text) }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didFinish u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.isSpeaking = false }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didCancel u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.isSpeaking = false }
    }
}

/// A "Read aloud" / "Stop" button.
struct ReadAloudButton: View {
    let text: () -> String
    var compact = false
    @ObservedObject private var speaker = Speaker.shared

    var body: some View {
        Button {
            speaker.toggle(text())
        } label: {
            if compact {
                Image(systemName: speaker.isSpeaking ? "stop.circle" : "speaker.wave.2")
            } else {
                Label(speaker.isSpeaking ? L("Stop reading") : L("Read aloud"),
                      systemImage: speaker.isSpeaking ? "stop.circle" : "speaker.wave.2.fill")
            }
        }
        .accessibilityLabel(speaker.isSpeaking ? L("Stop reading") : L("Read aloud"))
    }
}

struct SettingsView: View {
    @AppStorage(AppSettings.languageKey) private var language = "en"
    @AppStorage(AppSettings.textSizeKey) private var textSize = TextSizeOption.standard.rawValue
    @AppStorage(AppSettings.autoReadKey) private var autoRead = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Section(L("Language")) {
                Picker(L("Language"), selection: $language) {
                    Text("English").tag("en")
                    Text("Español").tag("es")
                }
                .pickerStyle(.segmented)
            }

            Section {
                Picker(L("Text size"), selection: $textSize) {
                    ForEach(TextSizeOption.allCases) { Text($0.label).tag($0.rawValue) }
                }
                .pickerStyle(.inline)
                .labelsHidden()
                Text(L("Preview: This is how text will look in RetinaScan."))
                    .padding(.vertical, 4)
            } header: {
                Text(L("Text size"))
            }

            Section {
                Toggle(L("Read results aloud automatically"), isOn: $autoRead)
                ReadAloudButton(text: { L("This is how RetinaScan sounds when it reads your results aloud.") })
            } header: {
                Text(L("Read aloud"))
            } footer: {
                Text(L("RetinaScan also works with VoiceOver and the system text size in the iPhone Settings app."))
            }
        }
        .navigationTitle(L("Settings"))
        .toolbar {
            ToolbarItem(placement: .confirmationAction) { Button(L("Done")) { dismiss() } }
        }
    }
}
