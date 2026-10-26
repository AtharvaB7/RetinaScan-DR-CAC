// ChatView.swift — RetinaScan
// "Ask RetinaScan": an assistant powered by DeepSeek that can see the active profile's saved scans.
// The API key goes in Secrets.swift.

import SwiftUI
import SwiftData

// MARK: - DeepSeek client

struct ChatTurn: Codable {
    let role: String       // "system", "user", "assistant"
    let content: String
}

enum DeepSeekError: LocalizedError {
    case noKey, http(Int, String), empty, network(String)
    var errorDescription: String? {
        switch self {
        case .noKey: return L("Add your DeepSeek API key in Secrets.swift to use the assistant.")
        case .http(401, _): return L("The DeepSeek API key was rejected. Check the key in Secrets.swift.")
        case .http(402, _): return L("Your DeepSeek account has no balance. Add credit on the DeepSeek website.")
        case .http(429, _): return L("Too many requests. Wait a moment and try again.")
        case .http(let c, let body): return "DeepSeek error \(c): \(body.prefix(200))"
        case .empty: return L("The assistant returned an empty answer. Please try again.")
        case .network(let m): return L("Could not reach the assistant. Check your internet connection.") + " (\(m))"
        }
    }
}

enum DeepSeekClient {
    private struct Request: Encodable {
        let model: String
        let messages: [ChatTurn]
        let max_tokens: Int
        let temperature: Double
        let stream: Bool
    }
    private struct Response: Decodable {
        struct Choice: Decodable {
            struct Message: Decodable { let content: String? }
            let message: Message
        }
        let choices: [Choice]
    }

    static func complete(_ messages: [ChatTurn]) async throws -> String {
        let key = Secrets.deepSeekAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty, !key.hasPrefix("PASTE_") else { throw DeepSeekError.noKey }

        var lastError: Error = DeepSeekError.empty
        for model in Secrets.deepSeekModels {
            var req = URLRequest(url: URL(string: Secrets.deepSeekURL)!)
            req.httpMethod = "POST"
            req.timeoutInterval = 90
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            req.httpBody = try JSONEncoder().encode(Request(model: model, messages: messages,
                                                            max_tokens: 2000, temperature: 0.3, stream: false))
            let result: (Data, URLResponse)
            do { result = try await URLSession.shared.data(for: req) }
            catch { throw DeepSeekError.network(error.localizedDescription) }
            let (data, resp) = result

            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let body = String(data: data, encoding: .utf8) ?? ""
            if code == 200 {
                let r = try JSONDecoder().decode(Response.self, from: data)
                let text = (r.choices.first?.message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                if text.isEmpty { throw DeepSeekError.empty }
                return text
            }
            lastError = DeepSeekError.http(code, body)
            // 400/404/422 usually mean an unknown model name -> try the next one. Anything else: stop.
            if ![400, 404, 422].contains(code) { break }
        }
        throw lastError
    }
}

// MARK: - Context given to the model

enum AssistantContext {
    static func systemPrompt(profile: Profile?, scans: [ScanRecord]) -> String {
        let c = AppConfig.shared.thresholds.cutoffs
        let names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd HH:mm"

        var p = """
        You are the RetinaScan Assistant, inside an iPhone app that screens retina photos for diabetic retinopathy (DR).
        You help users and community health workers understand their screening results and learn about diabetic eye health.

        RULES
        - You are NOT a doctor and this is NOT medical advice. Never diagnose, never say someone does or does not have disease, never suggest changing medication.
        - When a result is Moderate or Refer, encourage booking an eye exam with an optometrist or ophthalmologist. A Low result does not rule out disease; yearly dilated eye exams are still recommended for people with diabetes.
        - If the user mentions sudden vision loss, a curtain or shadow over vision, many new floaters, flashes of light, or eye pain, tell them to seek urgent medical care now.
        - Only talk about scans listed under SAVED SCANS. Never invent scans, dates or numbers. If there are none, say so and suggest doing a scan.
        - Keep answers short (under 150 words), warm and in plain language that a 12-year-old could follow. No markdown tables. Plain text or short bullet lists only.
        - Politely decline questions unrelated to eye health, diabetes, or the app.
        - Answer in \(AppSettings.isSpanish ? "Spanish" : "English").

        HOW THE APP WORKS
        - The user photographs the retina through a 20D lens on a phone (or picks an existing photo). The app checks image quality (blur, darkness, glare), enhances the image, and runs a MobileNetV3 neural network on the phone.
        - Model score = estimated likelihood of referable DR (moderate or worse). It ranks images; it is not a calibrated probability.
        - Triage: Low if score < \(String(format: "%.3f", c.moderate_min)); Moderate (borderline, eye exam within a few months) if between; Refer (see an eye doctor soon) if score >= \(String(format: "%.3f", c.refer_min)).
        - The grade estimates (No DR, Mild, Moderate, Severe, Proliferative) were only validated on the training dataset; the triage is the main output.
        - Testing: on its own dataset (APTOS) AUC 0.992, sensitivity 96%, specificity 96%. On a different hospital dataset (Messidor-2) AUC 0.844: it caught most severe disease but missed many moderate cases. Not yet validated on phone-lens photos. Be honest about these limits when asked.

        """

        if let pr = profile {
            p += "ACTIVE PROFILE\n- Name: \(pr.name)\n"
            if let a = pr.age { p += "- Age: \(a)\n" }
            if !pr.sex.isEmpty { p += "- Sex: \(pr.sex)\n" }
            p += "\n"
        }

        p += "SAVED SCANS (newest first, \(scans.count) total)\n"
        if scans.isEmpty { p += "- none yet\n" }
        for (i, s) in scans.prefix(15).enumerated() {
            let probs = zip(names, s.probs).map { "\($0) \(String(format: "%.2f", $1))" }.joined(separator: ", ")
            p += "\(i + 1). \(df.string(from: s.date)) | \(s.eye) eye | result \(s.triageValue.rawValue.capitalized)"
            p += " | model score \(String(format: "%.3f", s.score)) | grade estimates: \(probs)"
            p += " | image quality \(s.qualityOverridden ? "LOW (user overrode the quality check)" : "passed")\n"
        }
        return p
    }
}

// MARK: - View model

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let role: String
    let text: String
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var input = ""
    @Published var isLoading = false
    @Published var errorText: String?

    func send(_ text: String, systemPrompt: String) {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, !isLoading else { return }
        messages.append(ChatMessage(role: "user", text: q))
        input = ""
        errorText = nil
        isLoading = true

        var turns = [ChatTurn(role: "system", content: systemPrompt)]
        turns += messages.suffix(12).map { ChatTurn(role: $0.role, content: $0.text) }

        Task {
            do {
                let answer = try await DeepSeekClient.complete(turns)
                messages.append(ChatMessage(role: "assistant", text: answer))
                if UserDefaults.standard.bool(forKey: AppSettings.autoReadKey) { Speaker.shared.speak(answer) }
            } catch {
                errorText = error.localizedDescription
            }
            isLoading = false
        }
    }

    func clear() { messages = []; errorText = nil; Speaker.shared.stop() }
}

// MARK: - View

struct ChatView: View {
    let profile: Profile

    @Environment(\.modelContext) private var context
    @StateObject private var vm = ChatViewModel()
    @FocusState private var inputFocused: Bool

    private var suggestions: [String] {
        [L("What was my last result?"), L("What does a Moderate result mean?"),
         L("Compare my saved scans."), L("How can I lower my risk of diabetic retinopathy?"),
         L("How accurate is RetinaScan?")]
    }

    var body: some View {
        VStack(spacing: 0) {
            disclaimer
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if vm.messages.isEmpty { welcome }
                        ForEach(vm.messages) { m in bubble(m).id(m.id) }
                        if vm.isLoading {
                            HStack(spacing: 8) { ProgressView(); Text(L("Thinking…")).foregroundStyle(.secondary) }
                                .padding(.horizontal, 4).id("loading")
                        }
                        if let e = vm.errorText {
                            Label(e, systemImage: "exclamationmark.triangle").font(.footnote).foregroundStyle(.red)
                        }
                    }
                    .padding(16)
                }
                .onChange(of: vm.messages) { _, msgs in
                    if let last = msgs.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
                }
                .onChange(of: vm.isLoading) { _, loading in
                    if loading { withAnimation { proxy.scrollTo("loading", anchor: .bottom) } }
                }
            }
            inputBar
        }
        .navigationTitle(L("Ask RetinaScan"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if !vm.messages.isEmpty {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(L("Clear")) { vm.clear() }
                }
            }
        }
        .onDisappear { Speaker.shared.stop() }
    }

    private var disclaimer: some View {
        Label(L("AI assistant for information only. Not medical advice. For any concern, see an eye doctor."),
              systemImage: "info.circle")
            .font(.footnote)
            .padding(.horizontal, 16).padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.orange.opacity(0.15))
    }

    private var welcome: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "bubble.left.and.text.bubble.right.fill").font(.title).foregroundStyle(Brand.blue)
                Text(String(format: L("Hi %@! Ask me about your saved scans or about diabetic eye health."), profile.name))
                    .font(.headline)
            }
            ForEach(suggestions, id: \.self) { s in
                Button { send(s) } label: {
                    Text(s).font(.subheadline).multilineTextAlignment(.leading)
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(Brand.blue.opacity(0.1), in: Capsule())
                }
            }
        }
        .padding(.bottom, 8)
    }

    private func bubble(_ m: ChatMessage) -> some View {
        let mine = m.role == "user"
        return HStack(alignment: .bottom) {
            if mine { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 6) {
                Text(m.text).textSelection(.enabled)
                if !mine {
                    ReadAloudButton(text: { m.text }, compact: true).font(.subheadline)
                }
            }
            .padding(12)
            .foregroundStyle(mine ? Color.white : Color.primary)
            .background(mine ? AnyShapeStyle(Brand.blue) : AnyShapeStyle(Color(.secondarySystemBackground)),
                        in: RoundedRectangle(cornerRadius: 16))
            if !mine { Spacer(minLength: 40) }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 10) {
            TextField(L("Ask a question…"), text: $vm.input, axis: .vertical)
                .lineLimit(1...4)
                .focused($inputFocused)
                .padding(.horizontal, 14).padding(.vertical, 10)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20))
                .submitLabel(.send)
                .onSubmit { send(vm.input) }
            Button { send(vm.input) } label: {
                Image(systemName: "arrow.up.circle.fill").font(.system(size: 34))
            }
            .disabled(vm.input.trimmingCharacters(in: .whitespaces).isEmpty || vm.isLoading)
            .accessibilityLabel(L("Send"))
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.bar)
    }

    private func send(_ text: String) {
        let prompt = AssistantContext.systemPrompt(profile: profile, scans: Store.scans(for: profile.id, in: context))
        vm.send(text, systemPrompt: prompt)
    }
}
