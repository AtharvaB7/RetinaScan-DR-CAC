// Secrets.swift — RetinaScan
// ⬇️ PASTE YOUR DEEPSEEK API KEY BETWEEN THE QUOTES (starts with "sk-").
// Demo only: a key inside an app can be extracted. Don't publish the app with a key in it.

enum Secrets {
    static let deepSeekAPIKey = "PASTE_YOUR_DEEPSEEK_KEY_HERE"

    /// Tried in order. If DeepSeek rejects the first model name, the next one is used automatically.
    static let deepSeekModels = ["deepseek-chat", "deepseek-flash", "deepseek-v4-pro"]

    static let deepSeekURL = "https://api.deepseek.com/chat/completions"
}
