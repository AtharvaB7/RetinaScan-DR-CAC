// RetinaScanApp.swift — RetinaScan (REPLACES the Xcode template file of the same name)
// Sets up local storage (SwiftData) and the root view: profile picker on every launch,
// then the home screen. Language and text size are applied app-wide here.

import SwiftUI
import SwiftData

@main
struct RetinaScanApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
        .modelContainer(for: [Profile.self, ScanRecord.self])
    }
}

struct RootView: View {
    @AppStorage(AppSettings.activeProfileKey) private var activeProfileID = ""
    @AppStorage(AppSettings.languageKey) private var language = "en"
    @AppStorage(AppSettings.textSizeKey) private var textSize = TextSizeOption.standard.rawValue
    @State private var pickedThisLaunch = false
    @Query(sort: \Profile.createdAt) private var profiles: [Profile]

    var body: some View {
        Group {
            if pickedThisLaunch, profiles.contains(where: { $0.id.uuidString == activeProfileID }) {
                ContentView(onSwitchProfile: { pickedThisLaunch = false })
            } else {
                ProfilePickerView { p in
                    activeProfileID = p.id.uuidString
                    pickedThisLaunch = true
                }
            }
        }
        .id(language)                                    // rebuild every screen when the language changes
        .environment(\.locale, Locale(identifier: language == "es" ? "es_US" : "en_US"))
        .dynamicTypeSize((TextSizeOption(rawValue: textSize) ?? .standard).size)
        .tint(Brand.blue)
    }
}

enum Brand {
    static let blue = Color(red: 0.10, green: 0.36, blue: 0.55)
}
