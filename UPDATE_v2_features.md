# RetinaScan update 2: profiles, AI assistant, accessibility + Spanish

## Install (about 10 minutes)
1. Unzip `RetinaScan_Swift_v2.zip`. Everything you need is in its `App/` folder (19 .swift files).
2. In Finder, open your Xcode project folder (Desktop/RetinaScan/RetinaScan, the one that already contains
   ContentView.swift and RetinaScanDR.mlpackage). Copy all 19 files from `App/` into it → **Replace** when asked.
   This also replaces the template `RetinaScanApp.swift`; that is intended.
3. Back in Xcode, check the left sidebar shows the new files: Accessibility, ChatView, HistoryView, Localization,
   Persistence, Profiles, Secrets, RetinaScanApp.
   - If they appear automatically → done (Xcode 16 syncs the folder).
   - If not → right-click the yellow RetinaScan group → Add Files to "RetinaScan"… → select the missing files →
     untick "Copy items if needed", tick target **RetinaScan** → Add.
4. Do NOT replace `GoldenVectorTests.swift` (keep your edited version).

## DeepSeek API key
Open **Secrets.swift** in Xcode and paste your key between the quotes:
    static let deepSeekAPIKey = "sk-...."
Your DeepSeek account needs a small credit balance (a few dollars lasts far longer than the demo).
If DeepSeek has renamed its models, the app automatically tries the next name in `deepSeekModels`.

## Run
Product → Clean Build Folder (Cmd+Shift+K) → Cmd+R. Run Cmd+U once too: the tests should still be green.

## Demo checklist
- First launch: "Create a profile" → add name/photo/age/sex (all optional). Up to 5 profiles; Edit (top right) to change/delete.
- Every launch shows the profile picker; the home screen has "Switch profile".
- Analyze a photo → result → pick the eye → "Save to <name>'s history".
- History: score-over-time chart (2+ scans), swipe left to delete one, ••• → Delete all, tap a scan to reopen it.
- Ask RetinaScan: tap a suggested question, e.g. "What was my last result?" — it answers from the saved scans.
- Settings (gear on home, or the "Aa" icon on the profile picker): English/Español, 4 text sizes, auto read-aloud.
- Result screen: "Read aloud" button; chat answers have a speaker button.
- Refer/Moderate results show "Find an eye doctor nearby" (opens Apple Maps).

Simulator notes: the Spanish voice may not be installed in the Simulator (it then reads with an English voice);
a real iPhone has it. Chat needs the Mac to be online.
