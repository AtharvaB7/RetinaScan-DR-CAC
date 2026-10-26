// AboutView.swift — RetinaScan
// Plain-language explanation of what the app does, how it was tested, and its limits.

import SwiftUI

struct AboutView: View {
    var body: some View {
        List {
            Section(L("What it does")) {
                row("eye", "Looks for signs of diabetic retinopathy that need an eye doctor (moderate or worse), from one retina image.")
                row("camera.aperture", "Works with a phone plus a ~$30 20D condensing lens, instead of a $20,000+ fundus camera.")
                row("cpu", "Screening runs entirely on the phone: no internet needed and no image upload.")
                row("person.2", "Up to 5 profiles, each with its own saved scans and score history, stored only on this phone.")
                row("bubble.left.and.text.bubble.right", "Optional AI assistant explains saved results in plain language (needs internet; not medical advice).")
                row("textformat.size", "Larger text, read-aloud results, and English or Spanish.")
                row("checkmark.shield", "Checks every capture for blur, darkness and glare first, and asks for a retake instead of guessing.")
            }
            Section(L("How it was tested")) {
                stat("APTOS 2019 test set (586 images)", "AUC 0.992 · sensitivity 0.96 · specificity 0.96")
                stat("Untouched APTOS hold-out (732 images)", "AUC 0.985 · sensitivity 0.91 · specificity 0.95")
                stat("Messidor-2, different hospitals & camera (1,744 images)", "AUC 0.844 · catches 91% of severe cases but only 26% of moderate ones")
                stat("Simulated phone problems (blur, glare, low light)", "Quality check recovers most of the lost accuracy by rejecting bad captures")
            }
            Section(L("Limits")) {
                row("exclamationmark.triangle", "A screening aid, not a diagnosis. A Low result does not rule out eye disease.")
                row("iphone", "Trained on clinical camera images and not yet validated on images taken through a phone lens.")
                row("arrow.triangle.branch", "Accuracy drops on images from new clinics, mainly for moderate disease. Severe disease transfers well.")
                row("eye.slash", "Lens images usually need dilated pupils; undilated screening is a design goal, not yet shown.")
            }
            Section(L("Roadmap")) {
                row("square.stack.3d.up", "v2: one shared model with extra heads for glaucoma, hypertensive retinopathy and AMD.")
                row("arrow.triangle.2.circlepath", "Retrain with images from more clinics and cameras to fix the moderate-disease gap.")
            }
        }
        .navigationTitle(L("How it works"))
    }

    private func row(_ icon: String, _ text: String) -> some View {
        Label { Text(L(text)) } icon: { Image(systemName: icon).foregroundStyle(Color.accentColor) }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(L(title)).font(.subheadline.bold())
            Text(L(value)).font(.subheadline).foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}
