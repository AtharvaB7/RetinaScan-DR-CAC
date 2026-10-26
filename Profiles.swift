// Profiles.swift — RetinaScan
// "Who is being screened today?" picker (shown on every launch), profile editor, avatar.

import SwiftUI
import SwiftData
import PhotosUI

struct ProfilePickerView: View {
    static let maxProfiles = 5

    @Environment(\.modelContext) private var context
    @Query(sort: \Profile.createdAt) private var profiles: [Profile]
    @AppStorage(AppSettings.activeProfileKey) private var activeProfileID = ""
    var onSelect: (Profile) -> Void

    @State private var editingMode = false
    @State private var editing: Profile?
    @State private var creating = false
    @State private var showSettings = false

    private let columns = [GridItem(.adaptive(minimum: 130), spacing: 18)]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    VStack(spacing: 8) {
                        Image(systemName: "eye.circle.fill").font(.system(size: 56)).foregroundStyle(Brand.blue)
                        Text("RetinaScan").font(.largeTitle.bold())
                        Text(profiles.isEmpty ? L("Create a profile to get started.") : L("Who is being screened today?"))
                            .font(.title3).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding(.top, 30)

                    LazyVGrid(columns: columns, spacing: 22) {
                        ForEach(profiles) { p in
                            Button {
                                if editingMode { editing = p } else { onSelect(p) }
                            } label: {
                                VStack(spacing: 10) {
                                    ZStack(alignment: .topTrailing) {
                                        AvatarView(profile: p, size: 96)
                                        if editingMode {
                                            Image(systemName: "pencil.circle.fill").font(.title)
                                                .foregroundStyle(Color.white, Brand.blue)
                                        }
                                    }
                                    Text(p.name).font(.headline).foregroundStyle(.primary).lineLimit(1)
                                }
                            }
                            .accessibilityLabel(editingMode ? L("Edit") + " " + p.name : p.name)
                        }
                        if profiles.count < Self.maxProfiles {
                            Button { creating = true } label: {
                                VStack(spacing: 10) {
                                    Circle().strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6]))
                                        .foregroundStyle(.secondary)
                                        .frame(width: 96, height: 96)
                                        .overlay(Image(systemName: "plus").font(.title).foregroundStyle(.secondary))
                                    Text(L("Add profile")).font(.headline).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 24)

                    Text(String(format: L("Up to %d profiles. Everything stays on this phone."), Self.maxProfiles))
                        .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        .padding(.horizontal, 30)
                }
                .padding(.bottom, 30)
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showSettings = true } label: { Image(systemName: "textformat.size") }
                        .accessibilityLabel(L("Settings"))
                }
                if !profiles.isEmpty {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button(editingMode ? L("Done") : L("Edit")) { editingMode.toggle() }
                    }
                }
            }
            .sheet(isPresented: $creating) { ProfileEditorView(profile: nil) }
            .sheet(item: $editing) { ProfileEditorView(profile: $0) }
            .sheet(isPresented: $showSettings) { NavigationStack { SettingsView() } }
        }
    }
}

struct AvatarView: View {
    let profile: Profile
    var size: CGFloat = 44

    private static let palette: [Color] = [.blue, .teal, .orange, .purple, .pink, .green, .indigo]

    var body: some View {
        Group {
            if let data = profile.photo, let img = UIImage(data: data) {
                Image(uiImage: img).resizable().scaledToFill()
            } else {
                let color = Self.palette[Int(profile.id.uuid.0) % Self.palette.count]
                ZStack {
                    color.opacity(0.85)
                    Text(initials).font(.system(size: size * 0.4, weight: .semibold)).foregroundStyle(.white)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
        .accessibilityHidden(true)
    }

    private var initials: String {
        let parts = profile.name.split(separator: " ").prefix(2)
        let s = parts.compactMap { $0.first }.map { String($0) }.joined().uppercased()
        return s.isEmpty ? "?" : s
    }
}

struct ProfileEditorView: View {
    let profile: Profile?

    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @AppStorage(AppSettings.activeProfileKey) private var activeProfileID = ""
    @Query private var allProfiles: [Profile]

    @State private var name = ""
    @State private var ageText = ""
    @State private var sex = ""
    @State private var photo: Data?
    @State private var photoItem: PhotosPickerItem?
    @State private var confirmDelete = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack {
                        Spacer()
                        VStack(spacing: 10) {
                            Group {
                                if let d = photo, let img = UIImage(data: d) {
                                    Image(uiImage: img).resizable().scaledToFill()
                                } else {
                                    ZStack {
                                        Color(.systemGray5)
                                        Image(systemName: "person.fill").font(.system(size: 44)).foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .frame(width: 110, height: 110).clipShape(Circle())
                            PhotosPicker(selection: $photoItem, matching: .images) {
                                Text(photo == nil ? L("Add photo") : L("Change photo"))
                            }
                            if photo != nil {
                                Button(L("Remove photo"), role: .destructive) { photo = nil }.font(.footnote)
                            }
                        }
                        Spacer()
                    }
                }
                .listRowBackground(Color.clear)

                Section(L("Profile (all fields optional)")) {
                    TextField(L("Name"), text: $name)
                        .textContentType(.name)
                    TextField(L("Age"), text: $ageText)
                        .keyboardType(.numberPad)
                    Picker(L("Sex"), selection: $sex) {
                        Text(L("Not set")).tag("")
                        Text(L("Female")).tag("female")
                        Text(L("Male")).tag("male")
                        Text(L("Other")).tag("other")
                    }
                }

                Section {
                    Text(L("Profiles and scans are saved only on this phone and are never uploaded."))
                        .font(.footnote).foregroundStyle(.secondary)
                }

                if profile != nil {
                    Section {
                        Button(L("Delete profile and all its scans"), role: .destructive) { confirmDelete = true }
                    }
                }
            }
            .navigationTitle(profile == nil ? L("New profile") : L("Edit profile"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button(L("Cancel")) { dismiss() } }
                ToolbarItem(placement: .confirmationAction) { Button(L("Save")) { save() }.bold() }
            }
            .onAppear(perform: load)
            .onChange(of: photoItem) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self) {
                        photo = Store.thumbnailData(data)
                    }
                    photoItem = nil
                }
            }
            .confirmationDialog(L("Delete this profile?"), isPresented: $confirmDelete, titleVisibility: .visible) {
                Button(L("Delete"), role: .destructive) {
                    if let p = profile {
                        if activeProfileID == p.id.uuidString { activeProfileID = "" }
                        Store.delete(p, in: context)
                    }
                    dismiss()
                }
            } message: {
                Text(L("All saved scans for this profile will be permanently deleted."))
            }
        }
    }

    private func load() {
        guard let p = profile else { return }
        name = p.name
        ageText = p.age.map(String.init) ?? ""
        sex = p.sex
        photo = p.photo
    }

    private func save() {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        let finalName = trimmed.isEmpty ? String(format: L("Profile %d"), allProfiles.count + (profile == nil ? 1 : 0)) : trimmed
        let age = Int(ageText.trimmingCharacters(in: .whitespaces)).flatMap { (0...120).contains($0) ? $0 : nil }
        if let p = profile {
            p.name = finalName; p.age = age; p.sex = sex; p.photo = photo
        } else if allProfiles.count < ProfilePickerView.maxProfiles {
            context.insert(Profile(name: finalName, photo: photo, age: age, sex: sex))
        }
        try? context.save()
        dismiss()
    }
}
