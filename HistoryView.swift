// HistoryView.swift — RetinaScan
// Saved scans for the active profile: score trend chart, list, open, delete one, delete all.

import SwiftUI
import SwiftData
import Charts

struct HistoryView: View {
    let profile: Profile

    @Environment(\.modelContext) private var context
    @Query private var scans: [ScanRecord]
    @State private var confirmDeleteAll = false

    init(profile: Profile) {
        self.profile = profile
        let pid = profile.id
        _scans = Query(filter: #Predicate<ScanRecord> { $0.profileID == pid },
                       sort: \ScanRecord.date, order: .reverse)
    }

    var body: some View {
        List {
            if scans.isEmpty {
                ContentUnavailableView(L("No saved scans yet"), systemImage: "tray",
                                       description: Text(L("After a scan, tap “Save to history” on the result screen.")))
                    .listRowBackground(Color.clear)
            } else {
                if scans.count >= 2 {
                    Section(L("Model score over time")) { trendChart.padding(.vertical, 8) }
                }
                Section(String(format: L("%d saved scans"), scans.count)) {
                    ForEach(scans) { r in
                        NavigationLink {
                            ResultView(result: ScreeningResult(record: r), savedRecord: r)
                        } label: {
                            row(r)
                        }
                    }
                    .onDelete { idx in
                        for i in idx { context.delete(scans[i]) }
                        try? context.save()
                    }
                }
            }
        }
        .navigationTitle(String(format: L("%@'s scans"), profile.name))
        .toolbar {
            if !scans.isEmpty {
                ToolbarItem(placement: .topBarTrailing) { EditButton() }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button(L("Delete all scans"), role: .destructive) { confirmDeleteAll = true }
                    } label: { Image(systemName: "ellipsis.circle") }
                }
            }
        }
        .confirmationDialog(L("Delete all saved scans for this profile?"), isPresented: $confirmDeleteAll,
                            titleVisibility: .visible) {
            Button(L("Delete all"), role: .destructive) {
                for s in scans { context.delete(s) }
                try? context.save()
            }
        }
    }

    private func row(_ r: ScanRecord) -> some View {
        HStack(spacing: 12) {
            if let img = r.thumbnail {
                Image(uiImage: img).resizable().scaledToFill()
                    .frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 10))
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Image(systemName: r.triageValue.symbol).foregroundStyle(r.triageValue.color)
                    Text(L(AppConfig.shared.band(r.triageValue).label)).font(.headline).foregroundStyle(r.triageValue.color)
                    Text("· " + r.eyeSide.label).font(.subheadline).foregroundStyle(.secondary)
                }
                Text(r.date, format: .dateTime.month(.abbreviated).day().year().hour().minute())
                    .font(.subheadline).foregroundStyle(.secondary)
            }
            Spacer()
            Text(String(format: "%.2f", r.score)).font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }

    private var trendChart: some View {
        let cut = AppConfig.shared.thresholds.cutoffs
        let ordered = scans.reversed()
        return VStack(alignment: .leading, spacing: 6) {
            Chart {
                RuleMark(y: .value("Refer", cut.refer_min))
                    .foregroundStyle(Triage.refer.color.opacity(0.6))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 4]))
                RuleMark(y: .value("Moderate", cut.moderate_min))
                    .foregroundStyle(Triage.moderate.color.opacity(0.6))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 4]))
                ForEach(Array(ordered)) { r in
                    LineMark(x: .value("Date", r.date), y: .value("Score", r.score))
                        .foregroundStyle(by: .value("Eye", r.eyeSide.label))
                    PointMark(x: .value("Date", r.date), y: .value("Score", r.score))
                        .foregroundStyle(by: .value("Eye", r.eyeSide.label))
                }
            }
            .chartYScale(domain: 0...1)
            .frame(height: 180)
            Text(L("Dashed lines: Moderate and Refer cutoffs."))
                .font(.caption).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L("Chart of model score over time"))
    }
}
