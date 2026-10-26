// CaptureView.swift — RetinaScan
// Live camera with a circle guide, live quality hint, torch slider, and a 3-second scan that
// auto-selects the sharpest frame, then runs the full pipeline.

import SwiftUI
import AVFoundation

struct CaptureView: View {
    @StateObject private var camera = CameraManager()
    @State private var result: ScreeningResult?
    @State private var retakeMessage: String?
    @State private var errorMessage: String?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: camera.session) { p in camera.focus(at: p) }
                .ignoresSafeArea()
            GuideOverlay(fraction: CaptureROI.fraction, active: camera.hintOK)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack {
                hintPill.padding(.top, 12)
                Spacer()
                controls.padding(.bottom, 28)
            }
            .padding(.horizontal, 20)

            if case .unavailable(let msg) = camera.phase {
                VStack(spacing: 12) {
                    Image(systemName: "video.slash").font(.largeTitle)
                    Text(msg).multilineTextAlignment(.center)
                }
                .foregroundStyle(.white).padding(30)
            }
            if camera.phase == .processing { BusyOverlay(text: L("Analyzing the sharpest frame…")) }
        }
        .navigationTitle(L("Scan"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .onAppear {
            camera.onFrameSelected = { frame in analyze(frame) }
            camera.start()
        }
        .onDisappear { camera.stop() }
        .navigationDestination(item: $result) { ResultView(result: $0) }
        .alert(L("Retake needed"), isPresented: Binding(get: { retakeMessage != nil }, set: { if !$0 { retakeMessage = nil } })) {
            Button(L("Retake"), role: .cancel) { camera.start() }
        } message: { Text(L(retakeMessage ?? "")) }
        .alert(L("Something went wrong"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) { camera.start() }
        } message: { Text(errorMessage ?? "") }
    }

    private var hintPill: some View {
        Label(camera.hint, systemImage: camera.hintOK ? "checkmark.circle.fill" : "exclamationmark.circle")
            .font(.subheadline.weight(.semibold))
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background((camera.hintOK ? Color.green : Color.orange).opacity(0.85), in: Capsule())
            .foregroundStyle(.white)
    }

    private var controls: some View {
        VStack(spacing: 18) {
            HStack {
                Image(systemName: "flashlight.off.fill")
                Slider(value: $camera.torchLevel, in: Float(0.1)...Float(1.0),
                       onEditingChanged: { _ in camera.updateTorch() })
                Image(systemName: "flashlight.on.fill")
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(.black.opacity(0.45), in: Capsule())

            Button { camera.startScan() } label: {
                ZStack {
                    Circle().stroke(.white, lineWidth: 4).frame(width: 78, height: 78)
                    if case .scanning(let p) = camera.phase {
                        Circle().trim(from: 0, to: p).stroke(Color.green, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                            .rotationEffect(.degrees(-90)).frame(width: 78, height: 78)
                        Text(L("Hold")).font(.caption.bold()).foregroundStyle(.white)
                    } else {
                        Circle().fill(.white).frame(width: 64, height: 64)
                        Text(L("Scan")).font(.caption.bold()).foregroundStyle(.black)
                    }
                }
            }
            .disabled(camera.phase != .previewing)
            Text(L("Hold steady for 3 seconds — the sharpest frame is picked automatically."))
                .font(.caption).foregroundStyle(.white.opacity(0.8)).multilineTextAlignment(.center)
        }
    }

    private func analyze(_ frame: RGBImage) {
        ScreeningEngine.shared.analyze(raw: frame, source: .lens) { outcome in
            camera.stop()
            switch outcome {
            case .result(let r): result = r
            case .rejected(_, _, let msg): retakeMessage = msg
            case .failed(let msg): errorMessage = msg
            }
        }
    }
}

/// Circle guide matching CaptureROI: diameter = fraction x the frame's short side.
/// The preview uses aspect-fit so the on-screen circle maps exactly to the processed region.
struct GuideOverlay: View {
    let fraction: Double
    let active: Bool
    var body: some View {
        GeometryReader { geo in
            let frameAspect: CGFloat = 1080.0 / 1920.0                 // portrait buffer
            let shownW = min(geo.size.width, geo.size.height * frameAspect)
            let d = shownW * fraction
            ZStack {
                Rectangle().fill(.black.opacity(0.55))
                    .mask {
                        Rectangle().overlay(Circle().frame(width: d, height: d).blendMode(.destinationOut))
                            .compositingGroup()
                    }
                Circle().stroke(active ? Color.green : Color.white, lineWidth: 3).frame(width: d, height: d)
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var onTap: (CGPoint) -> Void

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        var onTap: ((CGPoint) -> Void)?
        @objc func tapped(_ g: UITapGestureRecognizer) {
            onTap?(previewLayer.captureDevicePointConverted(fromLayerPoint: g.location(in: self)))
        }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.backgroundColor = .black
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspect
        if let c = v.previewLayer.connection, c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        v.onTap = onTap
        v.addGestureRecognizer(UITapGestureRecognizer(target: v, action: #selector(PreviewView.tapped(_:))))
        return v
    }

    func updateUIView(_ v: PreviewView, context: Context) {
        v.onTap = onTap
        if let c = v.previewLayer.connection, c.isVideoRotationAngleSupported(90), c.videoRotationAngle != 90 {
            c.videoRotationAngle = 90
        }
    }
}
