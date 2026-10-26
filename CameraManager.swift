// CameraManager.swift — RetinaScan
// AVCaptureSession for capture through a 20D condensing lens:
//   * back wide camera, 1080p, 32BGRA frames (uncompressed: never JPEG/HEIC, see Phase 2/3 findings)
//   * torch on (adjustable level to control glare)
//   * continuous autofocus while aiming; focus + exposure LOCKED when the scan starts
//   * during a 3 s scan, every frame is scored for sharpness (Laplacian variance) and the
//     sharpest frame is kept in memory (equivalent to recording video and picking the best frame,
//     without writing a video file)
//   * live hints (centering / too dark / glare) from cheap metrics on subsampled frames

import Foundation
import AVFoundation
import UIKit

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    enum Phase: Equatable {
        case idle, previewing, scanning(Double), processing
        case unavailable(String)
    }

    @Published var phase: Phase = .idle
    @Published var hint: String = L("Center the retina inside the circle")
    @Published var hintOK = false
    @Published var torchLevel: Float = 0.6

    let session = AVCaptureSession()
    var onFrameSelected: ((RGBImage) -> Void)?
    var scanDuration: Double = 3.0

    private let sessionQueue = DispatchQueue(label: "retinascan.session")
    private let videoQueue = DispatchQueue(label: "retinascan.video")
    private let output = AVCaptureVideoDataOutput()
    private var device: AVCaptureDevice?
    private var configured = false

    // touched only on videoQueue
    private var scanning = false
    private var scanStart: CFAbsoluteTime = 0
    private var bestScore = -1.0
    private var bestFrame: RGBImage?
    private var frameCount = 0

    // MARK: - Lifecycle

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndRun()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { ok in
                if ok { self.configureAndRun() }
                else { DispatchQueue.main.async { self.phase = .unavailable(L("Camera access was denied. Enable it in Settings > RetinaScan.")) } }
            }
        default:
            phase = .unavailable(L("Camera access was denied. Enable it in Settings > RetinaScan."))
        }
    }

    func stop() {
        sessionQueue.async {
            self.setTorchLocked(on: false)
            if self.session.isRunning { self.session.stopRunning() }
        }
        videoQueue.async { self.scanning = false; self.bestFrame = nil }
        DispatchQueue.main.async { if case .unavailable = self.phase {} else { self.phase = .idle } }
    }

    private func configureAndRun() {
        sessionQueue.async {
            if !self.configured {
                self.session.beginConfiguration()
                self.session.sessionPreset = self.session.canSetSessionPreset(.hd1920x1080) ? .hd1920x1080 : .high
                guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                      let input = try? AVCaptureDeviceInput(device: dev),
                      self.session.canAddInput(input) else {
                    self.session.commitConfiguration()
                    DispatchQueue.main.async { self.phase = .unavailable(L("No back camera available (the Simulator has no camera — use 'Analyze a photo').")) }
                    return
                }
                self.session.addInput(input)
                self.device = dev
                self.output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                self.output.alwaysDiscardsLateVideoFrames = true
                self.output.setSampleBufferDelegate(self, queue: self.videoQueue)
                if self.session.canAddOutput(self.output) { self.session.addOutput(self.output) }
                if let conn = self.output.connection(with: .video), conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90       // portrait frames (1080 wide x 1920 tall)
                }
                self.session.commitConfiguration()
                self.configured = true
            }
            self.setAutoFocus(locked: false)
            if !self.session.isRunning { self.session.startRunning() }
            self.setTorchLocked(on: true)
            DispatchQueue.main.async { self.phase = .previewing }
        }
    }

    // MARK: - Device controls (sessionQueue)

    private func setTorchLocked(on: Bool) {
        guard let d = device, d.hasTorch else { return }
        do {
            try d.lockForConfiguration()
            if on { try d.setTorchModeOn(level: max(0.05, min(1.0, torchLevel))) } else { d.torchMode = .off }
            d.unlockForConfiguration()
        } catch { print("torch error: \(error)") }
    }

    func updateTorch() { sessionQueue.async { self.setTorchLocked(on: true) } }

    private func setAutoFocus(locked: Bool) {
        guard let d = device else { return }
        do {
            try d.lockForConfiguration()
            if locked {
                if d.isFocusModeSupported(.locked) { d.focusMode = .locked }
                if d.isExposureModeSupported(.locked) { d.exposureMode = .locked }
            } else {
                if d.isFocusPointOfInterestSupported { d.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5) }
                if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
                if d.isExposurePointOfInterestSupported { d.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5) }
                if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
            }
            d.unlockForConfiguration()
        } catch { print("focus error: \(error)") }
    }

    /// Tap-to-focus. `p` is in device coordinates (0-1), e.g. from previewLayer.captureDevicePointConverted.
    func focus(at p: CGPoint) {
        sessionQueue.async {
            guard let d = self.device else { return }
            do {
                try d.lockForConfiguration()
                if d.isFocusPointOfInterestSupported { d.focusPointOfInterest = p; d.focusMode = .autoFocus }
                if d.isExposurePointOfInterestSupported { d.exposurePointOfInterest = p; d.exposureMode = .autoExpose }
                d.unlockForConfiguration()
            } catch { print("focus error: \(error)") }
        }
    }

    // MARK: - Scan

    func startScan() {
        sessionQueue.async { self.setAutoFocus(locked: true) }
        videoQueue.async {
            self.bestScore = -1; self.bestFrame = nil
            self.scanStart = CFAbsoluteTimeGetCurrent()
            self.scanning = true
        }
        DispatchQueue.main.async { self.phase = .scanning(0) }
    }

    private func finishScan() {
        scanning = false
        let frame = bestFrame
        bestFrame = nil
        sessionQueue.async { self.setAutoFocus(locked: false) }
        DispatchQueue.main.async {
            self.phase = .processing
            if let f = frame { self.onFrameSelected?(f) }
            else { self.phase = .previewing; self.hint = L("No usable frame — try again") }
        }
    }

    // MARK: - Frames (videoQueue)

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        frameCount += 1

        if scanning {
            if let small = RGBImage.from(pixelBuffer: pb, step: 4) {
                let s = QuickMetrics.snapshot(CaptureROI.apply(small))
                if s.fovFrac > 0.05 && s.sharpness > bestScore, let full = RGBImage.from(pixelBuffer: pb) {
                    bestScore = s.sharpness
                    bestFrame = CaptureROI.apply(full)
                }
            }
            let progress = min(1, (CFAbsoluteTimeGetCurrent() - scanStart) / scanDuration)
            DispatchQueue.main.async { self.phase = .scanning(progress) }
            if progress >= 1 { finishScan() }
            return
        }

        guard frameCount % 6 == 0, let small = RGBImage.from(pixelBuffer: pb, step: 4) else { return }
        let s = QuickMetrics.snapshot(CaptureROI.apply(small))
        let t = AppConfig.shared.qualityGate.thresholds
        let msgs = AppConfig.shared.qualityGate.user_messages
        var text = L("Looks good — tap Scan and hold steady")
        var ok = true
        if s.fovFrac < 0.15 { text = L("Center the retina inside the circle"); ok = false }
        else if s.meanLum < t.mean_lum_min { text = L(msgs["too_dark"] ?? "Too dark"); ok = false }
        else if s.glareFrac > t.glare_frac_max { text = L(msgs["glare"] ?? "Too much glare"); ok = false }
        let finalText = text, finalOK = ok
        DispatchQueue.main.async { self.hint = finalText; self.hintOK = finalOK }
    }
}
