// ImageBuffer.swift — RetinaScan
// Packed 8-bit RGB image type used by every preprocessing step, plus conversions
// to/from CGImage, CVPixelBuffer (camera + Core ML) and UIImage.

import Foundation
import CoreGraphics
import CoreVideo
import UIKit

/// Packed 8-bit RGB, row-major, 3 bytes per pixel (R, G, B). No alpha, no padding.
struct RGBImage {
    let width: Int
    let height: Int
    var data: [UInt8]

    init(width: Int, height: Int, data: [UInt8]) {
        precondition(data.count == width * height * 3, "RGBImage: data size mismatch")
        self.width = width
        self.height = height
        self.data = data
    }

    init(width: Int, height: Int) {
        self.init(width: width, height: height, data: [UInt8](repeating: 0, count: width * height * 3))
    }

    // MARK: - Decoding

    /// Decode a CGImage to packed RGB. Draws in the image's own RGB color space so no
    /// color management is applied (matches cv2.imread, which ignores color profiles).
    static func from(cgImage: CGImage) -> RGBImage? {
        let w = cgImage.width, h = cgImage.height
        guard w > 0, h > 0 else { return nil }
        var spaces: [CGColorSpace] = []
        if let cs = cgImage.colorSpace, cs.model == .rgb { spaces.append(cs) }
        if let srgb = CGColorSpace(name: CGColorSpace.sRGB) { spaces.append(srgb) }
        spaces.append(CGColorSpaceCreateDeviceRGB())

        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        var drawn = false
        for space in spaces where !drawn {
            drawn = rgba.withUnsafeMutableBytes { buf -> Bool in
                guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h,
                                          bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
                ctx.interpolationQuality = .none
                ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
                return true
            }
        }
        guard drawn else { return nil }

        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        rgba.withUnsafeBufferPointer { s in
            rgb.withUnsafeMutableBufferPointer { d in
                for i in 0..<(w * h) {
                    d[i * 3] = s[i * 4]
                    d[i * 3 + 1] = s[i * 4 + 1]
                    d[i * 3 + 2] = s[i * 4 + 2]
                }
            }
        }
        return RGBImage(width: w, height: h, data: rgb)
    }

    /// From a camera frame (kCVPixelFormatType_32BGRA). `step` > 1 subsamples every Nth
    /// pixel — use only for fast live hints, never for the frame sent to the model.
    static func from(pixelBuffer pb: CVPixelBuffer, step: Int = 1) -> RGBImage? {
        guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let W = CVPixelBufferGetWidth(pb), H = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let s = max(1, step)
        let w = W / s, h = H / s
        var out = [UInt8](repeating: 0, count: w * h * 3)
        out.withUnsafeMutableBufferPointer { o in
            for y in 0..<h {
                let row = base + (y * s) * bpr
                for x in 0..<w {
                    let p = row + (x * s) * 4
                    let i = (y * w + x) * 3
                    o[i] = p[2]; o[i + 1] = p[1]; o[i + 2] = p[0]
                }
            }
        }
        return RGBImage(width: w, height: h, data: out)
    }

    // MARK: - Encoding

    /// 32BGRA pixel buffer with the exact bytes of this image (used as Core ML input).
    func toPixelBuffer() -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferCGImageCompatibilityKey: true,
                                      kCVPixelBufferCGBitmapContextCompatibilityKey: true]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess, let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let base = CVPixelBufferGetBaseAddress(buf)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(buf)
        data.withUnsafeBufferPointer { d in
            for y in 0..<height {
                let row = base + y * bpr
                for x in 0..<width {
                    let i = (y * width + x) * 3
                    let p = row + x * 4
                    p[0] = d[i + 2]; p[1] = d[i + 1]; p[2] = d[i]; p[3] = 255
                }
            }
        }
        return buf
    }

    func toCGImage() -> CGImage? {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        data.withUnsafeBufferPointer { s in
            rgba.withUnsafeMutableBufferPointer { d in
                for i in 0..<(width * height) {
                    d[i * 4] = s[i * 3]; d[i * 4 + 1] = s[i * 3 + 1]; d[i * 4 + 2] = s[i * 3 + 2]
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: space,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    var uiImage: UIImage? { toCGImage().map { UIImage(cgImage: $0) } }
}
