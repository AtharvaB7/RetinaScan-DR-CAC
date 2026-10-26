// PDFReport.swift — RetinaScan
// One-page US Letter PDF (in the app's language): profile, images, triage result, score, quality, disclaimer.

import UIKit

enum PDFReport {
    static func make(for r: ScreeningResult, profile: Profile? = nil, eye: EyeSide = .unknown) throws -> URL {
        let cfg = AppConfig.shared
        let band = cfg.band(r.triage)
        let page = CGRect(x: 0, y: 0, width: 612, height: 792)
        let margin: CGFloat = 48
        let width = page.width - 2 * margin

        let df = DateFormatter()
        df.locale = Locale(identifier: AppSettings.isSpanish ? "es_US" : "en_US")
        df.dateStyle = .medium; df.timeStyle = .short
        let stamp = DateFormatter(); stamp.dateFormat = "yyyyMMdd-HHmmss"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("RetinaScan-\(stamp.string(from: r.date)).pdf")

        let renderer = UIGraphicsPDFRenderer(bounds: page)
        try renderer.writePDF(to: url) { ctx in
            ctx.beginPage()
            var y = margin

            func text(_ s: String, size: CGFloat, weight: UIFont.Weight = .regular, color: UIColor = .black) {
                let p = NSMutableParagraphStyle(); p.lineBreakMode = .byWordWrapping
                let attrs: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: size, weight: weight),
                                                            .foregroundColor: color, .paragraphStyle: p]
                let str = NSAttributedString(string: s, attributes: attrs)
                let h = ceil(str.boundingRect(with: CGSize(width: width, height: .greatestFiniteMagnitude),
                                              options: [.usesLineFragmentOrigin], context: nil).height)
                str.draw(with: CGRect(x: margin, y: y, width: width, height: h), options: [.usesLineFragmentOrigin], context: nil)
                y += h + 6
            }

            text(L("RetinaScan — Diabetic Retinopathy Screening Report"), size: 18, weight: .bold)
            var meta = df.string(from: r.date) + "  ·  " + L(r.source.rawValue) + "  ·  " + eye.label
            if let p = profile {
                var who = p.name
                if let a = p.age { who += ", " + String(format: L("age %d"), a) }
                if !p.sex.isEmpty { who += ", " + p.sexLabel }
                meta = who + "  ·  " + meta
            }
            text(meta, size: 10, color: .darkGray)
            y += 6

            let imgSide = (width - 16) / 2
            r.displayImage.draw(in: CGRect(x: margin, y: y, width: imgSide, height: imgSide))
            r.enhancedImage.draw(in: CGRect(x: margin + imgSide + 16, y: y, width: imgSide, height: imgSide))
            y += imgSide + 4
            text(L("Left: captured image (cropped). Right: contrast-enhanced image analyzed by the model."),
                 size: 9, color: .darkGray)
            y += 4

            let ui = UIColor(r.triage.color)
            let cardH: CGFloat = 92
            let card = UIBezierPath(roundedRect: CGRect(x: margin, y: y, width: width, height: cardH), cornerRadius: 10)
            ui.withAlphaComponent(0.12).setFill(); card.fill()
            ui.setStroke(); card.lineWidth = 1.5; card.stroke()
            let top = y
            NSAttributedString(string: L("Result") + ": " + L(band.label),
                               attributes: [.font: UIFont.systemFont(ofSize: 20, weight: .bold), .foregroundColor: ui])
                .draw(at: CGPoint(x: margin + 14, y: top + 10))
            NSAttributedString(string: L(band.message),
                               attributes: [.font: UIFont.systemFont(ofSize: 11), .foregroundColor: UIColor.black])
                .draw(with: CGRect(x: margin + 14, y: top + 40, width: width - 28, height: 48),
                      options: [.usesLineFragmentOrigin], context: nil)
            y = top + cardH + 12

            let c = cfg.thresholds.cutoffs
            text(String(format: L("Model score (referable DR): %.3f · Cutoffs: Moderate ≥ %.3f, Refer ≥ %.3f"),
                        r.prediction.pReferable, c.moderate_min, c.refer_min), size: 11, weight: .semibold)
            text(L("The model score ranks images by how likely they are to need referral; it is not a calibrated probability."),
                 size: 9, color: .darkGray)
            let q = r.quality
            text(String(format: L("Image quality: sharpness %.0f · brightness %.0f · glare %.1f%%"),
                        q.lapVar, q.meanLum, q.glareFrac * 100)
                 + "  " + (r.qualityOverridden ? L("(quality check was overridden by the user)") : L("(passed)")), size: 10)
            text(String(format: L("On-device analysis time: %.0f ms (model %.0f ms)"), r.totalMs, r.inferenceMs), size: 10)
            y += 8

            text(L("Important"), size: 12, weight: .bold)
            text(L(cfg.thresholds.disclaimer), size: 10)
            text(L("Validation so far: internal APTOS test set AUC 0.992; external Messidor-2 set AUC 0.844, where the Refer cutoff caught 41% of referable eyes (61% including Moderate). Severe disease transferred much better than moderate disease. Not yet validated on images taken through a phone lens."),
                 size: 9, color: .darkGray)
        }
        return url
    }
}
