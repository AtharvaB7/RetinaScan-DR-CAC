// Localization.swift — RetinaScan
// English ⇄ Spanish. Every on-screen string is wrapped in L("English text").
// To change a translation, edit the Spanish text on the right. Missing entries fall back to English.

import Foundation

func L(_ english: String) -> String {
    guard AppSettings.isSpanish else { return english }
    if let es = Spanish.table[english] { return es }
    #if DEBUG
    print("⚠️ Missing Spanish translation: \(english)")
    #endif
    return english
}

enum Spanish {
    static let table: [String: String] = [
        "%@'s scans":
            "Escaneos de %@",
        "%d saved scans":
            "%d escaneos guardados",
        "(passed)":
            "(aprobada)",
        "(quality check was overridden by the user)":
            "(el usuario omitió la revisión de calidad)",
        "A screening aid, not a diagnosis. A Low result does not rule out eye disease.":
            "Es una ayuda de detección, no un diagnóstico. Un resultado Bajo no descarta enfermedad ocular.",
        "AI assistant for information only. Not medical advice. For any concern, see an eye doctor.":
            "Asistente de IA solo para información. No es consejo médico. Ante cualquier duda, consulte a un oftalmólogo.",
        "AI assistant that can explain your saved results":
            "Asistente de IA que explica sus resultados guardados",
        "APTOS 2019 test set (586 images)":
            "Conjunto de prueba APTOS 2019 (586 imágenes)",
        "AUC 0.844 · catches 91% of severe cases but only 26% of moderate ones":
            "AUC 0.844 · detecta el 91% de los casos graves pero solo el 26% de los moderados",
        "AUC 0.985 · sensitivity 0.91 · specificity 0.95":
            "AUC 0.985 · sensibilidad 0.91 · especificidad 0.95",
        "AUC 0.992 · sensitivity 0.96 · specificity 0.96":
            "AUC 0.992 · sensibilidad 0.96 · especificidad 0.96",
        "Accuracy drops on images from new clinics, mainly for moderate disease. Severe disease transfers well.":
            "La precisión baja con imágenes de clínicas nuevas, sobre todo en enfermedad moderada. La enfermedad grave se detecta bien.",
        "Add photo":
            "Agregar foto",
        "Add profile":
            "Agregar perfil",
        "Add your DeepSeek API key in Secrets.swift to use the assistant.":
            "Agregue su clave API de DeepSeek en Secrets.swift para usar el asistente.",
        "After a scan, tap “Save to history” on the result screen.":
            "Después de un escaneo, toque “Guardar en el historial” en la pantalla de resultados.",
        "Age":
            "Edad",
        "All saved scans for this profile will be permanently deleted.":
            "Todos los escaneos guardados de este perfil se eliminarán permanentemente.",
        "Analyze a photo":
            "Analizar una foto",
        "Analyze anyway":
            "Analizar de todos modos",
        "Analyzed on this device in %.0f ms (model %.0f ms). Source: %@.":
            "Analizado en este dispositivo en %.0f ms (modelo %.0f ms). Origen: %@.",
        "Analyzing on device…":
            "Analizando en el dispositivo…",
        "Analyzing the sharpest frame…":
            "Analizando la imagen más nítida…",
        "Ask RetinaScan":
            "Pregúntele a RetinaScan",
        "Ask a question…":
            "Haga una pregunta…",
        "Attach the 20D lens, align the eye, and capture":
            "Coloque la lente 20D, alinee el ojo y capture",
        "Borderline result. Possible early changes the screen can't rule out. Schedule an eye exam within the next few months.":
            "Resultado limítrofe. Posibles cambios tempranos que la prueba no puede descartar. Programe un examen de la vista en los próximos meses.",
        "Camera access was denied. Enable it in Settings > RetinaScan.":
            "Se negó el acceso a la cámara. Actívelo en Ajustes > RetinaScan.",
        "Can't see the retina. Center the lens over the pupil and move closer.":
            "No se ve la retina. Centre la lente sobre la pupila y acérquese.",
        "Cancel":
            "Cancelar",
        "Captured":
            "Capturada",
        "Center the retina inside the circle":
            "Centre la retina dentro del círculo",
        "Change photo":
            "Cambiar foto",
        "Chart of model score over time":
            "Gráfica de la puntuación del modelo a lo largo del tiempo",
        "Checks every capture for blur, darkness and glare first, and asks for a retake instead of guessing.":
            "Revisa cada captura en busca de desenfoque, oscuridad y reflejos, y pide repetirla en lugar de adivinar.",
        "Choose another":
            "Elegir otra",
        "Clear":
            "Borrar",
        "Compare my saved scans.":
            "Compare mis escaneos guardados.",
        "Could not reach the assistant. Check your internet connection.":
            "No se pudo conectar con el asistente. Revise su conexión a internet.",
        "Could not read that image.":
            "No se pudo leer esa imagen.",
        "Create a profile to get started.":
            "Cree un perfil para comenzar.",
        "Current profile %@. Switch profile":
            "Perfil actual %@. Cambiar de perfil",
        "Dashed lines: Moderate and Refer cutoffs.":
            "Líneas punteadas: umbrales de Moderado y Referir.",
        "Delete":
            "Eliminar",
        "Delete all":
            "Eliminar todo",
        "Delete all saved scans for this profile?":
            "¿Eliminar todos los escaneos guardados de este perfil?",
        "Delete all scans":
            "Eliminar todos los escaneos",
        "Delete profile and all its scans":
            "Eliminar el perfil y todos sus escaneos",
        "Delete this profile?":
            "¿Eliminar este perfil?",
        "Delete this saved scan?":
            "¿Eliminar este escaneo guardado?",
        "Delete this scan":
            "Eliminar este escaneo",
        "Details":
            "Detalles",
        "Done":
            "Listo",
        "Edit":
            "Editar",
        "Edit profile":
            "Editar perfil",
        "Enhanced (model input)":
            "Mejorada (entrada del modelo)",
        "Export PDF report":
            "Exportar informe PDF",
        "Extra large":
            "Extra grande",
        "Eye":
            "Ojo",
        "Female":
            "Femenino",
        "Find an eye doctor nearby":
            "Buscar un oftalmólogo cercano",
        "Fine grades were only validated on APTOS images; the triage uses the referable score above.":
            "Los grados detallados solo se validaron con imágenes de APTOS; la clasificación usa la puntuación de referencia de arriba.",
        "Grade estimate (APTOS scale)":
            "Estimación del grado (escala APTOS)",
        "Hi %@! Ask me about your saved scans or about diabetic eye health.":
            "¡Hola, %@! Pregúnteme sobre sus escaneos guardados o sobre la salud ocular en la diabetes.",
        "History":
            "Historial",
        "Hold":
            "Quieto",
        "Hold steady for 3 seconds — the sharpest frame is picked automatically.":
            "Mantenga firme por 3 segundos: la imagen más nítida se elige automáticamente.",
        "How accurate is RetinaScan?":
            "¿Qué tan preciso es RetinaScan?",
        "How can I lower my risk of diabetic retinopathy?":
            "¿Cómo puedo reducir mi riesgo de retinopatía diabética?",
        "How it was tested":
            "Cómo se probó",
        "How it works":
            "Cómo funciona",
        "Huge":
            "Enorme",
        "Image check failed":
            "La revisión de la imagen falló",
        "Image is blurry. Hold steady and let the camera refocus.":
            "La imagen está borrosa. Mantenga firme y deje que la cámara vuelva a enfocar.",
        "Image is too dark. Make sure the flashlight is on and the lens is aligned.":
            "La imagen está demasiado oscura. Asegúrese de que la linterna esté encendida y la lente alineada.",
        "Image quality":
            "Calidad de la imagen",
        "Image quality: sharpness %.0f · brightness %.0f · glare %.1f%%":
            "Calidad de la imagen: nitidez %.0f · brillo %.0f · reflejo %.1f%%",
        "Important":
            "Importante",
        "Language":
            "Idioma",
        "Large":
            "Grande",
        "Larger text, read-aloud results, and English or Spanish.":
            "Texto más grande, lectura en voz alta de resultados, y español o inglés.",
        "Left eye":
            "Ojo izquierdo",
        "Left: captured image (cropped). Right: contrast-enhanced image analyzed by the model.":
            "Izquierda: imagen capturada (recortada). Derecha: imagen con contraste mejorado analizada por el modelo.",
        "Lens capture":
            "Captura con lente",
        "Lens images usually need dilated pupils; undilated screening is a design goal, not yet shown.":
            "Las imágenes con lente suelen requerir pupilas dilatadas; la detección sin dilatar es una meta de diseño, aún no demostrada.",
        "Limits":
            "Limitaciones",
        "Looks for signs of diabetic retinopathy that need an eye doctor (moderate or worse), from one retina image.":
            "Busca signos de retinopatía diabética que requieren un oftalmólogo (moderada o peor) en una imagen de la retina.",
        "Looks good — tap Scan and hold steady":
            "Se ve bien: toque Escanear y mantenga firme",
        "Low":
            "Bajo",
        "Low-cost diabetic retinopathy screening with a phone and a lens. Runs fully on this device.":
            "Detección de retinopatía diabética de bajo costo con un teléfono y una lente. Funciona completamente en este dispositivo.",
        "Male":
            "Masculino",
        "Messidor-2, different hospitals & camera (1,744 images)":
            "Messidor-2, otros hospitales y otra cámara (1,744 imágenes)",
        "Mild":
            "Leve",
        "Model score":
            "Puntuación del modelo",
        "Model score %.2f out of 1":
            "Puntuación del modelo %.2f de 1",
        "Model score (referable DR): %.3f · Cutoffs: Moderate ≥ %.3f, Refer ≥ %.3f":
            "Puntuación del modelo (RD referible): %.3f · Umbrales: Moderado ≥ %.3f, Referir ≥ %.3f",
        "Model score over time":
            "Puntuación del modelo a lo largo del tiempo",
        "Moderate":
            "Moderado",
        "Name":
            "Nombre",
        "New profile":
            "Nuevo perfil",
        "No DR":
            "Sin RD",
        "No back camera available (the Simulator has no camera — use 'Analyze a photo').":
            "No hay cámara trasera disponible (el Simulador no tiene cámara; use 'Analizar una foto').",
        "No saved scans yet":
            "Aún no hay escaneos guardados",
        "No signs that need a specialist referral were found in this image. Keep your routine yearly eye exam.":
            "No se encontraron en esta imagen signos que requieran referencia a un especialista. Mantenga su examen de la vista anual de rutina.",
        "No usable frame — try again":
            "No hubo una imagen útil; intente de nuevo",
        "Not set":
            "Sin especificar",
        "Not specified":
            "Sin especificar",
        "On-device analysis time: %.0f ms (model %.0f ms)":
            "Tiempo de análisis en el dispositivo: %.0f ms (modelo %.0f ms)",
        "Optional AI assistant explains saved results in plain language (needs internet; not medical advice).":
            "Asistente de IA opcional que explica los resultados guardados en lenguaje sencillo (requiere internet; no es consejo médico).",
        "Other":
            "Otro",
        "Photo library":
            "Fototeca",
        "Preview: This is how text will look in RetinaScan.":
            "Vista previa: así se verá el texto en RetinaScan.",
        "Profile %d":
            "Perfil %d",
        "Profile (all fields optional)":
            "Perfil (todos los campos son opcionales)",
        "Profiles and scans are saved only on this phone and are never uploaded.":
            "Los perfiles y escaneos se guardan solo en este teléfono y nunca se suben a internet.",
        "Proliferative":
            "Proliferativa",
        "Quality check recovers most of the lost accuracy by rejecting bad captures":
            "La revisión de calidad recupera la mayor parte de la precisión perdida al rechazar malas capturas",
        "Ranks how likely the image needs referral. Not a calibrated probability.":
            "Indica qué tan probable es que la imagen requiera referencia. No es una probabilidad calibrada.",
        "Read aloud":
            "Leer en voz alta",
        "Read results aloud automatically":
            "Leer los resultados en voz alta automáticamente",
        "Refer":
            "Referir",
        "Remove photo":
            "Quitar foto",
        "Result":
            "Resultado",
        "Results on low-quality images are less reliable.":
            "Los resultados con imágenes de baja calidad son menos confiables.",
        "Retake":
            "Repetir",
        "Retake needed":
            "Se necesita repetir",
        "RetinaScan also works with VoiceOver and the system text size in the iPhone Settings app.":
            "RetinaScan también funciona con VoiceOver y con el tamaño de texto del sistema en la app Ajustes del iPhone.",
        "RetinaScan is a screening aid, not a diagnosis. It has not been validated on smartphone-lens images. A Low result does not rule out eye disease.":
            "RetinaScan es una ayuda de detección, no un diagnóstico. No ha sido validado con imágenes tomadas con lente en teléfono. Un resultado Bajo no descarta enfermedad ocular.",
        "RetinaScan — Diabetic Retinopathy Screening Report":
            "RetinaScan — Informe de detección de retinopatía diabética",
        "Retrain with images from more clinics and cameras to fix the moderate-disease gap.":
            "Reentrenar con imágenes de más clínicas y cámaras para cerrar la brecha en enfermedad moderada.",
        "Right eye":
            "Ojo derecho",
        "Roadmap":
            "Próximos pasos",
        "Save":
            "Guardar",
        "Save to %@'s history":
            "Guardar en el historial de %@",
        "Saved scan":
            "Escaneo guardado",
        "Saved to %@'s history":
            "Guardado en el historial de %@",
        "Saved · %@":
            "Guardado · %@",
        "Scan":
            "Escanear",
        "Scan with lens":
            "Escanear con lente",
        "Screening result":
            "Resultado de la detección",
        "Screening result: %@. %@ Model score: %.2f. %@":
            "Resultado de la detección: %@. %@ Puntuación del modelo: %.2f. %@",
        "Screening runs entirely on the phone: no internet needed and no image upload.":
            "La detección funciona completamente en el teléfono: no necesita internet ni sube imágenes.",
        "Send":
            "Enviar",
        "Settings":
            "Ajustes",
        "Severe":
            "Grave",
        "Sex":
            "Sexo",
        "Sharpness %.0f (min %.0f) · Brightness %.0f (min %.0f) · Glare %.1f%% (max %.1f%%)":
            "Nitidez %.0f (mín. %.0f) · Brillo %.0f (mín. %.0f) · Reflejo %.1f%% (máx. %.1f%%)",
        "Signs of diabetic retinopathy that should be checked by an eye doctor soon. Please make an appointment.":
            "Signos de retinopatía diabética que un oftalmólogo debe revisar pronto. Por favor, haga una cita.",
        "Simulated phone problems (blur, glare, low light)":
            "Problemas de teléfono simulados (desenfoque, reflejos, poca luz)",
        "Something went wrong":
            "Algo salió mal",
        "Standard":
            "Estándar",
        "Stop reading":
            "Dejar de leer",
        "Switch profile":
            "Cambiar de perfil",
        "Text size":
            "Tamaño del texto",
        "The DeepSeek API key was rejected. Check the key in Secrets.swift.":
            "La clave API de DeepSeek fue rechazada. Revise la clave en Secrets.swift.",
        "The assistant returned an empty answer. Please try again.":
            "El asistente devolvió una respuesta vacía. Intente de nuevo.",
        "The image quality check was overridden. This result is less reliable.":
            "Se omitió la revisión de calidad de la imagen. Este resultado es menos confiable.",
        "The model score ranks images by how likely they are to need referral; it is not a calibrated probability.":
            "La puntuación del modelo ordena las imágenes según la probabilidad de requerir referencia; no es una probabilidad calibrada.",
        "Thinking…":
            "Pensando…",
        "This is a screening aid, not a diagnosis.":
            "Esta es una ayuda de detección, no un diagnóstico.",
        "This is how RetinaScan sounds when it reads your results aloud.":
            "Así suena RetinaScan cuando lee sus resultados en voz alta.",
        "Too dark":
            "Demasiado oscuro",
        "Too many requests. Wait a moment and try again.":
            "Demasiadas solicitudes. Espere un momento e intente de nuevo.",
        "Too much glare":
            "Demasiado reflejo",
        "Too much glare. Tilt the lens slightly to move the reflection off the retina.":
            "Demasiado reflejo. Incline un poco la lente para quitar el reflejo de la retina.",
        "Trained on clinical camera images and not yet validated on images taken through a phone lens.":
            "Entrenado con imágenes de cámaras clínicas y aún no validado con imágenes tomadas con lente en teléfono.",
        "Untouched APTOS hold-out (732 images)":
            "Conjunto APTOS reservado y nunca usado (732 imágenes)",
        "Up to %d profiles. Everything stays on this phone.":
            "Hasta %d perfiles. Todo se queda en este teléfono.",
        "Up to 5 profiles, each with its own saved scans and score history, stored only on this phone.":
            "Hasta 5 perfiles, cada uno con sus escaneos guardados y su historial de puntuaciones, almacenados solo en este teléfono.",
        "Use an existing fundus image from your library":
            "Use una imagen de fondo de ojo de su fototeca",
        "Validation so far: internal APTOS test set AUC 0.992; external Messidor-2 set AUC 0.844, where the Refer cutoff caught 41% of referable eyes (61% including Moderate). Severe disease transferred much better than moderate disease. Not yet validated on images taken through a phone lens.":
            "Validación hasta ahora: conjunto de prueba interno APTOS AUC 0.992; conjunto externo Messidor-2 AUC 0.844, donde el umbral Referir detectó el 41% de los ojos referibles (61% incluyendo Moderado). La enfermedad grave se detectó mucho mejor que la moderada. Aún no validado con imágenes tomadas con lente en teléfono.",
        "What RetinaScan checks, how it was tested, and its limits":
            "Qué revisa RetinaScan, cómo se probó y sus limitaciones",
        "What does a Moderate result mean?":
            "¿Qué significa un resultado Moderado?",
        "What it does":
            "Qué hace",
        "What was my last result?":
            "¿Cuál fue mi último resultado?",
        "Which eye is this?":
            "¿Qué ojo es este?",
        "Who is being screened today?":
            "¿A quién se le hará la prueba hoy?",
        "Works with a phone plus a ~$30 20D condensing lens, instead of a $20,000+ fundus camera.":
            "Funciona con un teléfono y una lente condensadora 20D de unos $30, en lugar de una cámara de fondo de ojo de más de $20,000.",
        "Your DeepSeek account has no balance. Add credit on the DeepSeek website.":
            "Su cuenta de DeepSeek no tiene saldo. Agregue crédito en el sitio web de DeepSeek.",
        "age %d":
            "%d años",
        "v2: one shared model with extra heads for glaucoma, hypertensive retinopathy and AMD.":
            "v2: un modelo compartido con salidas adicionales para glaucoma, retinopatía hipertensiva y DMAE.",
    ]
}
