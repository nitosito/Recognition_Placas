# dashboard_final.py

import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import re

# ====================================================================================
# --- CONFIGURACIÓN ---
# ====================================================================================
# <<< ¡REVISA ESTA RUTA! Debe apuntar al archivo .pt de tu modelo de CARACTERES.
MODEL_PATH = 'PLACA_COLOMBIA.v9i.yolov11/runs/detect/train/weights/best.pt'
MIN_CHARS_IN_PLATE = 5
# ====================================================================================

def resize_for_display(image, width, height):
    """Redimensiona una imagen a un tamaño fijo, manejando color y escala de grises."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (width, height))

def reconstruct_plates(detections):
    """Toma una lista de detecciones de caracteres, los agrupa, los formatea y forma placas."""
    if not detections:
        return []

    detections.sort(key=lambda d: d['y_center'])
    lines = []
    current_line = []
    if detections:
        current_line.append(detections[0])
        avg_char_height = detections[0]['height']
        for i in range(1, len(detections)):
            if abs(detections[i]['y_center'] - detections[i-1]['y_center']) < avg_char_height * 0.7:
                current_line.append(detections[i])
            else:
                if current_line: lines.append(current_line)
                current_line = [detections[i]]
        if current_line: lines.append(current_line)

    reconstructed_plates = []
    for line in lines:
        if len(line) >= MIN_CHARS_IN_PLATE:
            line.sort(key=lambda d: d['x_center'])
            plate_text_raw = "".join([d['class'] for d in line])
            
            formatted_text = plate_text_raw
            match = re.match(r'^([A-Z]{3})([0-9]{3})$', plate_text_raw)
            if match:
                formatted_text = f"{match.group(1)}-{match.group(2)}"
            
            x1 = min(d['x_center'] - d['width'] / 2 for d in line)
            y1 = min(d['y_center'] - d['height'] / 2 for d in line)
            x2 = max(d['x_center'] + d['width'] / 2 for d in line)
            y2 = max(d['y_center'] + d['height'] / 2 for d in line)
            
            reconstructed_plates.append({
                "text": formatted_text,
                "box": (int(x1), int(y1), int(x2), int(y2))
            })
    return reconstructed_plates

# --- EJECUCIÓN PRINCIPAL DEL PROGRAMA ---
if __name__ == "__main__":
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encuentra el archivo del modelo en: {MODEL_PATH}")

        print("Cargando modelo detector de caracteres (YOLOv8)...")
        char_detector = YOLO(MODEL_PATH)
        print("Modelo cargado.")
        
        image_path = input("Arrastra una imagen aquí o escribe la ruta y presiona Enter: ").strip().replace("'", "").replace('"', '')
        if not os.path.exists(image_path): raise FileNotFoundError(f"El archivo no existe en la ruta: {image_path}")
            
        original_image = cv2.imread(image_path)
        if original_image is None: raise ValueError("OpenCV no pudo leer la imagen.")
            
        result_image = original_image.copy()

        print("\nDetectando caracteres y reconstruyendo placas...")
        results = char_detector(original_image)[0]
        
        detections_list = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if score > 0.4:
                detections_list.append({
                    'class': results.names[int(class_id)],
                    'x_center': (x1 + x2) / 2, 'y_center': (y1 + y2) / 2,
                    'width': x2 - x1, 'height': y2 - y1
                })

        final_plates = reconstruct_plates(detections_list)

        detected_plate_text = "No se pudo reconstruir"
        plate_crop_for_display = np.zeros((80, 200, 3), dtype=np.uint8)

        if final_plates:
            # Tomamos la primera placa reconstruida para mostrarla
            plate = final_plates[0]
            detected_plate_text = plate['text']
            x1, y1, x2, y2 = plate['box']
            print(f"  > Texto Reconstruido: {detected_plate_text}")
            
            # Dibujar el resultado en la imagen de resultado
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Recortar la placa para mostrarla en el dashboard
            plate_crop = original_image[y1:y2, x1:x2]
            if plate_crop.size > 0:
                plate_crop_for_display = plate_crop.copy()

        print("Generando imágenes del proceso...")
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray_image, 100, 200)
        _, background_removed_sim = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # --- CREACIÓN DEL DASHBOARD ---
        disp_width, disp_height = 400, 300
        margin = 30 
        img_original_disp = resize_for_display(original_image, disp_width, disp_height)
        img_gray_disp = resize_for_display(gray_image, disp_width, disp_height)
        img_edges_disp = resize_for_display(canny_edges, disp_width, disp_height)
        img_bg_removed_sim = resize_for_display(background_removed_sim, disp_width, disp_height)
        img_result_disp = resize_for_display(result_image, disp_width, disp_height)
        img_plate_disp = resize_for_display(plate_crop_for_display, 200, 80)

        canvas_height = disp_height * 2 + margin * 4
        canvas_width = disp_width * 3 + margin * 4 
        canvas = np.full((canvas_height, canvas_width, 3), (48, 48, 48), dtype=np.uint8)
        
        # --- "PEGAR" LAS IMÁGENES Y TEXTOS EN EL LIENZO ---
        cv2.putText(canvas, "Proceso", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_original_disp; cv2.putText(canvas, "Imagen cargada", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_gray_disp; cv2.putText(canvas, "Imagen escala de grises", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_pos_col2 = disp_width + margin * 2; y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_bg_removed_sim; cv2.putText(canvas, "Imagen con fondo eliminado (sim)", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_edges_disp; cv2.putText(canvas, "Imagen con solo bordes", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_pos_col3 = disp_width * 2 + margin * 3; cv2.putText(canvas, "Resultado", (x_pos_col3, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col3:x_pos_col3 + disp_width] = img_result_disp
        y_pos += disp_height + margin; cv2.putText(canvas, "Placa reconstruida", (x_pos_col3, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); canvas[y_pos:y_pos + 80, x_pos_col3:x_pos_col3 + 200] = img_plate_disp
        y_pos += 80 + margin; cv2.putText(canvas, "Texto detectado:", (x_pos_col3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); cv2.putText(canvas, detected_plate_text, (x_pos_col3, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
        
        cv2.imshow("Dashboard con Reconstruccion de Caracteres", canvas)
        print("\nVentana con el dashboard mostrada. Presiona 'q' para salir.")
        while cv2.waitKey(1) & 0xFF != ord('q'):
            if cv2.getWindowProperty("Dashboard con Reconstruccion de Caracteres", cv2.WND_PROP_VISIBLE) < 1: break
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\nHa ocurrido un error: {e}")