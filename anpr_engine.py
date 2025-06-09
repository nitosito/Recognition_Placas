# anpr_engine.py (Versión Final para Dashboard con Reconstrucción de Caracteres)

import cv2
import numpy as np
from ultralytics import YOLO
import re
import os

# --- CONFIGURACIÓN ---
# <<< ¡ASEGÚRATE DE QUE ESTA RUTA APUNTE A TU MODELO DE CARACTERES!
MODEL_PATH = 'runs/detect/yolov8_caracteres_kaggle/weights/best.pt'
MIN_CHARS_IN_PLATE = 5

# --- Carga de Modelos ---
print("Cargando modelo detector de caracteres...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"El modelo no se encuentra en: {MODEL_PATH}")
    char_detector = YOLO(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")
    char_detector = None

def resize_for_display(image, width, height):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (width, height))

def reconstruct_and_format_plate(detections, class_names):
    if not detections:
        return "", None
    
    detections.sort(key=lambda d: d[1]) # Ordenar por Y
    lines = []
    current_line = []
    if detections:
        current_line.append(detections[0])
        avg_char_height = detections[0][3] - detections[0][1]
        for i in range(1, len(detections)):
            if abs(((detections[i][1] + detections[i][3])/2) - ((detections[i-1][1] + detections[i-1][3])/2)) < avg_char_height * 0.7:
                current_line.append(detections[i])
            else:
                if current_line: lines.append(current_line)
                current_line = [detections[i]]
        if current_line: lines.append(current_line)

    if not lines: return "", None

    # Asumimos que la línea con más caracteres es la placa
    best_line = max(lines, key=len)
    
    if len(best_line) < MIN_CHARS_IN_PLATE:
        return "", None

    best_line.sort(key=lambda d: d[0]) # Ordenar por X
    
    raw_text = "".join([class_names[int(d[5])] for d in best_line])
    
    formatted_text = raw_text
    match = re.match(r'^([A-Z]{3})([0-9]{3})$', raw_text)
    if match:
        formatted_text = f"{match.group(1)}-{match.group(2)}"
        
    x1 = min(d[0] for d in best_line)
    y1 = min(d[1] for d in best_line)
    x2 = max(d[2] for d in best_line)
    y2 = max(d[3] for d in best_line)
            
    return formatted_text, (int(x1), int(y1), int(x2), int(y2))

def process_image_for_dashboard(image_object):
    """
    Toma un objeto de imagen, la procesa con el motor de reconstrucción
    y devuelve el canvas completo del dashboard.
    """
    if not char_detector:
        raise RuntimeError("El modelo no se cargó correctamente.")
        
    original_image = image_object
    result_image = original_image.copy()
    
    # --- Detección y Reconstrucción ---
    results = char_detector(original_image, verbose=False)[0]
    detected_text, plate_box = reconstruct_and_format_plate(results.boxes.data.tolist(), results.names)
    
    plate_crop_for_display = np.zeros((80, 200, 3), dtype=np.uint8) 

    if plate_box:
        x1, y1, x2, y2 = plate_box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plate_crop = original_image[y1:y2, x1:x2]
        if plate_crop.size > 0:
            plate_crop_for_display = plate_crop.copy()

    # --- Generación de Imágenes Intermedias ---
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    _, background_removed_sim = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    # --- Creación del Dashboard ---
    disp_width, disp_height = 400, 300
    margin = 30 
    img_original_disp = resize_for_display(original_image, disp_width, disp_height)
    img_gray_disp = resize_for_display(gray_image, disp_width, disp_height)
    img_edges_disp = resize_for_display(canny_edges, disp_width, disp_height)
    img_bg_removed_disp = resize_for_display(background_removed_sim, disp_width, disp_height)
    img_result_disp = resize_for_display(result_image, disp_width, disp_height)
    img_plate_disp = resize_for_display(plate_crop_for_display, 200, 80)

    canvas_height = disp_height * 2 + margin * 4
    canvas_width = disp_width * 3 + margin * 4 
    canvas = np.full((canvas_height, canvas_width, 3), (48, 48, 48), dtype=np.uint8)
    
    # --- Pegado de Elementos en el Canvas ---
    cv2.putText(canvas, "Proceso", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_original_disp; cv2.putText(canvas, "Imagen cargada", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_gray_disp; cv2.putText(canvas, "Imagen escala de grises", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    x_pos_col2 = disp_width + margin * 2; y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_bg_removed_sim; cv2.putText(canvas, "Imagen con fondo eliminado (sim)", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_edges_disp; cv2.putText(canvas, "Imagen con solo bordes", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    x_pos_col3 = disp_width * 2 + margin * 3; cv2.putText(canvas, "Resultado", (x_pos_col3, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col3:x_pos_col3 + disp_width] = img_result_disp
    y_pos += disp_height + margin; cv2.putText(canvas, "Placa reconstruida", (x_pos_col3, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); canvas[y_pos:y_pos + 80, x_pos_col3:x_pos_col3 + 200] = img_plate_disp
    y_pos += 80 + margin; cv2.putText(canvas, "Texto detectado:", (x_pos_col3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); cv2.putText(canvas, detected_text if detected_text else "N/A", (x_pos_col3, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
    
    return canvas