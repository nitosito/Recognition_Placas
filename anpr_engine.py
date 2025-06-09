# anpr_engine.py (Versión Definitiva con Doble Funcionalidad)

import cv2
import numpy as np
from ultralytics import YOLO
import re
import os

# ====================================================================================
# --- CONFIGURACIÓN ---
# ====================================================================================
# <<< ¡ASEGÚRATE DE QUE ESTA RUTA APUNTE A TU MODELO DE CARACTERES!
MODEL_PATH = 'PLACA_COLOMBIA.v9i.yolov11/runs/detect/train/weights/best.pt'
MIN_CHARS_IN_PLATE = 5
# ====================================================================================

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
    """Toma detecciones crudas y devuelve el texto formateado y el recuadro."""
    if len(detections) < MIN_CHARS_IN_PLATE: return None, None
    
    # Ordenar por X para el orden de lectura
    detections.sort(key=lambda d: d[0])
    
    raw_text = "".join([class_names[int(d[5])] for d in detections])
    
    # Formateo
    formatted_text = raw_text
    match = re.match(r'^([A-Z]{3})([0-9]{3})$', raw_text)
    if match:
        formatted_text = f"{match.group(1)}-{match.group(2)}"
        
    # Calcular recuadro
    x1 = min(d[0] for d in detections); y1 = min(d[1] for d in detections)
    x2 = max(d[2] for d in detections); y2 = max(d[3] for d in detections)
            
    return formatted_text, (int(x1), int(y1), int(x2), int(y2))

# =============================================================================
# --- FUNCIÓN 1: Para el Dashboard Detallado ---
# =============================================================================
def process_image_for_dashboard(image_object):
    if not char_detector: raise RuntimeError("El modelo no se cargó correctamente.")
        
    original_image = image_object
    result_image = original_image.copy()
    
    results = char_detector(original_image, verbose=False)[0]
    # Lógica de agrupación por líneas para imágenes estáticas (más precisa)
    detections = sorted(results.boxes.data.tolist(), key=lambda d: d[1])
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
    
    detected_plate_text = "No se pudo reconstruir"
    plate_crop_for_display = np.zeros((80, 200, 3), dtype=np.uint8)

    if lines:
        best_line = max(lines, key=len)
        detected_plate_text, plate_box = reconstruct_and_format_plate(best_line, results.names)
        if plate_box:
            x1, y1, x2, y2 = plate_box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            plate_crop = original_image[y1:y2, x1:x2]
            if plate_crop.size > 0: plate_crop_for_display = plate_crop.copy()

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    _, background_removed_sim = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    # Resto del código para crear el canvas... (sin cambios)
    disp_width, disp_height, margin = 400, 300, 30
    img_list = [resize_for_display(img, disp_width, disp_height) for img in [original_image, gray_image, background_removed_sim, canny_edges, result_image]]
    img_plate_disp = resize_for_display(plate_crop_for_display, 200, 80)
    canvas_height = disp_height * 2 + margin * 4
    canvas_width = disp_width * 3 + margin * 4 
    canvas = np.full((canvas_height, canvas_width, 3), (48, 48, 48), dtype=np.uint8)
    cv2.putText(canvas, "Proceso", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos+disp_height, margin:margin+disp_width] = img_list[0]; cv2.putText(canvas, "Imagen cargada", (margin, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos+disp_height, margin:margin+disp_width] = img_list[1]; cv2.putText(canvas, "Imagen escala de grises", (margin, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    x_pos_col2 = disp_width + margin * 2; y_pos = margin * 2; canvas[y_pos:y_pos+disp_height, x_pos_col2:x_pos_col2+disp_width] = img_list[2]; cv2.putText(canvas, "Imagen con fondo eliminado (sim)", (x_pos_col2, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos+disp_height, x_pos_col2:x_pos_col2+disp_width] = img_list[3]; cv2.putText(canvas, "Imagen con solo bordes", (x_pos_col2, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    x_pos_col3 = disp_width * 2 + margin * 3; cv2.putText(canvas, "Resultado", (x_pos_col3, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos+disp_height, x_pos_col3:x_pos_col3+disp_width] = img_list[4]
    y_pos += disp_height + margin; cv2.putText(canvas, "Placa reconstruida", (x_pos_col3, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); canvas[y_pos:y_pos+80, x_pos_col3:x_pos_col3+200] = img_plate_disp
    y_pos += 80 + margin; cv2.putText(canvas, "Texto detectado:", (x_pos_col3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cv2.putText(canvas, detected_plate_text, (x_pos_col3, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,128), 2)
    
    return canvas, detected_plate_text

# =============================================================================
# --- FUNCIÓN 2: LÓGICA PARA EL VIDEO EN TIEMPO REAL ---
# =============================================================================
def process_frame_for_realtime(frame):
    """
    Toma un fotograma de video y devuelve el fotograma con los resultados dibujados.
    """
    if not char_detector: return frame # Si el modelo no cargó, devuelve el frame original

    results = char_detector(frame, verbose=False)[0]
    plate_text, plate_box = reconstruct_and_format_plate(results.boxes.data.tolist(), results.names)
    
    if plate_text and plate_box:
        x1, y1, x2, y2 = plate_box
        (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1 - 5), (0, 100, 0), cv2.FILLED)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame