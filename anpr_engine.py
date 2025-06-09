# anpr_engine.py (Versión Final con Doble Funcionalidad)

import cv2
import numpy as np
from ultralytics import YOLO
import re
import os

# --- Carga de Modelos (Se hace una sola vez al iniciar) ---
print("Cargando modelo ANPR (esto puede tardar un momento)...")
try:
    # <<< ¡ASEGÚRATE DE QUE ESTA RUTA APUNTE A TU MODELO DE CARACTERES!
    MODEL_PATH = 'PLACA_COLOMBIA.v9i.yolov11/runs/detect/train/weights/best.pt'
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"El modelo no se encuentra en: {MODEL_PATH}")
    
    detector = YOLO(MODEL_PATH)
    print("Modelo de caracteres cargado exitosamente.")
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")
    detector = None

def resize_for_display(image, width, height):
    """Redimensiona una imagen a un tamaño fijo para el dashboard."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (width, height))

def reconstruct_and_format_plate(detections, class_names):
    """Toma detecciones de caracteres y devuelve el texto formateado y su recuadro."""
    if len(detections) < 5: return None, None
    detections.sort(key=lambda d: d[0])
    raw_text = "".join([class_names[int(d[5])] for d in detections])
    match = re.match(r'^([A-Z]{3})([0-9]{3})$', raw_text)
    formatted_text = f"{match.group(1)}-{match.group(2)}" if match else raw_text
    x1, y1 = min(d[0] for d in detections), min(d[1] for d in detections)
    x2, y2 = max(d[2] for d in detections), max(d[3] for d in detections)
    return formatted_text, (int(x1), int(y1), int(x2), int(y2))

# =============================================================================
# --- FUNCIÓN 1: LÓGICA PARA EL DASHBOARD DE ANÁLISIS DE IMAGEN ---
# =============================================================================
def process_image_for_dashboard(image_object):
    if not detector: raise RuntimeError("El modelo no se cargó correctamente.")
        
    original_image = image_object
    result_image = original_image.copy()
    
    results = detector(original_image, verbose=False)[0]
    detected_text, plate_box = reconstruct_and_format_plate(results.boxes.data.tolist(), results.names)
    
    plate_crop_for_display = np.zeros((80, 200, 3), dtype=np.uint8) 
    if plate_box:
        x1, y1, x2, y2 = plate_box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plate_crop = original_image[y1:y2, x1:x2]
        if plate_crop.size > 0: plate_crop_for_display = plate_crop.copy()
    else:
        detected_text = "No se pudo reconstruir"

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    _, background_removed_sim = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    disp_width, disp_height, margin = 400, 300, 30
    img_list = [resize_for_display(img, disp_width, disp_height) for img in [original_image, gray_image, background_removed_sim, canny_edges, result_image]]
    img_plate_disp = resize_for_display(plate_crop_for_display, 200, 80)

    canvas_height = disp_height * 2 + margin * 4
    canvas_width = disp_width * 3 + margin * 4 
    canvas = np.full((canvas_height, canvas_width, 3), (48, 48, 48), dtype=np.uint8)
    
    # Pegado de elementos en el Canvas... (lógica de dibujo)
    cv2.putText(canvas, "Proceso", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos+disp_height, margin:margin+disp_width] = img_list[0]; cv2.putText(canvas, "Imagen cargada", (margin, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos+disp_height, margin:margin+disp_width] = img_list[1]; cv2.putText(canvas, "Imagen escala de grises", (margin, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    x_pos_col2 = disp_width + margin * 2; y_pos = margin * 2; canvas[y_pos:y_pos+disp_height, x_pos_col2:x_pos_col2+disp_width] = img_list[2]; cv2.putText(canvas, "Imagen con fondo eliminado (sim)", (x_pos_col2, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos+disp_height, x_pos_col2:x_pos_col2+disp_width] = img_list[3]; cv2.putText(canvas, "Imagen con solo bordes", (x_pos_col2, y_pos+disp_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    x_pos_col3 = disp_width * 2 + margin * 3; cv2.putText(canvas, "Resultado", (x_pos_col3, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos+disp_height, x_pos_col3:x_pos_col3+disp_width] = img_list[4]
    y_pos += disp_height + margin; cv2.putText(canvas, "Placa reconstruida", (x_pos_col3, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); canvas[y_pos:y_pos+80, x_pos_col3:x_pos_col3+200] = img_plate_disp
    y_pos += 80 + margin; cv2.putText(canvas, "Texto detectado:", (x_pos_col3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cv2.putText(canvas, detected_text, (x_pos_col3, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,128), 2)
    
    return canvas

# =============================================================================
# --- FUNCIÓN 2: LÓGICA PARA EL VIDEO EN TIEMPO REAL ---
# =============================================================================
def process_frame_for_realtime(frame):
    """Toma un fotograma de video y devuelve el fotograma con los resultados dibujados."""
    results = detector(frame, verbose=False)[0]
    plate_text, plate_box = reconstruct_and_format_plate(results.boxes.data.tolist(), results.names)
    
    if plate_text and plate_box:
        x1, y1, x2, y2 = plate_box
        (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1 - 5), (0, 100, 0), cv2.FILLED)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame