# anpr_engine.py (Versión para Modelo Multi-Clase)

import cv2
import numpy as np
from ultralytics import YOLO
import re
import os

# --- Carga del Modelo ---
print("Cargando modelo ANPR multi-clase...")
try:
    # <<< ¡ASEGÚRATE DE QUE ESTA RUTA APUNTE A TU MODELO MULTI-CLASE!
    MODEL_PATH = 'C:/Users/hlcp2/OneDrive/Documents/DETECCION_CARACTERES/PLACA_COLOMBIA.v9i.yolov11/runs/detect/train/weights/best.pt' 
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"El modelo no se encuentra en: {MODEL_PATH}")
    detector = YOLO(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")
    detector = None

def format_plate_text_multiclass(plate_text_raw):
    """Aplica el formato ABC-123 a la placa reconstruida."""
    match = re.match(r'^([A-Z]{3})([0-9]{3})$', plate_text_raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return plate_text_raw

def process_image_multiclass(image_object):
    """
    Toma una imagen, la procesa con el modelo multi-clase, y devuelve
    la imagen con los resultados dibujados y el texto de la placa.
    """
    if not detector:
        raise RuntimeError("El modelo no se cargó correctamente.")

    result_image = image_object.copy()
    final_detected_text = "No se detecto placa"

    # Ejecutar la inferencia
    results = detector(image_object, verbose=False)[0]
    
    # Separar las detecciones en placas y caracteres
    detections = results.boxes.data.tolist()
    class_names = results.names
    
    plates = [d for d in detections if class_names[int(d[5])] == 'placa']
    characters = [d for d in detections if class_names[int(d[5])] != 'placa']

    # Procesar cada placa encontrada
    for plate_detection in plates:
        x1_p, y1_p, x2_p, y2_p, score_p, _ = plate_detection
        
        if score_p < 0.5: continue # Ignorar placas con baja confianza

        chars_in_plate = []
        # Encontrar qué caracteres están dentro de esta placa
        for char_detection in characters:
            x1_c, y1_c, x2_c, y2_c, score_c, class_id_c = char_detection
            # Calcular el centro del caracter
            center_x_c = (x1_c + x2_c) / 2
            center_y_c = (y1_c + y2_c) / 2
            
            # Si el centro del caracter está dentro de la caja de la placa
            if x1_p < center_x_c < x2_p and y1_p < center_y_c < y2_p:
                chars_in_plate.append({
                    'class': class_names[int(class_id_c)],
                    'x_center': center_x_c
                })
        
        if len(chars_in_plate) >= 5: # Si encontramos suficientes caracteres
            # Ordenar los caracteres por su posición horizontal
            chars_in_plate.sort(key=lambda c: c['x_center'])
            
            # Ensamblar y formatear el texto
            raw_text = "".join([c['class'] for c in chars_in_plate])
            final_detected_text = format_plate_text_multiclass(raw_text)
            
            # Dibujar el resultado en la imagen
            cv2.rectangle(result_image, (int(x1_p), int(y1_p)), (int(x2_p), int(y2_p)), (0, 100, 0), 3)
            (text_width, text_height), _ = cv2.getTextSize(final_detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(result_image, (int(x1_p), int(y1_p) - text_height - 15), (int(x1_p) + text_width, int(y1_p) - 5), (0, 100, 0), cv2.FILLED)
            cv2.putText(result_image, final_detected_text, (int(x1_p), int(y1_p) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Procesamos solo la primera placa con alta confianza
            break 

    return result_image, final_detected_text