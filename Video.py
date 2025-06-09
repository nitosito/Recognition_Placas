# reconstruir_video.py

import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import re
import time # Importamos la librería 'time' para el contador de FPS

# ====================================================================================
# --- CONFIGURACIÓN ---
# ====================================================================================
# <<< ¡REVISA ESTA RUTA! Debe apuntar al archivo .pt de tu modelo de CARACTERES.
MODEL_PATH = 'PLACA_COLOMBIA.v9i.yolov11/runs/detect/train/weights/best.pt'

# Para usar un archivo de video, pon la ruta: "videos/mi_video.mp4"
# Para usar la cámara web en vivo, pon el número: 0
VIDEO_SOURCE = 0 
# ====================================================================================

def reconstruct_plates(detections):
    """
    Toma una lista de detecciones de caracteres, los agrupa, los formatea y forma placas.
    """
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
        if current_line:
            lines.append(current_line)

    reconstructed_plates = []
    for line in lines:
        if len(line) >= 5: # Mínimo 5 caracteres para ser una placa
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

        # Abrir la fuente de video
        video_capture = cv2.VideoCapture(VIDEO_SOURCE)
        if not video_capture.isOpened():
            raise ConnectionError(f"No se pudo abrir la fuente de video: {VIDEO_SOURCE}")
        
        print("\nIniciando detección en video... Presiona 'q' en la ventana para salir.")
        
        prev_time = 0

        # Bucle principal para procesar cada fotograma del video
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                print("Fin del video o error al leer el fotograma.")
                break

            # --- Detección y Reconstrucción por Fotograma ---
            results = char_detector(frame, verbose=False)[0] # verbose=False para una consola limpia
            
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

            # Dibujar los resultados en el fotograma actual
            for plate in final_plates:
                x1, y1, x2, y2 = plate['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), 3)
                (text_width, text_height), _ = cv2.getTextSize(plate['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1 - 5), (0, 100, 0), cv2.FILLED)
                cv2.putText(frame, plate['text'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # --- Cálculo y Visualización de FPS ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Muestra el fotograma procesado
            cv2.imshow("Deteccion en Tiempo Real - Reconstruction", frame)

            # Si se presiona la tecla 'q', se cierra el video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera los recursos al finalizar
        video_capture.release()
        cv2.destroyAllWindows()
        print("Aplicación finalizada.")

    except Exception as e:
        print(f"\nHa ocurrido un error: {e}")