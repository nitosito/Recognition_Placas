# anpr_engine.py (Versión con Depuración de Rutas)

import cv2
import numpy as np
from ultralytics import YOLO
import re
import os

# --- INICIO DEL CÓDIGO DE DEPURACIÓN ---
# Imprimimos la estructura de archivos para ver qué hay en el servidor de Streamlit
print("--- INICIO DEPURACION DE RUTAS ---")
print(f"Directorio de trabajo actual: {os.getcwd()}")
print("Contenido del directorio raíz:")
try:
    print(os.listdir('.'))
except Exception as e:
    print(f"No se pudo listar el directorio raíz: {e}")

# Verificamos si la carpeta 'runs' existe
if os.path.exists('runs'):
    print("Contenido de la carpeta 'runs':")
    try:
        print(os.listdir('runs'))
        if os.path.exists('runs/detect'):
            print("Contenido de 'runs/detect':")
            print(os.listdir('runs/detect'))
    except Exception as e:
        print(f"No se pudo listar subdirectorios de 'runs': {e}")
print("--- FIN DEPURACION DE RUTAS ---")
# --- FIN DEL CÓDIGO DE DEPURACIÓN ---


# ====================================================================================
# --- CONFIGURACIÓN ---
# <<< ¡ASEGÚRATE DE QUE ESTA RUTA RELATIVA SEA CORRECTA SEGÚN TU GITHUB!
MODEL_PATH = 'PLACA_COLOMBIA.v9i.yolov11/runs/detect/train/weights/best.pt'
# ====================================================================================

# ... (El resto del archivo anpr_engine.py es el mismo que ya tenías) ...
# ... Pega aquí el resto de tus funciones: resize_for_display, format_plate_text, etc. ...
# (He omitido el resto para no hacer este bloque gigante, pero asegúrate de que esté completo)

# Carga de Modelos
print("Cargando modelos ANPR...")
try:
    if not os.path.exists(MODEL_PATH):
        # Este error nos dice si la ruta está mal o el archivo no existe
        raise FileNotFoundError(f"El modelo no se encuentra en la ruta especificada: '{MODEL_PATH}'")
    
    detector = YOLO(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")
    detector = None

def process_image_for_dashboard(image_object):
    if not detector:
        raise RuntimeError("El modelo no se cargó correctamente.")
    # ... (resto de la función)
    
# (Pega aquí el resto de tus funciones)