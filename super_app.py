# super_app.py (Versión para Modelo Multi-Clase)

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Importamos la función de nuestro nuevo motor
from anpr_engine import process_image_multiclass

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="ANPR Multi-Clase", page_icon="🤖")

# --- Título y Descripción ---
st.title("🤖 Detector de Placas (Modelo Multi-Clase)")
st.write("Esta aplicación utiliza un único modelo YOLOv8 para detectar tanto la placa como sus caracteres. Sube una imagen para comenzar.")
st.markdown("---")

# --- Widget para Subir Archivos ---
uploaded_file = st.file_uploader("Elige una imagen de un vehículo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convertir el archivo a una imagen que OpenCV pueda leer
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_to_process = cv2.imdecode(file_bytes, 1)

    st.image(image_to_process, channels="BGR", caption="Imagen Cargada", use_column_width=True)
    
    if st.button("Analizar Placa", use_container_width=True):
        with st.spinner('Procesando con modelo multi-clase...'):
            # Llamar a nuestro motor ANPR
            result_image, detected_text = process_image_multiclass(image_to_process)
            
            st.success('¡Análisis Completado!')
            
            # Mostrar el resultado
            st.image(result_image, channels="BGR", caption="Resultado de la Detección")
            
            st.markdown("---")
            st.header("Texto de la Placa Reconstruido:")
            st.title(f"`{detected_text}`")