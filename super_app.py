# super_app.py (Versión Simplificada - Solo Dashboard)

import streamlit as st
import numpy as np
import cv2
import os

# Importamos la única función que necesitamos de nuestro motor
from anpr_engine import process_image_for_dashboard

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Dashboard ANPR", page_icon="📈")

# --- Título Principal ---
st.title("📈 Dashboard de Análisis de Placas Vehiculares")
st.write("Esta aplicación utiliza un modelo YOLOv8 para detectar caracteres y reconstruir el texto de la placa, mostrando todo el proceso.")
st.markdown("---")

# --- Widget para Subir Archivos ---
uploaded_file = st.file_uploader("Sube una imagen de un vehículo para generar el dashboard:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convertir el archivo a una imagen que OpenCV pueda leer
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_to_process = cv2.imdecode(file_bytes, 1)

    # Mostrar la imagen original como confirmación
    st.image(image_to_process, channels="BGR", caption="Imagen Cargada", width=400)
    
    if st.button("Generar Dashboard de Análisis", use_container_width=True, type="primary"):
        with st.spinner('Realizando análisis profundo...'):
            # Llamar a nuestra única función que crea el dashboard completo
            dashboard_image, detected_text = process_image_for_dashboard(image_to_process)
        
        st.success('¡Análisis Completado!')
        
        # Mostrar el dashboard final
        st.image(dashboard_image, channels="BGR", caption="Dashboard de Resultados")
        st.subheader(f"Texto Reconstruido: `{detected_text}`")