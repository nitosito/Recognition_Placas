# super_app.py (Versión final con dos pestañas funcionales)

import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Importamos las dos funciones principales de nuestro motor
from anpr_engine import process_image_for_dashboard, process_frame_for_realtime

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Recognition-plate", page_icon="🇨🇴")

# --- Título Principal ---
st.title("🇨🇴 RECONOCIMIENTO DE PLACAS COLOMBIANAS")
st.write("Desarrollado por Helian Cepeda-Alain Espinosa-Andres Castelllanos. Este proyecto utiliza un modelo YOLO11l para ANPR.")
st.markdown("---")

# --- Creación de las Pestañas ---
tab1, tab2 = st.tabs(["📁 Análisis Detallado de Imagen", "📹 Detección en Tiempo Real (Webcam)"])

# --- Contenido de la Pestaña 1: Análisis con Dashboard ---
with tab1:
    st.header("Sube una imagen para un análisis completo tipo dashboard")
    uploaded_file = st.file_uploader("Elige una imagen de un vehículo...", type=["jpg", "png", "jpeg"], key="uploader")

    if uploaded_file is not None:
        # Convertimos el archivo a una imagen que OpenCV pueda leer
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)

        if st.button("Generar Dashboard de Análisis", use_container_width=True, type="primary"):
            with st.spinner('Realizando análisis profundo...'):
                # Llamamos a nuestra función que crea el dashboard completo
                dashboard_image = process_image_for_dashboard(image_to_process)
            
            st.success("¡Análisis completado!")
            st.image(dashboard_image, channels="BGR", caption="Dashboard de Resultados")

# --- Contenido de la Pestaña 2: Detección en Tiempo Real ---
with tab2:
    st.header("Detección en vivo desde tu cámara web")
    st.write("Presiona 'Start' para activar tu cámara. La detección se realizará en tiempo real.")
    st.warning("Asegúrate de permitir el acceso a tu cámara en el navegador.")

    # Clase para procesar cada fotograma que llega desde la cámara
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_counter = 0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Para optimizar, solo procesamos 1 de cada 3 fotogramas
            self.frame_counter += 1
            if self.frame_counter % 3 == 0:
                processed_img = process_frame_for_realtime(img)
                return processed_img
            else:
                return img # Devolvemos el fotograma sin procesar para mantener la fluidez

    # El componente que activa la cámara y muestra el video
    webrtc_streamer(
        key="realtime_detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )