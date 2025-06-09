# super_app.py (Versión Definitiva y Sincronizada)

import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Importamos las dos funciones principales de nuestro motor
# Esta línea ahora debería funcionar porque anpr_engine.py está correcto
from anpr_engine import process_image_for_dashboard, process_frame_for_realtime

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="ANPR Final", page_icon="🇨🇴")

# --- Título Principal ---
st.title("🇨🇴 Aplicación Final de Reconocimiento de Placas")
st.write("Implementación de un pipeline de Visión por Computador para ANPR.")
st.markdown("---")

# --- Creación de las Pestañas ---
tab1, tab2 = st.tabs(["📁 Análisis Detallado de Imagen", "📹 Detección en Tiempo Real"])

# --- Contenido de la Pestaña 1: Análisis con Dashboard ---
with tab1:
    st.header("Sube una imagen para generar un dashboard de análisis")
    uploaded_file = st.file_uploader("Elige una imagen de un vehículo...", type=["jpg", "png", "jpeg"], key="uploader")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)

        if st.button("Generar Dashboard", use_container_width=True, type="primary"):
            with st.spinner('Procesando con modelo de caracteres...'):
                dashboard_image, detected_text = process_image_for_dashboard(image_to_process)
            
            st.success('¡Análisis Completado!')
            st.image(dashboard_image, channels="BGR", caption="Dashboard de Resultados")
            st.subheader(f"Texto Reconstruido: `{detected_text}`")

# --- Contenido de la Pestaña 2: Detección en Tiempo Real ---
with tab2:
    st.header("Detección en vivo desde tu cámara web")
    st.warning("Asegúrate de permitir el acceso a tu cámara en el navegador.")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed_img = process_frame_for_realtime(img)
            return processed_img

    webrtc_streamer(
        key="realtime_detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )