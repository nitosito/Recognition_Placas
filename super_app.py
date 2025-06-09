# super_app.py (Versión de Diagnóstico de Rutas)

import streamlit as st
import os

st.set_page_config(layout="wide", page_title="Diagnóstico de Archivos")

st.title("🕵️‍♂️ Explorador de Archivos del Servidor de Streamlit")
st.write(
    "Esta herramienta nos ayudará a encontrar la ruta exacta de tu modelo 'best.pt' "
    "y a diagnosticar por qué no se está cargando."
)
st.markdown("---")

# Mostramos el directorio de trabajo actual
try:
    cwd = os.getcwd()
    st.header("Directorio de Trabajo Actual:")
    st.code(cwd)
except Exception as e:
    st.error(f"No se pudo obtener el directorio de trabajo actual: {e}")

# Creamos un árbol de archivos y directorios para visualizar todo
st.header("Estructura Completa de Carpetas del Proyecto:")
st.warning("Buscando el archivo 'best.pt'...")

output_lines = []
found_path = ""
try:
    for root, dirs, files in os.walk(".", topdown=True):
        # Ignoramos carpetas de caché para una vista más limpia
        dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__')]
        
        level = root.replace('.', '').count(os.sep)
        indent = " " * 4 * level
        output_lines.append(f"{indent}{os.path.basename(root)}/")
        
        subindent = " " * 4 * (level + 1)
        for f in files:
            output_lines.append(f"{subindent}{f}")
            # ¡Si encontramos el modelo, lo anunciamos!
            if f == 'best.pt':
                correct_path = os.path.join(root, f)
                found_path = correct_path
                
    st.code('\n'.join(output_lines))
    
    if found_path:
        st.success("¡MODELO ENCONTRADO!")
        st.subheader("La ruta correcta para tu MODEL_PATH es:")
        st.code(f"MODEL_PATH = '{found_path}'")
    else:
        st.error("ERROR CRÍTICO: No se encontró el archivo 'best.pt' en ninguna carpeta. Asegúrate de que se haya subido correctamente a GitHub y que Git LFS esté funcionando.")

except Exception as e:
    st.error(f"Ocurrió un error al explorar los archivos: {e}")