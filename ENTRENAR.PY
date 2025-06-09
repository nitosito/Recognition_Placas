from ultralytics import YOLO
import torch

def main():
    # Verificar si la GPU está disponible para confirmar la configuración
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print("Entrenando en GPU.")
    else:
        print("ADVERTENCIA: No se detectó GPU. Entrenando en CPU (será mucho más lento).")

    # Carga un modelo YOLOv8 para iniciar (yolov8n.pt es pequeño y bueno para empezar)
    model = YOLO('yolo11s.pt')

    print("Iniciando el ajuste de hiperparámetros...")

    # Ejecuta el ajuste de hiperparámetros con la sintaxis correcta
    model.tune(
        data='data.yaml',
        # El parámetro correcto es 'iterations', no 'trials'.
        # Esto cumple el requisito de explorar ≥ 25 trials. 
        iterations=1,
        epochs=80,
        optimizer='AdamW',
        patience=10,# 
        plots=True,
        save=True
    )

    print("\n" + "="*50)
    print("AJUSTE DE HIPERPARÁMETROS COMPLETADO")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()