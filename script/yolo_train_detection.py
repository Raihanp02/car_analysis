from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "yolo26n.pt"
MODEL_PATH = BASE_DIR / MODEL_NAME

model = YOLO(r"C:\Users\AQI\JOB\ppe_detection\yolo26n.pt")

def main():
    model.train(
        data=BASE_DIR / "image_raw/Smoking Person Detection.v12-2k8_histogramequalization.yolo26/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
        project=BASE_DIR / "image_raw/Smoking Person Detection.v12-2k8_histogramequalization.yolo26/output",

        device=0,

        optimizer="auto",
        lr0=0.01,

        cos_lr=True,

        patience=30,

        cache=True,

        amp=True,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        bgr=0.3,

        degrees=5,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        close_mosaic=10
    )

if __name__ == "__main__":
    main()