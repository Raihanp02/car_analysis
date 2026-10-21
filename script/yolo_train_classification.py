from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "yolo26s.pt"
MODEL_PATH = BASE_DIR / MODEL_NAME

model = YOLO("yolo26n-cls")

def main():
    model.train(
        data=BASE_DIR / "image_raw/classify/dataset2/Cars_Body_Type",
        epochs=100,
        imgsz=224,
        batch=16,
        workers=0,
        project=BASE_DIR / "image_raw/classify/dataset2",

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
        scale=0.1,
        fliplr=0.5,
        erasing=0.2
    )

if __name__ == "__main__":
    main()