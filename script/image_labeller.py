from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

model = YOLO(
    r"C:\Users\AQI\JOB\pakuwon\yolo26m.pt"
)

image_folder = BASE_DIR / "image_raw/combined2/images"

model.predict(
    source=str(image_folder),
    classes=[2],
    save_txt=True,
    save=False,
    project=BASE_DIR / "image_raw",
    name="combined2",
    exist_ok=True,
)

print("Finished labeling")