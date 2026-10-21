from ultralytics import YOLO
from pathlib import Path
import cv2
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]

# Load model
model = YOLO("yolo26m.pt")

INPUT_DIR = BASE_DIR / Path("image_raw/combined")
OUTPUT_DIR = BASE_DIR / Path("image_raw/filtered")
OUTPUT_DIR.mkdir(exist_ok=True)

# COCO class id for car
CAR_CLASS = 2

# Car bbox must occupy at least 1% of image
MIN_AREA_RATIO = 0.05
for img_path in INPUT_DIR.glob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    image = cv2.imread(str(img_path))
    if image is None:
        continue

    h, w = image.shape[:2]
    frame_area = h * w

    results = model(image, verbose=False)

    found_large_car = False

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])

            if cls != CAR_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            bbox_area = (x2 - x1) * (y2 - y1)
            area_ratio = bbox_area / frame_area

            if area_ratio >= MIN_AREA_RATIO:
                found_large_car = True
                break

        if found_large_car:
            break

    if found_large_car:
        destination = OUTPUT_DIR / img_path.name
        shutil.move(str(img_path), str(destination))
        print(f"Moved: {img_path.name}")