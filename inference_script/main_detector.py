from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str, class_ids: list[int] = None):
        self.model = YOLO(model_path)
        self.class_ids = class_ids

    def detect(self, frame):
        result = self.model(frame, classes=self.class_ids if self.class_ids else None)
        bbox = result[0].boxes.xyxy
        
        return bbox