from main_detector import YOLODetector
from sub_detector import YOLOImageClassification

class BodyTypeDetector:
    def __init__(self, detector_model, classification_model):
        self.detector = detector_model
        self.classificer = classification_model

    def process(self, image):
        bbox = self.detector.detect(image)

        crops = []
        for box in bbox:
            crops.append(image[int(box[1]):int(box[3]), int(box[0]): int(box[2])])

        predictions = self.classificer.classify(crops)

        return bbox, predictions
