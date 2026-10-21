from ultralytics import YOLO

class YOLOImageClassification:
    def __init__(self, model_path, batch_size=8):
        self.model = YOLO(model_path)
        self.class_mapping = self.model.names
        self.batch_size = batch_size

    def classify(self, images):
        results = self.model(images, batch=self.batch_size)
        predictions = []

        for result in results:
            prediction = self.class_mapping[result.probs.top1]
            predictions.append(prediction)

        return predictions