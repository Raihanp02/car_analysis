import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str, class_ids: list[int] = None):
        self.model = YOLO(model_path)
        self.class_ids = class_ids

    def detect(self, frame):
        result = self.model(frame, classes=self.class_ids if self.class_ids else None)
        bbox = result[0].boxes.xyxy
        
        return bbox

class CarBrandONNX:
    def __init__(self, model_path = r"C:\Users\AQI\JOB\pakuwon\script\ViT\model\resnet18_pruned.onnx"):
        self.model_path = model_path
        self.class_list = ["acura","audi","bmw","chevrolet","chrysler","dodge","ford","gmc","honda","hyundai","infiniti","jeep","kia","lexus","mazda","mercedes","nissan","subaru","toyota","volkswagen"]
        self._load_model()

    def _load_model(self):
        import onnxruntime as ort

        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, images: np.ndarray | list[np.ndarray]) -> np.ndarray:
        preprocessed = self.preprocess(images)

        scores = self.session.run(None, {self.input_name: preprocessed})[0]

        predicted = self.class_list[scores[0].argmax()]
        return scores, predicted
    
    def preprocess(self, image: np.ndarray | list[np.ndarray]) -> np.ndarray:
        if not isinstance(image, list):
            image = [image]
        
        blob = cv2.dnn.blobFromImages(
            images=image,
            scalefactor=1.0,                                     # net-scale-factor=1.0
            size=(224, 224),                        # resize to 224x224
            mean=(103.939, 116.779, 123.68),                       # B, G, R offsets
            swapRB=True,                                          # keep BGR order
            crop=False,
            ddepth=cv2.CV_32F,
        )
        return blob

class RealtimePipeline:
    def __init__(self, car_detector, car_classifier, record: bool = False):
        self.car_detector = car_detector
        self.car_classifier = car_classifier
        self.record = record

    def run(self, rtsp_url, region: list, output_path=None):
        cap = cv2.VideoCapture(rtsp_url)
        
        self.init_cap(cap)
        pixel_region = [
            int((region[0] / 100) * self.width),
            int((region[1] / 100) * self.height),
            int((region[2] / 100) * self.width),
            int((region[3] / 100) * self.height),
        ]

        if self.record:
            writer = self.init_record(cap, output_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results, bbox = self.process(frame, pixel_region)
            self.draw_bbox(frame, bbox, results)

            cv2.rectangle(frame, (pixel_region[0], pixel_region[1]), (pixel_region[2], pixel_region[3]), (255, 0, 0), 2)
            cv2.imshow("Inference Pipeline", frame)

            if self.record:
                writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
           
        cap.release()
        if self.record:
            writer.release()

    def process(self, image, region):
        bbox = self.car_detector.detect(image)

        if len(bbox)==0:
            return [], []

        filtered_bbox = [box for box in bbox if box[2] > region[0] and box[2] < region[2] and box[3] > region[1] and box[3] < region[3]]
        
        if len(filtered_bbox)==0:
            return [], []
        
        predictions = []
        for cropped in self.cropper_batch(image, filtered_bbox):
            scores, predicted = self.car_classifier.forward(cropped)
            predictions.append(predicted)
        
        return predictions, filtered_bbox

    def cropper_batch(self, frame, bbox, batch_size=16):
        cropped_images = []
        for box in bbox:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            
            cropped_image = frame[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
            if len(cropped_images) >= batch_size:
                yield cropped_images
                cropped_images = []
        
        yield cropped_images

    def draw_bbox(self, frame, bbox, results):
        for box, result in zip(bbox, results):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            print(result)
            label = result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    def init_record(self, cap, output_path, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else fps
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        return out
    
    def init_cap(self, cap):
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if __name__ == "__main__":
    car_classifier = CarBrandONNX()

    # img = cv2.imread(r"C:\Users\AQI\JOB\pakuwon\image_raw\staging_area\30b2d7cec5c943be8d6acbbfed395431.jpg")
    # preprocessed = classifier.preprocess(img)
    # print(classifier.forward(preprocessed))

    car_detector = YOLODetector(model_path="yolo26s.pt", class_ids=[2])

    pipeline = RealtimePipeline(car_detector=car_detector, car_classifier=car_classifier, record=True)

    rtsp_url = r"C:\Users\AQI\JOB\pakuwon\video\Car_cek North_pagi.mp4"
    #region = [45, 50, 85, 85]
    region = [15, 35, 75, 84] #305,387,1247,909 (1920,1080)

    pipeline.run(rtsp_url=rtsp_url, region=region, output_path="classification.mp4")