import cv2
from ultralytics import YOLO
from car_embedding_db.chroma_db import ChromaDB
from vision_transformers import SigLipTransformers, DinoV2, MobileClipONNX
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class YOLODetector:
    def __init__(self, model_path: str, class_ids: list[int] = None):
        self.model = YOLO(model_path)
        self.class_ids = class_ids

    def detect(self, frame):
        result = self.model(frame, classes=self.class_ids if self.class_ids else None)
        bbox = result[0].boxes.xyxy
        
        return bbox

class InferencePipeline:
    def __init__(self, car_detector, car_embedder, car_classifier_db, record: bool = False):
        self.car_detector = car_detector
        self.car_embedder = car_embedder
        self.car_classifier_db = car_classifier_db
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
        
        embeddings = []
        for cropped in self.cropper_batch(image, filtered_bbox):
            embedding = self.car_embedder.encode_image(cropped).cpu().numpy().tolist()
            embeddings.extend(embedding)
        
        results = self.car_classifier_db.query_db(embedding=embeddings, n_results=2)
        return results["metadatas"], filtered_bbox

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
            label = f"{result[0]['car_brand']} {result[0]['car_model']}"
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
    car_detector = YOLODetector(model_path="yolo26n.pt", class_ids=[2])
    #car_embedder_model = SigLipTransformers(model_name="google/siglip2-so400m-patch14-384")
    car_embedder_model = DinoV2()
    #car_embedder_model = MobileClipONNX()
    
    embedding_db = ChromaDB(client_path=BASE_DIR / "script/ViT/car_embedding_db", collection_name="cars")

    inference_pipeline = InferencePipeline(car_detector=car_detector, car_embedder=car_embedder_model, car_classifier_db=embedding_db, record=False)
    rtsp_url = r"C:\Users\AQI\JOB\pakuwon\video\car check Utara 12 Juli 2026-20260714T083613Z-1-001\car check Utara 12 Juli 2026\Camera_01_172.20.20.200_20260712125959_20260712130959_28174553.mp4"
    #region = [45, 50, 85, 85]
    #region = [15, 35, 75, 84] #305,387,1247,909 (1920,1080)
    #region = [35, 65, 75, 84]
    region = [35, 45, 75, 70]

    inference_pipeline.run(rtsp_url=rtsp_url, output_path="output_video5.mp4", region=region)