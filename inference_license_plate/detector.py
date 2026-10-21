from ultralytics import YOLO
import cv2
import onnxruntime as ort
import numpy as np

class YOLODetector:
    def __init__(self, model_path: str, class_ids: list[int] = None):
        self.model = YOLO(model_path)
        self.class_ids = class_ids

    def detect(self, frame):
        result = self.model(frame, classes=self.class_ids if self.class_ids else None)
        bbox = result[0].boxes.xyxy
        return bbox

class YOLOOnnx:
    """
    Runs a pretrained end2end YOLOv9 ONNX plate detector (NMS baked into the graph).
    Works with either single-tensor (1,N,6) output or the 4-tensor
    EfficientNMS layout (num_dets, det_boxes, det_scores, det_classes).
    """

    def __init__(self, model_path: str = r"C:\Users\AQI\.cache\open-image-models\yolo-v9-t-384-license-plate-end2end\yolo-v9-t-384-license-plates-end2end.onnx", 
                 input_size: int = 384,
                 conf_thres: float = 0.4, providers=None):
        self.input_size = input_size
        self.conf_thres = conf_thres

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        # for i in self.session.get_inputs():
        #     print("INPUT :", i.name, i.shape, i.type)
        # for o in self.session.get_outputs():
        #     print("OUTPUT:", o.name, o.shape, o.type)

    def preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_left = (self.input_size - nw) // 2
        pad_top = (self.input_size - nh) // 2
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # NCHW
        return blob, scale, pad_left, pad_top

    def postprocess(self, outputs, scale, pad_left, pad_top):
        preds = outputs[0]  # shape (N, 7)
        detections = []

        for row in preds:
            img_idx, x1, y1, x2, y2, score, cls = row
            if score < self.conf_thres:
                continue
            detections.append(self._unletterbox(x1, y1, x2, y2, score, scale, pad_left, pad_top))

        return detections

    def _unletterbox(self, x1, y1, x2, y2, score, scale, pad_left, pad_top):
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        return {"box": [float(x1), float(y1), float(x2), float(y2)], "score": float(score)}

    def detect(self, img: np.ndarray):
        blob, scale, pad_left, pad_top = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outputs, scale, pad_left, pad_top)

    def crop_plates(self, img: np.ndarray, pad_ratio: float = 0.0):
        """Convenience: returns list of (crop, detection_dict) ready for fast-plate-ocr."""
        h, w = img.shape[:2]
        crops = []
        for det in self.predict(img):
            x1, y1, x2, y2 = det["box"]
            if pad_ratio:
                bw, bh = x2 - x1, y2 - y1
                x1 -= bw * pad_ratio
                x2 += bw * pad_ratio
                y1 -= bh * pad_ratio
                y2 += bh * pad_ratio
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            crops.append((img[y1:y2, x1:x2], det))
        return crops

def draw_bbox(bbox, image, color=(0, 255, 0), thickness=2,
                 show_score=False, font_scale=0.5):
    
    for det in bbox:
        x1, y1, x2, y2 = map(int,det)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if show_score and "score" in det:
            label = f"{det['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(image, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


if __name__ == "__main__":
    model = YOLOOnnx(r"C:\Users\AQI\JOB\pakuwon\best.onnx")
    # model = YOLODetector(r"C:\Users\AQI\JOB\pakuwon\best.pt")

    img = cv2.imread(r"C:\Users\AQI\JOB\pakuwon\image_raw\screenshoot_cctv\97de3efa5ae7444d99dce7b18ee3992b-Camera_01_172.20.20.200_20260712125959_20260712130959_28174553-50.jpg")

    result = model.detect(img)
    draw_bbox(result, img)

    cv2.imshow("res", img)
    cv2.waitKey(0)          # 0 = wait indefinitely until a key is pressed
    cv2.destroyAllWindows()