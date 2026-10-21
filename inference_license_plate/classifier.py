import numpy as np
import onnxruntime as ort
import cv2

class SwinV2Classifier:
    def __init__(self, model_path: str, image_size: int = 256, threshold: float = 0.5):
        self._image_size = image_size
        self._threshold = threshold
        self.model_path = model_path
        self._session = ort.InferenceSession(
                self.model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
        self._input_name = self._session.get_inputs()[0].name
        self._mean = np.array([
                0.485,
                0.456,
                0.406
            ], dtype=np.float32)
        self._std = np.array([
                0.229,
                0.224,
                0.225
            ], dtype=np.float32)

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """Direct resize to image_size x image_size (matches preprocessor_config.json's
        ViTImageProcessor — no aspect-preserving crop, unlike DINOv2 in object_recognizer.py)
        + ImageNet normalize."""

        img = cv2.resize(rgb, (self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        return np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # HWC -> CHW -> NCHW

    def _classify_crop(self, crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pixel_values = self._preprocess(rgb)
        outputs = self._session.run(None, {self._input_name: pixel_values})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if idx >= len(self._labels) or conf < self._threshold:
            return None
        return {
            "predicted_label": self._labels[idx],
            "prediction_confidence": round(conf, 4),
        }