from email.mime import image

import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
import json
import onnxruntime as ort

class SigLipTransformers:
    def __init__(self, model_name: str = "google/siglip2-so400m-patch14-384"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model.eval()

    def forward(self, images: list[np.ndarray] | np.ndarray, text_prompts: list[str]):
        
        image_features = self.encode_image(images)
        self.text_features = self.encode_text(text_prompts)
    
        probs = torch.sigmoid(image_features @ self.text_features.T * self.model.logit_scale.exp() + self.model.logit_bias)

        return probs
    
    def similarity(self, image_features, text_features):
        # Compute cosine similarity between image and text features
        probs = torch.sigmoid(image_features @ text_features.T * self.model.logit_scale.exp() + self.model.logit_bias)

        return probs
    
    def encode_image(self, bgr_crops: list[np.ndarray] | np.ndarray):
        # bgr_crops: list of HWC numpy arrays, or a single HWC array
        if isinstance(bgr_crops, np.ndarray) and bgr_crops.ndim == 3:
            bgr_crops = [bgr_crops]          # single crop -> wrap as batch of 1

        batch = []
        for crop in bgr_crops:
            resized = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_CUBIC)
            rgb = resized[..., ::-1].copy()   # BGR->RGB, force contiguous copy
            batch.append(rgb)

        arr = np.stack(batch).astype(np.float32) / 255.   # (N, H, W, C), 0-1 range
        tensor = torch.from_numpy(arr)
        tensor = (tensor - 0.5) / 0.5                        # normalize
        tensor = tensor.permute(0, 3, 1, 2).contiguous()      # NHWC -> NCHW

        with torch.no_grad():
            output = self.model.get_image_features(pixel_values=tensor.to(self.device))  # (N, D)

        image_embeds = output.pooler_output
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds
    
    def encode_text(self, class_prompts: list[str]):
        # Manual tokenization — must match training: fixed length, padded
        tokens = self.tokenizer(class_prompts, padding="max_length", max_length=64, return_tensors="pt")

        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self.model.get_text_features(**tokens)   # (num_classes, D) — raw, unnormalized

        # L2-normalize — makes the dot product equal cosine similarity
        text_embeds = output.pooler_output
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds
    
class SigLIPVisionOnnx:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        import onnxruntime as ort

        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def encode_image(self, pixel_values: np.ndarray) -> np.ndarray:
        # pixel_values: (B, 3, 384, 384)
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: pixel_values.astype(np.float32)}
        )
        embedding = outputs[0]
        # L2-normalize
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding

    def preprocess(self, image: np.ndarray | list[np.ndarray]) -> np.ndarray:
        if not isinstance(image, list):
            image = [image]
        
        blob = cv2.dnn.blobFromImages(
            image,
            scalefactor=1.0 / 127.5,        # 1/255 * 2 -> combines rescale (1/255) and std (0.5) division
            size=(384, 384),
            mean=(127.5, 127.5, 127.5),     # subtract before scaling, in 0-255 space
            swapRB=True,                    # BGR -> RGB, handled in the same call
            crop=False,
        )
        return blob

class DinoV2:
    # DINOv2 uses standard ImageNet normalization, not 0.5/0.5
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_name: str = "facebook/dinov2-with-registers-small"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self.model.eval()

    def encode_image(self, bgr_crops: list[np.ndarray] | np.ndarray):
        if isinstance(bgr_crops, np.ndarray) and bgr_crops.ndim == 3:
            bgr_crops = [bgr_crops]

        img_size = self.model.config.image_size  # must be multiple of patch_size (14)

        batch = []
        for crop in bgr_crops:
            resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            rgb = resized[..., ::-1].copy()
            batch.append(rgb)

        arr = np.stack(batch).astype(np.float32) / 255.
        arr = (arr - self._IMAGENET_MEAN) / self._IMAGENET_STD          # ImageNet norm, still HWC
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW

        with torch.no_grad():
            output = self.model(pixel_values=tensor.to(self.device))

        last_hidden_state = output.last_hidden_state           # (N, 1+num_reg+num_patches, D)
        num_registers = self.model.config.num_register_tokens  # 4 by default

        cls_token = last_hidden_state[:, 0]                     # (N, D)
        patch_tokens = last_hidden_state[:, 1 + num_registers:] # (N, num_patches, D)
        patch_mean = patch_tokens.mean(dim=1)                   # (N, D)

        image_embeds = torch.cat([cls_token, patch_mean], dim=-1)  # (N, 2D) — DINOv2-recommended embedding
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

class MobileClipONNX:
    """
    ONNX Runtime wrapper around the MobileCLIP vision tower (Xenova / plhery
    community exports). Produces a single L2-normalized image embedding per
    crop -- NOT a DINOv2-style CLS+patch-mean concat, since MobileCLIP's ONNX
    export already returns a pooled `image_embeds` output.
 
    Example repo ids:
        "Xenova/mobileclip_s0"        (MobileCLIP-S0, transformers.js layout)
        "Xenova/mobileclip_s2"
        "plhery/mobileclip2-onnx"     (MobileCLIP2, subfoldered onnx/s0, onnx/s2, ...)
    """
 
    def __init__(
        self,
        repo_id: str = "Xenova/mobileclip_s0",
        vision_filename: str = "onnx/vision_model_int8.onnx",
        preprocessor_filename: str = "preprocessor_config.json",
        providers: list[str] | None = None,
    ):
        self.repo_id = repo_id
 
        vision_path = hf_hub_download(repo_id=repo_id, filename=vision_filename)
        preprocessor_path = hf_hub_download(repo_id=repo_id, filename=preprocessor_filename)
 
        with open(preprocessor_path) as f:
            self._preproc_cfg = json.load(f)
 
        self._image_size = self._get_image_size(self._preproc_cfg)
        self._mean = np.array(
            self._preproc_cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073]),
            dtype=np.float32,
        )
        self._std = np.array(
            self._preproc_cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711]),
            dtype=np.float32,
        )
        self._rescale = self._preproc_cfg.get("rescale_factor", 1 / 255.0)
 
        if providers is None:
            available = ort.get_available_providers()
            providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
 
        self.session = ort.InferenceSession(vision_path, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]

        # print("Actual providers used:", self.session.get_providers())
        # for i in self.session.get_inputs():
        #     print("input:", i.name, i.shape, i.type)
        # for o in self.session.get_outputs():
        #     print("output:", o.name, o.shape, o.type)
 
    @staticmethod
    def _get_image_size(cfg: dict) -> int:
        size = cfg.get("size") or cfg.get("crop_size")
        if isinstance(size, dict):
            return size.get("height") or size.get("shortest_edge") or 256
        if isinstance(size, int):
            return size
        return 256
 
    def _preprocess(self, bgr_crops: list[np.ndarray]) -> np.ndarray:
        batch = []
        for crop in bgr_crops:
            resized = cv2.resize(
                crop, (self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC
            )
            rgb = resized[..., ::-1].astype(np.float32)
            rgb *= self._rescale
            rgb = (rgb - self._mean) / self._std
            batch.append(rgb)
 
        arr = np.stack(batch)  # (N, H, W, C)
        arr = arr.transpose(0, 3, 1, 2).astype(np.float32)  # NCHW
        return np.ascontiguousarray(arr)
 
    def encode_image(self, bgr_crops: list[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(bgr_crops, np.ndarray) and bgr_crops.ndim == 3:
            bgr_crops = [bgr_crops]
 
        pixel_values = self._preprocess(bgr_crops)
 
        outputs = self.session.run(self._output_names, {self._input_name: pixel_values})
        # Vision projection models expose `image_embeds` as an output; fall back
        # to the first output if the export names it differently.
        if "image_embeds" in self._output_names:
            image_embeds = outputs[self._output_names.index("image_embeds")]
        else:
            image_embeds = outputs[0]
 
        image_embeds = image_embeds.astype(np.float32)
        norm = np.linalg.norm(image_embeds, axis=-1, keepdims=True)
        image_embeds = image_embeds / np.clip(norm, a_min=1e-12, a_max=None)
        return image_embeds
    
if __name__ == "__main__":
    mobilenet = MobileClipONNX()

    img = cv2.imread(r"C:\Users\AQI\JOB\pakuwon\image_raw\staging_area\camera3\8cf5550f314d4ec4b0cbd4479fbbf7d7.jpg")

    result = mobilenet.encode_image(img)
    print(result)

    