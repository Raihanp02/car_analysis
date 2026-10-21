import numpy as np
import torch

class SigLipImageClassifier:
    def __init__(self, model):
        self.model = model
    
    def classify_by_features(self, image_features, text_features):
        probs = self.model.similarity(image_features, text_features)
        pred = torch.argmax(probs, dim=1)

        return pred

    def classify(self, images: list[np.ndarray] | np.ndarray, text_prompts: list[str]):
        probs = self.model.forward(images, text_prompts)
        return probs