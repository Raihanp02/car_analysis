import cv2
import os
from prompt_mapping import prompt_list, class_mapping
import torch
from pathlib import Path
from uuid import uuid4
from brand_classifier import SigLipImageClassifier
from vision_transformers import SigLipTransformers
from script.ViT.naming_pipeline import BrandClassifierAnnotator
from pathlib import Path

class BrandClassifierAnnotator:
    def __init__(self, image_folder, new_directory, brand_classifier, batch_size=16, text_prompts=prompt_list, class_mapping=class_mapping):
        self.brand_classifier = brand_classifier
        self.image_folder = image_folder
        self.new_directory = new_directory
        self.batch_size = batch_size

        self.prompt_list = text_prompts
        self.text_features = self.brand_classifier.model.encode_text(text_prompts)
        self.class_mapping = class_mapping

    def process(self):
        # Classify each detected car

        for image, path in self.image_loader():
            image_features = self.brand_classifier.model.encode_image(image)
            probs = self.brand_classifier.model.similarity(image_features, self.text_features)
            pred_indices = torch.argmax(probs, dim=1)
            pred_brands = [self.class_mapping[self.prompt_list[idx]] for idx in pred_indices]

            for brand, image_path in zip(pred_brands, path):
                self.rename_and_move_file(image_path, brand, self.new_directory)
                print(f"Renamed and moved {image_path} to {brand} in {self.new_directory}")

    def image_loader(self):
        images = []
        path = []
        for filename in os.listdir(self.image_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.image_folder, filename)
                image = cv2.imread(image_path)
                images.append(image)
                path.append(image_path)
                if len(images) % self.batch_size == 0:
                    yield images, path
                    images = []
                    path = []

        if images:
            yield images, path

    def rename_and_move_file(self, original_path, brand, new_directory):
        original_directory = Path(original_path).parent.parent
        new_name = original_directory / new_directory / f"{brand}_{uuid4().hex}.jpg"

        Path(original_path).rename(new_name)

if __name__ == "__main__":
    brand_classifier_model = SigLipTransformers(model_name="google/siglip2-so400m-patch14-384")
    brand_classifier = SigLipImageClassifier(model=brand_classifier_model)

    BASE_DIR = Path(__file__).resolve().parents[2]
    image_folder = BASE_DIR / "image_raw" / "classify" / "dataset3" / "raw"
    new_directory = BASE_DIR / "image_raw" / "classify" / "dataset3" / "classified"
    brand_annotator = BrandClassifierAnnotator(image_folder=image_folder, new_directory="renamed", brand_classifier=brand_classifier)

    brand_annotator.process()
