import chromadb
import cv2
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

class ChromaDB:
    def __init__(self, client_path: str = "./chroma_db", collection_name: str = "images"):
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=client_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_to_db(self, image_path: list, embedding: list, car_brand: list, car_model: list):

        self.collection.add(
            ids=image_path,
            embeddings=embedding,
            metadatas=[
                {"car_brand": brand, "car_model": model}
                for brand, model in zip(car_brand, car_model)
            ]
        )

    def query_db(self, embedding: list, n_results=2):
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results
        )
        return results

    def initialize_collection(self, folder_path: str, batch_size: int, model):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)

        image_paths = []
        embeddings = []
        car_brands = []
        car_models = []

        for images, paths in self.image_loader(folder_path, batch_size):
            embedding = model.encode_image(images).cpu().numpy().tolist()
            image_paths.extend(paths)
            embeddings.extend(embedding)

            for path in paths:
                relative = path.relative_to(BASE_DIR / "image_raw/car_brands")
                name = relative.parts[0]
                
                car_brand, car_model = name.split("_", 1) if "_" in name else (name, "")
                car_brands.append(car_brand)
                car_models.append(car_model)

                print(f"Processed {path} -> Brand: {car_brand}, Model: {car_model}")

        self.add_to_db([str(path.as_posix()) for path in image_paths], embeddings, car_brands, car_models)
        print(f"Added {len(image_paths)} images to the ChromaDB collection '{self.collection_name}'.")

    def image_loader(self, folder_path: str, batch_size: int = 16):
        images = []
        path = []
        for image_path in folder_path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image = cv2.imread(str(image_path))
                images.append(image)
                path.append(image_path)
                if len(images) % batch_size == 0:
                    yield images, path
                    images = []
                    path = []

        if images:
            yield images, path
        
