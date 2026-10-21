from pathlib import Path
from vision_transformers import SigLipTransformers, DinoV2, MobileClipONNX
from car_embedding_db.chroma_db import ChromaDB
import cv2

BASE_DIR = Path(__file__).resolve().parents[2]

#siglip_embedder = SigLipTransformers(model_name="google/siglip2-so400m-patch14-384")
dinov2_embedder = DinoV2()
#mobileclip_embedder = MobileClipONNX()
db = ChromaDB(client_path=BASE_DIR / "script/ViT/car_embedding_db", collection_name="cars")

db.initialize_collection(folder_path=BASE_DIR / "image_raw/car_brands", batch_size=16, model=dinov2_embedder)
print(db.collection.count())
# result = db.collection.get()

# print(result)

# img_test = r"C:\Users\AQI\JOB\pakuwon\image_raw\classify\dataset3\renamed\Isuzu_af20608457674ff9be04bb5f1db0a3cb.jpg"
# image = cv2.imread(img_test)

# embedding = siglip_embedder.encode_image(image)

# print(db.query_db(embedding=embedding.cpu().numpy().tolist(), n_results=2))