from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]

def move_file(base_dataset, folder_name):

    for name in folder_name:
        src_dir = BASE_DIR / f"{base_dataset}/{name}/images"

        for image_path in src_dir.glob("*"):
            if not image_path.is_file():
                continue

            class_name = image_path.stem.split("_")[0]

            class_dir = BASE_DIR / Path(base_dataset) / name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            shutil.move(
                str(image_path),
                str(class_dir / image_path.name)
            )

            print("file moved")

if __name__ == "__main__":
    base_dataset = "image_raw/836cb13d-cartype-full"
    folder_name = ["train", "valid", "test"]

    move_file(base_dataset=base_dataset, folder_name=folder_name)
    