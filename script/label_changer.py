from pathlib import Path

def remap_yolo_labels(label_dir, mapping=None):
    """
    Remap class IDs in YOLO-format .txt label files.

    YOLO format:
        class_id x_center y_center width height

    Args:
        label_dir (str | Path):
            Directory containing YOLO .txt label files.

        mapping (dict | None):
            Class mapping, e.g. {1: 0, 2: 1, 3: 0}.
            If None, all classes are changed to 0.

    Example:
        remap_yolo_labels("labels")
        # All classes -> 0

        remap_yolo_labels(
            "labels",
            mapping={0: 0, 1: 0, 2: 1}
        )
    """
    label_dir = Path(label_dir)

    for txt_file in label_dir.glob("*.txt"):
        new_lines = []

        with txt_file.open("r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # First value is the class ID
                class_id = int(parts[0])

                if mapping is None:
                    new_class_id = 0
                else:
                    if class_id not in mapping:
                        raise ValueError(
                            f"Class {class_id} not found in mapping "
                            f"for {txt_file}"
                        )

                    new_class_id = mapping[class_id]

                parts[0] = str(new_class_id)
                new_lines.append(" ".join(parts))

        with txt_file.open("w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

def erase_yolo_class(label_dir, class_ids):
    """
    Remove annotations with specified class IDs from YOLO label files.

    YOLO format:
        class_id x_center y_center width height

    Args:
        label_dir: Directory containing .txt label files.
        class_ids: int or iterable of ints.

    Examples:
        erase_yolo_class("labels", 2)

        erase_yolo_class("labels", [1, 2, 3])
    """
    label_dir = Path(label_dir)

    if isinstance(class_ids, int):
        class_ids = {class_ids}
    else:
        class_ids = set(class_ids)

    for txt_file in label_dir.glob("*.txt"):
        new_lines = []

        with txt_file.open("r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])

                # Keep everything except the classes we want to erase
                if class_id not in class_ids:
                    new_lines.append(line)

        with txt_file.open("w") as f:
            f.write("\n".join(new_lines))

            if new_lines:
                f.write("\n")

if __name__ == "__main__":
    # All classes -> 0
    
    remap_yolo_labels(r"C:\Users\AQI\JOB\pakuwon\image_raw\Smoking Detection.v5i.yolo26\test\labels")
    remap_yolo_labels(r"C:\Users\AQI\JOB\pakuwon\image_raw\Smoking Detection.v5i.yolo26\train\labels")
    remap_yolo_labels(r"C:\Users\AQI\JOB\pakuwon\image_raw\Smoking Detection.v5i.yolo26\valid\labels")
