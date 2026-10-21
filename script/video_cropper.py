import cv2
import time
from ultralytics import YOLO
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parents[1]

class YOLODetector:
    def __init__(self, model_path: str, class_ids: list[int] = None):
        self.model = YOLO(model_path)
        self.class_ids = class_ids

    def detect(self, frame):
        result = self.model(frame, classes=self.class_ids if self.class_ids else None)
        bbox = result[0].boxes.xyxy
        
        return bbox

class VideoCropper:
    def __init__(self, model, saved_path, interval_crop_second=2.5, video_length_duration=180):
        self.model = model
        self.saved_path = saved_path
        self.interval_crop_second = interval_crop_second
        self.video_length_duration = video_length_duration

    def save_cropped(self, frame, bbox, filename):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        cv2.imwrite(self.saved_path / f"{filename}.jpg", frame[y1:y2, x1:x2])

        print(f"cropped frame saved for {filename}")

    def crop_video(self, video_path, target_fps=None):
        cap = cv2.VideoCapture(video_path)

        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or target_fps  # FIX: always defined, safe fallback
        except:
            print(f"error in processing {video_path}")
            return
        
        frame_time = 1.0 / fps

        frame_interval = int(fps * self.interval_crop_second)
        video_max_frame = int(fps * self.video_length_duration)
        frame_count = 0
        
        while True:
            if frame_count > video_max_frame:
                break

            ret, frame = cap.read()

            if not ret:
                break

            start = time.perf_counter()

            if frame_count % frame_interval == 0:
                bbox = self.model.detect(frame)

                for box in bbox:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                    # if (x2-x1) < 112 or (y2-y1) < 112:
                    #     continue

                    uuid_base = uuid4().hex
                    self.save_cropped(frame, box, filename=uuid_base)

            elapsed = time.perf_counter() - start
            sleep_time = frame_time - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

            frame_count = frame_count + 1

    def crop_image(self, image_folder):
        for image_path in image_folder.glob("*"):
            frame = cv2.imread(str(image_path))
            bbox = self.model.detect(frame)

            for box in bbox:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                # if (x2-x1) < 112 or (y2-y1) < 112:
                #     continue

                uuid_base = uuid4().hex
                self.save_cropped(frame, box, filename=uuid_base)

if __name__ == "__main__":
    # model = YOLODetector(r"C:\Users\AQI\JOB\pakuwon\yolo26m.pt", class_ids=[2])
    # cropper = VideoCropper(
    #     model,
    #     BASE_DIR / "image_raw/staging_area/camera3",
    #     interval_crop_second=2,
    #     video_length_duration=700
    # )

    # cropper.crop_video(video_path=r"C:\Users\AQI\JOB\pakuwon\video\car check Utara 12 Juli 2026-20260714T083613Z-1-001\car check Utara 12 Juli 2026\Camera_01_172.20.20.200_20260712125959_20260712130959_28174553.mp4")

    # video_paths = [
    #     "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_DeBritto_View_Utara.stream/chunklist_w984485905.m3u8",
    #     "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Ahmad-Jazuli.stream/chunklist_w2028646296.m3u8",
    #     "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SopMerah1.stream/chunklist_w633276639.m3u8",
    #     "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-FM-Noto-McDonalds.stream/chunklist_w762113331.m3u8",
    #     "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-FM-Noto-Gardu-Aniem.stream/chunklist_w788973340.m3u8",
    #     "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Wardhani.stream/chunklist_w2045180028.m3u8",
    #     "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Yos-Sudarso.stream/chunklist_w843162914.m3u8"]

    # for video_path in video_paths:
    #     print(f"processing video {video_path}")
    #     cropper.crop_video(video_path=video_path)
    #     print(f"{video_path} has been processed\n")

    # "image_raw/license_plate"
    # "image_raw/staging_area/cropped"

    model = YOLODetector(r"C:\Users\AQI\JOB\pakuwon\best.pt")
    cropper = VideoCropper(
        model,
        BASE_DIR / "image_raw/staging_area/cropped",
        video_length_duration=900,
        interval_crop_second=1
    )

    # video_path = r"C:\Users\AQI\JOB\pakuwon\video\2907\D32_20260728180949 (1).mp4"
    # cropper.crop_video(video_path=video_path)

    folder_path = r"C:\Users\AQI\JOB\pakuwon\image_raw\staging_area\cropped"
    cropper.crop_image(image_folder=Path(folder_path))
