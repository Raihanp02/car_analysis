import cv2
from uuid import uuid4
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
print(BASE_DIR)
def capture_frame(video_path, interval_in_seconds, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_in_seconds)

    frame_count = 0
    saved_count = 0

    uuid_base = uuid4().hex
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{uuid_base}-{video_path.stem}-{frame_count}.jpg"
            saved_path = BASE_DIR / Path(output_path) / filename
            cv2.imwrite(saved_path, frame)
            print(f"Frame saved: {saved_path}")
            saved_count += 1

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"Total frames processed: {frame_count}, Total frames saved: {saved_count}")

def read_video(path, interval_in_seconds, output_path):
    if Path(path).is_dir():
        for video_file in (BASE_DIR / Path(path)).glob("*"):
            if video_file.suffix in ['.mp4', '.avi', '.mkv']:
                print(f"Processing video: {video_file}")
                capture_frame(str(video_file), interval_in_seconds, output_path)

    elif Path(path).is_file():
        capture_frame((BASE_DIR / path), interval_in_seconds, output_path)

if __name__ == "__main__":
    video_path = r"C:\Users\AQI\JOB\pakuwon\video\car check Utara 12 Juli 2026-20260714T083613Z-1-001\car check Utara 12 Juli 2026\Camera_01_172.20.20.200_20260712185958_20260712191000_28359715.mp4"
    interval = 2.5
    output_path = "image_raw/screenshoot_cctv"

    read_video(path=video_path, interval_in_seconds=interval, output_path=output_path)