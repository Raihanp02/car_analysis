import cv2
from main_detector import YOLODetector
from sub_detector import YOLOImageClassification
from pipelines import BodyTypeDetector

def main(video_path, pipelines):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bbox, prediction = pipelines.process(frame)

        for box, predict in zip(bbox, prediction):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),  # BGR color
                2             # thickness
            )

           # draw text
            cv2.putText(
                frame,
                predict,
                (x1, y1 - 10),           # position
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,                     # font scale
                (0, 255, 0),             # color
                2                        # thickness
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("Result", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    car_detector = YOLODetector(model_path="yolo26n.pt", class_ids=[2])
    car_classifier = YOLOImageClassification(model_path=r"C:\Users\AQI\JOB\pakuwon\image_raw\classify\train-2\weights\best.pt")
    pipelines = BodyTypeDetector(detector_model=car_detector, classification_model=car_classifier)

    video_path = r"C:\Users\AQI\JOB\pakuwon\video\Car_cek West-pagi.mp4"

    main(video_path=video_path, pipelines=pipelines)


