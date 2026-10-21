import cv2
from detector import YOLODetector
from ocr import OCR, FastALPROCR
import time
from pathlib import Path
import cvzone
import re
import numpy as np
import supervision as sv  # pip install supervision>=0.16
from classifier import SwinV2Classifier


def main(plate_detector, plate_classifier, ocr, video_path, region, target_fps=None, record_filename=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # FIX: always defined, safe fallback

    if target_fps is None:
        target_fps = fps

    print(f"Video: {w}x{h} @ {fps:.1f} FPS | Target: {target_fps:.1f} FPS")

    # FIX: initialise out to None so the guard below always works
    out = None
    if record_filename:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(record_filename, fourcc, fps, (w, h))

    frame_time = 1.0 / target_fps

    pixel_region = [
        int((region[0] / 100) * w),
        int((region[1] / 100) * h),
        int((region[2] / 100) * w),
        int((region[3] / 100) * h),
    ]

    # ── ByteTrack ────────────────────────────────────────────────────────────
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,          # frames a lost track is kept alive
        minimum_matching_threshold=0.8,
        frame_rate=int(target_fps),
    )

    # best OCR result seen for each track id
    plate_texts: dict[int, str] = {}
    active_ids=set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()

        # ── Detection → supervision.Detections ───────────────────────────────
        raw_results = plate_detector.detect(frame)

        if raw_results is not None and len(raw_results) > 0:
            raw_np = raw_results.cpu().numpy()          # move to CPU once, reuse below
            xyxy = raw_np[:, :4].astype(np.float32)
            confidence = (
                raw_np[:, 4].astype(np.float32)
                if raw_np.shape[1] >= 5
                else np.ones(len(raw_np), dtype=np.float32)
            )
            detections = sv.Detections(xyxy=xyxy, confidence=confidence)
        else:
            detections = sv.Detections.empty()

        # ── ByteTrack update ─────────────────────────────────────────────────
        tracked = tracker.update_with_detections(detections)

        # ── Draw ROI ─────────────────────────────────────────────────────────
        cv2.rectangle(
            frame,
            (pixel_region[0], pixel_region[1]),
            (pixel_region[2], pixel_region[3]),
            (0, 255, 0),
            5,
        )

        # ── Per-track loop ────────────────────────────────────────────────────
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            x1e, y1e, x2e, y2e = _expand_bbox((x1, y1, x2, y2), (h, w), scale=0.25)

            # OCR only when the expanded box lies fully inside the ROI
            in_region = (
                x1e >= pixel_region[0]
                and y1e >= pixel_region[1]
                and x2e <= pixel_region[2]
                and y2e <= pixel_region[3]
            )
            if in_region:
                try:
                    # draw bbox only inside ROI
                    cv2.rectangle(frame, (x1e, y1e), (x2e, y2e), (0, 200, 255), 2)
                    
                    crop = frame[y1e:y2e, x1e:x2e]
                    if crop.size == 0:
                        raise ValueError("empty crop")
                    parsed = ocr.detect(crop)
                    plate_type = plate_classifier._classify_crop(crop)
                    print(plate_type)
                    cv2.putText(
                        frame,
                        f"{plate_type['predicted_label']} {plate_type['prediction_confidence']}",
                        (x1e, y1e - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                    )
                    raw_text = parsed.strip()
                    if len(raw_text) >= 3:
                        current_text = normalize_plate(raw_text)
                        prev = plate_texts.get(track_id, "")
                        if len(current_text) > len(prev):
                            plate_texts[track_id] = current_text
                except (IndexError, TypeError, AttributeError, ValueError):
                    pass

            plate_label = plate_texts.get(track_id, "")
            label = f"#{track_id}  {plate_label}" if plate_label else f"#{track_id}"
            # ── Corner panel ─────────────────────────────────────────────────────
            active_ids = {int(tid) for tid in tracked.tracker_id} if len(tracked) > 0 else set()
        _draw_plate_panel(frame, plate_texts, active_ids, w)

        # ── Display / record ──────────────────────────────────────────────────
        cv2.imshow("Result", frame)          # FIX: imshow before waitKey
        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed = time.perf_counter() - start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if plate_texts:
        print("\n── Detected plates ──")
        for tid, plate in sorted(plate_texts.items()):
            print(f"  Track #{tid}: {plate}")


def _expand_bbox(bbox, img_shape, scale=0.1):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1   # FIX: renamed from w/h to avoid shadowing outer scope
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    new_w = bw * (1 + scale)
    new_h = bh * (1 + scale)
    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(img_shape[1], cx + new_w / 2))
    y2 = int(min(img_shape[0], cy + new_h / 2))
    return x1, y1, x2, y2


def normalize_plate(text):
    text = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    match = re.match(r"([A-Z]+)(\d+)([A-Z]*)", text)
    if not match:
        return text
    prefix, number, suffix = match.groups()
    result = f"{prefix} {number}"
    if suffix:
        result += f" {suffix}"
    return result

def _draw_plate_panel(frame, plate_texts: dict, active_ids: set, frame_w: int):
    """Persistent upper-right panel showing every plate seen so far.
    Green  = car still in frame.
    Grey   = car has passed (track lost, text kept).
    """
    plates = [(tid, txt) for tid, txt in sorted(plate_texts.items()) if txt]
    if not plates:
        return

    PAD      = 12
    LINE_H   = 38
    PANEL_W  = 300
    PANEL_H  = PAD * 2 + LINE_H + len(plates) * LINE_H   # 1 header row + data rows
    X0       = frame_w - PANEL_W - 400
    Y0       = 20

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (X0, Y0), (X0 + PANEL_W, Y0 + PANEL_H), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Header
    cv2.putText(
        frame, "DETECTED PLATES",
        (X0 + PAD, Y0 + PAD + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (X0 + PAD, Y0 + PAD + 26),
        (X0 + PANEL_W - PAD, Y0 + PAD + 26),
        (80, 80, 80), 1,
    )

    for i, (tid, plate) in enumerate(plates):
        y = Y0 + PAD + LINE_H * (i + 1) + 20
        active = tid in active_ids
        color  = (0, 220, 80) if active else (140, 140, 140)   # green vs grey

        # Status dot
        cv2.circle(frame, (X0 + PAD + 6, y - 6), 5, color, cv2.FILLED)

        # Plate text
        cv2.putText(
            frame, plate,
            (X0 + PAD + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA,
        )


if __name__ == "__main__":
    # plate_detector = YOLODetector(r"C:\Users\AQI\JOB\pakuwon\license_plate.pt")
    plate_detector = YOLODetector(r"C:\Users\AQI\JOB\pakuwon\best.pt")
    plate_classifier = SwinV2Classifier(model_path=r"C:\Users\AQI\JOB\pakuwon\models\7fbfc29e02314063abcf8ffad621f65f-swinv2.onnx")
    ocr = FastALPROCR()
    # ocr = OCR()
    video_path = r"C:\Users\AQI\JOB\pakuwon\video\car check Utara 12 Juli 2026-20260714T083613Z-1-001\car check Utara 12 Juli 2026\Camera_01_172.20.20.200_20260712125959_20260712130959_28174553.mp4"
    region = [30, 35, 75, 70]

    main(
        plate_detector=plate_detector,
        plate_classifier=plate_classifier,
        ocr=ocr,
        video_path=video_path,
        region=region,
        record_filename="plate1.mp4",
    )

    # from ultralytics import YOLO
    # plate_detector = YOLO(r"C:\Users\AQI\JOB\pakuwon\license_plate.pt")
    # results = plate_detector(r"C:\Users\AQI\JOB\pakuwon\image_raw\screenshoot_cctv\97de3efa5ae7444d99dce7b18ee3992b-Camera_01_172.20.20.200_20260712125959_20260712130959_28174553-200.jpg")

    # # Draw bounding boxes, labels, confidence, etc.
    # annotated = results[0].plot()

    # cv2.imshow("Prediction", annotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()