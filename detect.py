from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Settings ──────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.40
IMG_SIZE       = 640
SKIP_FRAMES    = 1
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO("yolov8l.pt")
model.to(DEVICE)

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(model.names), 3), dtype=np.uint8)

def preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def extract_boxes(results):
    boxes = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, conf, cls_id))
    return boxes

def draw_detections(frame, boxes_data):
    for (x1, y1, x2, y2, conf, cls_id) in boxes_data:
        label = f"{model.names[cls_id]}  {conf:.0%}"
        color = [int(c) for c in COLORS[cls_id]]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def run_live():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {CAMERA_INDEX} opened at {w}x{h} — press 'q' to quit")

    cached_boxes = []
    frame_idx    = 0
    fps_display  = 0.0
    t_prev       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if frame_idx % SKIP_FRAMES == 0:
            results      = model(
                preprocess(frame),
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                stream=True,
                device=DEVICE,
                verbose=False,
            )
            cached_boxes = extract_boxes(results)

        output = draw_detections(frame.copy(), cached_boxes)

        now         = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev      = now

        cv2.putText(output, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, f"Objects: {len(cached_boxes)}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Live Object Detection", output)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    run_live()