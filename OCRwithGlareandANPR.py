from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import re
from collections import Counter
import time

# ── YOLO + OCR ─────────────────────────────────────────
model  = YOLO("runs/detect/train4/weights/best.onnx")
reader = easyocr.Reader(['en'], gpu=True)

# ── HISTORY ────────────────────────────────────────────
plate_history = []
seen_plates   = set()

# ── GLARE PARAMETERS ───────────────────────────────────
HEADLIGHT_AREA_HIGH     = 15000
HEADLIGHT_AREA_MID      = 10000
ROAD_GLARE_BRIGHT_RATIO = 0.005
SPREAD_RATIO_THRESH     = 0.35

ROI_START_RATIO  = 0.40
ROAD_START_RATIO = 0.70
CLOSE_KERNEL_SIZE = 45


# ── GLARE FUNCTION ─────────────────────────────────────
def detect_glare(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    h, w = gray.shape
    roi_y_start  = int(h * ROI_START_RATIO)
    road_y_start = int(h * ROAD_START_RATIO)

    headlight_roi = blur[roi_y_start:road_y_start, :]
    road_roi      = blur[road_y_start:h, :]

    _, thresh_hl = cv2.threshold(headlight_roi, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    closed_hl = cv2.morphologyEx(thresh_hl, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_hl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_spread = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        spread_ratio = w_box / w

        if area > max_area:
            max_area = area

        if spread_ratio > max_spread:
            max_spread = spread_ratio

    # ✅ Correct place for debug
    print("Max Area:", max_area)

    _, thresh_road = cv2.threshold(road_roi, 200, 255, cv2.THRESH_BINARY)
    road_glare_ratio = np.sum(thresh_road > 0) / road_roi.size

    high_beam = (
        (max_area >= HEADLIGHT_AREA_HIGH)
        or
        (max_area >= HEADLIGHT_AREA_MID and road_glare_ratio > ROAD_GLARE_BRIGHT_RATIO)
        or
        (max_spread > SPREAD_RATIO_THRESH)
    )

    return high_beam


# ── CAMERA ─────────────────────────────────────────────
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    high_beam = detect_glare(frame)
    if high_beam:
        cv2.putText(frame, "GLARE DETECTED", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(frame, "NO GLARE", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # 🔥 TEMP CHANGE FOR TESTING
    if True:   # <-- THIS IS THE ONLY CHANGE
        cv2.putText(frame, "TEST MODE (YOLO ALWAYS ON)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        results = model(frame)

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)

                # Crop plate
                pad = 10
                H, W = frame.shape[:2]
                plate = frame[max(0,y1-pad):min(H,y2+pad),
                              max(0,x1-pad):min(W,x2+pad)]

                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2.0, fy=2.0)

                # ── OCR ─────────────────
                ocr_results = reader.readtext(gray)

                lines = []
                for (bbox, text, conf) in ocr_results:
                    if conf < 0.5:
                        continue

                    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if not cleaned:
                        continue

                    lines.append((bbox, cleaned))

                if not lines:
                    continue

                # 2-line merge
                midpoint = gray.shape[0] / 2

                top_line = []
                bottom_line = []

                for (bbox, text) in lines:
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    x_left = bbox[0][0]

                    if y_center < midpoint:
                        top_line.append((x_left, text))
                    else:
                        bottom_line.append((x_left, text))

                top_line.sort(key=lambda x: x[0])
                bottom_line.sort(key=lambda x: x[0])

                top_text = "".join([t for _, t in top_line])
                bottom_text = "".join([t for _, t in bottom_line])

                if top_text and bottom_text:
                    plate_text = top_text + bottom_text
                else:
                    all_text = sorted(lines, key=lambda x: x[0][0][0])
                    plate_text = "".join([t for _, t in all_text])

                # history
                plate_history.append(plate_text)
                if len(plate_history) > 10:
                    plate_history.pop(0)

                stable_plate = Counter(plate_history).most_common(1)[0][0]

                if stable_plate not in seen_plates:
                    seen_plates.add(stable_plate)
                    print("✅ CAPTURED:", stable_plate)
                    with open("plates.txt", "a") as f:
                        f.write(f"{stable_plate}, {time.strftime('%H:%M:%S')}\n")
                    cv2.imwrite(f"captured_{stable_plate}.jpg", frame)

                # draw
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, stable_plate, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Glare + ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()