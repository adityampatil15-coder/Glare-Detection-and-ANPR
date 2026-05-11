import cv2
import numpy as np
import os

folder = "test_images"

# --- Tunable thresholds ---
BRIGHT_PIXEL_THRESHOLD  = 220
HEADLIGHT_AREA_HIGH     = 25000
HEADLIGHT_AREA_MID      = 12000
ROAD_GLARE_BRIGHT_RATIO = 0.005
SPREAD_RATIO_THRESH     = 0.35

# --- Zone split points ---
ROI_START_RATIO  = 0.40
ROAD_START_RATIO = 0.70

# --- Morphological closing ---
CLOSE_KERNEL_SIZE = 45

# --- Box drawing: only draw boxes in the HEADLIGHT band (not road noise) ---
# Boxes are only drawn above road_y_start, and must be large enough
BOX_DRAW_MIN_AREA  = 5000   # ignore small noise blobs
BOX_MAX_Y_RATIO    = 0.68   # don't draw boxes below this % of frame height

# --- Display settings ---
DISPLAY_WIDTH  = 600
DISPLAY_HEIGHT = 800

def resize_for_display(img, max_w=DISPLAY_WIDTH, max_h=DISPLAY_HEIGHT):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

for img_name in sorted(os.listdir(folder)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(folder, img_name)
    frame = cv2.imread(path)
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    h, w = gray.shape

    roi_y_start  = int(h * ROI_START_RATIO)
    road_y_start = int(h * ROAD_START_RATIO)
    box_max_y    = int(h * BOX_MAX_Y_RATIO)

    headlight_roi = blur[roi_y_start:road_y_start, :]
    road_roi      = blur[road_y_start:h, :]

    # --- Threshold + close ---
    _, thresh_hl = cv2.threshold(headlight_roi, 200, 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    closed_hl = cv2.morphologyEx(thresh_hl, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_hl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area   = 0
    max_spread = 0
    headlight_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        spread_ratio = w_box / w
        abs_y = y + roi_y_start   # y position in full frame

        if area > 800:
            headlight_boxes.append((x, abs_y, w_box, h_box, area))
            if area > max_area:
                max_area = area
            if spread_ratio > max_spread:
                max_spread = spread_ratio

    # --- Road glare ---
    _, thresh_road   = cv2.threshold(road_roi, 200, 255, cv2.THRESH_BINARY)
    road_glare_ratio = np.sum(thresh_road > 0) / road_roi.size

    # --- Decision ---
    high_beam = (
        (max_area >= HEADLIGHT_AREA_HIGH)
        or
        (max_area >= HEADLIGHT_AREA_MID and road_glare_ratio > ROAD_GLARE_BRIGHT_RATIO)
        or
        (max_spread > SPREAD_RATIO_THRESH)
    )

    result = "HIGH BEAM ON" if high_beam else "LOW BEAM / NORMAL"
    color  = (0, 0, 255) if high_beam else (0, 255, 0)

    # --- Annotate ---
    vis = frame.copy()
    cv2.line(vis, (0, roi_y_start),  (w, roi_y_start),  (0, 255, 255), 2)
    cv2.line(vis, (0, road_y_start), (w, road_y_start), (255, 200, 0), 2)
    cv2.putText(vis, "HEADLIGHT ZONE", (10, roi_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(vis, "ROAD GLARE ZONE", (10, road_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    for (x, y, wb, hb, a) in headlight_boxes:
        # Only draw if: large enough AND above road zone
        if a >= BOX_DRAW_MIN_AREA and y < box_max_y:
            cv2.rectangle(vis, (x, y), (x + wb, y + hb), color, 2)

    font_scale = max(0.6, h / 1500)
    thickness  = max(1, int(h / 600))
    label = (f"{result}  |  area={int(max_area)}  "
             f"road={road_glare_ratio:.4f}  spread={max_spread:.2f}")
    cv2.putText(vis, label, (10, int(h * 0.04)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    print(f"{img_name:30s} → {result}  "
          f"(area={int(max_area)}, road={road_glare_ratio:.4f}, spread={max_spread:.2f})")

    thresh_full = np.zeros_like(gray)
    thresh_full[roi_y_start:road_y_start, :] = thresh_hl
    thresh_full[road_y_start:h, :]           = thresh_road

    cv2.imshow("Detection",  resize_for_display(vis))
    cv2.imshow("Bright Mask (ROI)", resize_for_display(thresh_full))
    cv2.waitKey(0)

cv2.destroyAllWindows()