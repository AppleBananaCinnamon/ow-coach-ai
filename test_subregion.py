import cv2
import numpy as np
import mss

from killfeed_live_detector import Config, extract_killfeed_regions, roi_to_pixels

# Reuse the detector's parent ROI so subregion tuning stays aligned.
KILLFEED_ROI = Config(output_dir="").killfeed_roi
MONITOR_INDEX = Config(output_dir="").monitor_idx  # change if OW is on another monitor
PREVIEW_WIDTH = 1400

DX = 157
DY = 0

BASE_SUBREGIONS = {
    "left_name":  (8, 8, 91, 32),
    "left_icon":  (110, 3, 62, 43),
    "arrow":      (188, 3, 30, 43),
    "right_icon": (226, 3, 69, 43),
    "right_name": (300, 8, 150, 32),
}

def compute_roi_from_monitor(monitor: dict, roi_frac: tuple[float, float, float, float]) -> dict:
    x1_rel, y1_rel, x2_rel, y2_rel = roi_to_pixels(
        monitor["width"], monitor["height"], roi_frac
    )
    return {
        "left": monitor["left"] + x1_rel,
        "top": monitor["top"] + y1_rel,
        "width": x2_rel - x1_rel,
        "height": y2_rel - y1_rel,
    }

with mss.mss() as sct:
    monitor = sct.monitors[MONITOR_INDEX]
    roi = compute_roi_from_monitor(monitor, KILLFEED_ROI)

    print(f"MONITOR_INDEX={MONITOR_INDEX}")
    print(f"monitor={monitor}")
    print(f"roi={roi}")
    print("Controls: WASD move boxes | IJKL resize selected box | TAB next box | ESC quit")

    region_names = list(BASE_SUBREGIONS.keys())
    selected_idx = 0

    while True:
        img = np.array(sct.grab(roi))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        detector_regions = extract_killfeed_regions(frame)
        left_end = detector_regions["left"].shape[1]
        center_end = left_end + detector_regions["center"].shape[1]

        cv2.rectangle(frame, (0, 0), (roi["width"] - 1, roi["height"] - 1), (255, 0, 0), 1)
        cv2.line(frame, (left_end, 0), (left_end, roi["height"] - 1), (255, 128, 0), 1)
        cv2.line(frame, (center_end, 0), (center_end, roi["height"] - 1), (255, 128, 0), 1)
        cv2.putText(
            frame,
            "detector region guides",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 128, 0),
            1,
        )

        for i, name in enumerate(region_names):
            x, y, w, h = BASE_SUBREGIONS[name]
            x += DX
            y += DY

            color = (0, 255, 255) if i == selected_idx else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(
                frame,
                name,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        selected_name = region_names[selected_idx]
        cv2.putText(
            frame,
            f"DX={DX} DY={DY} selected={selected_name}",
            (8, roi["height"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        preview = frame
        if frame.shape[1] != PREVIEW_WIDTH:
            scale = PREVIEW_WIDTH / frame.shape[1]
            preview = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

        cv2.imshow("Killfeed ROI Tuning", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("w"):
            DY -= 1
        elif key == ord("s"):
            DY += 1
        elif key == ord("a"):
            DX -= 1
        elif key == ord("d"):
            DX += 1
        elif key == 9:  # TAB
            selected_idx = (selected_idx + 1) % len(region_names)
        elif key == ord("i"):  # shrink height
            x, y, w, h = BASE_SUBREGIONS[selected_name]
            BASE_SUBREGIONS[selected_name] = (x, y, w, max(1, h - 1))
        elif key == ord("k"):  # grow height
            x, y, w, h = BASE_SUBREGIONS[selected_name]
            BASE_SUBREGIONS[selected_name] = (x, y, w, h + 1)
        elif key == ord("j"):  # shrink width
            x, y, w, h = BASE_SUBREGIONS[selected_name]
            BASE_SUBREGIONS[selected_name] = (x, y, max(1, w - 1), h)
        elif key == ord("l"):  # grow width
            x, y, w, h = BASE_SUBREGIONS[selected_name]
            BASE_SUBREGIONS[selected_name] = (x, y, w + 1, h)
        elif key == 27:
            break

    print("\nFinal values:")
    print(f"DX = {DX}")
    print(f"DY = {DY}")
    print("BASE_SUBREGIONS = {")
    for name in region_names:
        print(f'    "{name}": {BASE_SUBREGIONS[name]},')
    print("}")

cv2.destroyAllWindows()

# reduce arrow box by about 30%
# increase left icon and right icon size by ~10% each
#  da
