from pathlib import Path
import cv2

CROPS_DIR = Path("artifacts/killfeed_live/crops")
OUTPUT_DIR = Path("artifacts/hero_test_hits")
DEBUG_DIR = OUTPUT_DIR / "debug"
TEMPLATE_PATH = Path("assets/soldier76_killfeed.png")

THRESHOLD = 0.35

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

template = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(f"Could not load template: {TEMPLATE_PATH}")

th, tw = template.shape
print(f"template size: h={th} w={tw}")

hits = 0

for crop_path in sorted(CROPS_DIR.glob("*.jpg")):
    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape
    x1 = 0
    x2 = int(w * 0.45)
    y1 = 0
    y2 = h
    roi = img[y1:y2, x1:x2]

    print(f"{crop_path.name} crop={h}x{w} roi={roi.shape[0]}x{roi.shape[1]}")

    if roi.shape[0] < th or roi.shape[1] < tw:
        print(f"{crop_path.name} skipped: ROI smaller than template")
        continue

    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)

    mx, my = loc
    matched_patch = roi[my:my+th, mx:mx+tw]

    cv2.imwrite(str(DEBUG_DIR / f"{crop_path.stem}_roi.jpg"), roi)
    cv2.imwrite(str(DEBUG_DIR / f"{crop_path.stem}_match_{score:.3f}.jpg"), matched_patch)

    print(f"{crop_path.name} score: {score:.3f}")

    if score >= THRESHOLD:
        cv2.imwrite(str(OUTPUT_DIR / f"{crop_path.stem}_score_{score:.3f}.jpg"), img)
        hits += 1

print(f"\nTotal hits: {hits}")