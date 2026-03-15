import cv2

from arrow_detector import ArrowDetector


detector = ArrowDetector()

frame = cv2.imread("test_frame.png")

if frame is None:
    raise RuntimeError("test_frame.png could not be loaded")

result = detector.find_arrow(frame)
print("Arrow location:", result)

if result is not None:
    cx, cy = result
    debug = frame.copy()
    cv2.circle(debug, (cx, cy), 8, (0, 255, 0), 2)
    cv2.imwrite("debug_arrow_match.png", debug)
