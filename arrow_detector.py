from pathlib import Path

import cv2
import numpy as np


TEMPLATE_PATH = Path(__file__).resolve().parent / "assets" / "templates" / "kill_arrow_white.png"


def compute_arrow_search_band(
    roi_width: int, roi_height: int
) -> tuple[int, int, int, int]:
    x1 = int(roi_width * 0.25)
    x2 = int(roi_width * 0.75)
    return (x1, 0, x2, roi_height)


class ArrowDetector:
    def __init__(self) -> None:
        self.template = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise RuntimeError("Kill arrow template not found")

        self.h, self.w = self.template.shape

    def find_arrow(
        self, frame: np.ndarray, search_band: tuple[int, int, int, int] | None = None
    ):
        # convert frame to HSV so we can isolate white pixels
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 40, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_white, upper_white)
        x1 = 0
        y1 = 0
        x2 = mask.shape[1]
        y2 = mask.shape[0]

        if search_band is not None:
            sx1, sy1, sx2, sy2 = search_band
            x1 = max(0, min(mask.shape[1], sx1))
            y1 = max(0, min(mask.shape[0], sy1))
            x2 = max(x1, min(mask.shape[1], sx2))
            y2 = max(y1, min(mask.shape[0], sy2))

        search_mask = mask[y1:y2, x1:x2]
        if (
            search_mask.shape[0] < self.h
            or search_mask.shape[1] < self.w
        ):
            return None

        result = cv2.matchTemplate(
            search_mask,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.65:
            return None

        x, y = max_loc
        cx = x + self.w // 2
        cy = y + self.h // 2

        return (cx + x1, cy + y1)
