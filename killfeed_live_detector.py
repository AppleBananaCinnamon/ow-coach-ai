from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse

import cv2
import mss
import numpy as np

from arrow_detector import ArrowDetector, compute_arrow_search_band
from killfeed_parser import run_parser


@dataclass
class FeedDetection:
    frame_idx: int
    ts_sec: float
    crop_path: str
    motion_score: float
    signal: float
    red_ratio: float
    cyan_ratio: float
    white_ratio: float
    bbox_xyxy: tuple[int, int, int, int]
    row_idx: int
    monitor_idx: int
    right_icon_box: Optional[tuple[int, int, int, int]]


@dataclass
class BurstCandidate:
    frame_idx: int
    ts_sec: float
    crop: np.ndarray
    motion_score: float
    signal: float
    red_ratio: float
    cyan_ratio: float
    white_ratio: float
    arrow_center: Optional[tuple[int, int]]
    crop_arrow_center: Optional[tuple[int, int]]
    right_icon_box: Optional[tuple[int, int, int, int]]


ARROW_ANCHORED_SUBREGIONS = {
    "left_name": (-180, 5, 91, 32),
    "left_icon": (-78, 0, 62, 43),
    "arrow": (0, 0, 30, 43),
    "right_icon": (38, 0, 69, 43),
    "right_name": (112, 5, 150, 32),
}

EXPORT_RIGHT_ICON_DEBUG = True


@dataclass
class Config:
    output_dir: str
    sample_fps: float = 4.0
    monitor_idx: int = 1
    # x1, y1, x2, y2 normalized to selected monitor
    killfeed_roi: tuple[float, float, float, float] = (0.75, 0.15, 0.985, 0.185)
    diff_threshold: float = 0.03
    min_gap_sec: float = 0.6
    resize_width: int = 700
    blur_kernel: int = 3
    debug: bool = False
    duration_sec: Optional[float] = None
    show_preview: bool = False
    burst_count: int = 3
    save_format: str = "jpg"  # "jpg" or "png"
    sat_threshold: float = 0.05  # fraction of saturated pixels required
    color_signal_threshold: float = 0.08
    event_cooldown_sec: float = 0.9
    save_burst_debug_candidates: bool = False
    startup_warmup_sec: float = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live killfeed detector for OW replay viewer"
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/killfeed_live")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument(
        "--monitor-idx",
        type=int,
        default=1,
        help="mss monitor index; usually 1 is primary",
    )
    parser.add_argument("--diff-threshold", type=float, default=0.03)
    parser.add_argument("--min-gap-sec", type=float, default=0.6)
    parser.add_argument("--event-cooldown-sec", type=float, default=0.6)
    parser.add_argument(
        "--duration-sec", type=float, default=None, help="Optional max run time"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show live preview window; press q to quit",
    )
    parser.add_argument(
        "--burst-count", type=int, default=3, help="Save N crops per detected hit"
    )
    parser.add_argument(
        "--save-format", type=str, default="jpg", choices=["jpg", "png"]
    )
    parser.add_argument("--save-burst-debug-candidates", action="store_true")
    parser.add_argument("--startup-warmup-sec", type=float, default=0.75)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path: Path, data: object) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data))
        f.write("\n")


def path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


def uri_to_local_path(path_or_uri: str) -> Path:
    parsed = urlparse(path_or_uri)
    if parsed.scheme != "file":
        return Path(path_or_uri)

    local_path = unquote(parsed.path)
    if local_path.startswith("/") and len(local_path) >= 3 and local_path[2] == ":":
        local_path = local_path[1:]
    return Path(local_path)


def roi_to_pixels(
    width: int, height: int, roi: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    x1 = int(roi[0] * width)
    y1 = int(roi[1] * height)
    x2 = int(roi[2] * width)
    y2 = int(roi[3] * height)
    return x1, y1, x2, y2


def preprocess_crop(
    crop_bgr: np.ndarray, resize_width: int, blur_kernel: int
) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    scale = resize_width / gray.shape[1]
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if blur_kernel > 1:
        resized = cv2.GaussianBlur(resized, (blur_kernel, blur_kernel), 0)
    return resized


def diff_score(curr: np.ndarray, prev: Optional[np.ndarray]) -> float:
    if prev is None:
        return 0.0
    diff = cv2.absdiff(curr, prev)
    return float(np.mean(diff))


def saturation_ratio(crop_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # count pixels with meaningful saturation
    sat_pixels = np.sum(saturation > 60)

    return sat_pixels / saturation.size


def killfeed_color_signal(crop_bgr: np.ndarray) -> tuple[float, float, float, float]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # OpenCV hue ranges 0-179
    # Red wraps around 0, so use two ranges
    red_mask = ((h <= 10) | (h >= 170)) & (s >= 90) & (v >= 80)

    # Cyan / blue team color
    cyan_mask = (h >= 80) & (h <= 110) & (s >= 70) & (v >= 80)

    # Bright white text / chevron
    white_mask = (s <= 45) & (v >= 180)

    red_ratio = float(np.mean(red_mask))
    cyan_ratio = float(np.mean(cyan_mask))
    white_ratio = float(np.mean(white_mask))
    signal = red_ratio + cyan_ratio + white_ratio

    return signal, red_ratio, cyan_ratio, white_ratio


def killfeed_ui_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    red_mask = ((h <= 10) | (h >= 170)) & (s >= 90) & (v >= 80)
    cyan_mask = (h >= 80) & (h <= 110) & (s >= 70) & (v >= 80)
    white_mask = (s <= 45) & (v >= 180)

    mask = (red_mask | cyan_mask | white_mask).astype(np.uint8) * 255

    # denoise / connect nearby UI bits
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def diff_score_binary(curr: np.ndarray, prev: Optional[np.ndarray]) -> float:
    if prev is None:
        return 0.0
    diff = cv2.absdiff(curr, prev)
    return float(np.mean(diff)) / 255.0


def has_horizontal_bar_structure(
    mask: np.ndarray, min_row_fill: float = 0.18, min_rows: int = 6
) -> bool:
    row_fill = np.mean(mask > 0, axis=1)  # fraction of "on" pixels in each row
    return int(np.sum(row_fill >= min_row_fill)) >= min_rows


def list_monitors() -> None:
    with mss.mss() as sct:
        print("Available monitors:")
        for idx, mon in enumerate(sct.monitors):
            if idx == 0:
                print(f"  {idx}: virtual bounding monitor {mon}")
            else:
                print(
                    f"  {idx}: {mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})"
                )


def save_crop_image(path: Path, crop: np.ndarray, save_format: str) -> None:
    if save_format == "jpg":
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    elif save_format == "png":
        cv2.imwrite(str(path), crop)
    else:
        raise ValueError(f"Unsupported save_format={save_format}")


def detection_rank(detection: FeedDetection) -> float:
    return detection.motion_score * detection.signal


def clip_box_to_image(
    box: tuple[int, int, int, int], image_shape: tuple[int, ...]
) -> tuple[int, int, int, int] | None:
    x, y, w, h = box
    img_h, img_w = image_shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def subregion_coverage(
    crop_shape: tuple[int, ...], box: tuple[int, int, int, int]
) -> float:
    clipped = clip_box_to_image(box, crop_shape)
    if clipped is None:
        return 0.0
    _, _, w, h = clipped
    expected_area = max(box[2] * box[3], 1)
    return float((w * h) / expected_area)


def candidate_structure_rank(
    candidate: BurstCandidate, arrow_detector: ArrowDetector
) -> tuple[int, int, int, int, int, float]:
    if candidate.crop_arrow_center is None:
        return (0, 0, 0, 0, 0, candidate.signal)

    arrow_x1 = int(round(candidate.crop_arrow_center[0] - (arrow_detector.w / 2)))
    arrow_y1 = int(round(candidate.crop_arrow_center[1] - (arrow_detector.h / 2)))

    visible_regions = 0
    icon_regions = 0
    name_regions = 0
    left_icon_coverage = 0.0
    right_icon_coverage = 0.0

    for name, (dx, dy, w, h) in ARROW_ANCHORED_SUBREGIONS.items():
        coverage = subregion_coverage(
            candidate.crop.shape, (arrow_x1 + dx, arrow_y1 + dy, w, h)
        )
        if name == "left_icon":
            left_icon_coverage = coverage
        elif name == "right_icon":
            right_icon_coverage = coverage
        if coverage >= 0.85:
            visible_regions += 1
            if "icon" in name:
                icon_regions += 1
            elif "name" in name:
                name_regions += 1

    both_icons_present = left_icon_coverage >= 0.85 and right_icon_coverage >= 0.85
    both_icons_full = left_icon_coverage >= 0.98 and right_icon_coverage >= 0.98
    both_names_present = name_regions == 2

    return (
        1,
        1 if both_icons_full else 0,
        1 if both_icons_present else 0,
        1 if both_names_present else 0,
        int(round((left_icon_coverage + right_icon_coverage) * 1000)),
        visible_regions + candidate.signal,
    )


def candidate_structure_debug(
    candidate: BurstCandidate, arrow_detector: ArrowDetector
) -> dict[str, object]:
    if candidate.crop_arrow_center is None:
        return {
            "arrow_present": False,
            "left_icon_coverage": 0.0,
            "right_icon_coverage": 0.0,
            "both_icons_full": False,
            "both_icons_present": False,
            "both_names_present": False,
            "visible_regions": 0,
            "signal": candidate.signal,
            "rank": list(candidate_structure_rank(candidate, arrow_detector)),
        }

    arrow_x1 = int(round(candidate.crop_arrow_center[0] - (arrow_detector.w / 2)))
    arrow_y1 = int(round(candidate.crop_arrow_center[1] - (arrow_detector.h / 2)))
    visible_regions = 0
    name_regions = 0
    left_icon_coverage = 0.0
    right_icon_coverage = 0.0

    for name, (dx, dy, w, h) in ARROW_ANCHORED_SUBREGIONS.items():
        coverage = subregion_coverage(
            candidate.crop.shape, (arrow_x1 + dx, arrow_y1 + dy, w, h)
        )
        if name == "left_icon":
            left_icon_coverage = coverage
        elif name == "right_icon":
            right_icon_coverage = coverage
        if coverage >= 0.85:
            visible_regions += 1
            if "name" in name:
                name_regions += 1

    return {
        "arrow_present": True,
        "left_icon_coverage": round(left_icon_coverage, 3),
        "right_icon_coverage": round(right_icon_coverage, 3),
        "both_icons_full": left_icon_coverage >= 0.98 and right_icon_coverage >= 0.98,
        "both_icons_present": left_icon_coverage >= 0.85 and right_icon_coverage >= 0.85,
        "both_names_present": name_regions == 2,
        "visible_regions": visible_regions,
        "signal": candidate.signal,
        "rank": list(candidate_structure_rank(candidate, arrow_detector)),
    }


def draw_arrow_overlay(
    image: np.ndarray,
    arrow_center: Optional[tuple[int, int]],
    origin_x: int = 0,
    origin_y: int = 0,
) -> np.ndarray:
    debug_img = image.copy()
    if arrow_center is None:
        return debug_img

    cx = arrow_center[0] - origin_x
    cy = arrow_center[1] - origin_y
    height, width = debug_img.shape[:2]

    if 0 <= cx < width and 0 <= cy < height:
        cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), 2)

    return debug_img


def extract_killfeed_regions(crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
    width = crop_bgr.shape[1]
    left_end = int(width * 0.42)
    center_end = int(width * 0.58)

    return {
        "left": crop_bgr[:, :left_end],
        "center": crop_bgr[:, left_end:center_end],
        "right": crop_bgr[:, center_end:],
    }


def region_fingerprint(region_bgr: np.ndarray) -> np.ndarray:
    mask = killfeed_ui_mask(region_bgr)
    return cv2.resize(mask, (32, 12), interpolation=cv2.INTER_AREA)


def compute_row_signature(crop_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop = cv2.imread(str(uri_to_local_path(crop_path)), cv2.IMREAD_COLOR)
    if crop is None or crop.size == 0:
        empty = np.zeros((12, 32), dtype=np.uint8)
        return empty, empty, empty

    regions = extract_killfeed_regions(crop)
    return (
        region_fingerprint(regions["left"]),
        region_fingerprint(regions["center"]),
        region_fingerprint(regions["right"]),
    )


def crop_box(
    image_bgr: np.ndarray, box: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    if box is None:
        return None
    x, y, w, h = box
    region = image_bgr[y : y + h, x : x + w]
    return region if region.size > 0 else None


def export_right_icon_debug(
    crop_path: str,
    right_icon: np.ndarray | None,
    accepted: bool,
    coverage: float,
) -> None:
    if not EXPORT_RIGHT_ICON_DEBUG:
        return

    local_crop_path = uri_to_local_path(crop_path)
    debug_dir = local_crop_path.parent.parent / "right_icon_debug"
    ensure_dir(debug_dir)

    status = "accepted" if accepted else "rejected"
    out_path = debug_dir / (
        f"{local_crop_path.stem}_righticon_{status}_cov{int(round(coverage * 1000)):04d}.png"
    )

    if right_icon is None or right_icon.size == 0:
        placeholder = np.zeros((43, 69, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), placeholder)
        return

    cv2.imwrite(str(out_path), right_icon)


def compute_arrow_anchored_subregions(
    crop_bgr: np.ndarray, arrow_center: tuple[int, int], arrow_detector: ArrowDetector
) -> dict[str, tuple[int, int, int, int]]:
    arrow_x1 = int(round(arrow_center[0] - (arrow_detector.w / 2)))
    arrow_y1 = int(round(arrow_center[1] - (arrow_detector.h / 2)))
    regions: dict[str, tuple[int, int, int, int]] = {}

    for name, (dx, dy, w, h) in ARROW_ANCHORED_SUBREGIONS.items():
        clipped = clip_box_to_image((arrow_x1 + dx, arrow_y1 + dy, w, h), crop_bgr.shape)
        if clipped is not None:
            regions[name] = clipped

    return regions


def subregion_fingerprint(region_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def compute_right_icon_signature(
    crop_path: str, right_icon_box: Optional[tuple[int, int, int, int]]
) -> np.ndarray | None:
    crop = cv2.imread(str(uri_to_local_path(crop_path)), cv2.IMREAD_COLOR)
    if crop is None or crop.size == 0:
        return None
    if right_icon_box is None:
        export_right_icon_debug(crop_path, None, accepted=False, coverage=0.0)
        return None
    right_icon_coverage = subregion_coverage(crop.shape, right_icon_box)
    right_icon = crop_box(crop, right_icon_box)
    if right_icon is None:
        export_right_icon_debug(crop_path, None, accepted=False, coverage=0.0)
        return None

    export_right_icon_debug(
        crop_path, right_icon, accepted=True, coverage=right_icon_coverage
    )
    return subregion_fingerprint(right_icon, (32, 32))


def structural_similarity(a: FeedDetection, b: FeedDetection) -> float:
    sig_a = compute_right_icon_signature(a.crop_path, a.right_icon_box)
    sig_b = compute_right_icon_signature(b.crop_path, b.right_icon_box)
    if sig_a is None or sig_b is None:
        return 0.0

    diff = cv2.absdiff(sig_a, sig_b)
    return 1.0 - (float(np.mean(diff)) / 255.0)


def crop_similarity(crop_path_a: str, crop_path_b: str) -> float:
    sig_a = compute_row_signature(crop_path_a)
    sig_b = compute_row_signature(crop_path_b)

    region_scores = []
    for region_a, region_b in zip(sig_a, sig_b):
        diff = cv2.absdiff(region_a, region_b)
        region_scores.append(1.0 - (float(np.mean(diff)) / 255.0))

    return float(np.mean(region_scores))


def dedupe_detections_visual(
    detections: List[FeedDetection],
    max_gap_sec: float = 5.0,
    similarity_threshold: float = 0.98,
) -> List[FeedDetection]:
    if not detections:
        return []

    deduped: List[FeedDetection] = []
    debug_path = Path("artifacts/killfeed_live/dedupe_debug.jsonl")

    for detection in detections:
        best_match_idx: int | None = None
        best_match_similarity = 0.0

        for idx, prev in enumerate(deduped):
            time_gap_sec = detection.ts_sec - prev.ts_sec
            if time_gap_sec < 0 or time_gap_sec > max_gap_sec:
                continue

            similarity = structural_similarity(prev, detection)
            merged = similarity >= similarity_threshold
            append_jsonl(
                debug_path,
                {
                    "prev_crop_path": prev.crop_path,
                    "curr_crop_path": detection.crop_path,
                    "prev_row_idx": prev.row_idx,
                    "curr_row_idx": detection.row_idx,
                    "time_gap_sec": round(time_gap_sec, 3),
                    "similarity": round(similarity, 4),
                    "threshold": similarity_threshold,
                    "merged": merged,
                },
            )

            if merged and similarity > best_match_similarity:
                best_match_idx = idx
                best_match_similarity = similarity

        if best_match_idx is not None:
            prev = deduped[best_match_idx]
            deduped[best_match_idx] = max(prev, detection, key=detection_rank)
        else:
            deduped.append(detection)

    return deduped


def estimate_row_idx_from_crop(
    crop_path: str, top_y: int = 0, row_height: int = 24, arrow_y: Optional[int] = None
) -> int:
    crop = cv2.imread(str(uri_to_local_path(crop_path)), cv2.IMREAD_GRAYSCALE)
    if crop is None or crop.size == 0:
        return 0

    norm = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)

    # Use the brightest horizontal band as a simple proxy for the active killfeed row.
    row_signal = np.mean(blur, axis=1)

    if row_height <= 0:
        return 0

    fallback_row_idx = int(np.argmax(row_signal) / row_height)

    if arrow_y is not None:
        row_idx = int(round((arrow_y - top_y) / row_height))
    else:
        row_idx = fallback_row_idx

    return max(0, min(row_idx, 2))


def detect_feed_changes_live(cfg: Config) -> List[FeedDetection]:
    out_dir = Path(cfg.output_dir)
    crops_dir = out_dir / "crops"
    debug_dir = out_dir / "debug_frames"
    arrow_debug_path = out_dir / "arrow_debug.jsonl"
    best_selection_debug_path = out_dir / "best_selection_debug.jsonl"
    ensure_dir(out_dir)
    ensure_dir(crops_dir)
    if cfg.debug:
        ensure_dir(debug_dir)

    detections: List[FeedDetection] = []
    prev_proc: Optional[np.ndarray] = None
    frame_idx = 0
    burst_remaining = 0
    burst_event_idx = 0
    last_event_ts = -999.0
    burst_candidates: List[BurstCandidate] = []
    arrow_detector = ArrowDetector()
    candidates_skipped_no_arrow = 0
    candidates_skipped_warmup = 0

    with mss.mss() as sct:
        if cfg.monitor_idx < 1 or cfg.monitor_idx >= len(sct.monitors):
            raise ValueError(
                f"Invalid monitor_idx={cfg.monitor_idx}. Run with a valid monitor index."
            )

        mon = sct.monitors[cfg.monitor_idx]
        mon_left = mon["left"]
        mon_top = mon["top"]
        mon_width = mon["width"]
        mon_height = mon["height"]

        x1_rel, y1_rel, x2_rel, y2_rel = roi_to_pixels(
            mon_width, mon_height, cfg.killfeed_roi
        )

        start_time = time.perf_counter()
        frame_period = 1.0 / cfg.sample_fps

        while True:
            loop_start = time.perf_counter()
            ts_sec = loop_start - start_time
            can_start_new_event = (ts_sec - last_event_ts) >= cfg.event_cooldown_sec
            in_startup_warmup = ts_sec < cfg.startup_warmup_sec

            if cfg.duration_sec is not None and ts_sec >= cfg.duration_sec:
                break

            raw = np.array(sct.grab(mon))
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            crop = frame_bgr[y1_rel:y2_rel, x1_rel:x2_rel]
            row_height = max(1, int(round(crop.shape[0] / 3)))

            mask = killfeed_ui_mask(crop)
            score = diff_score_binary(mask, prev_proc)
            has_bar = has_horizontal_bar_structure(mask)

            signal, red_ratio, cyan_ratio, white_ratio = killfeed_color_signal(crop)

            should_emit_base = (
                score >= cfg.diff_threshold and has_bar and can_start_new_event
            )
            if in_startup_warmup and should_emit_base:
                candidates_skipped_warmup += 1

            should_emit = should_emit_base and not in_startup_warmup

            if should_emit:
                burst_remaining = max(cfg.burst_count, 1)
                burst_event_idx += 1
                last_event_ts = ts_sec
                burst_candidates = []

            crop_arrow_center = None
            arrow_center = None
            if should_emit or burst_remaining > 0:
                arrow_search_band = compute_arrow_search_band(
                    crop.shape[1], crop.shape[0]
                )
                crop_arrow_center = arrow_detector.find_arrow(crop, arrow_search_band)
                if crop_arrow_center is not None:
                    arrow_center = (
                        crop_arrow_center[0] + x1_rel,
                        crop_arrow_center[1] + y1_rel,
                    )

            append_jsonl(
                arrow_debug_path,
                {
                    "frame_idx": frame_idx,
                    "ts_sec": ts_sec,
                    "arrow_center": arrow_center,
                    "arrow_detected": arrow_center is not None,
                },
            )

            if burst_remaining > 0:
                burst_seq = cfg.burst_count - burst_remaining

                should_keep_burst_candidate = burst_seq == 0 or (
                    has_bar and signal >= cfg.color_signal_threshold
                )

                if should_keep_burst_candidate:
                    if crop_arrow_center is None:
                        candidates_skipped_no_arrow += 1

                    candidate = BurstCandidate(
                        frame_idx=frame_idx,
                        ts_sec=round(ts_sec, 3),
                        crop=crop.copy(),
                        motion_score=round(score, 3),
                        signal=round(signal, 3),
                        red_ratio=round(red_ratio, 3),
                        cyan_ratio=round(cyan_ratio, 3),
                        white_ratio=round(white_ratio, 3),
                        arrow_center=arrow_center,
                        crop_arrow_center=crop_arrow_center,
                        right_icon_box=(
                            compute_arrow_anchored_subregions(
                                crop, crop_arrow_center, arrow_detector
                            ).get("right_icon")
                            if crop_arrow_center is not None
                            else None
                        ),
                    )
                    burst_candidates.append(candidate)

                    if cfg.save_burst_debug_candidates:
                        ext = cfg.save_format
                        debug_candidate_name = (
                            f"killfeed_evt{burst_event_idx:05d}_"
                            f"cand{burst_seq}_"
                            f"f{candidate.frame_idx:07d}_"
                            f"t{candidate.ts_sec:08.2f}."
                            f"{ext}"
                        )
                        debug_candidate_path = crops_dir / debug_candidate_name
                        debug_candidate_img = draw_arrow_overlay(
                            candidate.crop,
                            candidate.arrow_center,
                            origin_x=x1_rel,
                            origin_y=y1_rel,
                        )
                        save_crop_image(
                            debug_candidate_path,
                            debug_candidate_img,
                            cfg.save_format,
                        )

                burst_remaining -= 1

                if burst_remaining == 0 and burst_candidates:
                    candidate_debug_rows = []
                    for burst_idx, candidate in enumerate(burst_candidates):
                        candidate_debug_rows.append(
                            {
                                "event_idx": burst_event_idx,
                                "candidate_idx": burst_idx,
                                "frame_idx": candidate.frame_idx,
                                "ts_sec": candidate.ts_sec,
                                "motion_score": candidate.motion_score,
                                "signal": candidate.signal,
                                **candidate_structure_debug(candidate, arrow_detector),
                            }
                        )
                    best_candidate = max(
                        burst_candidates,
                        key=lambda candidate: candidate_structure_rank(
                            candidate, arrow_detector
                        ),
                    )
                    for row in candidate_debug_rows:
                        row["selected"] = row["frame_idx"] == best_candidate.frame_idx
                        append_jsonl(best_selection_debug_path, row)

                    ext = cfg.save_format
                    crop_name = (
                        f"killfeed_evt{burst_event_idx:05d}_"
                        f"best_"
                        f"f{best_candidate.frame_idx:07d}_"
                        f"t{best_candidate.ts_sec:08.2f}."
                        f"{ext}"
                    )
                    crop_path = crops_dir / crop_name
                    best_candidate_img = draw_arrow_overlay(
                        best_candidate.crop,
                        best_candidate.arrow_center,
                        origin_x=x1_rel,
                        origin_y=y1_rel,
                    )
                    save_crop_image(crop_path, best_candidate_img, cfg.save_format)

                    bbox_abs = (
                        mon_left + x1_rel,
                        mon_top + y1_rel,
                        mon_left + x2_rel,
                        mon_top + y2_rel,
                    )

                    detections.append(
                        FeedDetection(
                            frame_idx=best_candidate.frame_idx,
                            ts_sec=best_candidate.ts_sec,
                            crop_path=path_to_uri(crop_path),
                            motion_score=best_candidate.motion_score,
                            signal=best_candidate.signal,
                            red_ratio=best_candidate.red_ratio,
                            cyan_ratio=best_candidate.cyan_ratio,
                            white_ratio=best_candidate.white_ratio,
                            bbox_xyxy=bbox_abs,
                            row_idx=estimate_row_idx_from_crop(
                                path_to_uri(crop_path),
                                top_y=y1_rel,
                                row_height=row_height,
                                arrow_y=(
                                    best_candidate.arrow_center[1]
                                    if best_candidate.arrow_center is not None
                                    else None
                                ),
                            ),
                            monitor_idx=cfg.monitor_idx,
                            right_icon_box=best_candidate.right_icon_box,
                        )
                    )

                    h, w = best_candidate.crop.shape[:2]
                    print(
                        f"[save] evt={burst_event_idx:05d} "
                        f"t={best_candidate.ts_sec:7.2f}s "
                        f"score={best_candidate.motion_score:6.2f} "
                        f"sig={best_candidate.signal:0.3f} "
                        f"size={w}x{h} saved={crop_path.name}"
                    )

            if cfg.debug or cfg.show_preview:
                frame_dbg = frame_bgr.copy()
                frame_dbg = draw_arrow_overlay(frame_dbg, arrow_center)
                cv2.rectangle(
                    frame_dbg, (x1_rel, y1_rel), (x2_rel, y2_rel), (0, 255, 0), 2
                )
                label = f"t={ts_sec:.2f}s score={score:.1f}"
                cv2.putText(
                    frame_dbg,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                if cfg.debug:
                    debug_name = debug_dir / f"frame_{frame_idx:07d}.jpg"
                    cv2.imwrite(str(debug_name), frame_dbg)

                if cfg.show_preview:
                    preview = frame_dbg
                    max_preview_width = 1600
                    if preview.shape[1] > max_preview_width:
                        scale = max_preview_width / preview.shape[1]
                        preview = cv2.resize(
                            preview,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    cv2.imshow("OW Killfeed Detector", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            prev_proc = mask
            frame_idx += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, frame_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if cfg.show_preview:
        cv2.destroyAllWindows()

    setattr(cfg, "_candidates_skipped_no_arrow", candidates_skipped_no_arrow)
    setattr(cfg, "_candidates_skipped_warmup", candidates_skipped_warmup)
    return detections


def cluster_detections(
    detections: List[FeedDetection], gap_sec: float = 0.6
) -> List[FeedDetection]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.ts_sec)

    clusters = []
    current_cluster = [detections[0]]

    for d in detections[1:]:
        if d.ts_sec - current_cluster[-1].ts_sec <= gap_sec:
            current_cluster.append(d)
        else:
            clusters.append(current_cluster)
            current_cluster = [d]

    clusters.append(current_cluster)

    representatives = []

    for cluster in clusters:
        def cluster_rank(detection: FeedDetection) -> float:
            return detection_rank(detection)

        best = max(cluster, key=cluster_rank)
        representatives.append(best)

    return representatives


def main() -> None:
    args = parse_args()

    cfg = Config(
        output_dir=args.output_dir,
        sample_fps=args.sample_fps,
        monitor_idx=args.monitor_idx,
        diff_threshold=args.diff_threshold,
        min_gap_sec=args.min_gap_sec,
        duration_sec=args.duration_sec,
        debug=args.debug,
        show_preview=args.show_preview,
        burst_count=args.burst_count,
        save_format=args.save_format,
        event_cooldown_sec=args.event_cooldown_sec,
        save_burst_debug_candidates=args.save_burst_debug_candidates,
        startup_warmup_sec=args.startup_warmup_sec,
    )

    list_monitors()
    print(f"\nUsing monitor index: {cfg.monitor_idx}")
    print(
        "Start your Overwatch replay, make sure the kill feed is visible, then let this run."
    )
    print("If using --show-preview, press q to stop.\n")

    detections = detect_feed_changes_live(cfg)
    events = cluster_detections(detections, gap_sec=cfg.min_gap_sec)
    deduped_events = dedupe_detections_visual(events)
    out_dir = Path(cfg.output_dir)
    save_json(out_dir / "detections_raw.json", [asdict(d) for d in detections])
    save_json(out_dir / "detections_clustered.json", [asdict(d) for d in events])
    save_json(out_dir / "detections_deduped.json", [asdict(d) for d in deduped_events])
    save_json(out_dir / "config.json", asdict(cfg))
    run_parser(input_dir=out_dir, templates_dir=None)
    print(f"\nSaved {len(detections)} candidate killfeed crops to {out_dir}")
    print(f"Raw detections: {len(detections)}")
    print(f"Clustered events: {len(events)}")
    print(f"Deduped events: {len(deduped_events)}")
    print(
        f"candidates_skipped_no_arrow: {getattr(cfg, '_candidates_skipped_no_arrow', 0)} "
        f"candidates_skipped_warmup: {getattr(cfg, '_candidates_skipped_warmup', 0)}"
    )


if __name__ == "__main__":
    main()
