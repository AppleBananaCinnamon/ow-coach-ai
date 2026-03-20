from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from arrow_detector import ArrowDetector, compute_arrow_search_band


ARROW_ANCHORED_SUBREGIONS = {
    "left_name": (-180, 5, 91, 32),
    "left_icon": (-78, 0, 62, 43),
    "arrow": (0, 0, 30, 43),
    "right_icon": (38, 0, 69, 43),
    "right_name": (112, 5, 150, 32),
}


@dataclass
class DedupedDetection:
    ts_sec: float
    frame_idx: int
    crop_path: str
    right_icon_box: tuple[int, int, int, int] | None
    anchored_regions: dict[str, tuple[int, int, int, int]]


@dataclass
class ParsedKillfeedEvent:
    ts_sec: float
    crop_path: str
    killer_hero: str | None
    victim_hero: str | None
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse deduped OW killfeed crops")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="artifacts/killfeed_live",
        help="Directory containing detections_deduped.json and crops/",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default=None,
        help="Reserved for future hero icon templates",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def uri_to_local_path(path_or_uri: str) -> Path:
    parsed = urlparse(path_or_uri)
    if parsed.scheme != "file":
        return Path(path_or_uri)

    local_path = unquote(parsed.path)
    if local_path.startswith("/") and len(local_path) >= 3 and local_path[2] == ":":
        local_path = local_path[1:]
    return Path(local_path)


def load_deduped_detections(path: Path) -> list[DedupedDetection]:
    raw_items = load_json(path)
    detections: list[DedupedDetection] = []

    for item in raw_items:
        anchored_regions_raw = item.get("anchored_regions") or {}
        anchored_regions = {
            str(name): tuple(int(v) for v in box)
            for name, box in anchored_regions_raw.items()
            if isinstance(box, (list, tuple)) and len(box) == 4
        }
        right_icon_box_raw = item.get("right_icon_box")
        detections.append(
            DedupedDetection(
                ts_sec=float(item["ts_sec"]),
                frame_idx=int(item["frame_idx"]),
                crop_path=str(item["crop_path"]),
                right_icon_box=(
                    tuple(int(v) for v in right_icon_box_raw)
                    if isinstance(right_icon_box_raw, (list, tuple))
                    and len(right_icon_box_raw) == 4
                    else None
                ),
                anchored_regions=anchored_regions,
            )
        )

    return detections


def extract_ts_from_path(path: str) -> float:
    match = re.search(r"_t(\d+\.\d+)", path)
    return float(match.group(1)) if match else 0.0


def extract_event_id(path: str) -> int:
    match = re.search(r"evt(\d+)", path)
    return int(match.group(1)) if match else -1


def extract_killfeed_regions(
    crop_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = crop_bgr.shape[1]
    left_end = int(width * 0.42)
    center_end = int(width * 0.58)

    left_region = crop_bgr[:, :left_end]
    center_region = crop_bgr[:, left_end:center_end]
    right_region = crop_bgr[:, center_end:]
    return left_region, center_region, right_region


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


def crop_box(
    image_bgr: np.ndarray, box: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    if box is None:
        return None
    x, y, w, h = box
    region = image_bgr[y : y + h, x : x + w]
    return region if region.size > 0 else None


def compute_arrow_anchored_subregions(
    crop_bgr: np.ndarray, arrow_center: tuple[int, int], arrow_detector: ArrowDetector
) -> dict[str, tuple[int, int, int, int]]:
    arrow_x1 = int(round(arrow_center[0] - (arrow_detector.w / 2)))
    arrow_y1 = int(round(arrow_center[1] - (arrow_detector.h / 2)))
    regions: dict[str, tuple[int, int, int, int]] = {}

    for name, (dx, dy, w, h) in ARROW_ANCHORED_SUBREGIONS.items():
        box = (arrow_x1 + dx, arrow_y1 + dy, w, h)
        clipped = clip_box_to_image(box, crop_bgr.shape)
        if clipped is not None:
            regions[name] = clipped

    return regions


def _score_icon_bbox(
    bbox: tuple[int, int, int, int], region_shape: tuple[int, ...]
) -> float:
    x, y, w, h = bbox
    region_h, region_w = region_shape[:2]
    area = w * h
    squareness = 1.0 - abs(1.0 - (w / max(h, 1)))
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    x_bias = 1.0 - abs(center_x - (region_w / 2.0)) / max(region_w / 2.0, 1.0)
    y_bias = 1.0 - abs(center_y - (region_h / 2.0)) / max(region_h / 2.0, 1.0)
    return float(area * max(squareness, 0.0) * max(x_bias, 0.0) * max(y_bias, 0.0))


def find_icon_anchors(region_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 160)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_h, region_w = gray.shape
    min_side = max(8, int(region_h * 0.24))
    max_side = max(min_side + 1, int(region_h * 0.95))
    max_area = int(region_h * region_w * 0.22)

    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_side or h < min_side:
            continue
        if w > max_side or h > max_side:
            continue
        if (w * h) > max_area:
            continue

        ratio = w / max(h, 1)
        if ratio < 0.7 or ratio > 1.3:
            continue

        candidates.append((x, y, w, h))

    candidates.sort(
        key=lambda bbox: _score_icon_bbox(bbox, region_bgr.shape), reverse=True
    )
    return candidates[:3]


def crop_to_square_icon(
    region_bgr: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    x, y, w, h = bbox
    side = max(w, h)
    center_x = x + (w // 2)
    center_y = y + (h // 2)

    x1 = max(0, center_x - (side // 2))
    y1 = max(0, center_y - (side // 2))
    x2 = min(region_bgr.shape[1], x1 + side)
    y2 = min(region_bgr.shape[0], y1 + side)

    if (x2 - x1) < side:
        x1 = max(0, x2 - side)
    if (y2 - y1) < side:
        y1 = max(0, y2 - side)

    icon_bgr = region_bgr[y1:y2, x1:x2]
    if icon_bgr.size == 0:
        return np.zeros((32, 32), dtype=np.uint8)

    icon_gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(icon_gray, (32, 32), interpolation=cv2.INTER_AREA)


def match_icon_template(
    icon_bgr: np.ndarray, templates_dir: str | None
) -> tuple[str | None, float]:
    _ = icon_bgr
    _ = templates_dir
    return (None, 0.0)


def has_two_icons(
    left_boxes: list[tuple[int, int, int, int]],
    right_boxes: list[tuple[int, int, int, int]],
) -> bool:
    return len(left_boxes) >= 1 and len(right_boxes) >= 1


def region_from_box(
    crop_bgr: np.ndarray, box: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    clipped = clip_box_to_image(box, crop_bgr.shape) if box is not None else None
    if clipped is None:
        return None
    return crop_box(crop_bgr, clipped)


def fingerprint_victim_side(crop_bgr: np.ndarray) -> np.ndarray:
    _, _, right = extract_killfeed_regions(crop_bgr)
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 16))
    return small


def victim_side_similarity(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0
    return 1.0 - diff


def right_side_edge_signal(crop_bgr: np.ndarray) -> float:
    _, _, right = extract_killfeed_regions(crop_bgr)
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.mean(edges)) / 255.0


def parse_event(
    detection: DedupedDetection, workspace_dir: Path, templates_dir: str | None
) -> tuple[ParsedKillfeedEvent, str | None]:
    crop_path = uri_to_local_path(detection.crop_path)
    if not crop_path.is_absolute():
        crop_path = workspace_dir / crop_path
    crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop_bgr is None or crop_bgr.size == 0:
        return (
            ParsedKillfeedEvent(
                ts_sec=detection.ts_sec,
                crop_path=detection.crop_path,
                killer_hero=None,
                victim_hero=None,
                confidence=0.0,
            ),
            "missing_left_or_right_icon",
        )

    arrow_detector = ArrowDetector()
    arrow_band = compute_arrow_search_band(crop_bgr.shape[1], crop_bgr.shape[0])
    arrow_center = arrow_detector.find_arrow(crop_bgr, arrow_band)

    left_region = None
    right_region = None
    left_boxes: list[tuple[int, int, int, int]] = []
    right_boxes: list[tuple[int, int, int, int]] = []
    persisted_regions = detection.anchored_regions or {}

    persisted_left_icon_box = persisted_regions.get("left_icon")
    persisted_right_icon_box = persisted_regions.get("right_icon") or detection.right_icon_box

    if persisted_left_icon_box is not None and persisted_right_icon_box is not None:
        left_region = region_from_box(crop_bgr, persisted_left_icon_box)
        right_region = region_from_box(crop_bgr, persisted_right_icon_box)
        if left_region is not None:
            left_boxes = [(0, 0, left_region.shape[1], left_region.shape[0])]
        if right_region is not None:
            right_boxes = [(0, 0, right_region.shape[1], right_region.shape[0])]

    if (left_region is None or right_region is None) and arrow_center is not None:
        regions = compute_arrow_anchored_subregions(
            crop_bgr, arrow_center, arrow_detector
        )
        left_region = crop_box(crop_bgr, regions.get("left_icon"))
        right_region = crop_box(crop_bgr, regions.get("right_icon"))
        if left_region is not None:
            left_boxes = [(0, 0, left_region.shape[1], left_region.shape[0])]
        if right_region is not None:
            right_boxes = [(0, 0, right_region.shape[1], right_region.shape[0])]

    if left_region is None or right_region is None:
        left_region, _center_region, right_region = extract_killfeed_regions(crop_bgr)
        left_boxes = find_icon_anchors(left_region)
        right_boxes = find_icon_anchors(right_region)

    valid = has_two_icons(left_boxes, right_boxes)

    if not valid:
        reject_reason = "missing_left_or_right_icon"
    else:
        reject_reason = None

    killer_hero = None
    victim_hero = None
    killer_confidence = 0.0
    victim_confidence = 0.0

    if left_boxes:
        killer_icon = crop_to_square_icon(left_region, left_boxes[0])
        killer_hero, killer_confidence = match_icon_template(killer_icon, templates_dir)

    if right_boxes:
        victim_icon = crop_to_square_icon(right_region, right_boxes[0])
        victim_hero, victim_confidence = match_icon_template(victim_icon, templates_dir)

    confidence = float(max(killer_confidence, victim_confidence))
    return (
        ParsedKillfeedEvent(
            ts_sec=detection.ts_sec,
            crop_path=detection.crop_path,
            killer_hero=killer_hero,
            victim_hero=victim_hero,
            confidence=confidence,
        ),
        reject_reason,
    )


def merge_valid_events(
    events: list[ParsedKillfeedEvent], workspace_dir: Path
) -> list[ParsedKillfeedEvent]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda event: event.ts_sec)
    merged: list[ParsedKillfeedEvent] = [sorted_events[0]]

    prev_crop_path = uri_to_local_path(merged[0].crop_path)
    if not prev_crop_path.is_absolute():
        prev_crop_path = workspace_dir / prev_crop_path
    prev_crop = cv2.imread(str(prev_crop_path), cv2.IMREAD_COLOR)
    prev_fp = (
        fingerprint_victim_side(prev_crop)
        if prev_crop is not None and prev_crop.size > 0
        else None
    )

    for event in sorted_events[1:]:
        curr_crop_path = uri_to_local_path(event.crop_path)
        if not curr_crop_path.is_absolute():
            curr_crop_path = workspace_dir / curr_crop_path
        curr_crop = cv2.imread(str(curr_crop_path), cv2.IMREAD_COLOR)
        curr_fp = (
            fingerprint_victim_side(curr_crop)
            if curr_crop is not None and curr_crop.size > 0
            else None
        )

        ts_gap = abs(event.ts_sec - merged[-1].ts_sec)
        similarity = (
            victim_side_similarity(prev_fp, curr_fp)
            if prev_fp is not None and curr_fp is not None
            else 0.0
        )

        if ts_gap <= 2.0 and similarity >= 0.88:
            if prev_crop is not None and curr_crop is not None:
                prev_edge_signal = right_side_edge_signal(prev_crop)
                curr_edge_signal = right_side_edge_signal(curr_crop)
                if curr_edge_signal > prev_edge_signal:
                    merged[-1] = ParsedKillfeedEvent(
                        ts_sec=merged[-1].ts_sec,
                        crop_path=event.crop_path,
                        killer_hero=event.killer_hero,
                        victim_hero=event.victim_hero,
                        confidence=max(merged[-1].confidence, event.confidence),
                    )
                    prev_crop = curr_crop
                    prev_fp = curr_fp
            continue

        merged.append(event)
        prev_crop = curr_crop
        prev_fp = curr_fp

    return merged


def run_parser(input_dir: Path, templates_dir: str | None) -> list[ParsedKillfeedEvent]:
    detections_path = input_dir / "detections_deduped.json"
    detections = load_deduped_detections(detections_path)

    workspace_dir = Path.cwd()
    events: list[ParsedKillfeedEvent] = []
    event_records: list[dict[str, object]] = []
    valid_events: list[ParsedKillfeedEvent] = []

    for detection in detections:
        event_id = extract_event_id(detection.crop_path)
        candidate_paths = sorted(
            [
                * (input_dir / "crops").glob(f"killfeed_evt{event_id:05d}_cand*_t*.jpg"),
                * (input_dir / "crops").glob(f"killfeed_evt{event_id:05d}_cand*_t*.png"),
            ]
        )
        ts_candidates = [extract_ts_from_path(str(path)) for path in candidate_paths]
        earliest_ts = min(ts_candidates) if ts_candidates else detection.ts_sec

        event, reject_reason = parse_event(
            detection, workspace_dir=workspace_dir, templates_dir=templates_dir
        )
        event.ts_sec = earliest_ts
        events.append(event)
        event_records.append(
            {
                **asdict(event),
                "reject_reason": reject_reason,
            }
        )
        if reject_reason is None:
            valid_events.append(event)

    merged_events = merge_valid_events(valid_events, workspace_dir=workspace_dir)
    merged_events.sort(key=lambda event: event.ts_sec)

    save_json(input_dir / "parsed_events.json", event_records)
    save_json(
        input_dir / "parsed_events_valid.json",
        [asdict(event) for event in valid_events],
    )
    save_json(
        input_dir / "parsed_events_merged.json",
        [asdict(event) for event in merged_events],
    )
    return events


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    events = run_parser(input_dir=input_dir, templates_dir=args.templates_dir)
    print(f"Parsed {len(events)} deduped killfeed events")
    print(f"Saved: {input_dir / 'parsed_events.json'}")


if __name__ == "__main__":
    main()
