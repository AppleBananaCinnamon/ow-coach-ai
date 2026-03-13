from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import mss
import numpy as np


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
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    crop = cv2.imread(crop_path, cv2.IMREAD_COLOR)
    if crop is None or crop.size == 0:
        empty = np.zeros((12, 32), dtype=np.uint8)
        return empty, empty, empty

    regions = extract_killfeed_regions(crop)
    return (
        region_fingerprint(regions["left"]),
        region_fingerprint(regions["center"]),
        region_fingerprint(regions["right"]),
    )


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
    max_gap_sec: float = 4.0,
    similarity_threshold: float = 0.65,
) -> List[FeedDetection]:
    if not detections:
        return []

    deduped: List[FeedDetection] = []
    last_by_row: dict[int, FeedDetection] = {}
    last_idx_by_row: dict[int, int] = {}

    for detection in detections:
        row_idx = detection.row_idx
        prev = last_by_row.get(row_idx)
        prev_idx = last_idx_by_row.get(row_idx)

        if prev is not None and (detection.ts_sec - prev.ts_sec) <= max_gap_sec:
            similarity = crop_similarity(prev.crop_path, detection.crop_path)
        else:
            similarity = 0.0

        if prev is not None and prev_idx is not None and similarity >= similarity_threshold:
            best = max(prev, detection, key=detection_rank)
            last_by_row[row_idx] = best
            deduped[prev_idx] = best
        else:
            deduped.append(detection)
            last_by_row[row_idx] = detection
            last_idx_by_row[row_idx] = len(deduped) - 1

    return deduped


def estimate_row_idx_from_crop(
    crop_path: str, top_y: int = 0, row_height: int = 24
) -> int:
    crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    if crop is None or crop.size == 0:
        return 0

    norm = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)

    # Use the brightest horizontal band as a simple proxy for the active killfeed row.
    row_signal = np.mean(blur, axis=1)
    peak_y = int(np.argmax(row_signal))

    if row_height <= 0:
        return 0
    
    row_idx = int((peak_y - top_y) / row_height)

    return max(0, min(row_idx, 2))


def detect_feed_changes_live(cfg: Config) -> List[FeedDetection]:
    out_dir = Path(cfg.output_dir)
    crops_dir = out_dir / "crops"
    debug_dir = out_dir / "debug_frames"
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

            if cfg.duration_sec is not None and ts_sec >= cfg.duration_sec:
                break

            raw = np.array(sct.grab(mon))
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            crop = frame_bgr[y1_rel:y2_rel, x1_rel:x2_rel]

            mask = killfeed_ui_mask(crop)
            score = diff_score_binary(mask, prev_proc)
            has_bar = has_horizontal_bar_structure(mask)

            signal, red_ratio, cyan_ratio, white_ratio = killfeed_color_signal(crop)

            should_emit = (
                score >= cfg.diff_threshold and has_bar and can_start_new_event
            )

            if should_emit:
                burst_remaining = max(cfg.burst_count, 1)
                burst_event_idx += 1
                last_event_ts = ts_sec
                burst_candidates = []

            if burst_remaining > 0:
                burst_seq = cfg.burst_count - burst_remaining

                should_keep_burst_candidate = burst_seq == 0 or (
                    has_bar and signal >= cfg.color_signal_threshold
                )

                if should_keep_burst_candidate:
                    candidate = BurstCandidate(
                        frame_idx=frame_idx,
                        ts_sec=round(ts_sec, 3),
                        crop=crop.copy(),
                        motion_score=round(score, 3),
                        signal=round(signal, 3),
                        red_ratio=round(red_ratio, 3),
                        cyan_ratio=round(cyan_ratio, 3),
                        white_ratio=round(white_ratio, 3),
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
                        save_crop_image(debug_candidate_path, candidate.crop, cfg.save_format)

                burst_remaining -= 1

                if burst_remaining == 0 and burst_candidates:
                    best_candidate = max(
                        burst_candidates,
                        key=lambda candidate: candidate.motion_score * candidate.signal,
                    )

                    ext = cfg.save_format
                    crop_name = (
                        f"killfeed_evt{burst_event_idx:05d}_"
                        f"best_"
                        f"f{best_candidate.frame_idx:07d}_"
                        f"t{best_candidate.ts_sec:08.2f}."
                        f"{ext}"
                    )
                    crop_path = crops_dir / crop_name
                    save_crop_image(crop_path, best_candidate.crop, cfg.save_format)

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
                            crop_path=str(crop_path),
                            motion_score=best_candidate.motion_score,
                            signal=best_candidate.signal,
                            red_ratio=best_candidate.red_ratio,
                            cyan_ratio=best_candidate.cyan_ratio,
                            white_ratio=best_candidate.white_ratio,
                            bbox_xyxy=bbox_abs,
                            row_idx=estimate_row_idx_from_crop(str(crop_path)),
                            monitor_idx=cfg.monitor_idx,
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
    print(f"\nSaved {len(detections)} candidate killfeed crops to {out_dir}")
    print(f"Raw detections: {len(detections)}")
    print(f"Clustered events: {len(events)}")
    print(f"Deduped events: {len(deduped_events)}")


if __name__ == "__main__":
    main()
