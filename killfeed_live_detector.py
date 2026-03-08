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
    bbox_xyxy: tuple[int, int, int, int]
    monitor_idx: int


@dataclass
class Config:
    output_dir: str
    sample_fps: float = 4.0
    monitor_idx: int = 1
    # x1, y1, x2, y2 normalized to selected monitor
    killfeed_roi: tuple[float, float, float, float] = (0.68, 0.02, 0.99, 0.22)
    diff_threshold: float = 14.0
    min_gap_sec: float = 0.6
    resize_width: int = 700
    blur_kernel: int = 3
    debug: bool = False
    duration_sec: Optional[float] = None
    show_preview: bool = False
    burst_count: int = 3
    save_format: str = "jpg"   # "jpg" or "png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live killfeed detector for OW replay viewer")
    parser.add_argument("--output-dir", type=str, default="artifacts/killfeed_live")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--monitor-idx", type=int, default=1, help="mss monitor index; usually 1 is primary")
    parser.add_argument("--diff-threshold", type=float, default=14.0)
    parser.add_argument("--min-gap-sec", type=float, default=0.6)
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional max run time")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-preview", action="store_true", help="Show live preview window; press q to quit")
    parser.add_argument("--burst-count", type=int, default=3, help="Save N crops per detected hit")
    parser.add_argument("--save-format", type=str, default="jpg", choices=["jpg", "png"])
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def roi_to_pixels(width: int, height: int, roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x1 = int(roi[0] * width)
    y1 = int(roi[1] * height)
    x2 = int(roi[2] * width)
    y2 = int(roi[3] * height)
    return x1, y1, x2, y2


def preprocess_crop(crop_bgr: np.ndarray, resize_width: int, blur_kernel: int) -> np.ndarray:
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


def list_monitors() -> None:
    with mss.mss() as sct:
        print("Available monitors:")
        for idx, mon in enumerate(sct.monitors):
            if idx == 0:
                print(f"  {idx}: virtual bounding monitor {mon}")
            else:
                print(f"  {idx}: {mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})")


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
    last_hit_ts = -999.0
    frame_idx = 0

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

        x1_rel, y1_rel, x2_rel, y2_rel = roi_to_pixels(mon_width, mon_height, cfg.killfeed_roi)

        start_time = time.perf_counter()
        frame_period = 1.0 / cfg.sample_fps

        while True:
            loop_start = time.perf_counter()
            ts_sec = loop_start - start_time

            if cfg.duration_sec is not None and ts_sec >= cfg.duration_sec:
                break

            raw = np.array(sct.grab(mon))
            # mss returns BGRA
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            crop = frame_bgr[y1_rel:y2_rel, x1_rel:x2_rel]
            proc = preprocess_crop(crop, cfg.resize_width, cfg.blur_kernel)
            score = diff_score(proc, prev_proc)

            should_emit = score >= cfg.diff_threshold and (ts_sec - last_hit_ts) >= cfg.min_gap_sec
            if should_emit:
                crop_name = f"killfeed_{frame_idx:07d}_{ts_sec:08.2f}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                bbox_abs = (
                    mon_left + x1_rel,
                    mon_top + y1_rel,
                    mon_left + x2_rel,
                    mon_top + y2_rel,
                )

                detections.append(
                    FeedDetection(
                        frame_idx=frame_idx,
                        ts_sec=round(ts_sec, 3),
                        crop_path=str(crop_path),
                        motion_score=round(score, 3),
                        bbox_xyxy=bbox_abs,
                        monitor_idx=cfg.monitor_idx,
                    )
                )
                last_hit_ts = ts_sec
                print(f"[hit] t={ts_sec:7.2f}s score={score:6.2f} saved={crop_path.name}")

            if cfg.debug or cfg.show_preview:
                frame_dbg = frame_bgr.copy()
                cv2.rectangle(frame_dbg, (x1_rel, y1_rel), (x2_rel, y2_rel), (0, 255, 0), 2)
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
                        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("OW Killfeed Detector", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            prev_proc = proc
            frame_idx += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, frame_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if cfg.show_preview:
        cv2.destroyAllWindows()

    return detections


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
    )

    list_monitors()
    print(f"\nUsing monitor index: {cfg.monitor_idx}")
    print("Start your Overwatch replay, make sure the kill feed is visible, then let this run.")
    print("If using --show-preview, press q to stop.\n")

    detections = detect_feed_changes_live(cfg)
    out_dir = Path(cfg.output_dir)
    save_json(out_dir / "detections.json", [asdict(d) for d in detections])
    save_json(out_dir / "config.json", asdict(cfg))
    print(f"\nSaved {len(detections)} candidate killfeed crops to {out_dir}")


if __name__ == "__main__":
    main()