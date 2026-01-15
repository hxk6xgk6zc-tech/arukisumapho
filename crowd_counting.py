#!/usr/bin/env python3
"""Offline crowd counting and trajectory extraction.

This script detects and tracks people in a crowd video, projects their
positions onto an X-Y ground plane, and exports per-ID trajectories to CSV
with JSON-encoded arrays. It is designed for privacy-preserving analytics
in wide-angle public cameras where faces are not identifiable.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class TrackRecord:
    track_id: int
    video_name: str
    start_time_s: float
    end_time_s: float
    xy_points: List[Tuple[float, float]] = field(default_factory=list)
    head_brightness: List[float] = field(default_factory=list)

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "id": str(self.track_id),
            "video": self.video_name,
            "start_time_s": f"{self.start_time_s:.3f}",
            "end_time_s": f"{self.end_time_s:.3f}",
            "xy": json.dumps(self.xy_points, ensure_ascii=False),
            "head_brightness": json.dumps(self.head_brightness, ensure_ascii=False),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline crowd counting and trajectory extraction."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path or name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--calibration",
        help=(
            "Path to JSON calibration with image/world points. "
            "If omitted, raw image coordinates are used."
        ),
    )
    parser.add_argument(
        "--output",
        default="trajectories.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-video",
        help="Optional path to save anonymized mosaic video.",
    )
    parser.add_argument(
        "--mosaic",
        action="store_true",
        help="Apply mosaic to detected people in output video.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit export to top-K longest tracks.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detection IOU threshold.",
    )
    parser.add_argument(
        "--head-ratio",
        type=float,
        default=0.2,
        help="Top fraction of the box considered head region.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config for YOLO tracking (default: bytetrack.yaml).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, mps, cpu, or cuda (default: auto).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (default: 960).",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Process every Nth frame for speed (default: 1).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision on supported devices.",
    )
    return parser.parse_args()


def select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_homography(calibration_path: Optional[str]) -> Optional[np.ndarray]:
    if calibration_path is None:
        return None

    data = json.loads(Path(calibration_path).read_text())
    image_points = np.array(data["image_points"], dtype=np.float32)
    world_points = np.array(data["world_points"], dtype=np.float32)
    if image_points.shape != world_points.shape or image_points.shape[0] < 4:
        raise ValueError("Calibration must provide >=4 matching image/world points.")

    homography, _ = cv2.findHomography(image_points, world_points, method=0)
    return homography


def project_point(
    point: Tuple[float, float], homography: Optional[np.ndarray]
) -> Tuple[float, float]:
    if homography is None:
        return point
    pts = np.array([[point]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, homography)[0][0]
    return float(projected[0]), float(projected[1])


def head_brightness_from_box(
    frame: np.ndarray, box: Iterable[float], head_ratio: float
) -> float:
    x1, y1, x2, y2 = map(int, box)
    height = max(y2 - y1, 1)
    head_height = max(int(height * head_ratio), 1)
    head_roi = frame[y1 : y1 + head_height, x1:x2]
    if head_roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def apply_mosaic(frame: np.ndarray, box: Iterable[float], scale: int = 10) -> None:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    small = cv2.resize(roi, (max(1, roi.shape[1] // scale), max(1, roi.shape[0] // scale)))
    mosaic = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic


def iter_tracks(
    video_path: str,
    model: YOLO,
    tracker: str,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
    vid_stride: int,
    half: bool,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        results = model.track(
            frame,
            conf=conf,
            iou=iou,
            persist=True,
            tracker=tracker,
            classes=[0],
            device=device,
            imgsz=imgsz,
            vid_stride=vid_stride,
            half=half,
            verbose=False,
        )
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id
            if ids is not None:
                yield frame_idx, frame, boxes
            else:
                yield frame_idx, frame, None
        else:
            yield frame_idx, frame, None
        frame_idx += 1
    capture.release()


def export_csv(records: List[TrackRecord], output_path: str) -> None:
    fieldnames = ["id", "video", "start_time_s", "end_time_s", "xy", "head_brightness"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    model = YOLO(args.model)
    homography = load_homography(args.calibration)
    device = select_device(args.device)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.output_video, fourcc, fps, (width, height)
        )

    track_records: Dict[int, TrackRecord] = {}

    for frame_idx, frame, boxes in iter_tracks(
        str(video_path),
        model,
        args.tracker,
        args.conf,
        args.iou,
        device,
        args.imgsz,
        args.vid_stride,
        args.half,
    ):
        timestamp_s = frame_idx / fps
        if boxes is None or boxes.id is None:
            if video_writer is not None:
                video_writer.write(frame)
            continue

        for box, track_id in zip(boxes.xyxy, boxes.id.tolist()):
            x1, y1, x2, y2 = box.tolist()
            center_x = (x1 + x2) / 2.0
            center_y = y2
            xy = project_point((center_x, center_y), homography)
            brightness = head_brightness_from_box(frame, box, args.head_ratio)

            if args.mosaic:
                apply_mosaic(frame, box)

            if track_id not in track_records:
                track_records[track_id] = TrackRecord(
                    track_id=track_id,
                    video_name=video_path.name,
                    start_time_s=timestamp_s,
                    end_time_s=timestamp_s,
                )
            record = track_records[track_id]
            record.end_time_s = timestamp_s
            record.xy_points.append(xy)
            record.head_brightness.append(brightness)

        if video_writer is not None:
            video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()

    records = list(track_records.values())
    if args.max_tracks is not None:
        records.sort(key=lambda rec: len(rec.xy_points), reverse=True)
        records = records[: args.max_tracks]

    export_csv(records, args.output)


if __name__ == "__main__":
    main()
