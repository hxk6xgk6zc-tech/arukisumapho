# arukisumapho

# arukisumapho

Offline crowd counting and trajectory extraction for privacy-preserving urban analytics.

## Features
- Detects and tracks hundreds of people in wide-angle videos.
- Projects trajectories onto an X-Y ground plane using homography calibration.
- Exports per-ID tracks to CSV with JSON-encoded coordinates and head brightness.
- Optional mosaic anonymization in output video.
- Apple Silicon friendly: MPS auto-selection, configurable image size, and frame stride.

## Requirements
- Python 3.9+
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO (`ultralytics`)

## Usage
```bash
python crowd_tracking.py \
  --video /path/to/crowd.mp4 \
  --model yolov8n.pt \
  --calibration calibration.json \
  --output trajectories.csv \
  --output-video anonymized.mp4 \
  --mosaic \
  --device auto \
  --imgsz 960 \
  --vid-stride 1
```

### Performance notes (M2/M3)
- `--device auto` will prefer MPS when available.
- Increase `--vid-stride` (e.g., 2 or 3) to reduce per-frame inference cost.
- Reduce `--imgsz` if throughput is limited, at the expense of detection quality.

### Calibration JSON format
Provide at least four corresponding points between image coordinates and ground-plane coordinates.

```json
{
  "image_points": [[100, 200], [500, 210], [120, 700], [520, 710]],
  "world_points": [[0, 0], [10, 0], [0, 20], [10, 20]]
}
```

## Output CSV
Each row contains:
- `id`: tracking ID (used only for visual verification)
- `video`: video filename
- `start_time_s`, `end_time_s`: timestamp range
- `xy`: JSON array of X-Y coordinates
- `head_brightness`: JSON array of head-region brightness values
