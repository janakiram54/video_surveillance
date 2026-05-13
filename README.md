# AI Engineer Assignment: Video Surveillance Pipeline

This project detects people, tracks them with unique IDs, detects zone intrusion and loitering, and generates annotated video plus JSON event logs.

## Architecture

Video input -> YOLO person detection -> ByteTrack tracking -> zone/event logic -> annotated video + event log.

## Model Choices

- Detection: YOLOv8 via Ultralytics. It is fast, simple to run, and works well for person detection without training.
- Tracking: ByteTrack. It is lightweight and reliable for short surveillance clips.
- Event logic: Shapely polygon checks using the bottom-center of each person's bounding box.

Alternatives considered:
- Faster R-CNN: often accurate but slower.
- DeepSORT: includes appearance embeddings and can improve re-identification, but is heavier.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python run.py --video input.mp4 --zones configs/zones.json --output results
```

Optional:

```bash
python run.py --video input.mp4 --zones configs/zones.json --output results --model yolov8s.pt --conf 0.4 --show
```

## Zone Config

Edit `configs/zones.json`:

```json
{
  "loiter_seconds": 5,
  "stationary_pixel_threshold": 25,
  "dedup_seconds": 3,
  "zones": [
    {
      "name": "restricted_area",
      "type": "restricted",
      "polygon": [[100, 120], [450, 120], [450, 420], [100, 420]]
    },
    {
      "name": "waiting_area",
      "type": "loitering",
      "polygon": [[500, 150], [800, 150], [800, 500], [500, 500]]
    }
  ]
}
```

## Outputs

- `results/annotated_output.mp4`
- `results/events.json`

Each event contains event type, zone, track ID, frame number, timestamp, bounding box, and confidence.

## Known Limitations

- Re-identification is handled by ByteTrack, but long exits and re-entries may receive new IDs.
- Very crowded scenes can create ID switches.
- Loitering uses simple movement thresholding; pose/action recognition would improve it.
- Zone coordinates must match the video resolution.

## Performance Notes

Use `yolov8n.pt` for CPU or low-power machines. Use `yolov8s.pt` or larger models if you have a GPU and want better accuracy.
