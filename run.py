import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.event_detector import ZoneEventDetector
from src.visualizer import draw_zones, draw_tracks, draw_events


PERSON_CLASS_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Video Surveillance: Detection, Tracking & Event Recognition")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--zones", required=True, help="Zones JSON config path")
    parser.add_argument("--output", default="results", help="Output folder")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model: yolov8n.pt, yolov8s.pt, yolov8m.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output)

    zones_config = load_json(args.zones)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_video = output_dir / "annotated_output.mp4"
    out_json = output_dir / "events.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    model = YOLO(args.model)
    event_detector = ZoneEventDetector(zones_config, fps=fps)
    all_events = []
    frame_idx = 0

    progress = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                ids = boxes.id.cpu().numpy()

                for box, conf, track_id in zip(xyxy, confs, ids):
                    detections.append({
                        "track_id": int(track_id),
                        "bbox": [int(v) for v in box.tolist()],
                        "confidence": float(conf),
                    })

        frame_events = event_detector.update(detections, frame_idx)
        all_events.extend(frame_events)

        frame = draw_zones(frame, event_detector.zones)
        frame = draw_tracks(frame, detections)
        frame = draw_events(frame, frame_events)
        writer.write(frame)

        if args.show:
            cv2.imshow("Video Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        progress.update(1)

    progress.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2)

    print(f"Done")
    print(f"Annotated video: {out_video}")
    print(f"Event log: {out_json}")
    print(f"Total events: {len(all_events)}")


if __name__ == "__main__":
    main()
