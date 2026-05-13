from typing import Dict, List, Any

import cv2
import numpy as np


def draw_zones(frame, zones: List[Dict[str, Any]]):
    for zone in zones:
        pts = np.array(zone["polygon_points"], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        x, y = pts[0]
        cv2.putText(frame, zone["name"], (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def draw_tracks(frame, detections: List[Dict[str, Any]]):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        track_id = det["track_id"]
        conf = det["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {track_id} person {conf:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame


def draw_events(frame, events: List[Dict[str, Any]]):
    y = 30
    for event in events[-5:]:
        text = f"{event['event_type']} | ID {event['track_id']} | {event['zone']}"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        y += 28
    return frame
