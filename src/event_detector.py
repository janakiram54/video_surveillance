import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
from shapely.geometry import Point, Polygon


class ZoneEventDetector:
    """Detects zone intrusion and loitering events from tracked person boxes."""

    def __init__(self, config: Dict[str, Any], fps: float):
        self.fps = max(float(fps), 1.0)
        self.loiter_seconds = float(config.get("loiter_seconds", 10))
        self.stationary_pixel_threshold = float(config.get("stationary_pixel_threshold", 30))
        self.dedup_seconds = float(config.get("dedup_seconds", 5))

        self.zones = []
        for z in config.get("zones", []):
            self.zones.append({
                "name": z["name"],
                "type": z.get("type", "restricted"),
                "polygon_points": z["polygon"],
                "polygon": Polygon(z["polygon"]),
            })

        self.track_state = defaultdict(lambda: {
            "zone_enter_frame": {},
            "first_center_in_zone": {},
            "last_event_frame": {}
        })

    @staticmethod
    def bottom_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, float(y2))

    def _can_emit(self, track_id: int, event_key: str, frame_idx: int) -> bool:
        last = self.track_state[track_id]["last_event_frame"].get(event_key)
        if last is None:
            return True
        return (frame_idx - last) / self.fps >= self.dedup_seconds

    def _mark_emitted(self, track_id: int, event_key: str, frame_idx: int) -> None:
        self.track_state[track_id]["last_event_frame"][event_key] = frame_idx

    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        events = []

        for det in detections:
            track_id = int(det["track_id"])
            box = tuple(map(int, det["bbox"]))
            conf = float(det["confidence"])
            center = self.bottom_center(box)
            point = Point(center)

            for zone in self.zones:
                zone_name = zone["name"]
                zone_type = zone["type"]
                in_zone = zone["polygon"].contains(point)

                if not in_zone:
                    self.track_state[track_id]["zone_enter_frame"].pop(zone_name, None)
                    self.track_state[track_id]["first_center_in_zone"].pop(zone_name, None)
                    continue

                if zone_name not in self.track_state[track_id]["zone_enter_frame"]:
                    self.track_state[track_id]["zone_enter_frame"][zone_name] = frame_idx
                    self.track_state[track_id]["first_center_in_zone"][zone_name] = center

                    if zone_type == "restricted":
                        event_key = f"intrusion:{zone_name}"
                        if self._can_emit(track_id, event_key, frame_idx):
                            events.append(self._event("zone_intrusion", zone_name, track_id, frame_idx, box, conf))
                            self._mark_emitted(track_id, event_key, frame_idx)

                if zone_type == "loitering":
                    enter_frame = self.track_state[track_id]["zone_enter_frame"][zone_name]
                    duration = (frame_idx - enter_frame) / self.fps
                    first_center = self.track_state[track_id]["first_center_in_zone"][zone_name]
                    movement = float(np.linalg.norm(np.array(center) - np.array(first_center)))

                    if duration >= self.loiter_seconds and movement <= self.stationary_pixel_threshold:
                        event_key = f"loitering:{zone_name}"
                        if self._can_emit(track_id, event_key, frame_idx):
                            events.append(self._event("loitering", zone_name, track_id, frame_idx, box, conf, duration))
                            self._mark_emitted(track_id, event_key, frame_idx)

        return events

    def _event(self, event_type, zone_name, track_id, frame_idx, box, conf, duration=None):
        data = {
            "event_type": event_type,
            "zone": zone_name,
            "track_id": track_id,
            "frame": int(frame_idx),
            "timestamp_seconds": round(frame_idx / self.fps, 3),
            "bbox": [int(v) for v in box],
            "confidence": round(float(conf), 4),
        }
        if duration is not None:
            data["duration_seconds"] = round(float(duration), 3)
        return data
