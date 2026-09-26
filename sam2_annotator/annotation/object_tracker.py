"""Object tracking state, keyframe management, and temporal anomaly detection."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.utils.geometry import (
    calculate_polygon_area,
    calculate_polygon_centroid,
    compute_bounding_box,
    compute_bbox_iou,
)
from sam2_annotator.utils.logging_utils import logger


@dataclass
class TrackingAnomaly:
    frame_id: int
    object_id: str
    anomaly_type: str    # "area_jump", "centroid_jump", "iou_drop", "missing"
    description: str
    severity: str        # "warning", "error"


class ObjectTracker:
    """Manages object trajectories, keyframes, and flags temporal anomalies."""

    def __init__(self):
        # Keyframes: set of frame_ids where annotations were manually edited/confirmed
        self.keyframes: set[int] = set()

    def mark_keyframe(self, frame_id: int, is_keyframe: bool = True) -> None:
        if is_keyframe:
            self.keyframes.add(frame_id)
        else:
            self.keyframes.discard(frame_id)

    def is_keyframe(self, frame_id: int) -> bool:
        return frame_id in self.keyframes

    def detect_anomalies(
        self,
        frame_annotations: Dict[int, List[PolygonAnnotation]],
        area_jump_threshold: float = 2.5,
        max_centroid_jump_px: float = 250.0,
        min_consecutive_iou: float = 0.15,
    ) -> List[TrackingAnomaly]:
        """Detect sudden jumps in area, large displacements, or tracking losses across frames."""
        anomalies: List[TrackingAnomaly] = []
        sorted_frame_ids = sorted(frame_annotations.keys())
        if len(sorted_frame_ids) < 2:
            return anomalies

        # Group annotations by object_id along sequence
        trajectories: Dict[str, List[Tuple[int, PolygonAnnotation]]] = {}
        for fid in sorted_frame_ids:
            for anno in frame_annotations[fid]:
                if anno.object_id not in trajectories:
                    trajectories[anno.object_id] = []
                trajectories[anno.object_id].append((fid, anno))

        for obj_id, traj in trajectories.items():
            for i in range(len(traj) - 1):
                f1, a1 = traj[i]
                f2, a2 = traj[i + 1]

                # Check frame adjacency (only check if adjacent or close frames)
                if abs(f2 - f1) > 5:
                    continue

                # 1. Area jump
                area1 = max(1.0, a1.area)
                area2 = max(1.0, a2.area)
                ratio = max(area1 / area2, area2 / area1)
                if ratio > area_jump_threshold and min(area1, area2) > 100:
                    anomalies.append(TrackingAnomaly(
                        frame_id=f2,
                        object_id=obj_id,
                        anomaly_type="area_jump",
                        description=f"Object {obj_id} area changed significantly ({area1:.0f}px -> {area2:.0f}px) between frames {f1} and {f2}",
                        severity="warning",
                    ))

                # 2. Centroid jump
                c1 = calculate_polygon_centroid(a1.points)
                c2 = calculate_polygon_centroid(a2.points)
                if c1 and c2:
                    dist = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
                    if dist > max_centroid_jump_px:
                        anomalies.append(TrackingAnomaly(
                            frame_id=f2,
                            object_id=obj_id,
                            anomaly_type="centroid_jump",
                            description=f"Object {obj_id} centroid jumped {dist:.1f}px between frames {f1} and {f2}",
                            severity="warning",
                        ))

                # 3. Bounding box IoU drop
                iou = compute_bbox_iou(a1.bounding_box, a2.bounding_box)
                if iou < min_consecutive_iou:
                    anomalies.append(TrackingAnomaly(
                        frame_id=f2,
                        object_id=obj_id,
                        anomaly_type="iou_drop",
                        description=f"Object {obj_id} overlap dropped (IoU: {iou:.2f}) between frames {f1} and {f2}",
                        severity="warning",
                    ))

        return anomalies
