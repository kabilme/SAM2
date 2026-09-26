"""Polygon annotation data model."""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from sam2_annotator.utils.geometry import compute_bounding_box, calculate_polygon_area


@dataclass
class PolygonAnnotation:
    """Represents a single annotated object instance on a video frame."""

    object_id: str                      # e.g., "obj_00001" or uuid
    class_id: int                       # 0-indexed integer class ID
    class_name: str                     # Human-readable class name, e.g. "scooter"
    frame_id: int                       # Project frame ID (1-based)
    source_frame_index: int             # Video source frame index
    points: List[Tuple[float, float]]   # Polygon vertices in original image pixel coordinates [(x, y), ...]
    bounding_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0) # [x1, y1, x2, y2]
    area: float = 0.0
    confidence: Optional[float] = None  # Float [0.0 - 1.0] or None if not provided by model
    source: str = "manual"              # "manual", "sam2_text", "sam2_point", "sam2_box", "sam2_track", "interpolated"
    tracking_status: str = "confirmed"  # "confirmed", "tracked", "needs_review", "corrected"
    modified: bool = False
    visible: bool = True
    locked: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.object_id:
            self.object_id = f"obj_{uuid.uuid4().hex[:8]}"
        if self.points and self.bounding_box == (0.0, 0.0, 0.0, 0.0):
            self.update_geometry(self.points)

    def update_geometry(self, new_points: List[Tuple[float, float]]) -> None:
        """Update points, bounding box, and area."""
        self.points = [(float(x), float(y)) for x, y in new_points]
        self.bounding_box = compute_bounding_box(self.points)
        self.area = calculate_polygon_area(self.points)
        self.updated_at = time.time()
        self.modified = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize annotation to dictionary (JSON-compatible)."""
        data = asdict(self)
        data["bounding_box"] = list(self.bounding_box)
        data["points"] = [list(pt) for pt in self.points]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolygonAnnotation":
        """Reconstruct annotation from dictionary."""
        d = dict(data)
        d["bounding_box"] = tuple(d.get("bounding_box", (0.0, 0.0, 0.0, 0.0)))
        d["points"] = [tuple(p) for p in d.get("points", [])]
        return cls(**d)
