"""JSON serialization for frame-level and project-level annotations."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from sam3_annotator.annotation.polygon import PolygonAnnotation
from sam3_annotator.utils.logging_utils import logger


class AnnotationSerializer:
    """Serializes and deserializes PolygonAnnotations to and from JSON files."""

    @staticmethod
    def save_frame_annotations(
        annotations_dir: Path,
        frame_filename: str,
        annotations: List[PolygonAnnotation],
    ) -> bool:
        """Save list of annotations for a frame to JSON."""
        try:
            annotations_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(frame_filename).stem
            out_file = annotations_dir / f"{stem}.json"

            data = {
                "frame_filename": frame_filename,
                "objects_count": len(annotations),
                "objects": [a.to_dict() for a in annotations],
            }

            temp_file = out_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(out_file)
            return True
        except Exception as e:
            logger.error("Failed to save frame annotations for %s: %s", frame_filename, e)
            return False

    @staticmethod
    def load_frame_annotations(
        annotations_dir: Path,
        frame_filename: str,
    ) -> List[PolygonAnnotation]:
        """Load list of annotations for a frame from JSON."""
        stem = Path(frame_filename).stem
        json_file = annotations_dir / f"{stem}.json"
        if not json_file.exists():
            return []

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            objects = data.get("objects", [])
            return [PolygonAnnotation.from_dict(obj) for obj in objects]
        except Exception as e:
            logger.error("Failed to read frame annotations from %s: %s", json_file, e)
            return []
