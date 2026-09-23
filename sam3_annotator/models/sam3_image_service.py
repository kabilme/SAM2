"""High-level single-image SAM 3 segmentation service."""

from typing import List, Tuple, Optional
import numpy as np

from sam3_annotator.models.sam3_adapter import SAM3AdapterInterface, PointPrompt, BoxPrompt, TextPrompt
from sam3_annotator.annotation.polygon import PolygonAnnotation
from sam3_annotator.annotation.mask_to_polygon import mask_to_polygons
from sam3_annotator.utils.logging_utils import logger


class SAM3ImageService:
    """Coordinates prompt processing, SAM inference, and polygon generation for single frames."""

    def __init__(self, adapter: SAM3AdapterInterface):
        self.adapter = adapter

    def segment_points_to_annotation(
        self,
        image_bgr: np.ndarray,
        frame_id: int,
        source_frame_index: int,
        class_id: int,
        class_name: str,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
        simplify_tolerance: float = 0.005,
        min_area: float = 20.0,
        object_id: Optional[str] = None,
    ) -> Optional[PolygonAnnotation]:
        """Segment object from point prompts and return a PolygonAnnotation."""
        mask, conf = self.adapter.segment_with_points(image_bgr, positive_points, negative_points)
        if mask is None or np.count_nonzero(mask) == 0:
            return None

        polygons = mask_to_polygons(
            mask,
            tolerance_ratio=simplify_tolerance,
            min_area=min_area,
            only_largest_contour=True,
        )
        if not polygons:
            return None

        anno = PolygonAnnotation(
            object_id=object_id or "",
            class_id=class_id,
            class_name=class_name,
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            points=polygons[0],
            confidence=conf,
            source="sam3_point",
            tracking_status="confirmed",
        )
        return anno

    def segment_box_to_annotation(
        self,
        image_bgr: np.ndarray,
        frame_id: int,
        source_frame_index: int,
        class_id: int,
        class_name: str,
        box: Tuple[float, float, float, float],
        simplify_tolerance: float = 0.005,
        min_area: float = 20.0,
        object_id: Optional[str] = None,
    ) -> Optional[PolygonAnnotation]:
        """Segment object from bounding box prompt and return a PolygonAnnotation."""
        mask, conf = self.adapter.segment_with_box(image_bgr, box)
        if mask is None or np.count_nonzero(mask) == 0:
            return None

        polygons = mask_to_polygons(
            mask,
            tolerance_ratio=simplify_tolerance,
            min_area=min_area,
            only_largest_contour=True,
        )
        if not polygons:
            return None

        anno = PolygonAnnotation(
            object_id=object_id or "",
            class_id=class_id,
            class_name=class_name,
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            points=polygons[0],
            confidence=conf,
            source="sam3_box",
            tracking_status="confirmed",
        )
        return anno
