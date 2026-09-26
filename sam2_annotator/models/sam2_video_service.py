"""High-level video tracking and propagation service."""

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

from sam2_annotator.models.sam2_adapter import SAM2AdapterInterface
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.annotation.mask_to_polygon import mask_to_polygons
from sam2_annotator.video.frame_cache import FrameCache
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.utils.geometry import calculate_polygon_centroid
from sam2_annotator.utils.logging_utils import logger


class SAM2VideoService:
    """Manages multi-frame object propagation and tracking across video frames."""

    def __init__(self, adapter: SAM2AdapterInterface, frame_cache: FrameCache):
        self.adapter = adapter
        self.frame_cache = frame_cache

    def propagate_object(
        self,
        initial_annotation: PolygonAnnotation,
        target_frames: List[FrameMetadata],
        simplify_tolerance: float = 0.005,
        min_area: float = 20.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[PolygonAnnotation]:
        """Propagate an object polygon annotation across a sequence of target frames."""
        propagated_annotations: List[PolygonAnnotation] = []
        total = len(target_frames)
        if total == 0:
            return propagated_annotations

        # Track the last known bounding box and centroid
        current_box = initial_annotation.bounding_box
        current_centroid = calculate_polygon_centroid(initial_annotation.points)

        logger.info("Starting propagation for object %s across %d frames",
                    initial_annotation.object_id, total)

        for step, frame_meta in enumerate(target_frames):
            if is_cancelled and is_cancelled():
                logger.info("Video propagation cancelled at step %d", step)
                break

            img = self.frame_cache.get_frame(frame_meta.filename)
            if img is None:
                continue

            # Run inference using the propagated box or centroid prompt
            mask, conf = self.adapter.segment_with_box(img, current_box)
            if mask is None or np.count_nonzero(mask) == 0:
                # Try center point prompt as fallback
                if current_centroid:
                    mask, conf = self.adapter.segment_with_points(img, [current_centroid])

            if mask is None or np.count_nonzero(mask) == 0:
                logger.warning("Object %s lost on frame %d", initial_annotation.object_id, frame_meta.frame_id)
                continue

            polygons = mask_to_polygons(
                mask,
                tolerance_ratio=simplify_tolerance,
                min_area=min_area,
                only_largest_contour=True,
            )
            if not polygons:
                continue

            new_points = polygons[0]
            new_anno = PolygonAnnotation(
                object_id=initial_annotation.object_id,
                class_id=initial_annotation.class_id,
                class_name=initial_annotation.class_name,
                frame_id=frame_meta.frame_id,
                source_frame_index=frame_meta.source_frame_index,
                points=new_points,
                confidence=conf,
                source="sam2_track",
                tracking_status="tracked",
            )
            propagated_annotations.append(new_anno)

            # Update tracking anchor for next step
            current_box = new_anno.bounding_box
            current_centroid = calculate_polygon_centroid(new_anno.points)

            if progress_callback:
                progress_callback(
                    step + 1,
                    total,
                    f"Propagated object to frame {frame_meta.frame_id} ({step + 1}/{total})",
                )

        logger.info("Propagation complete: generated %d annotations", len(propagated_annotations))
        return propagated_annotations
