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
        prompt_type: str = "box",
        box_padding_ratio: float = 0.0,
        simplify_tolerance: float = 0.005,
        min_area: float = 20.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[PolygonAnnotation]:
        """Propagate an object polygon annotation across target frames using box or point prompts."""
        propagated_annotations: List[PolygonAnnotation] = []
        total = len(target_frames)
        if total == 0:
            return propagated_annotations

        # Ensure initial bounding box is valid
        current_box = initial_annotation.bounding_box
        if not current_box or current_box == (0.0, 0.0, 0.0, 0.0):
            from sam2_annotator.utils.geometry import compute_bounding_box
            current_box = compute_bounding_box(initial_annotation.points)

        current_centroid = calculate_polygon_centroid(initial_annotation.points)

        logger.info(
            "Starting propagation for object %s across %d frames using prompt_type='%s' (box_padding=%.2f)",
            initial_annotation.object_id, total, prompt_type, box_padding_ratio
        )

        for step, frame_meta in enumerate(target_frames):
            if is_cancelled and is_cancelled():
                logger.info("Video propagation cancelled at step %d", step)
                break

            img = self.frame_cache.get_frame(frame_meta.filename)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Use exact bounding box prompt clamped to image dimensions (no margin expansion by default)
            x1, y1, x2, y2 = current_box
            if box_padding_ratio > 0.0:
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                pad_x = bw * box_padding_ratio
                pad_y = bh * box_padding_ratio
                prompt_box = (
                    max(0.0, x1 - pad_x),
                    max(0.0, y1 - pad_y),
                    min(float(w), x2 + pad_x),
                    min(float(h), y2 + pad_y),
                )
            else:
                prompt_box = (
                    max(0.0, x1),
                    max(0.0, y1),
                    min(float(w), x2),
                    min(float(h), y2),
                )

            mask = None
            conf = None

            if prompt_type == "box":
                # Primary: Bounding box prompt (exact bounding box)
                mask, conf = self.adapter.segment_with_box(img, prompt_box)
                if mask is None or np.count_nonzero(mask) == 0:
                    # Fallback to centroid point if box produced no mask
                    if current_centroid:
                        mask, conf = self.adapter.segment_with_points(img, [current_centroid])
            elif prompt_type == "point":
                if current_centroid:
                    mask, conf = self.adapter.segment_with_points(img, [current_centroid])
                if mask is None or np.count_nonzero(mask) == 0:
                    mask, conf = self.adapter.segment_with_box(img, prompt_box)
            else:  # "combined"
                mask, conf = self.adapter.segment_with_box(img, prompt_box)
                if mask is None or np.count_nonzero(mask) == 0:
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
            source_tag = "sam2_box_track" if prompt_type == "box" else ("sam2_point_track" if prompt_type == "point" else "sam2_track")
            new_anno = PolygonAnnotation(
                object_id=initial_annotation.object_id,
                class_id=initial_annotation.class_id,
                class_name=initial_annotation.class_name,
                frame_id=frame_meta.frame_id,
                source_frame_index=frame_meta.source_frame_index,
                points=new_points,
                confidence=conf,
                source=source_tag,
                tracking_status="tracked",
            )
            from sam2_annotator.utils.geometry import compute_bounding_box
            new_anno.bounding_box = compute_bounding_box(new_points)
            propagated_annotations.append(new_anno)

            # Update tracking anchor for next step
            current_box = new_anno.bounding_box
            current_centroid = calculate_polygon_centroid(new_anno.points)

            if progress_callback:
                pt_label = "Box Prompt" if prompt_type == "box" else ("Point Prompt" if prompt_type == "point" else "Combined")
                progress_callback(
                    step + 1,
                    total,
                    f"Propagated object to frame {frame_meta.frame_id} ({step + 1}/{total}) [{pt_label}]",
                )

        logger.info("Propagation complete: generated %d annotations", len(propagated_annotations))
        return propagated_annotations
