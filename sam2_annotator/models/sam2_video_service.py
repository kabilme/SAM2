"""High-level video tracking and propagation service."""

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import cv2
from sam2_annotator.models.sam2_adapter import SAM2AdapterInterface
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.annotation.mask_to_polygon import mask_to_polygons
from sam2_annotator.video.frame_cache import FrameCache
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.utils.geometry import calculate_polygon_centroid, compute_bounding_box
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
        source_frame_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[PolygonAnnotation]:
        """Propagate an object polygon annotation across target frames using motion-tracked box and point prompts."""
        propagated_annotations: List[PolygonAnnotation] = []
        total = len(target_frames)
        if total == 0:
            return propagated_annotations

        # Ensure initial bounding box is valid
        current_box = initial_annotation.bounding_box
        if not current_box or current_box == (0.0, 0.0, 0.0, 0.0):
            current_box = compute_bounding_box(initial_annotation.points)

        current_centroid = calculate_polygon_centroid(initial_annotation.points)
        current_pts = np.array(initial_annotation.points, dtype=np.float32)

        # Load reference grayscale image for optical flow displacement
        prev_gray: Optional[np.ndarray] = None
        if source_frame_filename:
            ref_img = self.frame_cache.get_frame(source_frame_filename)
            if ref_img is not None:
                prev_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

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
            curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Estimate inter-frame object motion using Lucas-Kanade optical flow
            dx, dy = 0.0, 0.0
            if prev_gray is not None and len(current_pts) >= 3:
                try:
                    p0 = current_pts.reshape(-1, 1, 2)
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, p0, None, winSize=(21, 21), maxLevel=3
                    )
                    valid = (st.flatten() == 1)
                    if np.sum(valid) >= 3:
                        shifts = (p1[valid] - p0[valid]).reshape(-1, 2)
                        dx = float(np.median(shifts[:, 0]))
                        dy = float(np.median(shifts[:, 1]))
                        # Limit extreme displacement spikes (e.g. scene cut)
                        dx = max(-w * 0.20, min(w * 0.20, dx))
                        dy = max(-h * 0.20, min(h * 0.20, dy))
                except Exception as e:
                    logger.debug("Optical flow tracking step %d: %s", step, e)
                    dx, dy = 0.0, 0.0

            # 2. Shift bounding box and centroid by estimated object motion
            x1, y1, x2, y2 = current_box
            if box_padding_ratio > 0.0:
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                pad_x = bw * box_padding_ratio
                pad_y = bh * box_padding_ratio
                prompt_box = (
                    max(0.0, x1 + dx - pad_x),
                    max(0.0, y1 + dy - pad_y),
                    min(float(w), x2 + dx + pad_x),
                    min(float(h), y2 + dy + pad_y),
                )
            else:
                prompt_box = (
                    max(0.0, x1 + dx),
                    max(0.0, y1 + dy),
                    min(float(w), x2 + dx),
                    min(float(h), y2 + dy),
                )

            shifted_centroid = (
                max(0.0, min(float(w), current_centroid[0] + dx)),
                max(0.0, min(float(h), current_centroid[1] + dy)),
            )

            mask = None
            conf = None

            # 3. Model inference anchored to both bounding box and foreground centroid
            if prompt_type == "box":
                if hasattr(self.adapter, "segment_with_box_and_points"):
                    mask, conf = self.adapter.segment_with_box_and_points(
                        img, prompt_box, positive_points=[shifted_centroid]
                    )
                if mask is None or np.count_nonzero(mask) == 0:
                    mask, conf = self.adapter.segment_with_box(img, prompt_box)
                if mask is None or np.count_nonzero(mask) == 0:
                    mask, conf = self.adapter.segment_with_points(img, [shifted_centroid])
            elif prompt_type == "point":
                mask, conf = self.adapter.segment_with_points(img, [shifted_centroid])
                if mask is None or np.count_nonzero(mask) == 0:
                    mask, conf = self.adapter.segment_with_box(img, prompt_box)
            else:  # "combined"
                if hasattr(self.adapter, "segment_with_box_and_points"):
                    mask, conf = self.adapter.segment_with_box_and_points(
                        img, prompt_box, positive_points=[shifted_centroid]
                    )
                else:
                    mask, conf = self.adapter.segment_with_box(img, prompt_box)
                if mask is None or np.count_nonzero(mask) == 0:
                    mask, conf = self.adapter.segment_with_points(img, [shifted_centroid])

            if mask is None or np.count_nonzero(mask) == 0:
                logger.warning("Object %s lost on frame %d", initial_annotation.object_id, frame_meta.frame_id)
                prev_gray = curr_gray
                continue

            # 4. Extract all contours and pick the contour best matching the target object by IoU & proximity
            candidate_polys = mask_to_polygons(
                mask,
                tolerance_ratio=simplify_tolerance,
                min_area=min_area,
                only_largest_contour=False,
            )
            if not candidate_polys:
                prev_gray = curr_gray
                continue

            best_poly = None
            best_iou = -1.0
            for poly in candidate_polys:
                pb = compute_bounding_box(poly)
                ix1 = max(prompt_box[0], pb[0])
                iy1 = max(prompt_box[1], pb[1])
                ix2 = min(prompt_box[2], pb[2])
                iy2 = min(prompt_box[3], pb[3])
                inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union_area = (
                    (prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1])
                    + (pb[2] - pb[0]) * (pb[3] - pb[1])
                    - inter_area
                )
                iou = inter_area / max(1.0, union_area)
                if iou > best_iou:
                    best_iou = iou
                    best_poly = poly

            # Guard against jumping to another unrelated object when multiple proposals exist
            if best_poly is None or (best_iou < 0.05 and len(candidate_polys) > 1):
                logger.warning("Contour rejected due to low overlap (%.2f) on frame %d", best_iou, frame_meta.frame_id)
                prev_gray = curr_gray
                continue

            new_points = best_poly
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
            new_anno.bounding_box = compute_bounding_box(new_points)
            propagated_annotations.append(new_anno)

            # Update tracking anchors for subsequent steps
            current_box = new_anno.bounding_box
            current_pts = np.array(new_points, dtype=np.float32)
            current_centroid = calculate_polygon_centroid(new_points)
            prev_gray = curr_gray

            if progress_callback:
                pt_label = "Box Prompt" if prompt_type == "box" else ("Point Prompt" if prompt_type == "point" else "Combined")
                progress_callback(
                    step + 1,
                    total,
                    f"Propagated object to frame {frame_meta.frame_id} ({step + 1}/{total}) [{pt_label}]",
                )

        logger.info("Propagation complete: generated %d annotations", len(propagated_annotations))
        return propagated_annotations
