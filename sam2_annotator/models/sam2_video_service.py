"""High-level video tracking and propagation service utilizing native Meta SAM 2.1 Video Predictor with temporal memory attention."""

from collections import OrderedDict
import gc
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
import cv2
import torch

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
        """Propagate an object polygon annotation across target frames using Meta SAM 2.1 spatio-temporal memory attention."""
        if not target_frames:
            return []

        # 1. Attempt native Meta SAM 2.1 Video Predictor with temporal memory attention
        if hasattr(self.adapter, "get_video_predictor"):
            try:
                predictor = self.adapter.get_video_predictor()
                if predictor is not None:
                    native_results = self._propagate_with_native_video_predictor(
                        predictor=predictor,
                        initial_annotation=initial_annotation,
                        target_frames=target_frames,
                        prompt_type=prompt_type,
                        box_padding_ratio=box_padding_ratio,
                        simplify_tolerance=simplify_tolerance,
                        min_area=min_area,
                        source_frame_filename=source_frame_filename,
                        progress_callback=progress_callback,
                        is_cancelled=is_cancelled,
                    )
                    if native_results is not None:
                        return native_results
            except Exception as e:
                logger.warning("Native video predictor execution failed (%s); falling back to frame-by-frame tracker.", e)

        # 2. Fallback: Frame-by-frame tracker (for mock adapters or offline environments)
        return self._propagate_frame_by_frame(
            initial_annotation=initial_annotation,
            target_frames=target_frames,
            prompt_type=prompt_type,
            box_padding_ratio=box_padding_ratio,
            simplify_tolerance=simplify_tolerance,
            min_area=min_area,
            source_frame_filename=source_frame_filename,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

    def _propagate_with_native_video_predictor(
        self,
        predictor: Any,
        initial_annotation: PolygonAnnotation,
        target_frames: List[FrameMetadata],
        prompt_type: str,
        box_padding_ratio: float,
        simplify_tolerance: float,
        min_area: float,
        source_frame_filename: Optional[str],
        progress_callback: Optional[Callable[[int, int, str], None]],
        is_cancelled: Optional[Callable[[], bool]],
    ) -> Optional[List[PolygonAnnotation]]:
        """Propagate using Meta SAM 2.1 Video Predictor with spatio-temporal memory attention and bounded chunking."""
        if not source_frame_filename:
            return None

        chunk_size = 60
        num_targets = len(target_frames)
        propagated_annotations: List[PolygonAnnotation] = []

        current_anno = initial_annotation
        current_ref_filename = source_frame_filename

        for chunk_start in range(0, num_targets, chunk_size):
            if is_cancelled and is_cancelled():
                logger.info("Propagation cancelled before chunk at %d", chunk_start)
                break

            chunk_targets = target_frames[chunk_start : chunk_start + chunk_size]
            chunk_results = self._propagate_chunk(
                predictor=predictor,
                initial_annotation=current_anno,
                target_frames=chunk_targets,
                prompt_type=prompt_type,
                box_padding_ratio=box_padding_ratio,
                simplify_tolerance=simplify_tolerance,
                min_area=min_area,
                source_frame_filename=current_ref_filename,
                global_step_offset=chunk_start,
                total_target_count=num_targets,
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )

            if not chunk_results:
                logger.warning("Chunk starting at index %d produced no annotations; aborting remaining chunks.", chunk_start)
                break

            propagated_annotations.extend(chunk_results)
            # Chain the latest tracked annotation and frame as reference for the next chunk
            current_anno = chunk_results[-1]
            current_ref_filename = chunk_targets[len(chunk_results) - 1].filename

        logger.info(
            "Native SAM 2.1 video propagation finished: produced %d annotations for object %s across %d frames",
            len(propagated_annotations), initial_annotation.object_id, num_targets
        )
        return propagated_annotations

    def _propagate_chunk(
        self,
        predictor: Any,
        initial_annotation: PolygonAnnotation,
        target_frames: List[FrameMetadata],
        prompt_type: str,
        box_padding_ratio: float,
        simplify_tolerance: float,
        min_area: float,
        source_frame_filename: str,
        global_step_offset: int,
        total_target_count: int,
        progress_callback: Optional[Callable[[int, int, str], None]],
        is_cancelled: Optional[Callable[[], bool]],
    ) -> List[PolygonAnnotation]:
        """Run SAM 2.1 temporal memory propagation on a single bounded chunk of frames."""
        ref_img = self.frame_cache.get_frame(source_frame_filename)
        if ref_img is None:
            logger.warning("Could not load reference frame %s", source_frame_filename)
            return []

        orig_h, orig_w = ref_img.shape[:2]
        frame_imgs = [ref_img]

        for frame_meta in target_frames:
            img = self.frame_cache.get_frame(frame_meta.filename)
            if img is None:
                logger.warning("Target frame %s could not be loaded; aborting chunk.", frame_meta.filename)
                return []
            frame_imgs.append(img)

        # Build tensor batch
        img_size = getattr(predictor, "image_size", 1024)
        img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

        tensors = []
        for img in frame_imgs:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
            tensors.append(t)

        images = torch.stack(tensors, dim=0)
        images -= img_mean
        images /= img_std

        compute_device = getattr(predictor, "device", torch.device("cpu"))
        inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "video_height": orig_h,
            "video_width": orig_w,
            "device": compute_device,
            "storage_device": torch.device("cpu"),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        # Condition frame 0
        self._condition_frame_0(
            predictor=predictor,
            inference_state=inference_state,
            initial_annotation=initial_annotation,
            prompt_type=prompt_type,
            box_padding_ratio=box_padding_ratio,
            orig_w=orig_w,
            orig_h=orig_h,
        )

        chunk_annotations: List[PolygonAnnotation] = []
        source_tag = "sam2_box_track" if prompt_type == "box" else ("sam2_point_track" if prompt_type == "point" else "sam2_track")

        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if out_frame_idx == 0:
                    continue
                if is_cancelled and is_cancelled():
                    logger.info("Video propagation cancelled during chunk at frame %d", out_frame_idx)
                    break

                target_idx = out_frame_idx - 1
                if target_idx >= len(target_frames):
                    break
                frame_meta = target_frames[target_idx]

                mask_prob = torch.sigmoid(out_mask_logits[0, 0]).cpu().numpy()
                mask_bin = (mask_prob > 0.5).astype(np.uint8) * 255

                if np.count_nonzero(mask_bin) == 0:
                    logger.warning("Object %s lost or occluded on frame %d", initial_annotation.object_id, frame_meta.frame_id)
                    continue

                candidate_polys = mask_to_polygons(
                    mask_bin,
                    tolerance_ratio=simplify_tolerance,
                    min_area=min_area,
                    only_largest_contour=True,
                )
                if not candidate_polys:
                    candidate_polys = mask_to_polygons(
                        mask_bin,
                        tolerance_ratio=simplify_tolerance,
                        min_area=min_area,
                        only_largest_contour=False,
                    )
                if not candidate_polys:
                    continue

                new_points = candidate_polys[0]
                fg_probs = mask_prob[mask_bin > 0]
                conf = float(np.mean(fg_probs)) if len(fg_probs) > 0 else 0.95

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
                chunk_annotations.append(new_anno)

                if progress_callback:
                    cur_global = global_step_offset + out_frame_idx
                    pt_label = "Box Prompt" if prompt_type == "box" else ("Point Prompt" if prompt_type == "point" else "Combined")
                    progress_callback(
                        cur_global,
                        total_target_count,
                        f"Propagated object to frame {frame_meta.frame_id} ({cur_global}/{total_target_count}) [{pt_label}]",
                    )
        finally:
            try:
                predictor.reset_state(inference_state)
            except Exception:
                pass
            del inference_state
            gc.collect()

        return chunk_annotations

    def _condition_frame_0(
        self,
        predictor: Any,
        inference_state: Dict[str, Any],
        initial_annotation: PolygonAnnotation,
        prompt_type: str,
        box_padding_ratio: float,
        orig_w: int,
        orig_h: int,
    ) -> None:
        """Condition the keyframe on frame 0 with bounding box, interior centroid, or mask prompts."""
        current_box = initial_annotation.bounding_box
        if not current_box or current_box == (0.0, 0.0, 0.0, 0.0):
            current_box = compute_bounding_box(initial_annotation.points)

        cx, cy = calculate_polygon_centroid(initial_annotation.points)

        x1, y1, x2, y2 = current_box
        if box_padding_ratio > 0.0:
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pad_x = bw * box_padding_ratio
            pad_y = bh * box_padding_ratio
            prompt_box = (
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(orig_w), x2 + pad_x),
                min(float(orig_h), y2 + pad_y),
            )
        else:
            prompt_box = (
                max(0.0, x1),
                max(0.0, y1),
                min(float(orig_w), x2),
                min(float(orig_h), y2),
            )

        box_np = np.array(prompt_box, dtype=np.float32)
        pts_np = np.array([[cx, cy]], dtype=np.float32)
        labels_np = np.array([1], dtype=np.int32)

        if prompt_type == "box":
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=box_np,
                points=pts_np,
                labels=labels_np,
            )
        elif prompt_type == "point":
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=pts_np,
                labels=labels_np,
            )
        elif prompt_type == "mask" and len(initial_annotation.points) >= 3:
            mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
            poly_pts = np.array(initial_annotation.points, dtype=np.int32)
            cv2.fillPoly(mask_np, [poly_pts], 255)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                mask=(mask_np > 0),
            )
        else:  # "combined"
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=box_np,
                points=pts_np,
                labels=labels_np,
            )

    def _propagate_frame_by_frame(
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
        """Fallback optical-flow frame-by-frame propagation for MockSAM2Adapter or environments without native video weights."""
        propagated_annotations: List[PolygonAnnotation] = []
        total = len(target_frames)
        if total == 0:
            return propagated_annotations

        current_box = initial_annotation.bounding_box
        if not current_box or current_box == (0.0, 0.0, 0.0, 0.0):
            current_box = compute_bounding_box(initial_annotation.points)

        current_centroid = calculate_polygon_centroid(initial_annotation.points)
        current_pts = np.array(initial_annotation.points, dtype=np.float32)

        prev_gray: Optional[np.ndarray] = None
        if source_frame_filename:
            ref_img = self.frame_cache.get_frame(source_frame_filename)
            if ref_img is not None:
                prev_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        logger.info(
            "Starting fallback frame-by-frame propagation for object %s across %d frames using prompt_type='%s'",
            initial_annotation.object_id, total, prompt_type
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
                        dx = max(-w * 0.20, min(w * 0.20, dx))
                        dy = max(-h * 0.20, min(h * 0.20, dy))
                except Exception as e:
                    logger.debug("Optical flow tracking step %d: %s", step, e)
                    dx, dy = 0.0, 0.0

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

        logger.info("Fallback propagation complete: generated %d annotations", len(propagated_annotations))
        return propagated_annotations
