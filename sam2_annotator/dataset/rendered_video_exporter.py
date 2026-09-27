"""Annotated overlay video renderer and MP4 exporter."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import cv2
import numpy as np

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.utils.image_utils import load_image_bgr
from sam2_annotator.utils.logging_utils import logger
from sam2_annotator.dataset.base_exporter import BaseDatasetExporter


class RenderedVideoExporter(BaseDatasetExporter):
    """Renders polygons, labels, and tracking IDs directly onto video frames and exports an MP4 video."""

    def export(
        self,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        video_filename: str = "annotated_video.mp4",
        fps: float = 30.0,
        alpha: float = 0.45,
        draw_bbox: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute annotated video rendering."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = self.output_dir / video_filename

        # Collect all frames ordered by frame_id
        all_frames: List[FrameMetadata] = []
        for f_list in split_dict.values():
            all_frames.extend(f_list)
        all_frames = sorted(all_frames, key=lambda f: f.frame_id)

        total_frames = len(all_frames)
        if total_frames == 0:
            return {"status": "error", "error": "No frames provided to render."}

        # Determine frame dimensions from first frame
        first_frame_path = self.frames_dir / all_frames[0].filename
        sample_img = load_image_bgr(first_frame_path)
        if sample_img is None:
            return {"status": "error", "error": f"Could not read first frame {first_frame_path}"}

        height, width = sample_img.shape[:2]

        # Try video codecs in order of reliability across platforms
        writer = None
        codecs = ["mp4v", "avc1", "H264", "XVID"]
        for codec_str in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                test_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
                if test_writer.isOpened():
                    writer = test_writer
                    logger.info("Initialized cv2.VideoWriter with codec '%s'", codec_str)
                    break
                else:
                    test_writer.release()
            except Exception as e:
                logger.debug("Codec %s failed: %s", codec_str, e)

        if writer is None or not writer.isOpened():
            return {
                "status": "error",
                "error": f"Failed to initialize video writer with any of {codecs}."
            }

        logger.info("Beginning annotated video rendering to %s (%d frames, %.1f fps)",
                    out_video_path, total_frames, fps)

        processed_count = 0
        total_annotations_rendered = 0

        try:
            for frame in all_frames:
                if is_cancelled and is_cancelled():
                    logger.info("Video export cancelled by user.")
                    writer.release()
                    if out_video_path.exists():
                        out_video_path.unlink(missing_ok=True)
                    return {"status": "cancelled"}

                frame_path = self.frames_dir / frame.filename
                img_bgr = load_image_bgr(frame_path)
                if img_bgr is None:
                    # Fallback black frame if image missing
                    img_bgr = np.zeros((height, width, 3), dtype=np.uint8)

                overlay = img_bgr.copy()
                annos = annotations_by_frame.get(frame.frame_id, [])

                for anno in annos:
                    if len(anno.points) < 3:
                        continue

                    # Determine color (BGR for OpenCV)
                    rgb = self.class_color_map.get(anno.class_id, (0, 255, 0))
                    bgr_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

                    pts = np.array(anno.points, dtype=np.int32).reshape((-1, 1, 2))

                    # 1. Fill polygon on overlay
                    cv2.fillPoly(overlay, [pts], bgr_color)

                    # 2. Draw crisp boundary line
                    cv2.polylines(img_bgr, [pts], isClosed=True, color=bgr_color, thickness=2, lineType=cv2.LINE_AA)

                    # 3. Optional Bounding Box
                    xmin, ymin, xmax, ymax, w, h, _, _ = self.get_polygon_bbox(anno.points)
                    if draw_bbox:
                        cv2.rectangle(img_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), bgr_color, 1, lineType=cv2.LINE_AA)

                    # 4. Label Badge
                    c_name = self.class_map.get(anno.class_id, "object")
                    track_txt = f" [{anno.object_id[:6]}]" if anno.object_id else ""
                    badge_text = f"{c_name}{track_txt}"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.5
                    thick = 1
                    (tw, th), baseline = cv2.getTextSize(badge_text, font, scale, thick)

                    badge_x = int(max(0, min(width - tw - 6, xmin)))
                    badge_y = int(max(th + 6, ymin - 4))

                    # Badge background
                    cv2.rectangle(
                        img_bgr,
                        (badge_x, badge_y - th - 4),
                        (badge_x + tw + 6, badge_y + 4),
                        bgr_color,
                        -1,
                    )
                    # Text (black or white for contrast)
                    cv2.putText(
                        img_bgr,
                        badge_text,
                        (badge_x + 3, badge_y),
                        font,
                        scale,
                        (255, 255, 255),
                        thick,
                        cv2.LINE_AA,
                    )
                    total_annotations_rendered += 1

                # Blend filled overlay with original image
                if annos:
                    cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0, img_bgr)

                # Frame number watermark in top-left
                watermark = f"Frame: #{frame.frame_id:04d} | Objects: {len(annos)}"
                cv2.putText(img_bgr, watermark, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img_bgr, watermark, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                writer.write(img_bgr)
                processed_count += 1

                if progress_callback:
                    progress_callback(
                        processed_count,
                        total_frames,
                        f"Rendering video: {processed_count}/{total_frames} frames",
                    )
        finally:
            writer.release()

        if progress_callback:
            progress_callback(total_frames, total_frames, "Annotated video export complete!")

        summary = {
            "status": "success",
            "format": "rendered_video",
            "output_dir": str(self.output_dir),
            "video_path": str(out_video_path),
            "total_frames": processed_count,
            "total_objects": total_annotations_rendered,
            "fps": fps,
            "resolution": f"{width}x{height}",
        }
        logger.info("Video export completed successfully: %s", summary)
        return summary
