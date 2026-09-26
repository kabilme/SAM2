"""YOLOv8 instance segmentation dataset exporter."""

import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import cv2
import numpy as np
import yaml

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.annotation.mask_to_polygon import polygon_to_mask
from sam2_annotator.utils.geometry import CoordinateTransformer, validate_polygon
from sam2_annotator.utils.image_utils import load_image_bgr, save_image_bgr, create_colored_mask_overlay
from sam2_annotator.utils.logging_utils import logger


class YOLOExporter:
    """Exports annotations and frames into Ultralytics YOLOv8 instance segmentation format."""

    def __init__(
        self,
        output_dir: Path,
        classes: List[ClassItem],
        frames_dir: Path,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.classes = classes
        self.frames_dir = Path(frames_dir)
        self.class_map = {c.id: c.name for c in classes}

    def export_dataset(
        self,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        export_masks: bool = True,
        export_previews: bool = True,
        create_zip: bool = True,
        include_null_frames: bool = True,
        zip_name: str = "dataset.zip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Execute full dataset export process."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        masks_dir = self.output_dir / "masks"
        previews_dir = self.output_dir / "previews"

        total_frames = sum(len(f_list) for f_list in split_dict.values())
        processed_count = 0
        total_objects_exported = 0
        null_frames_exported = 0
        annotated_frames_exported = 0
        split_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {c.name: 0 for c in self.classes}
        warnings: List[str] = []

        logger.info("Beginning YOLOv8 segmentation export to %s (include_null_frames=%s)",
                    self.output_dir, include_null_frames)

        for split_name, frame_list in split_dict.items():
            if not frame_list:
                continue

            split_img_dir = images_dir / split_name
            split_lbl_dir = labels_dir / split_name
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            if export_masks:
                (masks_dir / split_name).mkdir(parents=True, exist_ok=True)
            if export_previews:
                (previews_dir / split_name).mkdir(parents=True, exist_ok=True)

            split_counts[split_name] = len(frame_list)

            for frame in frame_list:
                if is_cancelled and is_cancelled():
                    logger.info("Export cancelled by user.")
                    return {"status": "cancelled"}

                src_image_path = self.frames_dir / frame.filename
                if not src_image_path.exists():
                    warnings.append(f"Source frame missing: {frame.filename}")
                    continue

                dest_img_path = split_img_dir / frame.filename
                label_filename = f"{Path(frame.filename).stem}.txt"
                dest_lbl_path = split_lbl_dir / label_filename

                # Get annotations for this frame
                annos = annotations_by_frame.get(frame.frame_id, [])
                is_null_frame = (len(annos) == 0)

                # If unannotated and user requested excluding null frames
                if is_null_frame and not include_null_frames:
                    continue

                if is_null_frame:
                    null_frames_exported += 1
                else:
                    annotated_frames_exported += 1

                # Copy image
                shutil.copy2(src_image_path, dest_img_path)

                label_lines = []

                img_bgr = None
                if export_previews or export_masks:
                    img_bgr = load_image_bgr(src_image_path)

                preview_img = img_bgr.copy() if (export_previews and img_bgr is not None) else None
                combined_mask = np.zeros((frame.height, frame.width), dtype=np.uint8) if export_masks else None

                transformer = CoordinateTransformer(frame.width, frame.height)

                if is_null_frame and preview_img is not None:
                    # Subtle indicator on preview for background / null frames
                    cv2.putText(
                        preview_img,
                        "NULL / BACKGROUND",
                        (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (160, 160, 160),
                        2,
                        cv2.LINE_AA,
                    )

                for anno in annos:
                    # Validate polygon
                    is_valid, err = validate_polygon(
                        anno.points,
                        image_width=frame.width,
                        image_height=frame.height,
                        min_area=5.0,
                    )
                    if not is_valid:
                        warnings.append(f"Frame {frame.filename} object {anno.object_id}: {err}")
                        continue

                    # Normalize coordinates
                    norm_points = transformer.image_to_normalized(anno.points)
                    coord_strs = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
                    row = f"{anno.class_id} " + " ".join(coord_strs)
                    label_lines.append(row)

                    total_objects_exported += 1
                    c_name = self.class_map.get(anno.class_id, "unknown")
                    class_counts[c_name] = class_counts.get(c_name, 0) + 1

                    # Masks
                    if export_masks and combined_mask is not None:
                        obj_mask = polygon_to_mask(anno.points, frame.width, frame.height)
                        combined_mask = np.maximum(combined_mask, obj_mask)

                    # Preview overlay
                    if export_previews and preview_img is not None:
                        pts = np.array(anno.points, dtype=np.int32).reshape((-1, 1, 2))
                        color = anno.class_id * 50 % 255, (anno.class_id * 90 + 100) % 255, (anno.class_id * 140 + 50) % 255
                        cv2.polylines(preview_img, [pts], isClosed=True, color=color, thickness=2)
                        if len(anno.points) > 0:
                            label_pos = (int(anno.points[0][0]), max(15, int(anno.points[0][1]) - 5))
                            cv2.putText(
                                preview_img,
                                f"{c_name}:{anno.object_id[:6]}",
                                label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                # Write YOLO format label file (even if empty, for negative frames)
                with open(dest_lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

                # Save optional mask and preview
                if export_masks and combined_mask is not None:
                    mask_path = masks_dir / split_name / f"{Path(frame.filename).stem}.png"
                    save_image_bgr(mask_path, combined_mask)

                if export_previews and preview_img is not None:
                    prev_path = previews_dir / split_name / f"{Path(frame.filename).stem}_preview.jpg"
                    save_image_bgr(prev_path, preview_img, quality=85)

                processed_count += 1
                if progress_callback:
                    progress_callback(
                        processed_count,
                        total_frames,
                        f"Exported {processed_count}/{total_frames} frames",
                    )

        # Generate data.yaml
        data_yaml_path = self.output_dir / "data.yaml"
        yaml_content = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {c.id: c.name for c in self.classes},
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        # Generate dataset_manifest.json
        manifest_path = self.output_dir / "dataset_manifest.json"
        manifest_data = {
            "generated_at": time.time(),
            "classes": [c.to_dict() for c in self.classes],
            "total_images": processed_count,
            "annotated_images": annotated_frames_exported,
            "null_images": null_frames_exported,
            "total_objects": total_objects_exported,
            "split_counts": split_counts,
            "class_counts": class_counts,
            "warnings_count": len(warnings),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Generate dataset README.md
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# YOLOv8 Instance Segmentation Dataset\n\n")
            f.write(f"- Total Images: {processed_count}\n")
            f.write(f"- Annotated Images (with objects): {annotated_frames_exported}\n")
            f.write(f"- Null / Background Images (0 objects): {null_frames_exported}\n")
            f.write(f"- Total Objects: {total_objects_exported}\n")
            f.write(f"- Splits: {split_counts}\n")
            f.write(f"- Classes: {class_counts}\n\n")
            f.write(f"## Training with Ultralytics YOLO:\n")
            f.write("```bash\nyolo segment train data=data.yaml model=yolov8n-seg.pt epochs=50\n```\n")

        # Create ZIP archive if requested
        zip_path = None
        if create_zip:
            if progress_callback:
                progress_callback(processed_count, total_frames, f"Creating ZIP archive ({zip_name})...")
            base_zip_name = self.output_dir.parent / Path(zip_name).stem
            archive_format = "zip"
            created_zip = shutil.make_archive(str(base_zip_name), archive_format, self.output_dir)
            zip_path = str(created_zip)
            logger.info("Created dataset archive: %s", zip_path)

        if progress_callback:
            progress_callback(processed_count, total_frames, "Dataset export complete!")

        summary = {
            "status": "success",
            "output_dir": str(self.output_dir),
            "total_images": processed_count,
            "annotated_images": annotated_frames_exported,
            "null_images": null_frames_exported,
            "total_objects": total_objects_exported,
            "split_counts": split_counts,
            "class_counts": class_counts,
            "warnings": warnings,
            "zip_path": zip_path,
        }
        logger.info("Dataset export completed successfully: %s", summary)
        return summary
