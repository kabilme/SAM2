"""COCO 1.0 instance segmentation dataset exporter."""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.utils.logging_utils import logger
from sam2_annotator.dataset.base_exporter import BaseDatasetExporter


class COCOExporter(BaseDatasetExporter):
    """Exports annotations and frames into COCO 1.0 instance segmentation JSON format."""

    def export(
        self,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        create_zip: bool = True,
        include_null_frames: bool = True,
        zip_name: str = "coco_dataset.zip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute COCO dataset export."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images_base_dir = self.output_dir / "images"
        annotations_dir = self.output_dir / "annotations"
        images_base_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        total_frames = sum(len(f_list) for f_list in split_dict.values())
        processed_count = 0
        total_objects_exported = 0
        annotated_frames_exported = 0
        null_frames_exported = 0
        split_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {c.name: 0 for c in self.classes}
        warnings: List[str] = []

        # Build COCO categories (1-based category IDs as standard in COCO)
        categories = [
            {
                "id": c.id + 1,
                "name": c.name,
                "supercategory": "object",
            }
            for c in self.classes
        ]

        logger.info("Beginning COCO dataset export to %s", self.output_dir)

        global_anno_id = 1

        for split_name, frame_list in split_dict.items():
            if not frame_list:
                continue

            split_img_dir = images_base_dir / split_name
            split_img_dir.mkdir(parents=True, exist_ok=True)

            coco_images = []
            coco_annotations = []
            split_counts[split_name] = len(frame_list)

            for frame in frame_list:
                if is_cancelled and is_cancelled():
                    logger.info("COCO export cancelled by user.")
                    return {"status": "cancelled"}

                src_image_path = self.frames_dir / frame.filename
                if not src_image_path.exists():
                    warnings.append(f"Source frame missing: {frame.filename}")
                    continue

                dest_image_path = split_img_dir / frame.filename
                shutil.copy2(src_image_path, dest_image_path)

                annos = annotations_by_frame.get(frame.frame_id, [])
                is_null = (len(annos) == 0)

                if is_null and not include_null_frames:
                    continue

                if is_null:
                    null_frames_exported += 1
                else:
                    annotated_frames_exported += 1

                # Record image entry in COCO format
                image_id = frame.frame_id
                coco_images.append({
                    "id": image_id,
                    "file_name": frame.filename,
                    "width": frame.width,
                    "height": frame.height,
                    "date_captured": datetime.now().isoformat(),
                    "license": 1,
                })

                for anno in annos:
                    if len(anno.points) < 3:
                        continue

                    xmin, ymin, xmax, ymax, w, h, _, _ = self.get_polygon_bbox(anno.points)
                    area = self.get_polygon_area(anno.points)

                    # Flatten points for COCO: [x1, y1, x2, y2, ...]
                    flat_segmentation = []
                    for pt in anno.points:
                        flat_segmentation.extend([round(float(pt[0]), 2), round(float(pt[1]), 2)])

                    coco_annotations.append({
                        "id": global_anno_id,
                        "image_id": image_id,
                        "category_id": anno.class_id + 1,
                        "segmentation": [flat_segmentation],
                        "area": round(area, 2),
                        "bbox": [round(xmin, 2), round(ymin, 2), round(w, 2), round(h, 2)],
                        "iscrowd": 0,
                    })
                    global_anno_id += 1
                    total_objects_exported += 1
                    c_name = self.class_map.get(anno.class_id, "unknown")
                    class_counts[c_name] = class_counts.get(c_name, 0) + 1

                processed_count += 1
                if progress_callback:
                    progress_callback(
                        processed_count,
                        total_frames,
                        f"COCO export: processed {processed_count}/{total_frames} frames",
                    )

            # Write COCO JSON for this split
            split_json_path = annotations_dir / f"instances_{split_name}.json"
            coco_data = {
                "info": {
                    "description": "COCO dataset exported from SAM2 Video Polygon Annotator",
                    "url": "",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "SAM2 Annotator",
                    "date_created": datetime.now().isoformat(),
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "MIT License",
                        "url": "https://opensource.org/licenses/MIT",
                    }
                ],
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            }

            with open(split_json_path, "w", encoding="utf-8") as f:
                json.dump(coco_data, f, indent=2)

        # Generate dataset README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# COCO 1.0 Instance Segmentation Dataset\n\n")
            f.write(f"- Total Images: {processed_count}\n")
            f.write(f"- Annotated Images: {annotated_frames_exported}\n")
            f.write(f"- Null / Background Images: {null_frames_exported}\n")
            f.write(f"- Total Objects: {total_objects_exported}\n")
            f.write(f"- Splits: {split_counts}\n")
            f.write(f"- Classes: {class_counts}\n\n")
            f.write("## Usage with MMDetection or Detectron2:\n")
            f.write("```python\n# Load with torchvision or pycocotools:\n")
            f.write("from pycocotools.coco import COCO\n")
            f.write("coco = COCO('annotations/instances_train.json')\n```\n")

        # Create ZIP archive if requested
        zip_path = None
        if create_zip:
            zip_path = self.create_zip_archive(self.output_dir, zip_name, progress_callback, total_frames)

        if progress_callback:
            progress_callback(processed_count, total_frames, "COCO dataset export complete!")

        summary = {
            "status": "success",
            "format": "coco",
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
        logger.info("COCO dataset export completed successfully: %s", summary)
        return summary
