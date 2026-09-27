"""LabelMe JSON dataset exporter."""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.utils.logging_utils import logger
from sam2_annotator.dataset.base_exporter import BaseDatasetExporter


class LabelMeExporter(BaseDatasetExporter):
    """Exports annotations and frames into LabelMe JSON format (one JSON per image)."""

    def export(
        self,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        create_zip: bool = True,
        include_null_frames: bool = True,
        zip_name: str = "labelme_dataset.zip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute LabelMe dataset export."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images_base_dir = self.output_dir / "images"
        images_base_dir.mkdir(parents=True, exist_ok=True)

        total_frames = sum(len(f_list) for f_list in split_dict.values())
        processed_count = 0
        total_objects_exported = 0
        annotated_frames_exported = 0
        null_frames_exported = 0
        split_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {c.name: 0 for c in self.classes}
        warnings: List[str] = []

        logger.info("Beginning LabelMe dataset export to %s", self.output_dir)

        for split_name, frame_list in split_dict.items():
            if not frame_list:
                continue

            split_dir = images_base_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_counts[split_name] = len(frame_list)

            for frame in frame_list:
                if is_cancelled and is_cancelled():
                    logger.info("LabelMe export cancelled by user.")
                    return {"status": "cancelled"}

                src_image_path = self.frames_dir / frame.filename
                if not src_image_path.exists():
                    warnings.append(f"Source frame missing: {frame.filename}")
                    continue

                dest_image_path = split_dir / frame.filename
                shutil.copy2(src_image_path, dest_image_path)

                annos = annotations_by_frame.get(frame.frame_id, [])
                is_null = (len(annos) == 0)

                if is_null and not include_null_frames:
                    continue

                if is_null:
                    null_frames_exported += 1
                else:
                    annotated_frames_exported += 1

                # Construct LabelMe shapes
                shapes = []
                for anno in annos:
                    if len(anno.points) < 3:
                        continue

                    c_name = self.class_map.get(anno.class_id, "object")
                    points_list = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in anno.points]

                    shapes.append({
                        "label": c_name,
                        "points": points_list,
                        "group_id": None,
                        "description": f"track_id:{anno.object_id}" if anno.object_id else "",
                        "shape_type": "polygon",
                        "flags": {},
                    })
                    total_objects_exported += 1
                    class_counts[c_name] = class_counts.get(c_name, 0) + 1

                labelme_doc = {
                    "version": "5.3.1",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": frame.filename,
                    "imageData": None,
                    "imageHeight": frame.height,
                    "imageWidth": frame.width,
                }

                stem = Path(frame.filename).stem
                json_path = split_dir / f"{stem}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(labelme_doc, f, indent=2)

                processed_count += 1
                if progress_callback:
                    progress_callback(
                        processed_count,
                        total_frames,
                        f"LabelMe export: {processed_count}/{total_frames} frames",
                    )

        # Dataset README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# LabelMe Polygon Dataset\n\n")
            f.write(f"- Total Images: {processed_count}\n")
            f.write(f"- Annotated Images: {annotated_frames_exported}\n")
            f.write(f"- Null / Background Images: {null_frames_exported}\n")
            f.write(f"- Total Objects: {total_objects_exported}\n")
            f.write(f"- Splits: {split_counts}\n")
            f.write(f"- Classes: {class_counts}\n\n")
            f.write("## Usage:\n")
            f.write("Compatible with the open-source LabelMe desktop tool (`labelme images/train`).\n")

        # Create ZIP archive if requested
        zip_path = None
        if create_zip:
            zip_path = self.create_zip_archive(self.output_dir, zip_name, progress_callback, total_frames)

        if progress_callback:
            progress_callback(processed_count, total_frames, "LabelMe dataset export complete!")

        summary = {
            "status": "success",
            "format": "labelme",
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
        logger.info("LabelMe dataset export completed successfully: %s", summary)
        return summary
