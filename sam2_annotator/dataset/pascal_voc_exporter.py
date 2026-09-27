"""Pascal VOC and Semantic Segmentation Mask exporter."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import cv2
import numpy as np
from PIL import Image

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.annotation.mask_to_polygon import polygon_to_mask
from sam2_annotator.utils.logging_utils import logger
from sam2_annotator.dataset.base_exporter import BaseDatasetExporter


class PascalVOCExporter(BaseDatasetExporter):
    """Exports annotations and frames into Pascal VOC XML and indexed semantic segmentation masks."""

    def export(
        self,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        create_zip: bool = True,
        include_null_frames: bool = True,
        zip_name: str = "pascal_voc_dataset.zip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute Pascal VOC dataset export."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        jpeg_dir = self.output_dir / "JPEGImages"
        anno_dir = self.output_dir / "Annotations"
        seg_dir = self.output_dir / "SegmentationClass"
        main_sets_dir = self.output_dir / "ImageSets" / "Main"
        seg_sets_dir = self.output_dir / "ImageSets" / "Segmentation"

        jpeg_dir.mkdir(parents=True, exist_ok=True)
        anno_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        main_sets_dir.mkdir(parents=True, exist_ok=True)
        seg_sets_dir.mkdir(parents=True, exist_ok=True)

        total_frames = sum(len(f_list) for f_list in split_dict.values())
        processed_count = 0
        total_objects_exported = 0
        annotated_frames_exported = 0
        null_frames_exported = 0
        split_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {c.name: 0 for c in self.classes}
        warnings: List[str] = []

        # Create color palette for 8-bit indexed PNG masks
        # 0: background (black), 1..N: class colors
        palette = [0, 0, 0]
        for c in self.classes:
            rgb = c.color_rgb if c.color_rgb else [0, 255, 0]
            palette.extend(rgb)
        # Pad palette to 256 colors (768 integers)
        while len(palette) < 768:
            palette.extend([0, 0, 0])

        logger.info("Beginning Pascal VOC export to %s", self.output_dir)

        for split_name, frame_list in split_dict.items():
            if not frame_list:
                continue

            split_stems = []
            split_counts[split_name] = len(frame_list)

            for frame in frame_list:
                if is_cancelled and is_cancelled():
                    logger.info("Pascal VOC export cancelled by user.")
                    return {"status": "cancelled"}

                src_image_path = self.frames_dir / frame.filename
                if not src_image_path.exists():
                    warnings.append(f"Source frame missing: {frame.filename}")
                    continue

                stem = Path(frame.filename).stem
                dest_image_path = jpeg_dir / frame.filename
                shutil.copy2(src_image_path, dest_image_path)

                annos = annotations_by_frame.get(frame.frame_id, [])
                is_null = (len(annos) == 0)

                if is_null and not include_null_frames:
                    continue

                if is_null:
                    null_frames_exported += 1
                else:
                    annotated_frames_exported += 1

                split_stems.append(stem)

                # 1. Build Pascal VOC XML ElementTree
                root = ET.Element("annotation")
                ET.SubElement(root, "folder").text = "JPEGImages"
                ET.SubElement(root, "filename").text = frame.filename
                source = ET.SubElement(root, "source")
                ET.SubElement(source, "database").text = "SAM2 Video Polygon Annotator"

                size = ET.SubElement(root, "size")
                ET.SubElement(size, "width").text = str(frame.width)
                ET.SubElement(size, "height").text = str(frame.height)
                ET.SubElement(size, "depth").text = "3"
                ET.SubElement(root, "segmented").text = "1"

                # 2. Build Semantic Segmentation Palette Mask (indexed 8-bit)
                mask_array = np.zeros((frame.height, frame.width), dtype=np.uint8)

                for anno in annos:
                    if len(anno.points) < 3:
                        continue

                    xmin, ymin, xmax, ymax, _, _, _, _ = self.get_polygon_bbox(anno.points)
                    c_name = self.class_map.get(anno.class_id, "object")

                    obj_elem = ET.SubElement(root, "object")
                    ET.SubElement(obj_elem, "name").text = c_name
                    ET.SubElement(obj_elem, "pose").text = "Unspecified"
                    ET.SubElement(obj_elem, "truncated").text = "0"
                    ET.SubElement(obj_elem, "difficult").text = "0"

                    bndbox = ET.SubElement(obj_elem, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(int(round(xmin)))
                    ET.SubElement(bndbox, "ymin").text = str(int(round(ymin)))
                    ET.SubElement(bndbox, "xmax").text = str(int(round(xmax)))
                    ET.SubElement(bndbox, "ymax").text = str(int(round(ymax)))

                    # Polygon points in XML
                    poly_elem = ET.SubElement(obj_elem, "polygon")
                    for pt in anno.points:
                        pt_elem = ET.SubElement(poly_elem, "pt")
                        ET.SubElement(pt_elem, "x").text = str(int(round(pt[0])))
                        ET.SubElement(pt_elem, "y").text = str(int(round(pt[1])))

                    # Paint on semantic palette mask (class index + 1)
                    obj_mask = polygon_to_mask(anno.points, frame.width, frame.height)
                    mask_array[obj_mask > 0] = (anno.class_id + 1)

                    total_objects_exported += 1
                    class_counts[c_name] = class_counts.get(c_name, 0) + 1

                # Write XML
                xml_path = anno_dir / f"{stem}.xml"
                tree = ET.ElementTree(root)
                tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)

                # Write Palette PNG mask
                pil_mask = Image.fromarray(mask_array, mode="P")
                pil_mask.putpalette(palette)
                mask_png_path = seg_dir / f"{stem}.png"
                pil_mask.save(str(mask_png_path))

                processed_count += 1
                if progress_callback:
                    progress_callback(
                        processed_count,
                        total_frames,
                        f"Pascal VOC export: {processed_count}/{total_frames} frames",
                    )

            # Write ImageSets txt files
            with open(main_sets_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(split_stems) + ("\n" if split_stems else ""))
            with open(seg_sets_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(split_stems) + ("\n" if split_stems else ""))

        # Dataset README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Pascal VOC & Semantic Segmentation Dataset\n\n")
            f.write(f"- Total Images: {processed_count}\n")
            f.write(f"- Annotated Images: {annotated_frames_exported}\n")
            f.write(f"- Null / Background Images: {null_frames_exported}\n")
            f.write(f"- Total Objects: {total_objects_exported}\n")
            f.write(f"- Splits: {split_counts}\n")
            f.write(f"- Classes: {class_counts}\n\n")
            f.write("## Structure:\n")
            f.write("- `JPEGImages/`: Original frame images.\n")
            f.write("- `Annotations/`: Pascal VOC XML files containing bndbox and polygon vertices.\n")
            f.write("- `SegmentationClass/`: 8-bit indexed palette PNG masks (pixel value = class_id + 1).\n")
            f.write("- `ImageSets/`: Train / Val / Test split index files.\n")

        # Create ZIP archive if requested
        zip_path = None
        if create_zip:
            zip_path = self.create_zip_archive(self.output_dir, zip_name, progress_callback, total_frames)

        if progress_callback:
            progress_callback(processed_count, total_frames, "Pascal VOC export complete!")

        summary = {
            "status": "success",
            "format": "pascal_voc",
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
        logger.info("Pascal VOC dataset export completed successfully: %s", summary)
        return summary
