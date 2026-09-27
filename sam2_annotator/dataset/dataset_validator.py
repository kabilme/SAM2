"""Automated multi-format validator for exported datasets and video artifacts.

Supports:
- YOLOv8 Instance Segmentation & Object Detection
- COCO 1.0 JSON format
- Pascal VOC XML & Semantic Segmentation Masks
- LabelMe JSON format
- MOT / MOTChallenge video tracking format
- Rendered Video Overlays (MP4)
"""

import json
import configparser
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import cv2
from PIL import Image

from sam2_annotator.utils.logging_utils import logger


class DatasetValidator:
    """Verifies dataset structure, syntax, annotations, and metadata integrity across multiple formats."""

    def __init__(self, dataset_dir: Path, format_id: Optional[str] = None):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.format_id = format_id.lower() if format_id else None

    def detect_format(self) -> str:
        """Auto-detect the dataset or artifact format from files and directory signatures."""
        if self.format_id:
            return self.format_id

        # 1. Check dataset_manifest.json
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                m_fmt = m.get("format") or m.get("mode")
                if m_fmt:
                    if "coco" in m_fmt:
                        return "coco"
                    if "pascal" in m_fmt or "voc" in m_fmt:
                        return "pascal_voc"
                    if "labelme" in m_fmt:
                        return "labelme"
                    if "mot" in m_fmt:
                        return "mot"
                    if "rendered_video" in m_fmt or "video" in m_fmt:
                        return "rendered_video"
                    if "yolo" in m_fmt:
                        return "yolo"
            except Exception:
                pass

        # 2. Signature detection
        # COCO: annotations/ directory with instances_*.json or *.json
        if (self.dataset_dir / "annotations").is_dir():
            json_files = list((self.dataset_dir / "annotations").glob("*.json"))
            if json_files:
                return "coco"

        # Pascal VOC: Annotations/ and JPEGImages/
        if (self.dataset_dir / "Annotations").is_dir() and (self.dataset_dir / "JPEGImages").is_dir():
            return "pascal_voc"

        # MOT: seqinfo.ini or gt/gt.txt, or a child directory with seqinfo.ini
        if (self.dataset_dir / "seqinfo.ini").exists() or (self.dataset_dir / "gt" / "gt.txt").exists():
            return "mot"
        for child in self.dataset_dir.iterdir():
            if child.is_dir() and ((child / "seqinfo.ini").exists() or (child / "gt" / "gt.txt").exists()):
                return "mot"

        # YOLO: data.yaml or labels/ directory
        if (self.dataset_dir / "data.yaml").exists() or (self.dataset_dir / "labels").is_dir():
            return "yolo"

        # LabelMe: .json files alongside images
        json_candidates = [
            f for f in self.dataset_dir.rglob("*.json")
            if f.name not in ("dataset_manifest.json", "validation_report.json")
        ]
        if json_candidates:
            try:
                with open(json_candidates[0], "r", encoding="utf-8") as f:
                    cand_data = json.load(f)
                if "shapes" in cand_data or "imagePath" in cand_data:
                    return "labelme"
            except Exception:
                pass

        # Rendered Video: .mp4 video files
        if (self.dataset_dir / "annotated_video.mp4").exists() or list(self.dataset_dir.glob("*.mp4")):
            return "rendered_video"

        return "unknown"

    def validate(self) -> Dict[str, Any]:
        """Perform comprehensive validation and return summary report."""
        report: Dict[str, Any] = {
            "valid": True,
            "format": "unknown",
            "format_name": "Unknown Format",
            "errors": [],
            "warnings": [],
            "stats": {
                "images_count": 0,
                "labels_count": 0,
                "annotated_images_count": 0,
                "null_images_count": 0,
                "objects_count": 0,
                "splits": {},
                "classes": {},
                "extra": {},
            },
        }

        if not self.dataset_dir.exists():
            report["valid"] = False
            report["errors"].append(f"Directory does not exist: {self.dataset_dir}")
            return report

        detected = self.detect_format()
        report["format"] = detected

        if detected in ("yolo", "yolo_segmentation", "yolo_detection"):
            self._validate_yolo(report)
        elif detected == "coco":
            self._validate_coco(report)
        elif detected == "pascal_voc":
            self._validate_pascal_voc(report)
        elif detected == "labelme":
            self._validate_labelme(report)
        elif detected == "mot":
            self._validate_mot(report)
        elif detected == "rendered_video":
            self._validate_rendered_video(report)
        else:
            report["valid"] = False
            report["errors"].append(
                f"Could not automatically determine dataset format in '{self.dataset_dir}'. "
                "Expected YOLO (data.yaml), COCO (annotations/*.json), Pascal VOC (Annotations/*.xml), "
                "LabelMe (*.json), MOT (seqinfo.ini), or Rendered Video (*.mp4)."
            )

        if report["errors"]:
            report["valid"] = False

        self._write_reports(report)
        logger.info(
            "Validation finished for %s [%s]: valid=%s, %d errors, %d warnings",
            self.dataset_dir, report.get("format"), report["valid"],
            len(report["errors"]), len(report["warnings"])
        )
        return report

    # --------------------------------------------------------------------------
    # 1. YOLO Validator
    # --------------------------------------------------------------------------
    def _validate_yolo(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report["format_name"] = "YOLOv8 Dataset (Detection / Segmentation)"
        yaml_path = self.dataset_dir / "data.yaml"
        if not yaml_path.exists():
            report["valid"] = False
            report["errors"].append("Missing required data.yaml file.")
            class_names = {}
        else:
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                class_names = yaml_data.get("names", {})
                class_names = {int(k): str(v) for k, v in class_names.items()}
            except Exception as e:
                report["valid"] = False
                report["errors"].append(f"Failed to parse data.yaml: {e}")
                class_names = {}

        images_root = self.dataset_dir / "images"
        labels_root = self.dataset_dir / "labels"

        if not images_root.exists():
            report["valid"] = False
            report["errors"].append("Missing required 'images' directory.")
            return report

        if not labels_root.exists():
            report["valid"] = False
            report["errors"].append("Missing required 'labels' directory.")
            return report

        splits = ["train", "val", "test"]
        for split in splits:
            img_split_dir = images_root / split
            lbl_split_dir = labels_root / split

            if not img_split_dir.exists():
                if split in ["train", "val"]:
                    report["warnings"].append(f"Recommended split folder images/{split} not found.")
                continue

            img_files = list(img_split_dir.glob("*.*"))
            report["stats"]["splits"][split] = len(img_files)

            for img_file in img_files:
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue

                report["stats"]["images_count"] += 1
                stem = img_file.stem
                lbl_file = lbl_split_dir / f"{stem}.txt"

                if not lbl_file.exists():
                    report["valid"] = False
                    report["errors"].append(f"Image {img_file.name} in {split} has no matching label file {lbl_file.name}")
                    continue

                report["stats"]["labels_count"] += 1

                try:
                    with open(lbl_file, "r", encoding="utf-8") as lf:
                        lines = [line.strip() for line in lf if line.strip()]

                    if not lines:
                        report["stats"]["null_images_count"] += 1
                    else:
                        report["stats"]["annotated_images_count"] += 1

                    for line_idx, line in enumerate(lines):
                        parts = line.split()
                        if len(parts) < 5:
                            report["valid"] = False
                            report["errors"].append(
                                f"{lbl_file.name}:{line_idx + 1}: Row has fewer than 4 coordinates (expected bounding box or polygon)."
                            )
                            continue

                        try:
                            cid = int(parts[0])
                        except ValueError:
                            report["valid"] = False
                            report["errors"].append(f"{lbl_file.name}:{line_idx + 1}: Invalid class ID '{parts[0]}'")
                            continue

                        if class_names and cid not in class_names:
                            report["valid"] = False
                            report["errors"].append(f"{lbl_file.name}:{line_idx + 1}: Class ID {cid} not defined in data.yaml")

                        c_name = class_names.get(cid, str(cid))
                        report["stats"]["classes"][c_name] = report["stats"]["classes"].get(c_name, 0) + 1
                        report["stats"]["objects_count"] += 1

                        coords = parts[1:]
                        if len(coords) == 4:
                            report["stats"]["box_objects_count"] = report["stats"].get("box_objects_count", 0) + 1
                        elif len(coords) >= 6 and len(coords) % 2 == 0:
                            report["stats"]["polygon_objects_count"] = report["stats"].get("polygon_objects_count", 0) + 1
                        elif len(coords) < 6:
                            report["valid"] = False
                            report["errors"].append(
                                f"{lbl_file.name}:{line_idx + 1}: Invalid coordinate count ({len(coords)}). Expected 4 for bounding box (cx cy w h) or at least 6 for polygon."
                            )
                            continue
                        else:
                            report["valid"] = False
                            report["errors"].append(
                                f"{lbl_file.name}:{line_idx + 1}: Odd number of coordinate values ({len(coords)})."
                            )
                            continue

                        for val_str in coords:
                            try:
                                val = float(val_str)
                                if val < 0.0 or val > 1.0:
                                    if val < -0.01 or val > 1.01:
                                        report["valid"] = False
                                        report["errors"].append(
                                            f"{lbl_file.name}:{line_idx + 1}: Coordinate value {val} out of normalized range [0.0, 1.0]"
                                        )
                                    else:
                                        report["warnings"].append(
                                            f"{lbl_file.name}:{line_idx + 1}: Coordinate value {val} slightly out of bounds, will be clamped."
                                        )
                            except ValueError:
                                report["valid"] = False
                                report["errors"].append(f"{lbl_file.name}:{line_idx + 1}: Non-numeric coordinate '{val_str}'")
                except Exception as e:
                    report["valid"] = False
                    report["errors"].append(f"Failed to read label file {lbl_file.name}: {e}")

        return report

    # --------------------------------------------------------------------------
    # 2. COCO 1.0 Validator
    # --------------------------------------------------------------------------
    def _validate_coco(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report["format_name"] = "COCO 1.0 Instance Segmentation JSON"
        anno_dir = self.dataset_dir / "annotations"
        if not anno_dir.is_dir():
            report["valid"] = False
            report["errors"].append("Missing required 'annotations' directory.")
            return report

        json_files = list(anno_dir.glob("*.json"))
        if not json_files:
            report["valid"] = False
            report["errors"].append("No JSON annotation files found in 'annotations/' directory.")
            return report

        images_base = self.dataset_dir / "images"

        for jf in json_files:
            split_name = jf.stem.replace("instances_", "")
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    coco = json.load(f)
            except Exception as e:
                report["valid"] = False
                report["errors"].append(f"Failed to parse JSON file {jf.name}: {e}")
                continue

            for req_key in ["images", "annotations", "categories"]:
                if req_key not in coco:
                    report["valid"] = False
                    report["errors"].append(f"{jf.name}: Missing required top-level key '{req_key}'")

            cats = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", []) if "id" in c}
            images = {img["id"]: img for img in coco.get("images", []) if "id" in img}
            report["stats"]["splits"][split_name] = len(images)
            report["stats"]["images_count"] += len(images)

            img_dir = (
                images_base / split_name
                if (images_base / split_name).is_dir()
                else (images_base if images_base.is_dir() else self.dataset_dir)
            )

            for img_info in images.values():
                fn = img_info.get("file_name", "")
                if fn:
                    img_path = img_dir / fn
                    if not img_path.exists():
                        report["warnings"].append(f"{jf.name}: Image file '{fn}' not found in {img_dir.name}")

            annos = coco.get("annotations", [])
            report["stats"]["labels_count"] += len(annos)
            annotated_image_ids = set()

            for anno_idx, anno in enumerate(annos):
                img_id = anno.get("image_id")
                if img_id not in images:
                    report["valid"] = False
                    report["errors"].append(
                        f"{jf.name}: Annotation #{anno.get('id', anno_idx)} references non-existent image_id {img_id}"
                    )
                else:
                    annotated_image_ids.add(img_id)

                cat_id = anno.get("category_id")
                if cat_id not in cats:
                    report["valid"] = False
                    report["errors"].append(
                        f"{jf.name}: Annotation #{anno.get('id', anno_idx)} has unknown category_id {cat_id}"
                    )

                c_name = cats.get(cat_id, str(cat_id))
                report["stats"]["classes"][c_name] = report["stats"]["classes"].get(c_name, 0) + 1
                report["stats"]["objects_count"] += 1

                bbox = anno.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    report["valid"] = False
                    report["errors"].append(
                        f"{jf.name}: Annotation #{anno.get('id', anno_idx)} bbox must be [x, y, w, h]"
                    )
                else:
                    if bbox[2] < 0 or bbox[3] < 0:
                        report["valid"] = False
                        report["errors"].append(
                            f"{jf.name}: Annotation #{anno.get('id', anno_idx)} bbox has negative dimensions: {bbox}"
                        )

                seg = anno.get("segmentation")
                if isinstance(seg, list):
                    for poly in seg:
                        if isinstance(poly, list):
                            if len(poly) < 6:
                                report["valid"] = False
                                report["errors"].append(
                                    f"{jf.name}: Annotation #{anno.get('id', anno_idx)} polygon has fewer than 6 coordinates"
                                )
                            elif len(poly) % 2 != 0:
                                report["valid"] = False
                                report["errors"].append(
                                    f"{jf.name}: Annotation #{anno.get('id', anno_idx)} polygon has odd coordinate count ({len(poly)})"
                                )

            report["stats"]["annotated_images_count"] += len(annotated_image_ids)
            report["stats"]["null_images_count"] += max(0, len(images) - len(annotated_image_ids))

        return report

    # --------------------------------------------------------------------------
    # 3. Pascal VOC Validator
    # --------------------------------------------------------------------------
    def _validate_pascal_voc(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report["format_name"] = "Pascal VOC & Semantic Masks"
        anno_dir = self.dataset_dir / "Annotations"
        jpeg_dir = self.dataset_dir / "JPEGImages"

        if not anno_dir.is_dir():
            report["valid"] = False
            report["errors"].append("Missing required 'Annotations' directory.")
            return report

        if not jpeg_dir.is_dir():
            report["valid"] = False
            report["errors"].append("Missing required 'JPEGImages' directory.")
            return report

        xml_files = list(anno_dir.glob("*.xml"))
        if not xml_files:
            report["valid"] = False
            report["errors"].append("No XML annotation files found in 'Annotations/' directory.")
            return report

        seg_dir = self.dataset_dir / "SegmentationClass"

        for xml_file in xml_files:
            report["stats"]["labels_count"] += 1
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except Exception as e:
                report["valid"] = False
                report["errors"].append(f"Failed to parse XML file {xml_file.name}: {e}")
                continue

            fn_elem = root.find("filename")
            img_filename = fn_elem.text if fn_elem is not None and fn_elem.text else f"{xml_file.stem}.jpg"
            img_path = jpeg_dir / img_filename
            if not img_path.exists():
                report["valid"] = False
                report["errors"].append(f"{xml_file.name}: Referenced image {img_filename} not found in JPEGImages/")
            else:
                report["stats"]["images_count"] += 1

            size_elem = root.find("size")
            im_w = 0
            im_h = 0
            if size_elem is not None:
                w_elem = size_elem.find("width")
                h_elem = size_elem.find("height")
                im_w = int(w_elem.text) if w_elem is not None and w_elem.text else 0
                im_h = int(h_elem.text) if h_elem is not None and h_elem.text else 0

            objects = root.findall("object")
            if not objects:
                report["stats"]["null_images_count"] += 1
            else:
                report["stats"]["annotated_images_count"] += 1

            for obj in objects:
                report["stats"]["objects_count"] += 1
                name_elem = obj.find("name")
                c_name = name_elem.text if name_elem is not None and name_elem.text else "unknown"
                report["stats"]["classes"][c_name] = report["stats"]["classes"].get(c_name, 0) + 1

                bbox = obj.find("bndbox")
                if bbox is not None:
                    try:
                        xmin = float(bbox.find("xmin").text)
                        ymin = float(bbox.find("ymin").text)
                        xmax = float(bbox.find("xmax").text)
                        ymax = float(bbox.find("ymax").text)
                        if xmin > xmax or ymin > ymax:
                            report["valid"] = False
                            report["errors"].append(
                                f"{xml_file.name}: Inverted bounding box bounds [{xmin}, {ymin}, {xmax}, {ymax}]"
                            )
                    except Exception as e:
                        report["valid"] = False
                        report["errors"].append(f"{xml_file.name}: Error parsing bndbox: {e}")

                poly = obj.find("polygon")
                if poly is not None:
                    pts = poly.findall("pt")
                    if len(pts) < 3:
                        report["valid"] = False
                        report["errors"].append(
                            f"{xml_file.name}: Object polygon has fewer than 3 vertices ({len(pts)})"
                        )

            if seg_dir.is_dir():
                mask_path = seg_dir / f"{xml_file.stem}.png"
                if mask_path.exists():
                    try:
                        with Image.open(mask_path) as m_img:
                            if m_img.size != (im_w, im_h) and im_w > 0 and im_h > 0:
                                report["warnings"].append(
                                    f"Mask size {m_img.size} does not match XML size ({im_w}, {im_h}) for {xml_file.stem}"
                                )
                    except Exception as e:
                        report["warnings"].append(f"Failed to open mask {mask_path.name}: {e}")

        main_sets = self.dataset_dir / "ImageSets" / "Main"
        if main_sets.is_dir():
            for txt_file in main_sets.glob("*.txt"):
                split_name = txt_file.stem
                with open(txt_file, "r", encoding="utf-8") as f:
                    stems = [l.strip() for l in f if l.strip()]
                report["stats"]["splits"][split_name] = len(stems)

        return report

    # --------------------------------------------------------------------------
    # 4. LabelMe Validator
    # --------------------------------------------------------------------------
    def _validate_labelme(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report["format_name"] = "LabelMe JSON Polygon Dataset"
        json_files = [
            f for f in self.dataset_dir.rglob("*.json")
            if f.name not in ("dataset_manifest.json", "validation_report.json")
        ]

        if not json_files:
            report["valid"] = False
            report["errors"].append("No LabelMe JSON annotation files found.")
            return report

        for jf in json_files:
            report["stats"]["labels_count"] += 1
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                report["valid"] = False
                report["errors"].append(f"Failed to parse LabelMe JSON {jf.name}: {e}")
                continue

            img_path_rel = data.get("imagePath", "")
            img_found = False
            if img_path_rel:
                cand1 = jf.parent / img_path_rel
                cand2 = self.dataset_dir / img_path_rel
                if cand1.exists() or cand2.exists():
                    img_found = True

            if img_found:
                report["stats"]["images_count"] += 1
            else:
                report["warnings"].append(f"{jf.name}: Referenced image '{img_path_rel}' not found.")

            shapes = data.get("shapes", [])
            if not shapes:
                report["stats"]["null_images_count"] += 1
            else:
                report["stats"]["annotated_images_count"] += 1

            for s_idx, shape in enumerate(shapes):
                report["stats"]["objects_count"] += 1
                c_name = shape.get("label", "unknown")
                report["stats"]["classes"][c_name] = report["stats"]["classes"].get(c_name, 0) + 1

                pts = shape.get("points", [])
                stype = shape.get("shape_type", "polygon")
                if stype == "polygon":
                    if len(pts) < 3:
                        report["valid"] = False
                        report["errors"].append(f"{jf.name}: Shape #{s_idx + 1} ({c_name}) has fewer than 3 vertices.")
                elif stype in ("rectangle", "box"):
                    if len(pts) != 2:
                        report["valid"] = False
                        report["errors"].append(
                            f"{jf.name}: Bounding box shape #{s_idx + 1} must have 2 diagonal points."
                        )

        return report

    # --------------------------------------------------------------------------
    # 5. MOT / MOTChallenge Validator
    # --------------------------------------------------------------------------
    def _validate_mot(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report["format_name"] = "MOT / MOTChallenge Video Tracking"
        seq_dirs = []
        if (self.dataset_dir / "seqinfo.ini").exists():
            seq_dirs.append(self.dataset_dir)
        else:
            for d in self.dataset_dir.iterdir():
                if d.is_dir() and (d / "seqinfo.ini").exists():
                    seq_dirs.append(d)

        if not seq_dirs:
            report["valid"] = False
            report["errors"].append("Missing 'seqinfo.ini' file in sequence directory.")
            return report

        for s_dir in seq_dirs:
            ini_path = s_dir / "seqinfo.ini"
            seq_len = 0
            fps = 30.0
            im_dir_name = "img1"

            try:
                config = configparser.ConfigParser()
                config.read(str(ini_path))
                if "Sequence" in config:
                    sec = config["Sequence"]
                    seq_len = sec.getint("seqLength", fallback=0)
                    fps = sec.getfloat("frameRate", fallback=30.0)
                    im_dir_name = sec.get("imDir", fallback="img1")
            except Exception as e:
                report["warnings"].append(f"Failed to parse seqinfo.ini: {e}")

            im_dir = s_dir / im_dir_name
            if not im_dir.is_dir():
                report["valid"] = False
                report["errors"].append(f"Missing frame directory '{im_dir_name}' in sequence {s_dir.name}")
            else:
                img_files = list(im_dir.glob("*.*"))
                report["stats"]["images_count"] += len(img_files)
                if seq_len > 0 and len(img_files) != seq_len:
                    report["warnings"].append(
                        f"Image count ({len(img_files)}) does not match seqLength ({seq_len})"
                    )

            gt_path = s_dir / "gt" / "gt.txt"
            if not gt_path.exists():
                report["valid"] = False
                report["errors"].append(f"Missing ground truth file: {gt_path.relative_to(self.dataset_dir)}")
            else:
                report["stats"]["labels_count"] += 1
                try:
                    with open(gt_path, "r", encoding="utf-8") as f:
                        gt_lines = [l.strip() for l in f if l.strip()]

                    unique_tracks = set()
                    annotated_frames = set()

                    for line_idx, line in enumerate(gt_lines):
                        parts = line.split(",")
                        if len(parts) < 6:
                            report["valid"] = False
                            report["errors"].append(
                                f"gt.txt:{line_idx + 1}: Row has fewer than 6 comma-separated values"
                            )
                            continue

                        try:
                            f_idx = int(parts[0])
                            t_id = int(parts[1])
                            bw = float(parts[4])
                            bh = float(parts[5])

                            unique_tracks.add(t_id)
                            annotated_frames.add(f_idx)
                            report["stats"]["objects_count"] += 1

                            if bw < 0 or bh < 0:
                                report["valid"] = False
                                report["errors"].append(
                                    f"gt.txt:{line_idx + 1}: Negative bbox dimensions [{bw}, {bh}]"
                                )

                            cid = parts[7] if len(parts) >= 8 else "1"
                            report["stats"]["classes"][cid] = report["stats"]["classes"].get(cid, 0) + 1
                        except ValueError as ve:
                            report["valid"] = False
                            report["errors"].append(f"gt.txt:{line_idx + 1}: Non-numeric value: {ve}")

                    report["stats"]["annotated_images_count"] += len(annotated_frames)
                    report["stats"]["extra"]["unique_tracks"] = len(unique_tracks)
                    report["stats"]["extra"]["sequence_fps"] = fps
                except Exception as e:
                    report["valid"] = False
                    report["errors"].append(f"Failed to read gt.txt: {e}")

        return report

    # --------------------------------------------------------------------------
    # 6. Rendered Video Validator
    # --------------------------------------------------------------------------
    def _validate_rendered_video(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report["format_name"] = "Annotated Video Overlays (MP4)"
        mp4_files = list(self.dataset_dir.glob("*.mp4"))
        if not mp4_files:
            mp4_files = list(self.dataset_dir.rglob("*.mp4"))

        if not mp4_files:
            report["valid"] = False
            report["errors"].append("No MP4 video files found in output directory.")
            return report

        video_path = mp4_files[0]
        report["stats"]["labels_count"] = 1

        if video_path.stat().st_size == 0:
            report["valid"] = False
            report["errors"].append(f"Video file '{video_path.name}' is empty (0 bytes).")
            return report

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            report["valid"] = False
            report["errors"].append(f"Failed to open video file '{video_path.name}' with OpenCV.")
            return report

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        report["stats"]["images_count"] = total_frames
        report["stats"]["annotated_images_count"] = total_frames
        report["stats"]["extra"]["resolution"] = f"{w}x{h}"
        report["stats"]["extra"]["fps"] = fps

        if total_frames <= 0:
            report["valid"] = False
            report["errors"].append(f"Video has invalid frame count: {total_frames}")

        if w <= 0 or h <= 0:
            report["valid"] = False
            report["errors"].append(f"Video has invalid resolution: {w}x{h}")

        ret, frame = cap.read()
        if not ret or frame is None:
            report["valid"] = False
            report["errors"].append("Failed to decode first video frame.")

        cap.release()
        return report

    # --------------------------------------------------------------------------
    # Report Output Files
    # --------------------------------------------------------------------------
    def _write_reports(self, report: Dict[str, Any]) -> None:
        """Write validation_report.json and validation_report.txt to dataset root."""
        try:
            report_json_path = self.dataset_dir / "validation_report.json"
            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            report_txt_path = self.dataset_dir / "validation_report.txt"
            with open(report_txt_path, "w", encoding="utf-8") as f:
                f.write("DATASET / ARTIFACT VALIDATION REPORT\n")
                f.write("===================================\n")
                f.write(f"Format: {report.get('format_name', 'Unknown')}\n")
                f.write(f"Status: {'PASSED' if report['valid'] else 'FAILED'}\n")
                f.write(f"Total Images / Frames: {report['stats']['images_count']}\n")
                f.write(f" - Annotated: {report['stats']['annotated_images_count']}\n")
                f.write(f" - Null (Background): {report['stats']['null_images_count']}\n")
                f.write(f"Total Labels / Files: {report['stats']['labels_count']}\n")
                f.write(f"Total Objects / Detections: {report['stats']['objects_count']}\n")
                if report["stats"]["splits"]:
                    f.write(f"Splits: {report['stats']['splits']}\n")
                if report["stats"]["classes"]:
                    f.write(f"Classes: {report['stats']['classes']}\n")
                if report["stats"]["extra"]:
                    f.write(f"Extra Metadata: {report['stats']['extra']}\n")
                f.write(f"\nErrors ({len(report['errors'])}):\n")
                for err in report["errors"][:50]:
                    f.write(f" - {err}\n")
                f.write(f"\nWarnings ({len(report['warnings'])}):\n")
                for w in report["warnings"][:50]:
                    f.write(f" - {w}\n")
        except Exception as e:
            logger.warning("Failed to write validation report files: %s", e)
