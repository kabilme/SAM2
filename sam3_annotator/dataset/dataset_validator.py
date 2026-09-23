"""Automated validator for exported YOLOv8 instance segmentation datasets."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import cv2

from sam3_annotator.utils.logging_utils import logger


class DatasetValidator:
    """Verifies dataset structure, YAML configuration, and label polygon integrity."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir).resolve()

    def validate(self) -> Dict[str, Any]:
        """Perform comprehensive validation and return summary report."""
        report: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "images_count": 0,
                "labels_count": 0,
                "objects_count": 0,
                "splits": {},
                "classes": {},
            },
        }

        if not self.dataset_dir.exists():
            report["valid"] = False
            report["errors"].append(f"Dataset root directory does not exist: {self.dataset_dir}")
            return report

        # 1. Validate data.yaml
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
                # Normalize keys to ints
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

        # 2. Iterate splits
        splits = ["train", "val", "test"]
        found_splits = []

        for split in splits:
            img_split_dir = images_root / split
            lbl_split_dir = labels_root / split

            if not img_split_dir.exists():
                if split in ["train", "val"]:
                    report["warnings"].append(f"Recommended split folder images/{split} not found.")
                continue

            found_splits.append(split)
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

                # Parse label rows
                try:
                    with open(lbl_file, "r", encoding="utf-8") as lf:
                        lines = [line.strip() for line in lf if line.strip()]

                    for line_idx, line in enumerate(lines):
                        parts = line.split()
                        if len(parts) < 7:
                            report["valid"] = False
                            report["errors"].append(
                                f"{lbl_file.name}:{line_idx + 1}: Polygon row has fewer than 6 coordinates (minimum 3 points required)."
                            )
                            continue

                        # Check class ID
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
                        if len(coords) % 2 != 0:
                            report["valid"] = False
                            report["errors"].append(
                                f"{lbl_file.name}:{line_idx + 1}: Odd number of coordinate values ({len(coords)})."
                            )
                            continue

                        # Check coordinate ranges
                        for ci, val_str in enumerate(coords):
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

        # If errors were accumulated, mark invalid
        if report["errors"]:
            report["valid"] = False

        # Generate report files
        report_json_path = self.dataset_dir / "validation_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        report_txt_path = self.dataset_dir / "validation_report.txt"
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("DATASET VALIDATION REPORT\n")
            f.write("=========================\n")
            f.write(f"Status: {'PASSED' if report['valid'] else 'FAILED'}\n")
            f.write(f"Total Images: {report['stats']['images_count']}\n")
            f.write(f"Total Labels: {report['stats']['labels_count']}\n")
            f.write(f"Total Objects: {report['stats']['objects_count']}\n")
            f.write(f"Splits: {report['stats']['splits']}\n")
            f.write(f"Classes: {report['stats']['classes']}\n")
            f.write(f"Errors ({len(report['errors'])}):\n")
            for err in report["errors"][:50]:
                f.write(f" - {err}\n")
            f.write(f"Warnings ({len(report['warnings'])}):\n")
            for w in report["warnings"][:50]:
                f.write(f" - {w}\n")

        logger.info("Validation finished for %s: valid=%s, %d errors, %d warnings",
                    self.dataset_dir, report["valid"], len(report["errors"]), len(report["warnings"]))
        return report
