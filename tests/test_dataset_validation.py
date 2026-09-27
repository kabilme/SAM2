"""Unit tests for dataset validation."""

import pytest
import yaml
from pathlib import Path

from sam2_annotator.dataset.dataset_validator import DatasetValidator


def test_validator_on_valid_dataset(tmp_path):
    dataset_dir = tmp_path / "valid_ds"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)

    # Image
    img_path = dataset_dir / "images" / "train" / "img1.jpg"
    img_path.write_bytes(b"dummy")

    # Label
    lbl_path = dataset_dir / "labels" / "train" / "img1.txt"
    lbl_path.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n", encoding="utf-8")

    # data.yaml
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"names": {0: "scooter"}}, f)

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is True
    assert len(report["errors"]) == 0
    assert report["stats"]["objects_count"] == 1


def test_validator_detects_invalid_coords(tmp_path):
    dataset_dir = tmp_path / "invalid_ds"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)

    img_path = dataset_dir / "images" / "train" / "img1.jpg"
    img_path.write_bytes(b"dummy")

    # Out of range coordinate 2.5
    lbl_path = dataset_dir / "labels" / "train" / "img1.txt"
    lbl_path.write_text("0 0.1 0.1 2.5 0.1 0.5 0.5 0.1 0.5\n", encoding="utf-8")

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"names": {0: "scooter"}}, f)

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is False
    assert any("out of normalized range" in err for err in report["errors"])


def test_validator_on_valid_detection_dataset(tmp_path):
    """Test validator successfully validates YOLO bounding box detection format."""
    dataset_dir = tmp_path / "valid_detection_ds"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)

    # Image
    img_path = dataset_dir / "images" / "train" / "img1.jpg"
    img_path.write_bytes(b"dummy")

    # Bounding box label: class_id cx cy w h (5 tokens)
    lbl_path = dataset_dir / "labels" / "train" / "img1.txt"
    lbl_path.write_text("0 0.35 0.45 0.20 0.30\n", encoding="utf-8")

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"names": {0: "scooter"}}, f)

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is True
    assert len(report["errors"]) == 0
    assert report["stats"]["objects_count"] == 1
    assert report["stats"]["box_objects_count"] == 1

