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


def test_validator_on_coco_dataset(tmp_path):
    """Test validator on COCO 1.0 JSON format."""
    import json
    dataset_dir = tmp_path / "valid_coco"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "annotations").mkdir(parents=True)

    img_path = dataset_dir / "images" / "train" / "img1.jpg"
    img_path.write_bytes(b"dummy")

    coco_data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 50.0, 50.0],
                "segmentation": [[100.0, 100.0, 150.0, 100.0, 150.0, 150.0, 100.0, 150.0]],
                "area": 2500.0,
            }
        ],
        "categories": [{"id": 1, "name": "vehicle"}],
    }
    with open(dataset_dir / "annotations" / "instances_train.json", "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is True
    assert report["format"] == "coco"
    assert report["stats"]["images_count"] == 1
    assert report["stats"]["objects_count"] == 1
    assert (dataset_dir / "validation_report.json").exists()
    assert (dataset_dir / "validation_report.txt").exists()


def test_validator_on_pascal_voc(tmp_path):
    """Test validator on Pascal VOC format."""
    dataset_dir = tmp_path / "valid_voc"
    (dataset_dir / "JPEGImages").mkdir(parents=True)
    (dataset_dir / "Annotations").mkdir(parents=True)

    img_path = dataset_dir / "JPEGImages" / "frame_0001.jpg"
    img_path.write_bytes(b"dummy")

    xml_content = (
        "<annotation>\n"
        "  <filename>frame_0001.jpg</filename>\n"
        "  <size><width>640</width><height>480</height></size>\n"
        "  <object>\n"
        "    <name>car</name>\n"
        "    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>100</xmax><ymax>120</ymax></bndbox>\n"
        "  </object>\n"
        "</annotation>\n"
    )
    (dataset_dir / "Annotations" / "frame_0001.xml").write_text(xml_content, encoding="utf-8")

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is True
    assert report["format"] == "pascal_voc"
    assert report["stats"]["objects_count"] == 1


def test_validator_on_labelme(tmp_path):
    """Test validator on LabelMe JSON format."""
    import json
    dataset_dir = tmp_path / "valid_labelme"
    dataset_dir.mkdir(parents=True)

    img_path = dataset_dir / "frame_0001.jpg"
    img_path.write_bytes(b"dummy")

    labelme_data = {
        "imagePath": "frame_0001.jpg",
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [
            {
                "label": "pedestrian",
                "shape_type": "polygon",
                "points": [[10.0, 10.0], [50.0, 10.0], [50.0, 80.0], [10.0, 80.0]],
            }
        ],
    }
    with open(dataset_dir / "frame_0001.json", "w", encoding="utf-8") as f:
        json.dump(labelme_data, f)

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is True
    assert report["format"] == "labelme"
    assert report["stats"]["objects_count"] == 1


def test_validator_on_mot(tmp_path):
    """Test validator on MOT format."""
    dataset_dir = tmp_path / "valid_mot"
    seq_dir = dataset_dir / "seq01"
    (seq_dir / "img1").mkdir(parents=True)
    (seq_dir / "gt").mkdir(parents=True)

    (seq_dir / "img1" / "000001.jpg").write_bytes(b"dummy")
    ini_content = "[Sequence]\nname=seq01\nimDir=img1\nframeRate=30.0\nseqLength=1\nimWidth=640\nimHeight=480\n"
    (seq_dir / "seqinfo.ini").write_text(ini_content, encoding="utf-8")

    gt_content = "1,1,10.0,20.0,50.0,60.0,1,1,1.0\n"
    (seq_dir / "gt" / "gt.txt").write_text(gt_content, encoding="utf-8")

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is True
    assert report["format"] == "mot"
    assert report["stats"]["objects_count"] == 1


def test_validator_on_rendered_video(tmp_path):
    """Test validator on rendered video MP4 format."""
    import cv2
    import numpy as np

    dataset_dir = tmp_path / "valid_video"
    dataset_dir.mkdir(parents=True)
    video_path = dataset_dir / "annotated_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (100, 100))
    for _ in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    validator = DatasetValidator(dataset_dir)
    report = validator.validate()

    assert report["valid"] is True
    assert report["format"] == "rendered_video"
    assert report["stats"]["images_count"] == 5


