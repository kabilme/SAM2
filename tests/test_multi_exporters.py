"""Unit and integration tests for multiple dataset and video exporters."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.dataset.split_manager import DatasetSplitter
from sam2_annotator.dataset.base_exporter import BaseDatasetExporter
from sam2_annotator.dataset.yolo_exporter import YOLOExporter
from sam2_annotator.dataset.coco_exporter import COCOExporter
from sam2_annotator.dataset.pascal_voc_exporter import PascalVOCExporter
from sam2_annotator.dataset.labelme_exporter import LabelMeExporter
from sam2_annotator.dataset.mot_exporter import MOTExporter
from sam2_annotator.dataset.rendered_video_exporter import RenderedVideoExporter
from sam2_annotator.dataset.exporter_factory import (
    create_exporter,
    get_available_formats,
    EXPORT_FORMATS,
)
from sam2_annotator.utils.image_utils import save_image_bgr


@pytest.fixture
def mock_dataset_env(tmp_path):
    """Fixture providing a mock frame repository and annotations."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    frames = []
    annos = {}
    for i in range(1, 7):
        filename = f"frame_{i:04d}.jpg"
        img = np.full((120, 160, 3), 100 + i * 15, dtype=np.uint8)
        save_image_bgr(frames_dir / filename, img)

        meta = FrameMetadata(
            frame_id=i,
            source_frame_index=i * 5,
            timestamp_seconds=float(i * 0.2),
            filename=filename,
            width=160,
            height=120,
        )
        frames.append(meta)

        if i in (1, 2, 3, 5):
            # Frame with tracked object 1
            anno1 = PolygonAnnotation(
                object_id="vehicle_track_1",
                class_id=0,
                class_name="car",
                frame_id=i,
                source_frame_index=i * 5,
                points=[(20.0, 20.0), (80.0, 20.0), (80.0, 70.0), (20.0, 70.0)],
            )
            # Frame with tracked object 2 (on frames 2 and 3)
            if i in (2, 3):
                anno2 = PolygonAnnotation(
                    object_id="pedestrian_track_2",
                    class_id=1,
                    class_name="person",
                    frame_id=i,
                    source_frame_index=i * 5,
                    points=[(90.0, 30.0), (120.0, 30.0), (120.0, 90.0), (90.0, 90.0)],
                )
                annos[i] = [anno1, anno2]
            else:
                annos[i] = [anno1]
        else:
            # Null/negative frames (4, 6)
            annos[i] = []

    classes = [
        ClassItem(id=0, name="car", color_rgb=[255, 0, 0]),
        ClassItem(id=1, name="person", color_rgb=[0, 255, 0]),
    ]
    return frames, annos, classes, frames_dir


def test_exporter_factory_registry():
    """Test exporter factory format enumeration and creation."""
    formats = get_available_formats()
    format_ids = [f["id"] for f in formats]

    assert "yolo_segmentation" in format_ids
    assert "yolo_detection" in format_ids
    assert "coco" in format_ids
    assert "pascal_voc" in format_ids
    assert "labelme" in format_ids
    assert "mot" in format_ids
    assert "rendered_video" in format_ids

    # Unknown format raises ValueError
    with pytest.raises(ValueError):
        create_exporter("invalid_format", Path("out"), [], Path("frames"))


def test_coco_exporter(tmp_path, mock_dataset_env):
    """Test COCO 1.0 JSON format export."""
    frames, annos, classes, frames_dir = mock_dataset_env
    out_dir = tmp_path / "coco_out"

    split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.67, val_ratio=0.33, test_ratio=0.0)
    exporter = create_exporter("coco", out_dir, classes, frames_dir)
    assert isinstance(exporter, COCOExporter)

    result = exporter.export(
        split_dict=split_dict,
        annotations_by_frame=annos,
        create_zip=True,
        include_null_frames=True,
    )

    assert result["status"] == "success"
    assert result["total_images"] == 6
    assert (out_dir / "annotations" / "instances_train.json").exists()
    assert (out_dir / "annotations" / "instances_val.json").exists()
    assert result["zip_path"] is not None
    assert Path(result["zip_path"]).exists()

    # Verify JSON content
    with open(out_dir / "annotations" / "instances_train.json", "r", encoding="utf-8") as f:
        coco_train = json.load(f)

    assert "images" in coco_train
    assert "annotations" in coco_train
    assert "categories" in coco_train
    assert len(coco_train["categories"]) == 2
    assert coco_train["categories"][0]["name"] == "car"
    assert coco_train["categories"][0]["id"] == 1  # 1-based

    if coco_train["annotations"]:
        anno0 = coco_train["annotations"][0]
        assert "bbox" in anno0
        assert len(anno0["bbox"]) == 4  # [x, y, w, h]
        assert "segmentation" in anno0
        assert "area" in anno0
        assert anno0["area"] > 0


def test_pascal_voc_exporter(tmp_path, mock_dataset_env):
    """Test Pascal VOC XML and palette PNG mask export."""
    frames, annos, classes, frames_dir = mock_dataset_env
    out_dir = tmp_path / "voc_out"

    split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
    exporter = create_exporter("pascal_voc", out_dir, classes, frames_dir)
    assert isinstance(exporter, PascalVOCExporter)

    result = exporter.export(
        split_dict=split_dict,
        annotations_by_frame=annos,
        create_zip=False,
    )

    assert result["status"] == "success"
    assert (out_dir / "JPEGImages").exists()
    assert (out_dir / "Annotations").exists()
    assert (out_dir / "SegmentationClass").exists()
    assert (out_dir / "ImageSets" / "Main" / "train.txt").exists()

    # Verify XML content
    xml_file = out_dir / "Annotations" / "frame_0001.xml"
    assert xml_file.exists()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    assert root.tag == "annotation"
    assert root.find("filename").text == "frame_0001.jpg"
    obj = root.find("object")
    assert obj is not None
    assert obj.find("name").text == "car"
    assert obj.find("bndbox") is not None
    assert obj.find("polygon") is not None

    # Verify 8-bit palette PNG mask
    mask_file = out_dir / "SegmentationClass" / "frame_0001.png"
    assert mask_file.exists()
    with Image.open(mask_file) as img:
        assert img.mode == "P"  # Palette indexed
        arr = np.array(img)
        # Class 0 mapped to value 1
        assert 1 in arr


def test_labelme_exporter(tmp_path, mock_dataset_env):
    """Test LabelMe JSON export."""
    frames, annos, classes, frames_dir = mock_dataset_env
    out_dir = tmp_path / "labelme_out"

    split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
    exporter = create_exporter("labelme", out_dir, classes, frames_dir)
    assert isinstance(exporter, LabelMeExporter)

    result = exporter.export(
        split_dict=split_dict,
        annotations_by_frame=annos,
        create_zip=False,
    )

    assert result["status"] == "success"
    json_path = out_dir / "images" / "train" / "frame_0001.json"
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["imagePath"] == "frame_0001.jpg"
    assert len(data["shapes"]) == 1
    assert data["shapes"][0]["label"] == "car"
    assert data["shapes"][0]["shape_type"] == "polygon"
    assert len(data["shapes"][0]["points"]) == 4


def test_mot_exporter(tmp_path, mock_dataset_env):
    """Test MOT tracking format export."""
    frames, annos, classes, frames_dir = mock_dataset_env
    out_dir = tmp_path / "mot_out"

    split_dict = {"all": frames}
    exporter = create_exporter("mot", out_dir, classes, frames_dir)
    assert isinstance(exporter, MOTExporter)

    result = exporter.export(
        split_dict=split_dict,
        annotations_by_frame=annos,
        sequence_name="test_seq",
        fps=25.0,
        create_zip=False,
    )

    assert result["status"] == "success"
    seq_dir = out_dir / "test_seq"
    assert (seq_dir / "img1" / "000001.jpg").exists()
    assert (seq_dir / "seqinfo.ini").exists()
    assert (seq_dir / "gt" / "gt.txt").exists()

    # Verify seqinfo.ini
    with open(seq_dir / "seqinfo.ini", "r", encoding="utf-8") as f:
        ini_content = f.read()
    assert "name=test_seq" in ini_content
    assert "frameRate=25.0" in ini_content
    assert "seqLength=6" in ini_content

    # Verify gt.txt content
    with open(seq_dir / "gt" / "gt.txt", "r", encoding="utf-8") as f:
        gt_lines = [line.strip() for line in f if line.strip()]

    assert len(gt_lines) > 0
    parts = gt_lines[0].split(",")
    assert len(parts) == 9  # frame, id, x, y, w, h, conf, class, visibility
    assert parts[0] == "1"  # frame 1


def test_rendered_video_exporter(tmp_path, mock_dataset_env):
    """Test MP4 annotated overlay video export."""
    frames, annos, classes, frames_dir = mock_dataset_env
    out_dir = tmp_path / "video_out"

    split_dict = {"all": frames}
    exporter = create_exporter("rendered_video", out_dir, classes, frames_dir)
    assert isinstance(exporter, RenderedVideoExporter)

    result = exporter.export(
        split_dict=split_dict,
        annotations_by_frame=annos,
        video_filename="annotated_output.mp4",
        fps=10.0,
        alpha=0.5,
        draw_bbox=True,
    )

    assert result["status"] == "success"
    video_path = Path(result["video_path"])
    assert video_path.exists()
    assert video_path.stat().st_size > 0

    # Verify OpenCV can read the output video
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    read_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        read_frames += 1
    cap.release()
    assert read_frames == 6


def test_yolo_detection_mode(tmp_path, mock_dataset_env):
    """Test YOLO exporter in object detection (bounding box) mode."""
    frames, annos, classes, frames_dir = mock_dataset_env
    out_dir = tmp_path / "yolo_det_out"

    split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
    exporter = create_exporter("yolo_detection", out_dir, classes, frames_dir)
    assert isinstance(exporter, YOLOExporter)
    assert exporter.mode == "detection"

    result = exporter.export(
        split_dict=split_dict,
        annotations_by_frame=annos,
        export_masks=False,
        export_previews=False,
        create_zip=False,
    )

    assert result["status"] == "success"
    lbl_file = out_dir / "labels" / "train" / "frame_0001.txt"
    assert lbl_file.exists()

    with open(lbl_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    assert len(lines) == 1
    tokens = lines[0].split()
    assert len(tokens) == 5  # class_id cx cy w h
    assert tokens[0] == "0"
    for val_str in tokens[1:]:
        val = float(val_str)
        assert 0.0 <= val <= 1.0


def test_dataset_export_worker_with_new_exporters(tmp_path, mock_dataset_env):
    """Test DatasetExportWorker with COCOExporter and RenderedVideoExporter."""
    from PySide6.QtWidgets import QApplication
    from sam2_annotator.ui.main_window import DatasetExportWorker

    app = QApplication.instance() or QApplication([])

    frames, annos, classes, frames_dir = mock_dataset_env
    out_dir = tmp_path / "worker_coco_out"
    split_dict = {"all": frames}

    coco_exporter = create_exporter("coco", out_dir, classes, frames_dir)
    worker = DatasetExportWorker(
        exporter=coco_exporter,
        split_dict=split_dict,
        annotations_by_frame=annos,
        params={"create_zip": False, "include_null_frames": True},
    )

    completed_result = {}

    def on_finished(res):
        completed_result.update(res)

    worker.finished.connect(on_finished)
    worker.start()
    worker.wait(5000)
    app.processEvents()

    assert completed_result.get("status") == "success"
    assert completed_result.get("format") == "coco"
