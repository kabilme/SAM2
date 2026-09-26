"""Unit tests for YOLOv8 dataset export."""

import pytest
import numpy as np
import yaml
from pathlib import Path

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.dataset.split_manager import DatasetSplitter
from sam2_annotator.dataset.yolo_exporter import YOLOExporter
from sam2_annotator.utils.image_utils import save_image_bgr


@pytest.fixture
def mock_project_env(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    frames = []
    annos = {}
    for i in range(1, 11):
        filename = f"frame_{i:04d}.jpg"
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        save_image_bgr(frames_dir / filename, img)

        meta = FrameMetadata(
            frame_id=i,
            source_frame_index=i * 10,
            timestamp_seconds=float(i),
            filename=filename,
            width=100,
            height=100,
        )
        frames.append(meta)

        # Add one object on odd frames
        if i % 2 == 1:
            anno = PolygonAnnotation(
                object_id=f"obj_{i}",
                class_id=0,
                class_name="scooter",
                frame_id=i,
                source_frame_index=i * 10,
                points=[(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)],
            )
            annos[i] = [anno]
        else:
            annos[i] = [] # Negative frame

    classes = [ClassItem(id=0, name="scooter")]
    return frames, annos, classes, frames_dir


def test_dataset_splitter(mock_project_env):
    frames, _, _, _ = mock_project_env
    split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, strategy="sequential")

    assert len(split_dict["train"]) == 7
    assert len(split_dict["val"]) == 2
    assert len(split_dict["test"]) == 1


def test_yolo_exporter(tmp_path, mock_project_env):
    frames, annos, classes, frames_dir = mock_project_env
    out_dir = tmp_path / "dataset_out"

    split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    exporter = YOLOExporter(output_dir=out_dir, classes=classes, frames_dir=frames_dir)

    res = exporter.export_dataset(
        split_dict=split_dict,
        annotations_by_frame=annos,
        export_masks=True,
        export_previews=True,
        create_zip=True,
    )

    assert res["status"] == "success"
    assert (out_dir / "data.yaml").exists()
    assert (out_dir / "images" / "train").exists()
    assert (out_dir / "labels" / "train").exists()

    # Check data.yaml contents
    with open(out_dir / "data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    assert 0 in data_cfg["names"]
    assert data_cfg["names"][0] == "scooter"

    # Check a label file format
    label_file = out_dir / "labels" / "train" / "frame_0001.txt"
    assert label_file.exists()
    with open(label_file, "r") as f:
        content = f.read().strip()
    assert content.startswith("0 ")
    # Coordinates should be normalized [0, 1]
    tokens = content.split()
    assert len(tokens) == 9 # 1 class_id + 4 pairs of coords
    for v in tokens[1:]:
        assert 0.0 <= float(v) <= 1.0


def test_dataset_export_worker(tmp_path, mock_project_env):
    from PySide6.QtWidgets import QApplication
    from sam2_annotator.ui.main_window import DatasetExportWorker

    # Ensure QApplication exists for signal-slot processing
    app = QApplication.instance() or QApplication([])

    frames, annos, classes, frames_dir = mock_project_env
    out_dir = tmp_path / "worker_dataset_out"

    split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
    exporter = YOLOExporter(output_dir=out_dir, classes=classes, frames_dir=frames_dir)

    worker = DatasetExportWorker(
        exporter=exporter,
        split_dict=split_dict,
        annotations_by_frame=annos,
        export_masks=False,
        export_previews=False,
        create_zip=False,
        include_null_frames=True,
    )

    progress_events = []
    worker.progress.connect(lambda cur, tot, msg: progress_events.append((cur, tot, msg)))

    finished_result = []
    worker.finished.connect(lambda res: finished_result.append(res))

    worker.start()
    assert worker.wait(5000), "Worker did not finish in time"
    app.processEvents()

    assert len(progress_events) > 0
    assert len(finished_result) == 1
    assert finished_result[0]["status"] == "success"
    assert (out_dir / "data.yaml").exists()
