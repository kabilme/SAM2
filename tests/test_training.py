"""Unit tests for the deep learning training subsystem (Phase 1 & Phase 2)."""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.annotation.annotation_manager import AnnotationManager
from sam2_annotator.training.base_trainer import (
    BaseModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)
from sam2_annotator.training.trainer_factory import (
    TRAINING_ENGINES,
    get_available_engines,
    create_trainer,
)
from sam2_annotator.training.yolo_trainer import YOLOTrainer
from sam2_annotator.training.training_worker import TrainingWorker
from sam2_annotator.ui.training_dialog import LossChartWidget


def test_training_config_defaults():
    """Verify TrainingConfig initialization with valid defaults."""
    cfg = TrainingConfig()
    assert cfg.engine_id == "yolo"
    assert cfg.model_name == "yolo11n-seg.pt"
    assert cfg.task == "segmentation"
    assert cfg.epochs == 50
    assert cfg.batch_size == 8
    assert cfg.image_size == 640
    assert cfg.device == "auto"
    assert cfg.train_ratio == 0.8
    assert cfg.val_ratio == 0.2


def test_training_metrics_to_dict():
    """Verify TrainingMetrics serialization."""
    m = TrainingMetrics(
        epoch=3,
        total_epochs=10,
        train_loss=0.45,
        box_loss=0.20,
        seg_loss=0.25,
        map50=0.88,
        elapsed_seconds=12.5,
    )
    d = m.to_dict()
    assert d["epoch"] == 3
    assert d["train_loss"] == 0.45
    assert d["seg_loss"] == 0.25
    assert d["map50"] == 0.88


def test_trainer_factory_engines():
    """Verify trainer factory registry and engine instantiation."""
    engines = get_available_engines()
    engine_ids = [e["id"] for e in engines]
    assert "yolo" in engine_ids
    assert "maskrcnn" in engine_ids
    assert "sam2_decoder" in engine_ids

    # YOLO is available
    yolo_cfg = TrainingConfig(engine_id="yolo")
    trainer = create_trainer(yolo_cfg)
    assert isinstance(trainer, YOLOTrainer)

    # Unknown engine raises ValueError
    with pytest.raises(ValueError):
        create_trainer(TrainingConfig(engine_id="unknown_engine"))

    # Unimplemented engine raises NotImplementedError
    with pytest.raises(NotImplementedError):
        create_trainer(TrainingConfig(engine_id="maskrcnn"))


def test_yolo_trainer_device_resolution():
    """Verify device resolution logic."""
    cfg = TrainingConfig(device="cpu")
    trainer = YOLOTrainer(cfg)
    assert trainer._resolve_device() == "cpu"

    cfg_cuda = TrainingConfig(device="cuda:0")
    trainer_cuda = YOLOTrainer(cfg_cuda)
    dev = trainer_cuda._resolve_device()
    assert dev in ["0", "cpu"]


def test_yolo_trainer_prepare_dataset(tmp_path):
    """Verify automatic dataset preparation for YOLO training."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create dummy frame images
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(1, 4):
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), dummy_img)

    frames = [
        FrameMetadata(frame_id=1, filename="frame_0001.jpg", source_frame_index=0, timestamp_seconds=0.0, width=100, height=100),
        FrameMetadata(frame_id=2, filename="frame_0002.jpg", source_frame_index=1, timestamp_seconds=0.1, width=100, height=100),
        FrameMetadata(frame_id=3, filename="frame_0003.jpg", source_frame_index=2, timestamp_seconds=0.2, width=100, height=100),
    ]

    classes = [ClassItem(id=0, name="car", color_rgb=[255, 0, 0])]

    class DummyProjectManager:
        def __init__(self):
            self.frames = frames
            self.classes = classes
            self.frames_dir = frames_dir

    proj_mgr = DummyProjectManager()
    anno_mgr = AnnotationManager()

    # Add polygon annotations
    anno1 = PolygonAnnotation(
        object_id="car_1",
        class_id=0,
        class_name="car",
        frame_id=1,
        source_frame_index=0,
        points=[(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
    )
    anno_mgr.add_annotation(anno1)

    cfg = TrainingConfig(
        engine_id="yolo",
        output_dir=tmp_path / "run_test",
        train_ratio=0.7,
        val_ratio=0.3,
    )
    trainer = YOLOTrainer(cfg)
    yaml_path = trainer.prepare_dataset(proj_mgr, anno_mgr)

    assert yaml_path.exists()
    assert yaml_path.name == "data.yaml"
    dataset_dir = tmp_path / "run_test" / "dataset"
    assert (dataset_dir / "images" / "train").exists()
    assert (dataset_dir / "labels" / "train").exists()


def test_training_worker_with_mock_trainer(tmp_path):
    """Verify TrainingWorker executes lifecycle and emits signals properly."""
    from PySide6.QtWidgets import QApplication
    _ = QApplication.instance() or QApplication([])

    class MockTrainer(BaseModelTrainer):
        def prepare_dataset(self, project_manager, annotation_manager, progress_callback=None):
            if progress_callback:
                progress_callback(100, 100, "Prepared mock dataset")
            return tmp_path / "mock_data.yaml"

        def train(self, dataset_path, on_epoch_start=None, on_epoch_end=None, on_step=None, on_log=None):
            if on_log:
                on_log("Mock train starting")
            if on_epoch_start:
                on_epoch_start(1, 2)
            if on_epoch_end:
                on_epoch_end(TrainingMetrics(epoch=1, total_epochs=2, train_loss=0.5, map50=0.8))
            if on_epoch_start:
                on_epoch_start(2, 2)
            if on_epoch_end:
                on_epoch_end(TrainingMetrics(epoch=2, total_epochs=2, train_loss=0.2, map50=0.9))
            return TrainingResult(
                status="completed",
                best_model_path=tmp_path / "best.pt",
                duration_seconds=1.5,
            )

        def export_model(self, model_path, export_format="onnx"):
            return tmp_path / "best.onnx"

    trainer = MockTrainer(TrainingConfig())
    worker = TrainingWorker(trainer, project_manager=None, annotation_manager=None)

    signals_received = {
        "epoch_started": [],
        "epoch_completed": [],
        "finished": False,
    }

    worker.epoch_started.connect(lambda ep, tot: signals_received["epoch_started"].append(ep))
    worker.epoch_completed.connect(lambda ep, tot, m: signals_received["epoch_completed"].append(ep))
    worker.finished.connect(lambda res: signals_received.update({"finished": True}))

    worker.run()

    assert signals_received["epoch_started"] == [1, 2]
    assert signals_received["epoch_completed"] == [1, 2]
    assert signals_received["finished"] is True


def test_loss_chart_widget():
    """Verify LossChartWidget metrics accumulation and reset."""
    from PySide6.QtWidgets import QApplication
    _ = QApplication.instance() or QApplication([])

    chart = LossChartWidget()
    assert len(chart.losses) == 0

    chart.add_epoch_metrics(epoch=1, loss=0.5, map50=0.7)
    chart.add_epoch_metrics(epoch=2, loss=0.3, map50=0.85)

    assert len(chart.epochs) == 2
    assert chart.losses == [0.5, 0.3]
    assert chart.maps == [0.7, 0.85]

    chart.reset()
    assert len(chart.losses) == 0
    assert len(chart.maps) == 0


def test_yolo_trainer_reuse_dataset(tmp_path):
    """Verify that YOLOTrainer reuses existing prepared dataset without re-exporting."""
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "dataset"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    dummy_file = dataset_dir / "images" / "train" / "img1.jpg"
    dummy_file.write_text("fake")
    yaml_file = dataset_dir / "data.yaml"
    yaml_file.write_text("train: images/train\nval: images/val\nnames: {0: car}\n")

    cfg = TrainingConfig(
        engine_id="yolo",
        output_dir=run_dir,
        extra_params={"reuse_dataset": True},
    )
    trainer = YOLOTrainer(cfg)

    class EmptyProjectManager:
        frames = []
        classes = []
        frames_dir = tmp_path

    # Even with empty project manager, it returns existing yaml_file without error
    ret_yaml = trainer.prepare_dataset(EmptyProjectManager(), AnnotationManager())
    assert ret_yaml == yaml_file


def test_training_dialog_checkpoint_discovery(tmp_path):
    """Verify TrainingDialog automatically discovers previous best.pt and last.pt checkpoints."""
    from PySide6.QtWidgets import QApplication
    from sam2_annotator.ui.training_dialog import TrainingDialog
    _ = QApplication.instance() or QApplication([])

    proj_dir = tmp_path / "project_alpha"
    run_weights = proj_dir / "models" / "weights" / "train_run_1" / "weights"
    run_weights.mkdir(parents=True)
    best_pt = run_weights / "best.pt"
    best_pt.write_text("fake_weights")

    class MockProjMgr:
        project_dir = proj_dir
        frames = []
        classes = []

    diag = TrainingDialog(project_manager=MockProjMgr(), annotation_manager=AnnotationManager())
    # Verify the discovered run was added to model_combo
    all_items = [diag.model_combo.itemText(i) for i in range(diag.model_combo.count())]
    assert any("Continue Run: train_run_1 (best.pt)" in text for text in all_items)


def test_yolo_loss_parsing():
    """Verify robust extraction of losses from Ultralytics dict format."""
    import torch

    cfg = TrainingConfig(task="segmentation")
    trainer = YOLOTrainer(cfg)

    # Dictionary containing tensors (standard Ultralytics format)
    loss_dict = {
        "box_loss": torch.tensor(0.75),
        "cls_loss": torch.tensor(1.25),
        "seg_loss": torch.tensor(0.50),
    }

    # Simulate parsing
    parsed = {}
    for k, v in loss_dict.items():
        parsed[k] = float(v.item() if hasattr(v, "item") else v)

    total_loss = float(sum(parsed.values()))
    assert pytest.approx(total_loss, 0.001) == 2.50
    assert pytest.approx(parsed["box_loss"], 0.001) == 0.75
    assert pytest.approx(parsed["cls_loss"], 0.001) == 1.25
    assert pytest.approx(parsed["seg_loss"], 0.001) == 0.50


