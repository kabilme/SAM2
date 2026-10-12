"""Modern training dialog with live telemetry, dynamic loss chart, and background worker."""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QWidget,
    QFrame,
    QSplitter,
    QCheckBox,
)

from sam2_annotator.training.base_trainer import TrainingConfig, TrainingResult
from sam2_annotator.training.trainer_factory import (
    TRAINING_ENGINES,
    get_available_engines,
    create_trainer,
)
from sam2_annotator.training.training_worker import TrainingWorker
from sam2_annotator.utils.logging_utils import logger


class LossChartWidget(QWidget):
    """Custom high-performance QPainter chart displaying live loss and mAP curves."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.losses: List[float] = []
        self.maps: List[float] = []
        self.epochs: List[int] = []

    def reset(self) -> None:
        self.losses.clear()
        self.maps.clear()
        self.epochs.clear()
        self.update()

    def add_epoch_metrics(self, epoch: int, loss: float, map50: float) -> None:
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.maps.append(map50)
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # Background
        painter.fillRect(rect, QColor(26, 26, 32))

        # Margin
        pad_l, pad_r, pad_t, pad_b = 40, 40, 25, 25
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        if plot_w <= 10 or plot_h <= 10:
            return

        # Grid guidelines
        grid_pen = QPen(QColor(50, 50, 62), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        for i in range(5):
            y = pad_t + int(plot_h * (i / 4.0))
            painter.drawLine(pad_l, y, pad_l + plot_w, y)

        # Labels & legend
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.setPen(QColor(160, 160, 180))
        painter.drawText(pad_l, pad_t - 8, "Training Loss (Orange) vs. mAP50 (Green)")

        if not self.losses:
            painter.setPen(QColor(100, 100, 120))
            painter.drawText(rect, Qt.AlignCenter, "Live loss & mAP curves will appear here as epochs complete...")
            return

        max_loss = max(1.0, max(self.losses) * 1.1)
        num_pts = len(self.losses)

        # Plot Loss Curve (Orange)
        loss_pen = QPen(QColor(230, 126, 34), 2)
        painter.setPen(loss_pen)
        prev_pt: Optional[QPointF] = None
        for i, val in enumerate(self.losses):
            x = pad_l + (plot_w * (i / max(1, num_pts - 1)))
            y = pad_t + plot_h - (plot_h * (min(max_loss, max(0.0, val)) / max_loss))
            pt = QPointF(x, y)
            if prev_pt is not None:
                painter.drawLine(prev_pt, pt)
            prev_pt = pt

        # Plot mAP Curve (Emerald Green)
        map_pen = QPen(QColor(46, 204, 113), 2)
        painter.setPen(map_pen)
        prev_map_pt: Optional[QPointF] = None
        for i, val in enumerate(self.maps):
            x = pad_l + (plot_w * (i / max(1, num_pts - 1)))
            # mAP is in [0.0, 1.0]
            y = pad_t + plot_h - (plot_h * min(1.0, max(0.0, val)))
            pt = QPointF(x, y)
            if prev_map_pt is not None:
                painter.drawLine(prev_map_pt, pt)
            prev_map_pt = pt

        # Min / Max labels
        painter.setPen(QColor(230, 126, 34))
        painter.drawText(5, pad_t + 10, f"{max_loss:.2f}")
        painter.drawText(5, pad_t + plot_h, "0.0")

        painter.setPen(QColor(46, 204, 113))
        painter.drawText(pad_l + plot_w + 5, pad_t + 10, "1.0")
        painter.drawText(pad_l + plot_w + 5, pad_t + plot_h, "0.0")


class TrainingDialog(QDialog):
    """Integrated training studio dialog supporting multiple deep learning engines."""

    def __init__(
        self,
        project_manager: Any,
        annotation_manager: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.project_manager = project_manager
        self.annotation_manager = annotation_manager
        self.worker: Optional[TrainingWorker] = None
        self.last_result: Optional[TrainingResult] = None

        self.setWindowTitle("Train Vision Model — SAM 2 Annotation Studio")
        self.resize(880, 750)
        self.setMinimumSize(800, 680)

        self._init_ui()
        self._populate_devices()
        self._on_engine_changed()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # ---------------- 1. Configuration Section ----------------
        config_group = QGroupBox("Model & Training Configuration")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)

        # Engine selector
        config_layout.addWidget(QLabel("Training Engine:"), 0, 0)
        self.engine_combo = QComboBox()
        for eng in get_available_engines():
            self.engine_combo.addItem(eng["name"], eng["id"])
            if not eng["is_available"]:
                idx = self.engine_combo.count() - 1
                self.engine_combo.model().setData(
                    self.engine_combo.model().index(idx, 0),
                    False,
                    Qt.UserRole - 1,
                )
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        config_layout.addWidget(self.engine_combo, 0, 1, 1, 3)

        # Task selector
        config_layout.addWidget(QLabel("Task:"), 1, 0)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Instance Segmentation", "Object Detection"])
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        config_layout.addWidget(self.task_combo, 1, 1)

        # Pretrained Model / Checkpoint Selector
        config_layout.addWidget(QLabel("Base Checkpoint:"), 1, 2)
        model_layout = QHBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.model_combo = QComboBox()
        self.browse_ckpt_btn = QPushButton("Browse...")
        self.browse_ckpt_btn.setToolTip("Select custom trained checkpoint (*.pt) to continue/resume training")
        self.browse_ckpt_btn.clicked.connect(self._on_browse_checkpoint)
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.browse_ckpt_btn)
        config_layout.addLayout(model_layout, 1, 3)

        # Hyperparameters Row 1: Epochs & Batch Size
        config_layout.addWidget(QLabel("Epochs:"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        config_layout.addWidget(self.epochs_spin, 2, 1)

        config_layout.addWidget(QLabel("Batch Size:"), 2, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(8)
        config_layout.addWidget(self.batch_spin, 2, 3)

        # Hyperparameters Row 2: Image Size & Device
        config_layout.addWidget(QLabel("Image Size:"), 3, 0)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["640", "1024", "320", "512"])
        config_layout.addWidget(self.imgsz_combo, 3, 1)

        config_layout.addWidget(QLabel("Device:"), 3, 2)
        self.device_combo = QComboBox()
        config_layout.addWidget(self.device_combo, 3, 3)

        # Hyperparameters Row 3: Split Strategy & Train/Val Split
        config_layout.addWidget(QLabel("Split Strategy:"), 4, 0)
        self.split_combo = QComboBox()
        self.split_combo.addItems(["Sequential (Prevents Video Leakage)", "Random Split", "Grouped"])
        config_layout.addWidget(self.split_combo, 4, 1)

        config_layout.addWidget(QLabel("Train / Val Ratio:"), 4, 2)
        self.split_ratio_combo = QComboBox()
        self.split_ratio_combo.addItems(["80% Train / 20% Val", "70% Train / 30% Val", "90% Train / 10% Val"])
        config_layout.addWidget(self.split_ratio_combo, 4, 3)

        # Hyperparameters Row 4: Output Directory
        config_layout.addWidget(QLabel("Output Directory:"), 5, 0)
        out_layout = QHBoxLayout()
        default_out = Path(self.project_manager.project_dir) / "models" if self.project_manager.project_dir else Path("runs/train")
        self.output_dir_edit = QLineEdit(str(default_out))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_output_dir)
        out_layout.addWidget(self.output_dir_edit)
        out_layout.addWidget(browse_btn)
        config_layout.addLayout(out_layout, 5, 1, 1, 3)

        # Row 5: Dataset Options
        self.reuse_dataset_cb = QCheckBox("Reuse existing prepared dataset (skip re-export if data.yaml exists)")
        self.reuse_dataset_cb.setChecked(True)
        self.reuse_dataset_cb.setToolTip("Speeds up training startup by reusing already partitioned and exported images/labels")
        config_layout.addWidget(self.reuse_dataset_cb, 6, 1, 1, 3)

        main_layout.addWidget(config_group)

        # ---------------- 2. Live Telemetry & Dashboard ----------------
        telemetry_group = QGroupBox("Training Telemetry & Progress")
        telemetry_layout = QVBoxLayout(telemetry_group)
        telemetry_layout.setSpacing(8)

        # Progress bar & Status
        self.status_label = QLabel("Ready to train. Configure parameters and click 'Start Training'.")
        self.status_label.setStyleSheet("font-weight: bold; color: #a0a0c0;")
        telemetry_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        telemetry_layout.addWidget(self.progress_bar)

        # Metric Badges
        badge_layout = QHBoxLayout()
        badge_layout.setSpacing(10)
        self.badge_epoch = self._create_metric_card("Epoch", "0 / 50")
        self.badge_loss = self._create_metric_card("Train Loss", "—")
        self.badge_map = self._create_metric_card("Val mAP50", "—")
        self.badge_time = self._create_metric_card("Elapsed / ETA", "00:00 / —")
        badge_layout.addWidget(self.badge_epoch)
        badge_layout.addWidget(self.badge_loss)
        badge_layout.addWidget(self.badge_map)
        badge_layout.addWidget(self.badge_time)
        telemetry_layout.addLayout(badge_layout)

        # Dynamic Loss Chart
        self.chart_widget = LossChartWidget()
        telemetry_layout.addWidget(self.chart_widget)

        # Console Output Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet(
            "background-color: #15151a; color: #d0d0e0; font-family: 'Consolas', monospace; font-size: 11px;"
        )
        telemetry_layout.addWidget(self.log_text)

        main_layout.addWidget(telemetry_group)

        # ---------------- 3. Bottom Controls ----------------
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.start_btn = QPushButton("🚀 Start Training")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px 16px;")
        self.start_btn.clicked.connect(self._on_start_training)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop Training")
        self.stop_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 8px 16px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_training)
        btn_layout.addWidget(self.stop_btn)

        btn_layout.addStretch()

        self.export_onnx_btn = QPushButton("📦 Export to ONNX...")
        self.export_onnx_btn.setStyleSheet("background-color: #3d5afe; color: white; padding: 8px 14px;")
        self.export_onnx_btn.setEnabled(False)
        self.export_onnx_btn.clicked.connect(self._on_export_onnx)
        btn_layout.addWidget(self.export_onnx_btn)

        self.open_folder_btn = QPushButton("📁 Open Output Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._on_open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        main_layout.addLayout(btn_layout)

    def _create_metric_card(self, title: str, initial_val: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #282832; border-radius: 6px; padding: 4px;")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(2)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #9090a0; font-size: 11px;")
        lbl_val = QLabel(initial_val)
        lbl_val.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        lbl_val.setObjectName("val")
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        return frame

    def _update_card_val(self, frame: QFrame, text: str) -> None:
        lbl = frame.findChild(QLabel, "val")
        if lbl:
            lbl.setText(text)

    def _populate_devices(self) -> None:
        self.device_combo.clear()
        self.device_combo.addItem("Auto (Recommended)", "auto")
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                self.device_combo.addItem(f"CUDA:{i} ({name})", f"cuda:{i}")
        self.device_combo.addItem("CPU Only", "cpu")

    def _on_engine_changed(self) -> None:
        eng_id = self.engine_combo.currentData()
        eng_info = TRAINING_ENGINES.get(eng_id, {})
        tasks = eng_info.get("tasks", ["segmentation"])

        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        if "segmentation" in tasks:
            self.task_combo.addItem("Instance Segmentation", "segmentation")
        if "detection" in tasks:
            self.task_combo.addItem("Object Detection", "detection")
        self.task_combo.blockSignals(False)

        self._on_task_changed()

    def _on_task_changed(self) -> None:
        eng_id = self.engine_combo.currentData()
        eng_info = TRAINING_ENGINES.get(eng_id, {})
        task = self.task_combo.currentData() or "segmentation"

        models_dict = eng_info.get("default_models", {})
        models_list = models_dict.get(task, [])

        self.model_combo.clear()
        for m in models_list:
            self.model_combo.addItem(m["name"], m["id"])

        # Auto-discover previous checkpoints in the project's models/weights directory
        if self.project_manager.project_dir:
            weights_root = Path(self.project_manager.project_dir) / "models" / "weights"
            if weights_root.exists():
                runs = sorted(
                    [d for d in weights_root.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True,
                )
                for run_dir in runs:
                    w_dir = run_dir / "weights"
                    if w_dir.exists():
                        resumable_pt = w_dir / "resumable_last.pt"
                        best_pt = w_dir / "best.pt"
                        last_pt = w_dir / "last.pt"
                        target_pt = resumable_pt if resumable_pt.exists() else (best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None))
                        if target_pt:
                            self.model_combo.addItem(f"Continue Run: {run_dir.name} ({target_pt.name})", str(target_pt))

    def _on_browse_checkpoint(self) -> None:
        init_dir = str(Path(self.project_manager.project_dir) / "models" / "weights") if self.project_manager.project_dir else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint Weights to Continue / Resume Training",
            init_dir,
            "PyTorch Weights (*.pt *.pth);;All Files (*)",
        )
        if file_path:
            p = Path(file_path)
            label = f"Custom: {p.name} ({p.parent.parent.name})"
            self.model_combo.addItem(label, str(p))
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def _on_browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir_edit.text())
        if d:
            self.output_dir_edit.setText(d)

    def _on_start_training(self) -> None:
        # Validation checks
        if not self.project_manager.frames:
            QMessageBox.warning(self, "No Video Frames", "There are no video frames loaded in the current project.")
            return

        annotated_count = sum(
            1 for f in self.project_manager.frames if len(self.annotation_manager.get_annotations_for_frame(f.frame_id)) > 0
        )
        if annotated_count == 0:
            QMessageBox.warning(
                self,
                "No Annotations",
                "There are no annotated frames in this project. Please annotate at least a few keyframes before training."
            )
            return

        eng_id = self.engine_combo.currentData()
        eng_info = TRAINING_ENGINES.get(eng_id, {})
        if not eng_info.get("is_available"):
            QMessageBox.information(
                self,
                "Engine Not Available",
                f"{eng_info.get('name')} is currently scheduled for upcoming phases. Please select Ultralytics YOLO."
            )
            return

        # Build config
        ratio_text = self.split_ratio_combo.currentText()
        if "90%" in ratio_text:
            t_ratio, v_ratio = 0.9, 0.1
        elif "70%" in ratio_text:
            t_ratio, v_ratio = 0.7, 0.3
        else:
            t_ratio, v_ratio = 0.8, 0.2

        strategy_idx = self.split_combo.currentIndex()
        strategy = "sequential" if strategy_idx == 0 else ("random" if strategy_idx == 1 else "grouped")

        config = TrainingConfig(
            engine_id=eng_id,
            model_name=self.model_combo.currentData() or "yolo11n-seg.pt",
            task=self.task_combo.currentData() or "segmentation",
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            image_size=int(self.imgsz_combo.currentText()),
            device=self.device_combo.currentData() or "auto",
            train_ratio=t_ratio,
            val_ratio=v_ratio,
            split_strategy=strategy,
            output_dir=Path(self.output_dir_edit.text()),
            project_name=f"train_{int(time.time())}",
            extra_params={"reuse_dataset": self.reuse_dataset_cb.isChecked()},
        )

        try:
            trainer = create_trainer(config)
        except Exception as e:
            QMessageBox.critical(self, "Trainer Initialization Error", str(e))
            return

        # Prepare UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_onnx_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.chart_widget.reset()
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Preparing dataset and starting {eng_info['name']}...")
        self._update_card_val(self.badge_loss, "—")
        self._update_card_val(self.badge_map, "—")
        self._update_card_val(self.badge_time, "00:00 / —")

        # Launch worker thread
        self.worker = TrainingWorker(
            trainer=trainer,
            project_manager=self.project_manager,
            annotation_manager=self.annotation_manager,
        )

        self.worker.dataset_prep_progress.connect(self._on_prep_progress)
        self.worker.epoch_started.connect(self._on_epoch_started)
        self.worker.epoch_completed.connect(self._on_epoch_completed)
        self.worker.step_progress.connect(self._on_step_progress)
        self.worker.log_message.connect(self._on_log_message)
        self.worker.finished.connect(self._on_training_finished)
        self.worker.error.connect(self._on_training_error)

        self.worker.start()

    def _on_stop_training(self) -> None:
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Stopping training session, saving latest checkpoint...")
            self.stop_btn.setEnabled(False)
            self.worker.cancel()

    def _on_prep_progress(self, cur: int, tot: int, msg: str) -> None:
        self.status_label.setText(f"Dataset Prep: {msg}")
        self.progress_bar.setValue(cur)

    def _on_epoch_started(self, epoch: int, total_epochs: int) -> None:
        pct = int((epoch / max(1, total_epochs)) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"Training Epoch {epoch} of {total_epochs} ({pct}%)")
        self._update_card_val(self.badge_epoch, f"{epoch} / {total_epochs}")

    def _on_step_progress(self, step: int, total_steps: int, losses: Dict[str, float]) -> None:
        if "train_loss" in losses:
            self._update_card_val(self.badge_loss, f"{losses['train_loss']:.4f}")
        elif losses:
            tot = sum(losses.values())
            self._update_card_val(self.badge_loss, f"{tot:.4f}")

    def _on_epoch_completed(self, epoch: int, total_epochs: int, metrics: Dict[str, Any]) -> None:
        loss = float(metrics.get("train_loss", 0.0))
        if loss == 0.0:
            loss = float(metrics.get("box_loss", 0.0)) + float(metrics.get("cls_loss", 0.0)) + float(metrics.get("dfl_loss", 0.0))
        m50 = float(metrics.get("map50", 0.0))
        elapsed = float(metrics.get("elapsed_seconds", 0.0))
        eta = float(metrics.get("eta_seconds", 0.0))

        self.chart_widget.add_epoch_metrics(epoch, loss, m50)
        self._update_card_val(self.badge_loss, f"{loss:.4f}")
        self._update_card_val(self.badge_map, f"{m50 * 100:.1f}%")

        el_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}" if eta > 0 else "00:00"
        self._update_card_val(self.badge_time, f"{el_str} / {eta_str}")

    def _on_log_message(self, msg: str) -> None:
        self.log_text.append(msg)
        # Auto scroll to bottom
        sb = self.log_text.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_training_finished(self, result: TrainingResult) -> None:
        self.last_result = result
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)

        if result.status == "completed":
            self.progress_bar.setValue(100)
            self.status_label.setText(f"✅ Training completed! Best weights saved at: {result.best_model_path.name if result.best_model_path else 'weights/'}")
            self.export_onnx_btn.setEnabled(result.best_model_path is not None)
            self.open_folder_btn.setEnabled(result.output_dir is not None and result.output_dir.exists())
            QMessageBox.information(
                self,
                "Training Complete",
                f"Model training finished successfully!\n\n"
                f"Best Weights: {result.best_model_path}\n"
                f"Duration: {int(result.duration_seconds)}s\n"
                f"Total Epochs: {len(result.metrics_history)}"
            )
        elif result.status == "cancelled":
            self.status_label.setText("Training was stopped by user.")
            self.export_onnx_btn.setEnabled(result.last_model_path is not None)
            self.open_folder_btn.setEnabled(result.output_dir is not None and result.output_dir.exists())
            QMessageBox.information(self, "Training Stopped", "Training stopped. Latest weights were preserved.")
        else:
            self.status_label.setText(f"❌ Training error: {result.error_message}")
            QMessageBox.critical(self, "Training Error", f"An error occurred:\n\n{result.error_message}")

    def _on_training_error(self, err_msg: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.status_label.setText(f"❌ Training Failed: {err_msg}")
        QMessageBox.critical(self, "Training Failed", f"Failed to execute training:\n\n{err_msg}")

    def _on_export_onnx(self) -> None:
        if not self.last_result or not self.last_result.best_model_path:
            QMessageBox.warning(self, "No Model", "No trained model weights are available to export.")
            return

        model_path = self.last_result.best_model_path
        self.status_label.setText(f"Exporting {model_path.name} to ONNX format...")
        self.export_onnx_btn.setEnabled(False)

        try:
            config = TrainingConfig(engine_id=self.engine_combo.currentData())
            trainer = create_trainer(config)
            onnx_path = trainer.export_model(model_path, export_format="onnx")
            self.status_label.setText(f"✅ Exported ONNX model to: {onnx_path.name}")
            QMessageBox.information(
                self,
                "ONNX Export Complete",
                f"Successfully exported model to ONNX format!\n\nFile: {onnx_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export to ONNX: {e}")
        finally:
            self.export_onnx_btn.setEnabled(True)

    def _on_open_output_folder(self) -> None:
        if self.last_result and self.last_result.output_dir and self.last_result.output_dir.exists():
            folder = str(self.last_result.output_dir.resolve())
            if os.name == "nt":
                os.startfile(folder)
            else:
                subprocess.Popen(["xdg-open", folder])
