"""Dataset and artifact export configuration dialog."""

from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QGroupBox, QMessageBox
)

from sam2_annotator.dataset.exporter_factory import get_available_formats, EXPORT_FORMATS


class ExportDialog(QDialog):
    """Dialog configuring dataset and artifact exports with multi-format support."""

    def __init__(
        self,
        default_output_dir: Path,
        total_frames: int,
        classes_count: int,
        annotated_frames_count: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Dataset / Artifacts")
        self.setMinimumSize(560, 580)

        self.default_dir = Path(default_output_dir)
        self.total_frames = total_frames
        self.classes_count = classes_count
        self.annotated_frames_count = annotated_frames_count
        self.null_frames_count = max(0, total_frames - annotated_frames_count)
        self.formats_list = get_available_formats()

        self._setup_ui()
        self._on_format_changed(0)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Format Selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout(format_group)

        self.format_combo = QComboBox()
        for fmt in self.formats_list:
            self.format_combo.addItem(fmt["name"], fmt["id"])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_layout.addWidget(self.format_combo)

        self.format_desc_label = QLabel()
        self.format_desc_label.setWordWrap(True)
        self.format_desc_label.setStyleSheet("color: #9e9e9e; font-size: 11px; margin-top: 2px;")
        format_layout.addWidget(self.format_desc_label)

        layout.addWidget(format_group)

        # 2. Output Folder
        form = QFormLayout()
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit(str(self.default_dir))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        form.addRow("<b>Output Folder:</b>", dir_layout)
        layout.addLayout(form)

        # 3. Split Ratios Group
        self.split_group = QGroupBox("Train / Validation / Test Split")
        split_form = QFormLayout(self.split_group)

        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix(" %")
        split_form.addRow("Train Ratio:", self.train_spin)

        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(20)
        self.val_spin.setSuffix(" %")
        split_form.addRow("Validation Ratio:", self.val_spin)

        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(10)
        self.test_spin.setSuffix(" %")
        split_form.addRow("Test Ratio:", self.test_spin)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Sequential (Minimize Video Leakage)",
            "Grouped Segments",
            "Random Shuffled"
        ])
        split_form.addRow("Split Strategy:", self.strategy_combo)

        layout.addWidget(self.split_group)

        # 4. Video Rendering Options Group
        self.video_opts_group = QGroupBox("Video Rendering Options")
        video_form = QFormLayout(self.video_opts_group)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setSuffix(" fps")
        video_form.addRow("Playback Frame Rate:", self.fps_spin)

        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(45)
        self.opacity_spin.setSuffix(" %")
        video_form.addRow("Polygon Fill Opacity:", self.opacity_spin)

        self.chk_video_bbox = QCheckBox("Draw bounding boxes and class tracking badges")
        self.chk_video_bbox.setChecked(True)
        video_form.addRow("", self.chk_video_bbox)

        layout.addWidget(self.video_opts_group)

        # 5. Export Artifacts Group
        self.opts_group = QGroupBox("Export Artifacts & Options")
        opts_layout = QVBoxLayout(self.opts_group)

        self.chk_null_frames = QCheckBox("Export unannotated frames as null / negative frames")
        self.chk_null_frames.setChecked(True)
        self.chk_null_frames.setToolTip("Generates empty label entries for unannotated frames to reduce model false positives")
        opts_layout.addWidget(self.chk_null_frames)

        self.chk_masks = QCheckBox("Export Binary Masks (.png)")
        self.chk_masks.setChecked(True)
        opts_layout.addWidget(self.chk_masks)

        self.chk_previews = QCheckBox("Export Colored Preview Images (.jpg)")
        self.chk_previews.setChecked(True)
        opts_layout.addWidget(self.chk_previews)

        self.chk_zip = QCheckBox("Create ZIP Archive")
        self.chk_zip.setChecked(True)
        opts_layout.addWidget(self.chk_zip)

        layout.addWidget(self.opts_group)

        # Summary Info
        summary = QLabel(
            f"<b>Total Frames:</b> {self.total_frames}  |  "
            f"<b>Annotated:</b> {self.annotated_frames_count}  |  "
            f"<b>Null / Background:</b> {self.null_frames_count}  |  "
            f"<b>Classes:</b> {self.classes_count}"
        )
        summary.setStyleSheet("color: #3d5afe; font-size: 11px;")
        layout.addWidget(summary)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #444455;")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Start Export")
        export_btn.setStyleSheet("font-weight: bold; background-color: #1976d2;")
        export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)

    def _on_format_changed(self, index: int) -> None:
        """Dynamically update visible controls based on selected export format."""
        if index < 0 or index >= len(self.formats_list):
            return

        fmt = self.formats_list[index]
        fmt_id = fmt["id"]
        self.format_desc_label.setText(fmt["description"])

        supports_splits = fmt.get("supports_splits", False)
        self.split_group.setVisible(supports_splits)

        is_video = (fmt_id == "rendered_video")
        self.video_opts_group.setVisible(is_video)
        self.opts_group.setVisible(not is_video)

        if not is_video:
            self.chk_masks.setVisible(fmt.get("supports_masks", False))
            self.chk_previews.setVisible(fmt.get("supports_previews", False))
            self.chk_zip.setVisible(fmt.get("supports_zip", False))
            self.chk_null_frames.setVisible(fmt.get("is_dataset", True))

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if path:
            self.dir_edit.setText(path)

    def _on_export(self) -> None:
        fmt_id = self.format_combo.currentData()
        info = EXPORT_FORMATS.get(fmt_id, {})
        if info.get("supports_splits", False):
            total_pct = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
            if total_pct == 0:
                QMessageBox.warning(self, "Ratio Error", "Split ratios must sum to greater than 0.")
                return
        self.accept()

    def get_export_params(self) -> Dict[str, Any]:
        strat_map = {0: "sequential", 1: "grouped", 2: "random"}
        fmt_id = self.format_combo.currentData()

        return {
            "format": fmt_id,
            "output_dir": Path(self.dir_edit.text().strip()),
            "train_ratio": self.train_spin.value() / 100.0,
            "val_ratio": self.val_spin.value() / 100.0,
            "test_ratio": self.test_spin.value() / 100.0,
            "split_strategy": strat_map.get(self.strategy_combo.currentIndex(), "sequential"),
            "include_null_frames": self.chk_null_frames.isChecked(),
            "export_masks": self.chk_masks.isChecked(),
            "export_previews": self.chk_previews.isChecked(),
            "create_zip": self.chk_zip.isChecked(),
            # Video specific
            "fps": float(self.fps_spin.value()),
            "alpha": float(self.opacity_spin.value() / 100.0),
            "draw_bbox": self.chk_video_bbox.isChecked(),
        }
