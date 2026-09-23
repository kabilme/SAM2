"""Dataset export configuration dialog."""

from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QSpinBox, QCheckBox,
    QComboBox, QGroupBox, QMessageBox
)


class ExportDialog(QDialog):
    """Dialog configuring YOLOv8 instance segmentation dataset export."""

    def __init__(self, default_output_dir: Path, total_frames: int, classes_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export YOLOv8 Dataset")
        self.setMinimumSize(480, 460)

        self.default_dir = Path(default_output_dir)
        self.total_frames = total_frames
        self.classes_count = classes_count

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Output Folder
        form = QFormLayout()
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit(str(self.default_dir))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        form.addRow("<b>Output Folder:</b>", dir_layout)
        layout.addLayout(form)

        # 2. Split Ratios Group
        split_group = QGroupBox("Train / Validation / Test Split")
        split_form = QFormLayout(split_group)

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

        layout.addWidget(split_group)

        # 3. Export Components Group
        opts_group = QGroupBox("Export Artifacts")
        opts_layout = QVBoxLayout(opts_group)

        self.chk_labels = QCheckBox("Export YOLOv8 Polygons & data.yaml")
        self.chk_labels.setChecked(True)
        self.chk_labels.setEnabled(False) # Required
        opts_layout.addWidget(self.chk_labels)

        self.chk_masks = QCheckBox("Export Binary Masks (.png)")
        self.chk_masks.setChecked(True)
        opts_layout.addWidget(self.chk_masks)

        self.chk_previews = QCheckBox("Export Colored Preview Images (.jpg)")
        self.chk_previews.setChecked(True)
        opts_layout.addWidget(self.chk_previews)

        self.chk_zip = QCheckBox("Create ZIP Archive (dataset.zip)")
        self.chk_zip.setChecked(True)
        opts_layout.addWidget(self.chk_zip)

        layout.addWidget(opts_group)

        # Summary Info
        summary = QLabel(f"Total Frames: {self.total_frames} | Defined Classes: {self.classes_count}")
        summary.setStyleSheet("color: #3d5afe; font-weight: bold;")
        layout.addWidget(summary)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #444455;")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Start Export")
        export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if path:
            self.dir_edit.setText(path)

    def _on_export(self) -> None:
        total_pct = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if total_pct == 0:
            QMessageBox.warning(self, "Ratio Error", "Split ratios must sum to greater than 0.")
            return
        self.accept()

    def get_export_params(self) -> Dict[str, Any]:
        strat_map = {0: "sequential", 1: "grouped", 2: "random"}
        return {
            "output_dir": Path(self.dir_edit.text().strip()),
            "train_ratio": self.train_spin.value() / 100.0,
            "val_ratio": self.val_spin.value() / 100.0,
            "test_ratio": self.test_spin.value() / 100.0,
            "split_strategy": strat_map.get(self.strategy_combo.currentIndex(), "sequential"),
            "export_masks": self.chk_masks.isChecked(),
            "export_previews": self.chk_previews.isChecked(),
            "create_zip": self.chk_zip.isChecked(),
        }
