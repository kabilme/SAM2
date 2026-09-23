"""Settings and configuration dialog."""

from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QComboBox, QDoubleSpinBox,
    QSpinBox, QGroupBox
)

from sam3_annotator.config.config import AppConfig


class SettingsDialog(QDialog):
    """Application preferences and configuration dialog."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings - SAM3 Annotator")
        self.setMinimumSize(480, 420)
        self.config = config

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Model Settings
        model_group = QGroupBox("SAM 3 Model Settings")
        m_form = QFormLayout(model_group)

        # Checkpoint path
        ckpt_layout = QHBoxLayout()
        self.ckpt_edit = QLineEdit(self.config.model.checkpoint_path)
        self.ckpt_edit.setPlaceholderText("e.g. sam2.1_t.pt or custom SAM 3 weights...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_ckpt)
        ckpt_layout.addWidget(self.ckpt_edit)
        ckpt_layout.addWidget(browse_btn)
        m_form.addRow("Checkpoint (.pt):", ckpt_layout)

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(self.config.model.device)
        m_form.addRow("Compute Device:", self.device_combo)

        # Precision
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp32", "fp16", "bf16"])
        self.precision_combo.setCurrentText(self.config.model.precision)
        m_form.addRow("Precision:", self.precision_combo)

        layout.addWidget(model_group)

        # 2. Polygon & Geometry
        poly_group = QGroupBox("Polygon & Simplification")
        p_form = QFormLayout(poly_group)

        self.simplify_spin = QDoubleSpinBox()
        self.simplify_spin.setRange(0.0005, 0.05)
        self.simplify_spin.setSingleStep(0.001)
        self.simplify_spin.setDecimals(4)
        self.simplify_spin.setValue(self.config.polygon.simplify_tolerance)
        p_form.addRow("Simplification Tolerance:", self.simplify_spin)

        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(1.0, 1000.0)
        self.min_area_spin.setValue(self.config.polygon.min_area)
        self.min_area_spin.setSuffix(" px")
        p_form.addRow("Minimum Area:", self.min_area_spin)

        layout.addWidget(poly_group)

        # 3. UI & Autosave
        ui_group = QGroupBox("Interface & Autosave")
        u_form = QFormLayout(ui_group)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.1, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(self.config.ui.mask_opacity)
        u_form.addRow("Mask Opacity:", self.opacity_spin)

        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(5, 600)
        self.autosave_spin.setValue(self.config.ui.autosave_interval_seconds)
        self.autosave_spin.setSuffix(" sec")
        u_form.addRow("Autosave Interval:", self.autosave_spin)

        layout.addWidget(ui_group)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #444455;")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _browse_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "", "PyTorch Models (*.pt *.pth);;All Files (*.*)"
        )
        if path:
            self.ckpt_edit.setText(path)

    def _on_save(self) -> None:
        self.config.model.checkpoint_path = self.ckpt_edit.text().strip()
        self.config.model.device = self.device_combo.currentText()
        self.config.model.precision = self.precision_combo.currentText()
        self.config.polygon.simplify_tolerance = self.simplify_spin.value()
        self.config.polygon.min_area = self.min_area_spin.value()
        self.config.ui.mask_opacity = self.opacity_spin.value()
        self.config.ui.autosave_interval_seconds = self.autosave_spin.value()
        self.accept()
