"""Startup diagnostic panel displaying hardware and library detection."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox
)

from sam2_annotator.utils.device_utils import get_system_diagnostics, SystemDiagnostics


class StartupDialog(QDialog):
    """Diagnostic panel shown on launch or via Help menu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Diagnostics - SAM2 Annotator")
        self.setFixedSize(480, 420)
        self.diagnostics: SystemDiagnostics = get_system_diagnostics()
        self.selected_device: str = self.diagnostics.recommended_device

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("<h2>Hardware & Environment Diagnostics</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Specs Group
        group = QGroupBox("Detected Hardware & Libraries")
        form = QFormLayout(group)
        form.setSpacing(8)

        form.addRow("<b>Python:</b>", QLabel(self.diagnostics.python_version))
        form.addRow("<b>PyTorch:</b>", QLabel(self.diagnostics.pytorch_version))
        cuda_status = "Available (CUDA)" if self.diagnostics.cuda_available else "Not Available (CPU mode)"
        form.addRow("<b>CUDA:</b>", QLabel(cuda_status))
        form.addRow("<b>GPU Controller:</b>", QLabel(self.diagnostics.gpu_name))
        if self.diagnostics.gpu_vram_gb:
            form.addRow("<b>GPU VRAM:</b>", QLabel(f"{self.diagnostics.gpu_vram_gb:.2f} GB"))
        form.addRow("<b>CPU:</b>", QLabel(self.diagnostics.cpu_info))
        form.addRow("<b>System RAM:</b>", QLabel(f"{self.diagnostics.available_ram_gb:.1f} GB available / {self.diagnostics.total_ram_gb:.1f} GB total"))
        layout.addWidget(group)

        # Device Selection
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("<b>Compute Device:</b>"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        if self.diagnostics.cuda_available:
            self.device_combo.addItem("CUDA (NVIDIA GPU)", "cuda")
            self.device_combo.setCurrentIndex(1)
        else:
            self.device_combo.setCurrentIndex(0)
        dev_layout.addWidget(self.device_combo)
        layout.addLayout(dev_layout)

        # Info note
        note = QLabel("<i>Note: CPU execution is fully supported. SAM inference uses optimized torch threads.</i>")
        note.setStyleSheet("color: #888888; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addStretch()

        # OK button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("Continue to Workspace")
        ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def _on_ok(self) -> None:
        self.selected_device = self.device_combo.currentData()
        self.accept()
