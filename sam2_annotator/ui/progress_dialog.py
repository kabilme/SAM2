"""Modal progress dialog for long-running asynchronous background operations."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton
)


class ProgressDialog(QDialog):
    """Progress dialog with cancellation support for video extraction, propagation, and export."""

    cancelled = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(460, 160)
        self.setWindowModality(Qt.ApplicationModal)

        self._is_cancelled = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #dddddd;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #444455; border-radius: 4px; text-align: center; height: 22px; }"
            "QProgressBar::chunk { background-color: #1976d2; border-radius: 3px; }"
        )
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; padding: 4px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e74c3c; }"
        )
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress bar and status text."""
        if total > 0:
            pct = min(100, max(0, int((current / total) * 100)))
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}% ({current}/{total})")
        if message:
            self.status_label.setText(message)

    def _on_cancel(self) -> None:
        self._is_cancelled = True
        self.status_label.setText("Cancelling operation...")
        self.cancel_btn.setEnabled(False)
        self.cancelled.emit()

    def is_cancelled(self) -> bool:
        return self._is_cancelled
