"""Left dock panel displaying extracted frame thumbnails, timestamps, and review status."""

from typing import List, Optional
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QComboBox, QLineEdit
)

from sam3_annotator.video.frame_extractor import FrameMetadata
from sam3_annotator.video.frame_cache import FrameCache
from sam3_annotator.utils.image_utils import bgr_to_qimage
from sam3_annotator.utils.logging_utils import logger


class VideoPanel(QWidget):
    """Left dock panel for browsing frames by thumbnail."""

    frame_selected = Signal(int)  # frame_id

    def __init__(self, frame_cache: FrameCache, parent=None):
        super().__init__(parent)
        self.frame_cache = frame_cache
        self.frames: List[FrameMetadata] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Header with filter
        top_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Frames", "Annotated Only", "Reviewed Only", "Unreviewed Only"])
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        top_layout.addWidget(QLabel("Filter:"))
        top_layout.addWidget(self.filter_combo)
        layout.addLayout(top_layout)

        # Frame List Widget
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(120, 75))
        self.list_widget.setSpacing(4)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.list_widget)

        # Info summary label
        self.summary_label = QLabel("0 frames")
        self.summary_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.summary_label)

    def set_frames(self, frames: List[FrameMetadata]) -> None:
        """Populate the list with extracted frames."""
        self.frames = frames
        self.apply_filter()

    def apply_filter(self) -> None:
        """Filter items based on combo selection."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        filter_mode = self.filter_combo.currentText()

        displayed_count = 0
        for frame in self.frames:
            if filter_mode == "Annotated Only" and frame.review_status not in ["annotated", "reviewed"]:
                continue
            elif filter_mode == "Reviewed Only" and frame.review_status != "reviewed":
                continue
            elif filter_mode == "Unreviewed Only" and frame.review_status != "unreviewed":
                continue

            item = QListWidgetItem()
            # Title: Frame ID + Timestamp + Status
            status_symbol = {
                "unreviewed": "⚪",
                "annotated": "🔵",
                "reviewed": "🟢",
                "negative": "⚫",
                "rejected": "🔴",
            }.get(frame.review_status, "")

            item.setText(f"{status_symbol} #{frame.frame_id} ({frame.timestamp_seconds:.2f}s)")
            item.setData(Qt.UserRole, frame.frame_id)

            # Load thumbnail icon
            thumb_bgr = self.frame_cache.get_thumbnail(frame.thumbnail_filename)
            if thumb_bgr is not None:
                qimg = bgr_to_qimage(thumb_bgr)
                if qimg is not None:
                    item.setIcon(QIcon(QPixmap.fromImage(qimg)))

            self.list_widget.addItem(item)
            displayed_count += 1

        self.summary_label.setText(f"Showing {displayed_count} of {len(self.frames)} frames")
        self.list_widget.blockSignals(False)

    def select_frame(self, frame_id: int) -> None:
        """Select item matching frame_id in list."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == frame_id:
                self.list_widget.blockSignals(True)
                self.list_widget.setCurrentRow(i)
                self.list_widget.blockSignals(False)
                break

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            item = self.list_widget.item(row)
            if item:
                frame_id = item.data(Qt.UserRole)
                self.frame_selected.emit(frame_id)
