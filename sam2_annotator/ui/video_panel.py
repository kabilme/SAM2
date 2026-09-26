"""Left dock panel displaying extracted frame thumbnails, timestamps, and review status."""

from typing import List, Optional
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QComboBox, QMenu, QAbstractItemView
)

from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.video.frame_cache import FrameCache
from sam2_annotator.utils.image_utils import bgr_to_qimage
from sam2_annotator.utils.logging_utils import logger


class FrameListWidget(QListWidget):
    """QListWidget subclass with keyboard shortcut handling for deletion."""
    delete_pressed = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_pressed.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class VideoPanel(QWidget):
    """Left dock panel for browsing frames by thumbnail with multi-selection and deletion."""

    frame_selected = Signal(int)  # frame_id
    delete_requested = Signal(list)  # list of frame_ids
    mark_null_requested = Signal(list)  # list of frame_ids
    mark_reviewed_requested = Signal(list)  # list of frame_ids

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
        self.filter_combo.addItems([
            "All Frames",
            "Annotated Only",
            "Null / Negative Frames Only",
            "Reviewed Only",
            "Unreviewed Only"
        ])
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        top_layout.addWidget(QLabel("Filter:"))
        top_layout.addWidget(self.filter_combo)
        layout.addLayout(top_layout)

        # Frame List Widget (with multi-select & context menu)
        self.list_widget = FrameListWidget()
        self.list_widget.setIconSize(QSize(120, 75))
        self.list_widget.setSpacing(4)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.delete_pressed.connect(self._on_delete_clicked)
        layout.addWidget(self.list_widget)

        # Bottom info and actions bar
        bottom_layout = QHBoxLayout()
        self.summary_label = QLabel("0 frames")
        self.summary_label.setStyleSheet("color: #888888; font-size: 11px;")
        bottom_layout.addWidget(self.summary_label)
        bottom_layout.addStretch()

        self.delete_btn = QPushButton("🗑 Delete")
        self.delete_btn.setToolTip("Delete selected frame(s) from project (Ctrl+Delete)")
        self.delete_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 3px 8px; }"
            "QPushButton:hover { background-color: #5c1d1d; border-color: #d32f2f; color: #ffcdd2; }"
        )
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        bottom_layout.addWidget(self.delete_btn)

        layout.addLayout(bottom_layout)

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
            elif filter_mode == "Null / Negative Frames Only" and frame.review_status != "negative":
                continue
            elif filter_mode == "Reviewed Only" and frame.review_status != "reviewed":
                continue
            elif filter_mode == "Unreviewed Only" and frame.review_status != "unreviewed":
                continue

            item = QListWidgetItem()
            status_symbol = {
                "unreviewed": "⚪",
                "annotated": "🔵",
                "reviewed": "🟢",
                "negative": "⚫",
                "rejected": "🔴",
            }.get(frame.review_status, "")

            if frame.review_status == "negative":
                item_label = f"⚫ [Null] #{frame.frame_id} ({frame.timestamp_seconds:.2f}s)"
            else:
                item_label = f"{status_symbol} #{frame.frame_id} ({frame.timestamp_seconds:.2f}s)"

            item.setText(item_label)
            item.setData(Qt.UserRole, frame.frame_id)
            tip_v = f"Video: {frame.video_name} | " if frame.video_name else ""
            item.setToolTip(f"{tip_v}Frame #{frame.frame_id} (Source #{frame.source_frame_index}) | Status: {frame.review_status}")

            # Load thumbnail icon
            thumb_bgr = self.frame_cache.get_thumbnail(frame.thumbnail_filename)
            if thumb_bgr is not None:
                qimg = bgr_to_qimage(thumb_bgr)
                if qimg is not None:
                    item.setIcon(QIcon(QPixmap.fromImage(qimg)))

            self.list_widget.addItem(item)
            displayed_count += 1

        self.summary_label.setText(f"Showing {displayed_count} of {len(self.frames)} frames")
        self.delete_btn.setEnabled(len(self.frames) > 0)
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

    def get_selected_frame_ids(self) -> List[int]:
        """Return list of selected frame IDs in the list widget."""
        items = self.list_widget.selectedItems()
        if not items:
            cur = self.list_widget.currentItem()
            if cur:
                items = [cur]
        fids = []
        for item in items:
            fid = item.data(Qt.UserRole)
            if fid is not None:
                fids.append(fid)
        return fids

    def _show_context_menu(self, pos) -> None:
        """Show right-click context menu on frame list items."""
        selected_ids = self.get_selected_frame_ids()
        if not selected_ids:
            item = self.list_widget.itemAt(pos)
            if item:
                fid = item.data(Qt.UserRole)
                if fid is not None:
                    selected_ids = [fid]
        if not selected_ids:
            return

        menu = QMenu(self)
        if len(selected_ids) == 1:
            del_label = f"🗑 Delete Frame #{selected_ids[0]}..."
        else:
            del_label = f"🗑 Delete {len(selected_ids)} Selected Frames..."

        del_act = menu.addAction(del_label)
        del_act.triggered.connect(lambda: self.delete_requested.emit(selected_ids))

        menu.addSeparator()
        null_act = menu.addAction("⚫ Mark as Null Frame (No Objects)")
        null_act.triggered.connect(lambda: self.mark_null_requested.emit(selected_ids))

        rev_act = menu.addAction("🟢 Mark as Reviewed")
        rev_act.triggered.connect(lambda: self.mark_reviewed_requested.emit(selected_ids))

        menu.exec(self.list_widget.viewport().mapToGlobal(pos))

    def _on_delete_clicked(self) -> None:
        selected_ids = self.get_selected_frame_ids()
        if selected_ids:
            self.delete_requested.emit(selected_ids)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            item = self.list_widget.item(row)
            if item:
                frame_id = item.data(Qt.UserRole)
                if frame_id is not None:
                    self.frame_selected.emit(frame_id)
