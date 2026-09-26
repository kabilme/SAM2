"""Confirmation dialog for deleting video frames from a project."""

from typing import List, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QGroupBox, QFormLayout
)

from sam2_annotator.video.frame_extractor import FrameMetadata


class DeleteFrameDialog(QDialog):
    """Dialog prompting confirmation and disk cleanup options before deleting frames."""

    def __init__(
        self,
        frames_to_delete: List[FrameMetadata],
        annotation_counts: Optional[Dict[int, int]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.frames_to_delete = frames_to_delete
        self.annotation_counts = annotation_counts or {}

        count = len(frames_to_delete)
        if count == 1:
            self.setWindowTitle(f"Delete Frame #{frames_to_delete[0].frame_id}")
        else:
            self.setWindowTitle(f"Delete {count} Video Frames")

        self.setMinimumWidth(460)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        count = len(self.frames_to_delete)

        # Header warning
        header = QLabel("<h3>Confirm Frame Deletion</h3>")
        layout.addWidget(header)

        # Frame details group
        details_group = QGroupBox("Target Frame Details")
        form = QFormLayout(details_group)
        form.setSpacing(6)

        total_annos = 0
        for f in self.frames_to_delete:
            total_annos += self.annotation_counts.get(f.frame_id, 0)

        if count == 1:
            f = self.frames_to_delete[0]
            form.addRow("<b>Frame ID:</b>", QLabel(f"#{f.frame_id} (Source frame #{f.source_frame_index})"))
            form.addRow("<b>Timestamp:</b>", QLabel(f"{f.timestamp_seconds:.2f}s"))
            if f.video_name:
                form.addRow("<b>Source Video:</b>", QLabel(f.video_name))
            form.addRow("<b>Image File:</b>", QLabel(f.filename))
            form.addRow("<b>Annotations:</b>", QLabel(f"{self.annotation_counts.get(f.frame_id, 0)} object(s)"))
        else:
            frame_ids_str = ", ".join(f"#{f.frame_id}" for f in self.frames_to_delete[:8])
            if count > 8:
                frame_ids_str += f", ... (+{count - 8} more)"
            form.addRow("<b>Frames to Delete:</b>", QLabel(f"{count} frames ({frame_ids_str})"))
            form.addRow("<b>Total Annotations:</b>", QLabel(f"{total_annos} object(s)"))

        layout.addWidget(details_group)

        # Warning banner if annotations will be lost
        if total_annos > 0:
            anno_warn = QLabel(
                f"⚠️ <b>Warning:</b> Deleting will permanently remove <b>{total_annos} annotation(s)</b> "
                "associated with these frame(s)."
            )
            anno_warn.setStyleSheet("color: #ffb74d; background-color: #2b2210; padding: 8px; border-radius: 4px;")
            anno_warn.setWordWrap(True)
            layout.addWidget(anno_warn)

        # Options
        options_group = QGroupBox("Deletion Options")
        opt_layout = QVBoxLayout(options_group)
        opt_layout.setSpacing(6)

        self.chk_delete_files = QCheckBox("Also delete extracted image (.jpg) and thumbnail files from disk")
        self.chk_delete_files.setChecked(True)
        self.chk_delete_files.setToolTip(
            "If checked, frees up disk storage by deleting the image files from frames/ and thumbnails/.\n"
            "If unchecked, keeps the extracted image files on disk and only removes them from the project."
        )
        opt_layout.addWidget(self.chk_delete_files)

        note_label = QLabel(
            "ℹ️ Remaining frames in the project will be automatically re-indexed continuously (1..N)."
        )
        note_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        note_label.setWordWrap(True)
        opt_layout.addWidget(note_label)

        layout.addWidget(options_group)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)
        btn_layout.addWidget(self.cancel_btn)

        del_title = "Delete Frame" if count == 1 else f"Delete {count} Frames"
        self.delete_btn = QPushButton(f"🗑 {del_title}")
        self.delete_btn.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e53935; }"
            "QPushButton:pressed { background-color: #b71c1c; }"
        )
        self.delete_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

    def should_delete_files(self) -> bool:
        """Return True if user chose to delete physical image/thumbnail files from disk."""
        return self.chk_delete_files.isChecked()
