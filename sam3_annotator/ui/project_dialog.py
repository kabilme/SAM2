"""Project creation wizard dialog."""

from pathlib import Path
from typing import List, Optional, Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem, QMessageBox,
    QGroupBox
)

from sam3_annotator.video.video_reader import VideoReader, VideoMetadata
from sam3_annotator.utils.logging_utils import logger


class ProjectDialog(QDialog):
    """Wizard for creating a new video annotation project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Annotation Project")
        self.setMinimumSize(560, 520)

        self.video_metadata: Optional[VideoMetadata] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Video & Path Form
        form = QFormLayout()
        self.name_edit = QLineEdit("MyVideoProject")
        form.addRow("<b>Project Name:</b>", self.name_edit)

        # Video File Picker
        vid_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Select video (.mp4, .avi, .mov, .mkv)...")
        vid_browse = QPushButton("Browse...")
        vid_browse.clicked.connect(self._browse_video)
        vid_layout.addWidget(self.video_path_edit)
        vid_layout.addWidget(vid_browse)
        form.addRow("<b>Video File:</b>", vid_layout)

        # Project Location Picker
        dir_layout = QHBoxLayout()
        self.dir_path_edit = QLineEdit(str(Path.cwd() / "projects" / "MyVideoProject"))
        dir_browse = QPushButton("Browse...")
        dir_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self.dir_path_edit)
        dir_layout.addWidget(dir_browse)
        form.addRow("<b>Project Directory:</b>", dir_layout)

        layout.addLayout(form)

        # Video Info preview
        self.video_info_label = QLabel("No video selected.")
        self.video_info_label.setStyleSheet("color: #3d5afe; font-size: 11px;")
        layout.addWidget(self.video_info_label)

        # 2. Frame Sampling Group
        sampling_group = QGroupBox("Frame Extraction Sampling")
        sample_layout = QFormLayout(sampling_group)

        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["Every Nth Frame", "Fixed Count", "Time Interval (seconds)", "Every Frame"])
        self.sampling_combo.currentIndexChanged.connect(self._on_sampling_changed)
        sample_layout.addRow("Sampling Method:", self.sampling_combo)

        self.every_n_spin = QSpinBox()
        self.every_n_spin.setRange(1, 1000)
        self.every_n_spin.setValue(10)
        sample_layout.addRow("Frame Step (N):", self.every_n_spin)

        self.fixed_count_spin = QSpinBox()
        self.fixed_count_spin.setRange(1, 10000)
        self.fixed_count_spin.setValue(100)
        self.fixed_count_spin.setEnabled(False)
        sample_layout.addRow("Fixed Count:", self.fixed_count_spin)

        self.interval_sec_spin = QDoubleSpinBox()
        self.interval_sec_spin.setRange(0.1, 60.0)
        self.interval_sec_spin.setValue(1.0)
        self.interval_sec_spin.setEnabled(False)
        sample_layout.addRow("Interval (sec):", self.interval_sec_spin)

        layout.addWidget(sampling_group)

        # 3. Classes Group
        classes_group = QGroupBox("Dataset Classes")
        class_layout = QVBoxLayout(classes_group)

        self.classes_list = QListWidget()
        default_classes = ["scooter", "person", "helmet", "car"]
        for c in default_classes:
            self.classes_list.addItem(c)
        class_layout.addWidget(self.classes_list)

        btn_c_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter new class name...")
        add_c_btn = QPushButton("+ Add Class")
        add_c_btn.clicked.connect(self._add_class)
        remove_c_btn = QPushButton("Remove Selected")
        remove_c_btn.clicked.connect(self._remove_class)
        btn_c_layout.addWidget(self.class_input)
        btn_c_layout.addWidget(add_c_btn)
        btn_c_layout.addWidget(remove_c_btn)
        class_layout.addLayout(btn_c_layout)

        layout.addWidget(classes_group)

        # Dialog Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #444455;")
        cancel_btn.clicked.connect(self.reject)
        create_btn = QPushButton("Create Project & Extract Frames")
        create_btn.clicked.connect(self._on_create)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(create_btn)
        layout.addLayout(btn_layout)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*.*)"
        )
        if path:
            self.video_path_edit.setText(path)
            try:
                reader = VideoReader(Path(path))
                meta = reader.metadata
                if meta:
                    self.video_metadata = meta
                    self.video_info_label.setText(
                        f"Video: {meta.filename} | Resolution: {meta.width}x{meta.height} | "
                        f"FPS: {meta.fps} | Frames: {meta.total_frames:,} | Duration: {meta.formatted_duration}"
                    )
                    # Auto update project dir name if default
                    v_stem = Path(path).stem
                    self.name_edit.setText(v_stem)
                    self.dir_path_edit.setText(str(Path.cwd() / "projects" / v_stem))
                reader.close()
            except Exception as e:
                QMessageBox.critical(self, "Video Error", f"Could not read video: {e}")

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Project Output Folder")
        if path:
            self.dir_path_edit.setText(path)

    def _on_sampling_changed(self, idx: int) -> None:
        self.every_n_spin.setEnabled(idx == 0)
        self.fixed_count_spin.setEnabled(idx == 1)
        self.interval_sec_spin.setEnabled(idx == 2)

    def _add_class(self) -> None:
        text = self.class_input.text().strip()
        if text:
            self.classes_list.addItem(text)
            self.class_input.clear()

    def _remove_class(self) -> None:
        row = self.classes_list.currentRow()
        if row >= 0:
            self.classes_list.takeItem(row)

    def _on_create(self) -> None:
        v_path = self.video_path_edit.text().strip()
        if not v_path or not Path(v_path).exists():
            QMessageBox.warning(self, "Validation Error", "Please select a valid video file.")
            return

        p_name = self.name_edit.text().strip()
        if not p_name:
            QMessageBox.warning(self, "Validation Error", "Please enter a project name.")
            return

        if self.classes_list.count() == 0:
            QMessageBox.warning(self, "Validation Error", "Please define at least one class.")
            return

        self.accept()

    def get_project_params(self) -> Dict[str, Any]:
        """Return parameters dictionary for project creation."""
        sampling_idx = self.sampling_combo.currentIndex()
        strat_map = {0: "every_n", 1: "fixed_count", 2: "interval_seconds", 3: "every_frame"}
        strategy = strat_map.get(sampling_idx, "every_n")

        classes = [self.classes_list.item(i).text() for i in range(self.classes_list.count())]

        return {
            "name": self.name_edit.text().strip(),
            "video_path": Path(self.video_path_edit.text().strip()),
            "project_dir": Path(self.dir_path_edit.text().strip()),
            "classes": classes,
            "sampling_strategy": strategy,
            "every_n": self.every_n_spin.value(),
            "fixed_count": self.fixed_count_spin.value(),
            "interval_seconds": self.interval_sec_spin.value(),
        }
