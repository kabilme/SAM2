"""Project creation wizard dialog supporting single and multi-video uploads."""

from pathlib import Path
from typing import List, Optional, Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem, QMessageBox,
    QGroupBox
)

from sam2_annotator.video.video_reader import VideoReader, VideoMetadata
from sam2_annotator.utils.logging_utils import logger


class ProjectDialog(QDialog):
    """Wizard for creating a new video annotation project with multi-video support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Annotation Project")
        self.setMinimumSize(620, 600)

        self.video_items: List[Dict[str, Any]] = []  # List of {"path": Path, "metadata": VideoMetadata}

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Project Info Form
        form = QFormLayout()
        self.name_edit = QLineEdit("MyVideoProject")
        self.name_edit.textChanged.connect(self._on_name_changed)
        form.addRow("<b>Project Name:</b>", self.name_edit)

        # Project Location Picker
        dir_layout = QHBoxLayout()
        self.dir_path_edit = QLineEdit(str(Path.cwd() / "projects" / "MyVideoProject"))
        dir_browse = QPushButton("Browse...")
        dir_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self.dir_path_edit)
        dir_layout.addWidget(dir_browse)
        form.addRow("<b>Project Directory:</b>", dir_layout)

        layout.addLayout(form)

        # 2. Video Sources Group (Multi-Video Support)
        video_group = QGroupBox("Video Sources (Select one or more videos)")
        v_layout = QVBoxLayout(video_group)

        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.video_list.setMinimumHeight(100)
        v_layout.addWidget(self.video_list)

        btn_v_layout = QHBoxLayout()
        add_v_btn = QPushButton("+ Add Video(s)...")
        add_v_btn.setStyleSheet("font-weight: bold; background-color: #2e7d32;")
        add_v_btn.clicked.connect(self._browse_videos)
        remove_v_btn = QPushButton("Remove Selected")
        remove_v_btn.clicked.connect(self._remove_video)
        clear_v_btn = QPushButton("Clear All")
        clear_v_btn.clicked.connect(self._clear_videos)

        btn_v_layout.addWidget(add_v_btn)
        btn_v_layout.addWidget(remove_v_btn)
        btn_v_layout.addWidget(clear_v_btn)
        btn_v_layout.addStretch()
        v_layout.addLayout(btn_v_layout)

        # Video Info summary preview
        self.video_info_label = QLabel("No videos selected. Click '+ Add Video(s)...' to choose video files.")
        self.video_info_label.setStyleSheet("color: #3d5afe; font-size: 11px;")
        v_layout.addWidget(self.video_info_label)

        layout.addWidget(video_group)

        # 3. Frame Sampling Group
        sampling_group = QGroupBox("Frame Extraction Sampling")
        sample_layout = QFormLayout(sampling_group)

        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["Every Nth Frame", "Fixed Count per Video", "Time Interval (seconds)", "Every Frame"])
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
        sample_layout.addRow("Fixed Count per Video:", self.fixed_count_spin)

        self.interval_sec_spin = QDoubleSpinBox()
        self.interval_sec_spin.setRange(0.1, 60.0)
        self.interval_sec_spin.setValue(1.0)
        self.interval_sec_spin.setEnabled(False)
        sample_layout.addRow("Interval (sec):", self.interval_sec_spin)

        layout.addWidget(sampling_group)

        # 4. Classes Group
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
        create_btn.setStyleSheet("font-weight: bold; background-color: #1976d2;")
        create_btn.clicked.connect(self._on_create)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(create_btn)
        layout.addLayout(btn_layout)

    def _browse_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video File(s)",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*.*)"
        )
        if paths:
            for p in paths:
                self._add_video_file(Path(p))
            self._update_video_summary()

    def _add_video_file(self, path: Path) -> None:
        # Check duplicate
        for item in self.video_items:
            if item["path"].resolve() == path.resolve():
                return

        try:
            reader = VideoReader(path)
            meta = reader.metadata
            reader.close()
            if meta:
                self.video_items.append({"path": path, "metadata": meta})
                item_text = (
                    f"🎬 {path.name}  |  {meta.width}x{meta.height}  |  "
                    f"{meta.fps:.2f} fps  |  {meta.total_frames:,} frames  |  {meta.formatted_duration}"
                )
                list_item = QListWidgetItem(item_text)
                list_item.setToolTip(str(path))
                list_item.setData(Qt.UserRole, str(path))
                self.video_list.addItem(list_item)

                # If first video, auto-populate project name
                if len(self.video_items) == 1:
                    v_stem = path.stem
                    self.name_edit.setText(v_stem)
                    self.dir_path_edit.setText(str(Path.cwd() / "projects" / v_stem))
        except Exception as e:
            QMessageBox.critical(self, "Video Error", f"Could not read video '{path.name}': {e}")

    def _remove_video(self) -> None:
        selected = self.video_list.selectedItems()
        for it in selected:
            p_str = it.data(Qt.UserRole)
            self.video_items = [v for v in self.video_items if str(v["path"]) != p_str]
            row = self.video_list.row(it)
            self.video_list.takeItem(row)
        self._update_video_summary()

    def _clear_videos(self) -> None:
        self.video_items.clear()
        self.video_list.clear()
        self._update_video_summary()

    def _update_video_summary(self) -> None:
        if not self.video_items:
            self.video_info_label.setText("No videos selected. Click '+ Add Video(s)...' to choose video files.")
            return

        total_frames = sum(v["metadata"].total_frames for v in self.video_items)
        total_seconds = sum(v["metadata"].duration_seconds for v in self.video_items)
        m = int(total_seconds // 60)
        s = int(total_seconds % 60)
        duration_str = f"{m:02d}:{s:02d}"

        self.video_info_label.setText(
            f"<b>Selected Videos:</b> {len(self.video_items)}  |  "
            f"<b>Combined Frames:</b> {total_frames:,}  |  "
            f"<b>Total Duration:</b> {duration_str}"
        )

    def _on_name_changed(self, text: str) -> None:
        clean_name = text.strip() or "MyVideoProject"
        self.dir_path_edit.setText(str(Path.cwd() / "projects" / clean_name))

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
        if not self.video_items:
            QMessageBox.warning(self, "Validation Error", "Please add at least one video file.")
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
        video_paths = [v["path"] for v in self.video_items]
        video_metadatas = [v["metadata"] for v in self.video_items]

        return {
            "name": self.name_edit.text().strip(),
            "video_paths": video_paths,
            "video_path": video_paths[0] if video_paths else None,
            "video_metadatas": video_metadatas,
            "video_metadata": video_metadatas[0] if video_metadatas else None,
            "project_dir": Path(self.dir_path_edit.text().strip()),
            "classes": classes,
            "sampling_strategy": strategy,
            "every_n": self.every_n_spin.value(),
            "fixed_count": self.fixed_count_spin.value(),
            "interval_seconds": self.interval_sec_spin.value(),
        }
