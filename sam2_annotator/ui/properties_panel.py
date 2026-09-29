"""Right dock panel for object properties, visibility toggles, and propagation."""

from typing import List, Optional, Tuple
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QComboBox, QMessageBox,
    QGroupBox, QSpinBox
)

from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.project.project_schema import ClassItem


class PropertiesPanel(QWidget):
    """Panel showing active frame objects, visibility toggles, and metadata."""

    object_selected = Signal(str)      # object_id
    object_deleted = Signal(str)       # object_id
    object_class_changed = Signal(str, int, str) # object_id, class_id, class_name
    propagate_requested = Signal(str, int, str, str)  # object_id, frame_count, mode, prompt_type

    def __init__(self, parent=None):
        super().__init__(parent)
        self.classes: List[ClassItem] = []
        self.annotations: List[PolygonAnnotation] = []
        self.selected_object_id: Optional[str] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Frame Objects</b>"))

        # Table: Visible, Object ID, Class, Source
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Vis", "ID", "Class", "Source"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self.table)

        # Object Details
        self.details_label = QLabel("Select an object to inspect")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self.details_label)

        # Propagation & Tracking Controls
        prop_group = QGroupBox("Tracking & Propagation")
        prop_vlayout = QVBoxLayout(prop_group)
        prop_vlayout.setContentsMargins(8, 8, 8, 8)
        prop_vlayout.setSpacing(6)

        # Scope selector
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Scope:")
        mode_label.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 11px;")
        self.prop_mode_combo = QComboBox()
        self.prop_mode_combo.addItems([
            "Fixed Frame Count",
            "Until Next Keyframe",
            "Until End of Video",
        ])
        self.prop_mode_combo.currentIndexChanged.connect(self._on_prop_mode_changed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.prop_mode_combo, 1)
        prop_vlayout.addLayout(mode_layout)

        # Prompt type selector
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("Prompt:")
        prompt_label.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 11px;")
        self.prop_prompt_combo = QComboBox()
        self.prop_prompt_combo.addItems([
            "🔲 Box Prompt",
            "📍 Centroid Point",
            "🔲+📍 Box + Center Point",
        ])
        self.prop_prompt_combo.setToolTip("Prompt type used by SAM 2 to track and segment the object forward")
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prop_prompt_combo, 1)
        prop_vlayout.addLayout(prompt_layout)

        # Frame count spinbox
        self.count_container = QWidget()
        count_layout = QHBoxLayout(self.count_container)
        count_layout.setContentsMargins(0, 0, 0, 0)
        count_layout.setSpacing(6)
        self.count_label = QLabel("Frames:")
        self.count_label.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 11px;")
        self.prop_frames_spin = QSpinBox()
        self.prop_frames_spin.setRange(1, 9999)
        self.prop_frames_spin.setValue(30)
        self.prop_frames_spin.setSingleStep(5)
        self.prop_frames_spin.setSuffix(" frames")
        self.prop_frames_spin.setToolTip("Number of subsequent frames to track and propagate this object through")
        count_layout.addWidget(self.count_label)
        count_layout.addWidget(self.prop_frames_spin, 1)
        prop_vlayout.addWidget(self.count_container)

        # Quick preset buttons for common intervals
        self.preset_container = QWidget()
        preset_layout = QHBoxLayout(self.preset_container)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)
        preset_label = QLabel("Presets:")
        preset_label.setStyleSheet("color: #888888; font-size: 10px;")
        preset_layout.addWidget(preset_label)
        for preset_val in [10, 30, 60, 100]:
            btn = QPushButton(str(preset_val))
            btn.setFixedWidth(36)
            btn.setStyleSheet("padding: 2px 4px; font-size: 10px; background-color: #2b2b36; border-radius: 3px;")
            btn.clicked.connect(lambda checked=False, val=preset_val: self._set_preset(val))
            preset_layout.addWidget(btn)
        preset_layout.addStretch()
        prop_vlayout.addWidget(self.preset_container)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.propagate_btn = QPushButton("Propagate ⏩")
        self.propagate_btn.setStyleSheet("background-color: #2e7d32; font-weight: bold;")
        self.propagate_btn.setToolTip("Propagate selected object through subsequent frames using SAM 2")
        self.propagate_btn.clicked.connect(self._on_propagate_clicked)
        btn_layout.addWidget(self.propagate_btn)

        self.delete_btn = QPushButton("Delete 🗑️")
        self.delete_btn.setStyleSheet("background-color: #c0392b;")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        btn_layout.addWidget(self.delete_btn)

        prop_vlayout.addLayout(btn_layout)
        layout.addWidget(prop_group)

    def set_classes(self, classes: List[ClassItem]) -> None:
        self.classes = classes

    def set_annotations(self, annotations: List[PolygonAnnotation], selected_id: Optional[str] = None) -> None:
        """Update table items with annotations for the active frame."""
        self.annotations = annotations
        self.selected_object_id = selected_id

        self.table.blockSignals(True)
        self.table.setRowCount(len(annotations))

        for row, anno in enumerate(annotations):
            # Vis checkbox
            vis_item = QTableWidgetItem("👁️" if anno.visible else "❌")
            vis_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, vis_item)

            # ID
            id_item = QTableWidgetItem(anno.object_id[:8])
            id_item.setData(Qt.UserRole, anno.object_id)
            self.table.setItem(row, 1, id_item)

            # Class
            cls_item = QTableWidgetItem(anno.class_name)
            self.table.setItem(row, 2, cls_item)

            # Source
            src_item = QTableWidgetItem(anno.source.replace("sam2_", ""))
            self.table.setItem(row, 3, src_item)

            if anno.object_id == selected_id:
                self.table.selectRow(row)

        self.table.blockSignals(False)
        self._update_details()

    def select_object(self, object_id: str) -> None:
        self.selected_object_id = object_id
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if item and item.data(Qt.UserRole) == object_id:
                self.table.selectRow(row)
                break
        self._update_details()

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if 0 <= row < len(self.annotations):
            anno = self.annotations[row]
            if col == 0:
                # Toggle visibility
                anno.visible = not anno.visible
                self.set_annotations(self.annotations, self.selected_object_id)
            else:
                self.selected_object_id = anno.object_id
                self.object_selected.emit(anno.object_id)
                self._update_details()

    def _update_details(self) -> None:
        anno = next((a for a in self.annotations if a.object_id == self.selected_object_id), None)
        if not anno:
            self.details_label.setText("No object selected.")
            self.delete_btn.setEnabled(False)
            self.propagate_btn.setEnabled(False)
            return

        self.delete_btn.setEnabled(True)
        self.propagate_btn.setEnabled(True)

        conf_str = f"{anno.confidence * 100:.1f}%" if anno.confidence is not None else "N/A"
        txt = (
            f"<b>Object:</b> {anno.object_id}<br>"
            f"<b>Class:</b> {anno.class_name} (ID: {anno.class_id})<br>"
            f"<b>Vertices:</b> {len(anno.points)} | <b>Area:</b> {anno.area:.0f} px<br>"
            f"<b>Confidence:</b> {conf_str} | <b>Status:</b> {anno.tracking_status}<br>"
            f"<b>Source:</b> {anno.source}"
        )
        self.details_label.setText(txt)

    @property
    def propagation_mode(self) -> str:
        """Return the current propagation scope: 'fixed', 'next_keyframe', or 'end_of_video'."""
        idx = self.prop_mode_combo.currentIndex()
        if idx == 1:
            return "next_keyframe"
        elif idx == 2:
            return "end_of_video"
        return "fixed"

    @property
    def propagation_prompt_type(self) -> str:
        """Return the selected prompt type: 'box', 'point', or 'combined'."""
        idx = self.prop_prompt_combo.currentIndex()
        if idx == 1:
            return "point"
        elif idx == 2:
            return "combined"
        return "box"

    @property
    def propagation_frames(self) -> int:
        """Return the selected frame count for fixed-frame propagation."""
        return self.prop_frames_spin.value()

    def set_default_propagation_frames(self, frames: int) -> None:
        """Update the default frame count in the spinbox."""
        if frames > 0:
            self.prop_frames_spin.setValue(frames)

    def set_default_propagation_prompt_type(self, prompt_type: str) -> None:
        """Set the default prompt type in the dropdown."""
        if prompt_type == "point":
            self.prop_prompt_combo.setCurrentIndex(1)
        elif prompt_type == "combined":
            self.prop_prompt_combo.setCurrentIndex(2)
        else:
            self.prop_prompt_combo.setCurrentIndex(0)

    def _set_preset(self, val: int) -> None:
        self.prop_mode_combo.setCurrentIndex(0)  # Fixed Frame Count
        self.prop_frames_spin.setValue(val)

    def _on_prop_mode_changed(self, index: int) -> None:
        is_fixed = (index == 0)
        self.count_container.setEnabled(is_fixed)
        self.preset_container.setEnabled(is_fixed)
        if index == 1:
            self.propagate_btn.setToolTip("Propagate selected object forward until the next reference keyframe")
        elif index == 2:
            self.propagate_btn.setToolTip("Propagate selected object forward through all remaining video frames")
        else:
            self.propagate_btn.setToolTip(f"Propagate selected object forward by {self.prop_frames_spin.value()} frames")

    def _on_propagate_clicked(self) -> None:
        if self.selected_object_id:
            count = self.prop_frames_spin.value()
            mode = self.propagation_mode
            prompt_type = self.propagation_prompt_type
            self.propagate_requested.emit(self.selected_object_id, count, mode, prompt_type)

    def _on_delete_clicked(self) -> None:
        if self.selected_object_id:
            self.object_deleted.emit(self.selected_object_id)
