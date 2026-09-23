"""Right dock panel for object properties, visibility toggles, and propagation."""

from typing import List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QComboBox, QMessageBox
)

from sam3_annotator.annotation.polygon import PolygonAnnotation
from sam3_annotator.project.project_schema import ClassItem


class PropertiesPanel(QWidget):
    """Panel showing active frame objects, visibility toggles, and metadata."""

    object_selected = Signal(str)      # object_id
    object_deleted = Signal(str)       # object_id
    object_class_changed = Signal(str, int, str) # object_id, class_id, class_name
    propagate_requested = Signal(str)  # object_id

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

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.propagate_btn = QPushButton("Propagate ⏩")
        self.propagate_btn.setToolTip("Propagate this object through subsequent frames using SAM 3")
        self.propagate_btn.clicked.connect(self._on_propagate_clicked)
        btn_layout.addWidget(self.propagate_btn)

        self.delete_btn = QPushButton("Delete 🗑️")
        self.delete_btn.setStyleSheet("background-color: #c0392b;")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

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
            src_item = QTableWidgetItem(anno.source.replace("sam3_", ""))
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

    def _on_propagate_clicked(self) -> None:
        if self.selected_object_id:
            self.propagate_requested.emit(self.selected_object_id)

    def _on_delete_clicked(self) -> None:
        if self.selected_object_id:
            self.object_deleted.emit(self.selected_object_id)
