"""Right dock panel for managing dataset classes."""

from typing import List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QInputDialog, QMessageBox
)

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.utils.image_utils import get_deterministic_color


class ClassPanel(QWidget):
    """Panel for listing, adding, and selecting dataset object classes."""

    class_selected = Signal(int, str)  # class_id, class_name
    classes_modified = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.classes: List[ClassItem] = []
        self.active_class_id: int = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Project Classes</b>"))

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.list_widget)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self._add_class_dialog)
        btn_layout.addWidget(self.add_btn)

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self._rename_class_dialog)
        btn_layout.addWidget(self.rename_btn)

        self.delete_btn = QPushButton("Remove")
        self.delete_btn.clicked.connect(self._remove_class)
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

    def set_classes(self, classes: List[ClassItem]) -> None:
        """Populate the class list."""
        self.classes = classes
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        for c in self.classes:
            item = QListWidgetItem()
            item.setText(f"{c.id}: {c.name}")
            item.setData(Qt.UserRole, c.id)

            # Color icon
            color_rgb = c.color_rgb or list(get_deterministic_color(c.id))
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(*color_rgb))
            item.setIcon(QIcon(pixmap))

            self.list_widget.addItem(item)

        if self.classes:
            self.list_widget.setCurrentRow(0)
            self.active_class_id = self.classes[0].id
        self.list_widget.blockSignals(False)

    def get_active_class(self) -> Optional[ClassItem]:
        for c in self.classes:
            if c.id == self.active_class_id:
                return c
        return self.classes[0] if self.classes else None

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self.classes):
            c = self.classes[row]
            self.active_class_id = c.id
            self.class_selected.emit(c.id, c.name)

    def _add_class_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Class", "Class Name:")
        if ok and name.strip():
            new_id = len(self.classes)
            item = ClassItem(id=new_id, name=name.strip())
            self.classes.append(item)
            self.set_classes(self.classes)
            self.classes_modified.emit()

    def _rename_class_dialog(self) -> None:
        c = self.get_active_class()
        if not c:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Class", "New Name:", text=c.name)
        if ok and new_name.strip():
            c.name = new_name.strip()
            self.set_classes(self.classes)
            self.classes_modified.emit()

    def _remove_class(self) -> None:
        if len(self.classes) <= 1:
            QMessageBox.warning(self, "Warning", "Project must have at least one class.")
            return

        c = self.get_active_class()
        if not c:
            return

        reply = QMessageBox.question(
            self,
            "Remove Class",
            f"Are you sure you want to remove class '{c.name}'?\nExisting annotations will need reassignment.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.classes = [cls for cls in self.classes if cls.id != c.id]
            # Reindex classes
            for idx, cls in enumerate(self.classes):
                cls.id = idx
            self.set_classes(self.classes)
            self.classes_modified.emit()
