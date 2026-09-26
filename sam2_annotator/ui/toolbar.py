"""Main action toolbar for SAM2 Video Polygon Annotator."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QToolBar, QToolButton, QButtonGroup, QWidget, QHBoxLayout,
    QLineEdit, QLabel, QPushButton
)

from sam2_annotator.ui.annotation_canvas import (
    MODE_SELECT, MODE_POINT_POS, MODE_POINT_NEG, MODE_BOX, MODE_POLYGON, MODE_EDIT
)


class MainToolBar(QToolBar):
    """Primary application toolbar for tools, modes, prompts, and actions."""

    mode_changed = Signal(str)
    text_prompt_submitted = Signal(str)

    def __init__(self, parent=None, model_name: str = "sam2.1_hiera_tiny.pt"):
        super().__init__("Main Toolbar", parent)
        self.setMovable(False)
        self.model_name = model_name

        self._setup_tools()

    def set_model_name(self, model_name: str) -> None:
        """Update the prompt label with active model name."""
        self.model_name = model_name
        if hasattr(self, "prompt_label"):
            self.prompt_label.setText(f"{model_name} Prompt:")

    def _setup_tools(self) -> None:
        # Group for exclusive mode selection
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        self.btn_select = self._add_mode_btn("Select", MODE_SELECT, "Select object or canvas (S)")
        self.btn_point_pos = self._add_mode_btn("Point (+)", MODE_POINT_POS, "Add positive SAM prompt point")
        self.btn_point_neg = self._add_mode_btn("Point (-)", MODE_POINT_NEG, "Add negative SAM prompt point")
        self.btn_box = self._add_mode_btn("Box Prompt", MODE_BOX, "Draw bounding box SAM prompt (B)")
        self.btn_polygon = self._add_mode_btn("Manual Poly", MODE_POLYGON, "Draw manual polygon points (M)")
        self.btn_edit = self._add_mode_btn("Edit Poly", MODE_EDIT, "Move / insert / delete vertices (E)")

        self.btn_select.setChecked(True)
        self.addSeparator()

        # Text prompt widget
        prompt_widget = QWidget()
        p_layout = QHBoxLayout(prompt_widget)
        p_layout.setContentsMargins(4, 0, 4, 0)
        p_layout.setSpacing(4)

        self.prompt_label = QLabel(f"{self.model_name} Prompt:")
        p_layout.addWidget(self.prompt_label)
        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setPlaceholderText("e.g. scooter, helmet...")
        self.text_prompt_edit.setMaximumWidth(160)
        self.text_prompt_edit.returnPressed.connect(self._on_prompt_submit)
        p_layout.addWidget(self.text_prompt_edit)

        self.run_prompt_btn = QPushButton("Segment")
        self.run_prompt_btn.clicked.connect(self._on_prompt_submit)
        p_layout.addWidget(self.run_prompt_btn)

        self.addWidget(prompt_widget)
        self.addSeparator()

        # Zoom buttons
        self.btn_zoom_fit = self.addAction("Fit")
        self.btn_zoom_fit.setToolTip("Fit image to window")
        self.btn_zoom_100 = self.addAction("100%")
        self.btn_zoom_200 = self.addAction("200%")

    def _add_mode_btn(self, label: str, mode: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(label)
        btn.setCheckable(True)
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda: self.mode_changed.emit(mode))
        self.mode_group.addButton(btn)
        self.addWidget(btn)
        return btn

    def set_active_mode(self, mode: str) -> None:
        btn_map = {
            MODE_SELECT: self.btn_select,
            MODE_POINT_POS: self.btn_point_pos,
            MODE_POINT_NEG: self.btn_point_neg,
            MODE_BOX: self.btn_box,
            MODE_POLYGON: self.btn_polygon,
            MODE_EDIT: self.btn_edit,
        }
        btn = btn_map.get(mode)
        if btn:
            btn.setChecked(True)

    def _on_prompt_submit(self) -> None:
        text = self.text_prompt_edit.text().strip()
        if text:
            self.text_prompt_submitted.emit(text)
