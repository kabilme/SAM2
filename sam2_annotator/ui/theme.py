"""Modern dark theme stylesheet and styling tokens for PySide6 application."""

DARK_THEME_QSS = """
QMainWindow, QDialog {
    background-color: #1e1e24;
    color: #e0e0e6;
}

QWidget {
    color: #e0e0e6;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #18181c;
    color: #cccccc;
    border-bottom: 1px solid #2d2d35;
}

QMenuBar::item {
    background: transparent;
    padding: 6px 12px;
}

QMenuBar::item:selected {
    background-color: #2b2b36;
    color: #ffffff;
}

QMenu {
    background-color: #24242d;
    color: #e0e0e6;
    border: 1px solid #3c3c4a;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px 6px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #3d5afe;
    color: #ffffff;
}

QToolBar {
    background-color: #1a1a20;
    border-bottom: 1px solid #2d2d35;
    padding: 4px;
    spacing: 6px;
}

QToolButton {
    background-color: #282832;
    color: #ffffff;
    border: 1px solid #3c3c4a;
    border-radius: 4px;
    padding: 6px 10px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #353545;
    border-color: #55556a;
}

QToolButton:checked {
    background-color: #3d5afe;
    border-color: #536dfe;
    color: #ffffff;
}

QDockWidget {
    color: #e0e0e6;
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(undock.png);
}

QDockWidget::title {
    background-color: #1c1c22;
    padding: 6px;
    font-weight: bold;
    border-bottom: 1px solid #2d2d35;
}

QListWidget, QTreeWidget, QTableWidget {
    background-color: #19191e;
    border: 1px solid #2d2d35;
    border-radius: 4px;
    color: #e0e0e6;
    outline: none;
}

QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
    background-color: #2e3b6e;
    color: #ffffff;
}

QPushButton {
    background-color: #3d5afe;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #536dfe;
}

QPushButton:pressed {
    background-color: #304ffe;
}

QPushButton:disabled {
    background-color: #2b2b36;
    color: #707080;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #16161a;
    border: 1px solid #3a3a48;
    border-radius: 4px;
    padding: 5px 8px;
    color: #ffffff;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #3d5afe;
}

QScrollBar:vertical {
    background: #18181c;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #3c3c4a;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #55556a;
}

QScrollBar:horizontal {
    background: #18181c;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #3c3c4a;
    min-width: 20px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #55556a;
}

QStatusBar {
    background-color: #141418;
    color: #aaaaaa;
    border-top: 1px solid #282832;
}

QProgressBar {
    background-color: #16161a;
    border: 1px solid #2d2d35;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
}

QProgressBar::chunk {
    background-color: #3d5afe;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #2a2a35;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #3d5afe;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #3d5afe;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
"""
