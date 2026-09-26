"""Application launcher and Qt lifecycle manager."""

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from sam2_annotator.ui.theme import DARK_THEME_QSS
from sam2_annotator.ui.main_window import MainWindow
from sam2_annotator.utils.logging_utils import logger


def run_app(project_path: Optional[str] = None, device: Optional[str] = None) -> int:
    """Initialize PySide6 application, configure high-DPI scaling, and show main window."""
    # High-DPI scaling attributes
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("SAM2 Video Polygon Annotator")
    app.setOrganizationName("DeepMind")

    # Apply modern dark theme stylesheet
    app.setStyleSheet(DARK_THEME_QSS)

    window = MainWindow(project_path=project_path, device_override=device)
    window.show()

    logger.info("Application started successfully.")
    return app.exec()
