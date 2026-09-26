"""Logging configuration and utilities for SAM2 Video Polygon Annotator."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "sam2_annotator",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure application logger with console and rotating file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_dir is None:
        log_dir = Path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning("Could not set up file logging to %s: %s", log_dir, e)

    return logger


logger = setup_logger()
