"""Image processing utilities for loading, saving, thumbnailing, and display."""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from sam2_annotator.utils.logging_utils import logger


def load_image_bgr(path: Path) -> Optional[np.ndarray]:
    """Load image from disk using OpenCV with UTF-8/Windows path safety."""
    if not path.exists():
        logger.error("Image file does not exist: %s", path)
        return None
    try:
        # np.fromfile handles Windows unicode paths robustly
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error("Failed to load image %s: %s", path, e)
        return None


def save_image_bgr(path: Path, image: np.ndarray, quality: int = 95) -> bool:
    """Save image to disk in JPEG or PNG format with UTF-8/Windows path safety."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif ext == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
        else:
            params = []

        is_success, encoded_img = cv2.imencode(ext, image, params)
        if is_success:
            with open(path, "wb") as f:
                encoded_img.tofile(f)
            return True
        return False
    except Exception as e:
        logger.error("Failed to save image %s: %s", path, e)
        return False


def generate_thumbnail(
    image: np.ndarray,
    max_size: Tuple[int, int] = (160, 100),
) -> np.ndarray:
    """Create a scaled thumbnail preserving aspect ratio."""
    h, w = image.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / max(1, w), max_h / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_colored_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a semi-transparent colored mask over an RGB image."""
    overlay = image_rgb.copy()
    binary_mask = (mask > 0)
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask,
            image_rgb[:, :, c] * (1.0 - alpha) + color_rgb[c] * alpha,
            image_rgb[:, :, c],
        )
    return overlay.astype(np.uint8)


def get_deterministic_color(class_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, deterministic RGB color for a given class ID."""
    palette = [
        (46, 204, 113),   # Emerald Green
        (52, 152, 219),   # Peter River Blue
        (155, 89, 182),   # Amethyst Purple
        (241, 196, 15),   # Sun Yellow
        (230, 126, 34),   # Carrot Orange
        (231, 76, 60),    # Alizarin Red
        (26, 188, 156),   # Turquoise
        (243, 156, 18),   # Orange
        (211, 84, 0),     # Pumpkin
        (192, 57, 43),    # Pomegranate
        (142, 68, 173),   # Wisteria
        (41, 128, 185),   # Belize Hole
        (39, 174, 96),    # Nephritis
        (22, 160, 133),   # Green Sea
    ]
    return palette[class_id % len(palette)]


def bgr_to_qimage(image_bgr: np.ndarray):
    """Convert an OpenCV BGR image into a Qt QImage with C-contiguous buffer and memory safety."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    try:
        from PySide6.QtGui import QImage
        h, w = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        # .copy() ensures the QImage owns its memory buffer independently
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    except Exception as e:
        logger.error("Failed to convert BGR image to QImage: %s", e)
        return None
