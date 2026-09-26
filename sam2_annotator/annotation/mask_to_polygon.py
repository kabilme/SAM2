"""Converts SAM binary segmentation masks into validated polygons."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from sam2_annotator.utils.geometry import simplify_polygon, calculate_polygon_area, validate_polygon
from sam2_annotator.utils.logging_utils import logger


def clean_binary_mask(
    mask: np.ndarray,
    apply_morphology: bool = True,
    kernel_size: int = 3,
) -> np.ndarray:
    """Normalize mask to uint8 binary (0 or 255) and apply morphological noise removal."""
    if mask.dtype == bool:
        binary = (mask.astype(np.uint8)) * 255
    elif mask.max() <= 1:
        binary = (mask * 255).astype(np.uint8)
    else:
        binary = mask.astype(np.uint8)

    if apply_morphology and kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Morphological opening removes speckles/spurs
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # Morphological closing fills tiny holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def mask_to_polygons(
    mask: np.ndarray,
    tolerance_ratio: float = 0.005,
    min_area: float = 20.0,
    apply_morphology: bool = True,
    only_largest_contour: bool = True,
) -> List[List[Tuple[float, float]]]:
    """Extract simplified polygon coordinates from a 2D binary mask.

    Returns a list of polygons, where each polygon is a list of (x, y) coordinates.
    """
    if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
        return []

    h, w = mask.shape[:2]
    binary = clean_binary_mask(mask, apply_morphology=apply_morphology)

    # Find external contours
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS,
    )

    if not contours:
        return []

    # Sort contours by area descending
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area and len(cnt) >= 3:
            valid_contours.append((area, cnt))

    valid_contours.sort(key=lambda x: x[0], reverse=True)

    if not valid_contours:
        return []

    if only_largest_contour:
        selected_contours = [valid_contours[0][1]]
    else:
        selected_contours = [c[1] for c in valid_contours]

    polygons: List[List[Tuple[float, float]]] = []

    for cnt in selected_contours:
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        # Simplify contour
        simplified = simplify_polygon(pts, tolerance_ratio=tolerance_ratio, min_vertices=3)

        # Validate
        is_valid, err = validate_polygon(simplified, image_width=w, image_height=h, min_area=min_area)
        if is_valid:
            polygons.append(simplified)
        else:
            # If simplification produced invalid polygon, try with original points
            is_orig_valid, _ = validate_polygon(pts, image_width=w, image_height=h, min_area=min_area)
            if is_orig_valid:
                polygons.append(pts)
            else:
                logger.warning("Rejected invalid polygon from mask: %s", err)

    return polygons


def polygon_to_mask(
    points: List[Tuple[float, float]],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Rasterize polygon vertices into a binary uint8 mask (0 or 255)."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask
