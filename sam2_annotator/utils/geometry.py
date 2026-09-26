"""Geometry and coordinate transformation utilities for polygon annotation."""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2


class CoordinateTransformer:
    """Handles mapping between Image, Canvas/Display, and Normalized coordinate spaces."""

    def __init__(self, image_width: int, image_height: int):
        self.image_width = max(1, image_width)
        self.image_height = max(1, image_height)

    def image_to_normalized(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Convert pixel coordinates (x, y) to normalized coordinates in [0.0, 1.0]."""
        normalized = []
        for x, y in points:
            nx = max(0.0, min(1.0, float(x) / self.image_width))
            ny = max(0.0, min(1.0, float(y) / self.image_height))
            normalized.append((round(nx, 6), round(ny, 6)))
        return normalized

    def normalized_to_image(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Convert normalized [0.0, 1.0] coordinates back to original image pixel coordinates."""
        image_points = []
        for nx, ny in points:
            x = float(nx) * self.image_width
            y = float(ny) * self.image_height
            image_points.append((round(x, 2), round(y, 2)))
        return image_points

    def canvas_to_image(
        self,
        canvas_point: Tuple[float, float],
        zoom_factor: float,
        pan_offset: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Convert canvas/view point to original image pixel coordinates."""
        cx, cy = canvas_point
        ox, oy = pan_offset
        ix = (cx - ox) / max(1e-6, zoom_factor)
        iy = (cy - oy) / max(1e-6, zoom_factor)
        # Clamp to image bounds
        ix = max(0.0, min(float(self.image_width), ix))
        iy = max(0.0, min(float(self.image_height), iy))
        return (ix, iy)

    def image_to_canvas(
        self,
        image_point: Tuple[float, float],
        zoom_factor: float,
        pan_offset: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Convert original image pixel coordinates to canvas/view coordinates."""
        ix, iy = image_point
        ox, oy = pan_offset
        cx = (ix * zoom_factor) + ox
        cy = (iy * zoom_factor) + oy
        return (cx, cy)


def calculate_polygon_area(points: List[Tuple[float, float]]) -> float:
    """Calculate the area of a polygon using the Shoelace formula."""
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    return float(cv2.contourArea(pts))


def calculate_polygon_centroid(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Calculate the center of mass (centroid) of a polygon."""
    if len(points) < 3:
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return None

    pts = np.array(points, dtype=np.float32)
    M = cv2.moments(pts)
    if abs(M["m00"]) > 1e-5:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        # Fallback to mean of vertices
        return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))


def compute_bounding_box(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Calculate axis-aligned bounding box [x_min, y_min, x_max, y_max]."""
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def simplify_polygon(
    points: List[Tuple[float, float]],
    tolerance_ratio: float = 0.005,
    min_vertices: int = 3,
) -> List[Tuple[float, float]]:
    """Simplify polygon using the Ramer-Douglas-Peucker algorithm via cv2.approxPolyDP."""
    if len(points) <= min_vertices:
        return points

    pts = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(pts, closed=True)
    epsilon = max(0.5, perimeter * tolerance_ratio)

    simplified = cv2.approxPolyDP(pts, epsilon, closed=True)
    out_pts = [(float(p[0][0]), float(p[0][1])) for p in simplified]

    # Clean consecutive duplicate vertices
    cleaned = []
    for pt in out_pts:
        if not cleaned or math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 0.5:
            cleaned.append(pt)

    # Ensure minimum vertices preserved
    if len(cleaned) < min_vertices:
        return points

    return cleaned


def validate_polygon(
    points: List[Tuple[float, float]],
    image_width: int,
    image_height: int,
    min_area: float = 20.0,
) -> Tuple[bool, Optional[str]]:
    """Validate polygon points against geometric integrity requirements."""
    if len(points) < 3:
        return False, f"Polygon has only {len(points)} vertices. At least 3 are required."

    for i, (x, y) in enumerate(points):
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return False, f"Vertex {i} has invalid coordinates ({x}, {y})."
        if x < -5 or x > image_width + 5 or y < -5 or y > image_height + 5:
            return False, f"Vertex {i} ({x}, {y}) is outside image bounds ({image_width}x{image_height})."

    area = calculate_polygon_area(points)
    if area < min_area:
        return False, f"Polygon area ({area:.1f} px) is below minimum threshold ({min_area:.1f} px)."

    return True, None


def compute_bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area
