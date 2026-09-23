"""Interactive polygon editing logic and vertex manipulation."""

import math
from typing import List, Tuple, Optional
from sam3_annotator.annotation.polygon import PolygonAnnotation
from sam3_annotator.utils.geometry import validate_polygon


class PolygonEditor:
    """Provides algorithmic operations for modifying vertices and polygons."""

    @staticmethod
    def find_nearest_vertex(
        points: List[Tuple[float, float]],
        target_point: Tuple[float, float],
        max_distance: float = 12.0,
    ) -> Optional[int]:
        """Find the index of the closest vertex to a target point within max_distance."""
        if not points:
            return None
        tx, ty = target_point
        min_dist = float("inf")
        nearest_idx = None
        for i, (px, py) in enumerate(points):
            d = math.hypot(px - tx, py - ty)
            if d < min_dist and d <= max_distance:
                min_dist = d
                nearest_idx = i
        return nearest_idx

    @staticmethod
    def find_nearest_edge(
        points: List[Tuple[float, float]],
        target_point: Tuple[float, float],
        max_distance: float = 10.0,
    ) -> Optional[int]:
        """Find the edge index (i to (i+1)%n) closest to target_point to insert a vertex."""
        n = len(points)
        if n < 2:
            return None

        tx, ty = target_point
        min_dist = float("inf")
        best_edge_idx = None

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            dist = PolygonEditor._point_to_segment_distance(tx, ty, p1[0], p1[1], p2[0], p2[1])
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                best_edge_idx = i

        return best_edge_idx

    @staticmethod
    def _point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate minimum perpendicular distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return math.hypot(px - x1, py - y1)

        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    @staticmethod
    def move_vertex(
        annotation: PolygonAnnotation,
        vertex_idx: int,
        new_pos: Tuple[float, float],
    ) -> bool:
        """Update position of a single vertex."""
        if 0 <= vertex_idx < len(annotation.points):
            new_points = list(annotation.points)
            new_points[vertex_idx] = (round(new_pos[0], 2), round(new_pos[1], 2))
            annotation.update_geometry(new_points)
            return True
        return False

    @staticmethod
    def insert_vertex(
        annotation: PolygonAnnotation,
        edge_idx: int,
        new_point: Tuple[float, float],
    ) -> bool:
        """Insert a new vertex along edge_idx."""
        if 0 <= edge_idx < len(annotation.points):
            new_points = list(annotation.points)
            new_points.insert(edge_idx + 1, (round(new_point[0], 2), round(new_point[1], 2)))
            annotation.update_geometry(new_points)
            return True
        return False

    @staticmethod
    def delete_vertex(
        annotation: PolygonAnnotation,
        vertex_idx: int,
    ) -> bool:
        """Delete vertex at vertex_idx if polygon will still have at least 3 points."""
        if len(annotation.points) > 3 and 0 <= vertex_idx < len(annotation.points):
            new_points = list(annotation.points)
            new_points.pop(vertex_idx)
            annotation.update_geometry(new_points)
            return True
        return False

    @staticmethod
    def translate_polygon(
        annotation: PolygonAnnotation,
        dx: float,
        dy: float,
    ) -> None:
        """Shift all vertices of polygon by (dx, dy)."""
        new_points = [(p[0] + dx, p[1] + dy) for p in annotation.points]
        annotation.update_geometry(new_points)
