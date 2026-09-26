"""High-performance interactive annotation canvas using QGraphicsView & QGraphicsScene."""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QPainterPath,
    QPolygonF, QFont, QCursor, QWheelEvent, QMouseEvent, QKeyEvent
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsPolygonItem, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsTextItem, QMenu
)

from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.annotation.polygon_editor import PolygonEditor
from sam2_annotator.utils.geometry import CoordinateTransformer
from sam2_annotator.utils.image_utils import get_deterministic_color, bgr_to_qimage
from sam2_annotator.utils.logging_utils import logger


# Interaction Modes
MODE_SELECT = "select"
MODE_POINT_POS = "point_pos"
MODE_POINT_NEG = "point_neg"
MODE_BOX = "box"
MODE_POLYGON = "polygon"
MODE_EDIT = "edit"


class AnnotationCanvas(QGraphicsView):
    """Interactive canvas supporting zooming, panning, SAM prompts, and polygon editing."""

    # Signals for UI integration
    prompt_point_added = Signal(float, float, bool)  # x, y, is_positive
    prompt_box_completed = Signal(float, float, float, float)  # x1, y1, x2, y2
    manual_polygon_completed = Signal(list)  # [(x, y), ...]
    object_selected = Signal(str)  # object_id
    annotation_changed = Signal(str)  # object_id
    cursor_moved = Signal(float, float)  # image coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Rendering options
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Image state
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.image_width: int = 1
        self.image_height: int = 1
        self.zoom_factor: float = 1.0

        # Mode and interaction state
        self.current_mode: str = MODE_SELECT
        self.is_panning: bool = False
        self.pan_start_pos: QPointF = QPointF()

        # Temporary drawing state
        self.box_start_point: Optional[QPointF] = None
        self.box_current_rect: Optional[QGraphicsRectItem] = None
        self.in_progress_polygon_points: List[Tuple[float, float]] = []
        self.active_prompt_points: List[Tuple[float, float, bool]] = []  # (x, y, is_pos)

        # Display preferences
        self.show_masks: bool = True
        self.show_polygons: bool = True
        self.show_vertices: bool = True
        self.show_boxes: bool = True
        self.show_labels: bool = True
        self.mask_opacity: float = 0.45

        # Dragging vertex in edit mode
        self.dragged_vertex_idx: Optional[int] = None
        self.active_annotation_id: Optional[str] = None

        # Data cache
        self.current_annotations: List[PolygonAnnotation] = []

    def set_image(self, image_bgr: np.ndarray) -> None:
        """Set the active background frame."""
        self.scene.clear()
        self.pixmap_item = None
        self.dragged_vertex_idx = None
        self.box_start_point = None
        self.box_current_rect = None
        self.in_progress_polygon_points.clear()
        self.active_prompt_points.clear()

        if image_bgr is None or image_bgr.size == 0:
            self.scene.clear()
            self.pixmap_item = None
            self.current_annotations = []
            self.image_width = 0
            self.image_height = 0
            return

        h, w = image_bgr.shape[:2]
        self.image_width = w
        self.image_height = h

        # Convert OpenCV BGR to RGB QImage
        qimg = bgr_to_qimage(image_bgr)
        if qimg is None:
            return
        pixmap = QPixmap.fromImage(qimg)

        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene.setSceneRect(0, 0, w, h)

        self.redraw_annotations()

    def set_annotations(self, annotations: List[PolygonAnnotation], selected_id: Optional[str] = None) -> None:
        """Update visible annotations list and redraw."""
        self.current_annotations = annotations
        self.active_annotation_id = selected_id
        self.redraw_annotations()

    def redraw_annotations(self) -> None:
        """Reconstruct vector graphic items for polygons, vertices, and labels."""
        if not self.pixmap_item:
            return

        # Keep pixmap item, remove overlay items
        for item in list(self.scene.items()):
            if item != self.pixmap_item:
                self.scene.removeItem(item)

        for anno in self.current_annotations:
            if not anno.visible or len(anno.points) < 3:
                continue

            is_selected = (anno.object_id == self.active_annotation_id)
            base_color = get_deterministic_color(anno.class_id)
            qcolor = QColor(*base_color)

            # Draw semi-transparent filled polygon
            if self.show_masks:
                poly = QPolygonF([QPointF(x, y) for x, y in anno.points])
                brush_alpha = int(self.mask_opacity * 255)
                fill_color = QColor(qcolor.red(), qcolor.green(), qcolor.blue(), brush_alpha)
                poly_item = self.scene.addPolygon(poly, QPen(Qt.NoPen), QBrush(fill_color))
                poly_item.setZValue(1)

            # Draw polygon boundary line
            if self.show_polygons:
                path = QPainterPath()
                pts = [QPointF(x, y) for x, y in anno.points]
                path.moveTo(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                path.closeSubpath()

                pen_width = 3 if is_selected else 2
                pen = QPen(qcolor if not is_selected else QColor(255, 255, 255), pen_width)
                outline_item = self.scene.addPath(path, pen, QBrush(Qt.NoBrush))
                outline_item.setZValue(2)

            # Draw vertices in Edit mode or if selected
            if self.show_vertices and (is_selected or self.current_mode == MODE_EDIT):
                v_radius = 4
                for vi, (vx, vy) in enumerate(anno.points):
                    v_pen = QPen(QColor(0, 0, 0), 1)
                    v_brush = QBrush(QColor(255, 255, 255) if not is_selected else QColor(255, 215, 0))
                    vert_item = self.scene.addEllipse(
                        vx - v_radius, vy - v_radius, v_radius * 2, v_radius * 2,
                        v_pen, v_brush
                    )
                    vert_item.setZValue(4)

            # Draw Bounding Box & Label
            if self.show_boxes or self.show_labels:
                bx1, by1, bx2, by2 = anno.bounding_box
                if self.show_boxes and is_selected:
                    box_rect = QRectF(bx1, by1, bx2 - bx1, by2 - by1)
                    dash_pen = QPen(qcolor, 1, Qt.DashLine)
                    b_item = self.scene.addRect(box_rect, dash_pen)
                    b_item.setZValue(2)

                if self.show_labels:
                    txt = f"{anno.class_name}:{anno.object_id[:6]}"
                    txt_item = self.scene.addText(txt, QFont("Segoe UI", 9, QFont.Bold))
                    txt_item.setDefaultTextColor(QColor(255, 255, 255))
                    txt_item.setPos(bx1, max(0, by1 - 20))
                    txt_item.setZValue(5)

        # Draw active SAM prompt points
        for px, py, is_pos in self.active_prompt_points:
            p_color = QColor(46, 204, 113) if is_pos else QColor(231, 76, 60)
            p_pen = QPen(QColor(255, 255, 255), 2)
            p_brush = QBrush(p_color)
            p_item = self.scene.addEllipse(px - 5, py - 5, 10, 10, p_pen, p_brush)
            p_item.setZValue(10)

        # Draw in-progress manual polygon lines
        if len(self.in_progress_polygon_points) > 0:
            pts = [QPointF(x, y) for x, y in self.in_progress_polygon_points]
            for p in pts:
                self.scene.addEllipse(p.x() - 3, p.y() - 3, 6, 6, QPen(Qt.white), QBrush(Qt.yellow)).setZValue(10)
            if len(pts) > 1:
                path = QPainterPath()
                path.moveTo(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                self.scene.addPath(path, QPen(QColor(255, 215, 0), 2, Qt.DashLine)).setZValue(9)

    # ---------------- Mouse Events ----------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.pos())
        ix = max(0.0, min(float(self.image_width), pos.x()))
        iy = max(0.0, min(float(self.image_height), pos.y()))

        # Middle mouse click or Space key -> Pan
        if event.button() == Qt.MiddleButton or (event.buttons() & Qt.MiddleButton):
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            if self.current_mode == MODE_POINT_POS:
                self.active_prompt_points.append((ix, iy, True))
                self.prompt_point_added.emit(ix, iy, True)
                self.redraw_annotations()

            elif self.current_mode == MODE_POINT_NEG:
                self.active_prompt_points.append((ix, iy, False))
                self.prompt_point_added.emit(ix, iy, False)
                self.redraw_annotations()

            elif self.current_mode == MODE_BOX:
                self.box_start_point = QPointF(ix, iy)
                self.box_current_rect = self.scene.addRect(
                    QRectF(self.box_start_point, self.box_start_point),
                    QPen(QColor(52, 152, 219), 2, Qt.DashLine),
                )
                self.box_current_rect.setZValue(10)

            elif self.current_mode == MODE_POLYGON:
                self.in_progress_polygon_points.append((ix, iy))
                self.redraw_annotations()

            elif self.current_mode in [MODE_SELECT, MODE_EDIT]:
                # Check vertex hit first
                selected_anno = self._find_active_annotation()
                if selected_anno:
                    v_idx = PolygonEditor.find_nearest_vertex(selected_anno.points, (ix, iy), max_distance=10.0)
                    if v_idx is not None:
                        self.dragged_vertex_idx = v_idx
                        return

                # Otherwise check object hit
                clicked_obj = self._find_annotation_at(ix, iy)
                if clicked_obj:
                    self.active_annotation_id = clicked_obj.object_id
                    self.object_selected.emit(clicked_obj.object_id)
                    self.redraw_annotations()
                else:
                    self.active_annotation_id = None
                    self.redraw_annotations()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.pos())
        ix = max(0.0, min(float(self.image_width), pos.x()))
        iy = max(0.0, min(float(self.image_height), pos.y()))
        self.cursor_moved.emit(ix, iy)

        if self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self.current_mode == MODE_BOX and self.box_start_point and self.box_current_rect:
            rect = QRectF(self.box_start_point, QPointF(ix, iy)).normalized()
            self.box_current_rect.setRect(rect)

        elif self.current_mode == MODE_EDIT and self.dragged_vertex_idx is not None:
            selected_anno = self._find_active_annotation()
            if selected_anno:
                PolygonEditor.move_vertex(selected_anno, self.dragged_vertex_idx, (ix, iy))
                self.redraw_annotations()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton or self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            if self.current_mode == MODE_BOX and self.box_start_point:
                pos = self.mapToScene(event.pos())
                ix = max(0.0, min(float(self.image_width), pos.x()))
                iy = max(0.0, min(float(self.image_height), pos.y()))

                x1 = min(self.box_start_point.x(), ix)
                y1 = min(self.box_start_point.y(), iy)
                x2 = max(self.box_start_point.x(), ix)
                y2 = max(self.box_start_point.y(), iy)

                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.prompt_box_completed.emit(x1, y1, x2, y2)

                if self.box_current_rect:
                    self.scene.removeItem(self.box_current_rect)
                    self.box_current_rect = None
                self.box_start_point = None

            elif self.current_mode == MODE_EDIT and self.dragged_vertex_idx is not None:
                selected_anno = self._find_active_annotation()
                if selected_anno:
                    self.annotation_changed.emit(selected_anno.object_id)
                self.dragged_vertex_idx = None

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            if self.current_mode == MODE_POLYGON:
                # Complete manual polygon
                if len(self.in_progress_polygon_points) >= 3:
                    pts = list(self.in_progress_polygon_points)
                    self.in_progress_polygon_points.clear()
                    self.manual_polygon_completed.emit(pts)
                    self.redraw_annotations()
            elif self.current_mode == MODE_EDIT:
                # Double click on edge to insert a vertex
                pos = self.mapToScene(event.pos())
                ix, iy = pos.x(), pos.y()
                selected_anno = self._find_active_annotation()
                if selected_anno:
                    edge_idx = PolygonEditor.find_nearest_edge(selected_anno.points, (ix, iy), max_distance=8.0)
                    if edge_idx is not None:
                        PolygonEditor.insert_vertex(selected_anno, edge_idx, (ix, iy))
                        self.annotation_changed.emit(selected_anno.object_id)
                        self.redraw_annotations()

        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Smooth zoom in/out anchored under the mouse cursor."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
            self.zoom_factor *= factor
        else:
            factor = zoom_out_factor
            self.zoom_factor *= factor

        self.scale(factor, factor)
        event.accept()

    def fit_image(self) -> None:
        """Fit entire image within current view viewport."""
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0

    def zoom_to(self, scale: float) -> None:
        """Set zoom scale relative to fit."""
        self.resetTransform()
        self.scale(scale, scale)
        self.zoom_factor = scale

    def set_mode(self, mode: str) -> None:
        """Switch canvas interaction mode."""
        self.current_mode = mode
        if mode in [MODE_POINT_POS, MODE_POINT_NEG]:
            self.setCursor(Qt.CrossCursor)
        elif mode == MODE_BOX:
            self.setCursor(Qt.CrossCursor)
        elif mode == MODE_POLYGON:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def select_object(self, object_id: str) -> None:
        """Select an object by ID and redraw canvas overlays."""
        self.active_annotation_id = object_id
        self.redraw_annotations()

    def clear_active_prompts(self) -> None:
        self.active_prompt_points.clear()
        self.in_progress_polygon_points.clear()
        self.redraw_annotations()

    # ---------------- Helpers ----------------

    def _find_active_annotation(self) -> Optional[PolygonAnnotation]:
        if not self.active_annotation_id:
            return None
        for a in self.current_annotations:
            if a.object_id == self.active_annotation_id:
                return a
        return None

    def _find_annotation_at(self, x: float, y: float) -> Optional[PolygonAnnotation]:
        """Hit test annotations containing point (x, y)."""
        pt = QPointF(x, y)
        for a in reversed(self.current_annotations):
            poly = QPolygonF([QPointF(px, py) for px, py in a.points])
            if poly.containsPoint(pt, Qt.OddEvenFill):
                return a
        return None
