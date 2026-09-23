"""Unit tests for geometry, coordinate transformation, mask conversion, and editing."""

import pytest
import numpy as np

from sam3_annotator.utils.geometry import (
    CoordinateTransformer, calculate_polygon_area,
    compute_bounding_box, simplify_polygon, validate_polygon
)
from sam3_annotator.annotation.polygon import PolygonAnnotation
from sam3_annotator.annotation.mask_to_polygon import mask_to_polygons, polygon_to_mask
from sam3_annotator.annotation.polygon_editor import PolygonEditor


def test_coordinate_transformer():
    transformer = CoordinateTransformer(image_width=1000, image_height=500)
    pts = [(100.0, 50.0), (500.0, 250.0)]
    normalized = transformer.image_to_normalized(pts)
    assert normalized == [(0.1, 0.1), (0.5, 0.5)]

    reconstructed = transformer.normalized_to_image(normalized)
    assert reconstructed == pts


def test_polygon_geometry_and_validation():
    # Valid square
    pts = [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)]
    area = calculate_polygon_area(pts)
    assert area == 2500.0

    bbox = compute_bounding_box(pts)
    assert bbox == (10.0, 10.0, 60.0, 60.0)

    is_valid, err = validate_polygon(pts, image_width=100, image_height=100, min_area=20.0)
    assert is_valid
    assert err is None


def test_mask_to_polygon_and_roundtrip():
    # Create synthetic binary mask with a filled rectangle
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    polygons = mask_to_polygons(mask, tolerance_ratio=0.005, min_area=20.0)
    assert len(polygons) == 1
    poly = polygons[0]
    assert len(poly) >= 4

    # Rasterize back and check IoU
    raster = polygon_to_mask(poly, w, h)
    intersection = np.logical_and(mask, raster).sum()
    union = np.logical_or(mask, raster).sum()
    iou = intersection / union
    assert iou > 0.95


def test_polygon_editor():
    anno = PolygonAnnotation(
        object_id="test_obj",
        class_id=0,
        class_name="scooter",
        frame_id=1,
        source_frame_index=0,
        points=[(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
    )

    # Move vertex
    assert PolygonEditor.move_vertex(anno, 0, (15.0, 15.0))
    assert anno.points[0] == (15.0, 15.0)

    # Insert vertex
    assert PolygonEditor.insert_vertex(anno, 0, (30.0, 12.0))
    assert len(anno.points) == 5

    # Delete vertex
    assert PolygonEditor.delete_vertex(anno, 1)
    assert len(anno.points) == 4
