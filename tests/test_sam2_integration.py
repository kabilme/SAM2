"""Isolated integration test for SAM 2 adapter per Section 109 of specifications."""

from pathlib import Path
import pytest
import numpy as np
import cv2

from sam2_annotator.models.sam2_adapter import SAM2AdapterInterface, SAM2LocalAdapter, MockSAM2Adapter
from sam2_annotator.annotation.mask_to_polygon import mask_to_polygons


def test_mock_sam2_integration_pipeline():
    """Verify end-to-end SAM 2 pipeline criteria using MockSAM2Adapter:
    1. Model can initialize.
    2. Model can load a test image/frame.
    3. A simple prompt can be submitted.
    4. A mask can be returned.
    5. Mask dimensions can be retrieved.
    6. Mask can be converted to NumPy.
    7. Polygon extraction succeeds.
    """
    # 1. Model can initialize
    adapter = MockSAM2Adapter()
    init_ok = adapter.load_model(device="cpu")
    assert init_ok is True
    assert adapter.is_loaded() is True

    # 2. Model can load a test image/frame
    h, w = 480, 640
    test_image = np.full((h, w, 3), 100, dtype=np.uint8)
    cv2.circle(test_image, (320, 240), 50, (200, 200, 200), -1)

    # 3. A simple prompt can be submitted
    prompt_point = [(320.0, 240.0)]
    mask, conf = adapter.segment_with_points(test_image, positive_points=prompt_point)

    # 4. A mask can be returned
    assert mask is not None

    # 5. Mask dimensions can be retrieved
    mask_h, mask_w = mask.shape[:2]
    assert mask_h == h
    assert mask_w == w

    # 6. Mask can be converted to NumPy
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0

    # 7. Polygon extraction succeeds
    polygons = mask_to_polygons(mask, tolerance_ratio=0.005, min_area=20.0)
    assert len(polygons) >= 1
    assert len(polygons[0]) >= 3  # Valid polygon has at least 3 vertices


@pytest.mark.skipif(
    not (Path("sam2.1_hiera_tiny.pt").exists() or Path("sam2.1_t.pt").exists()),
    reason="SAM model checkpoint not present locally",
)
def test_real_sam_local_adapter_pipeline():
    """Verify Section 109 criteria with real SAM model checkpoint (sam2.1_hiera_tiny.pt)."""
    # 1. Model can initialize
    ckpt = "sam2.1_hiera_tiny.pt" if Path("sam2.1_hiera_tiny.pt").exists() else "sam2.1_t.pt"
    adapter = SAM2LocalAdapter()
    init_ok = adapter.load_model(checkpoint_path=ckpt, device="cpu")
    assert init_ok is True
    assert adapter.is_loaded() is True

    # 2. Model can load a test image/frame
    h, w = 360, 640
    test_image = np.full((h, w, 3), 40, dtype=np.uint8)
    # Draw a distinct bright rectangle
    cv2.rectangle(test_image, (200, 100), (440, 260), (220, 220, 220), -1)

    # 3. A simple prompt can be submitted
    prompt_point = [(320.0, 180.0)]
    mask, conf = adapter.segment_with_points(test_image, positive_points=prompt_point)

    # 4. A mask can be returned
    assert mask is not None

    # 5. Mask dimensions match original image
    mask_h, mask_w = mask.shape[:2]
    assert mask_h == h
    assert mask_w == w

    # 6. Mask is NumPy array with valid binary elements
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0
    assert conf is not None and conf > 0.0

    # 7. Polygon extraction succeeds
    polygons = mask_to_polygons(mask, tolerance_ratio=0.005, min_area=50.0)
    assert len(polygons) >= 1
    assert len(polygons[0]) >= 3
