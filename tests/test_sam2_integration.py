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


@pytest.mark.skipif(
    not (Path("sam2.1_hiera_tiny.pt").exists() or Path("sam2.1_t.pt").exists()),
    reason="SAM model checkpoint not present locally",
)
def test_native_video_predictor_propagation(tmp_path):
    """Verify that SAM2VideoService uses native Meta SAM 2.1 Video Predictor with spatio-temporal memory."""
    from sam2_annotator.models.sam2_video_service import SAM2VideoService
    from sam2_annotator.video.frame_cache import FrameCache
    from sam2_annotator.video.frame_extractor import FrameMetadata
    from sam2_annotator.annotation.polygon import PolygonAnnotation

    frames_dir = tmp_path / "frames"
    thumbs_dir = tmp_path / "thumbs"
    frames_dir.mkdir()
    thumbs_dir.mkdir()

    # Create 3 synthetic frames with a moving square
    h, w = 360, 640
    frame_metas = []
    for i in range(3):
        fname = f"frame_{i:04d}.jpg"
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # moving bright square
        x0, y0 = 100 + i * 15, 80 + i * 10
        cv2.rectangle(img, (x0, y0), (x0 + 80, y0 + 80), (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / fname), img)
        frame_metas.append(FrameMetadata(
            frame_id=i + 1,
            source_frame_index=i,
            timestamp_seconds=i * 0.033,
            filename=fname,
            width=w,
            height=h,
            is_keyframe=(i == 0),
        ))

    ckpt = "sam2.1_hiera_tiny.pt" if Path("sam2.1_hiera_tiny.pt").exists() else "sam2.1_t.pt"
    adapter = SAM2LocalAdapter()
    adapter.load_model(checkpoint_path=ckpt, device="cpu")
    assert adapter.is_loaded()

    # Verify get_video_predictor returns valid predictor
    vp = adapter.get_video_predictor()
    assert vp is not None

    frame_cache = FrameCache(frames_dir, thumbs_dir)
    service = SAM2VideoService(adapter, frame_cache)

    initial_anno = PolygonAnnotation(
        object_id="video_prop_obj_1",
        frame_id=1,
        source_frame_index=0,
        class_id=0,
        class_name="target",
        points=[(100, 80), (180, 80), (180, 160), (100, 160)],
        bounding_box=(100.0, 80.0, 180.0, 160.0),
    )

    # Propagate across frames 2 and 3
    results = service.propagate_object(
        initial_annotation=initial_anno,
        target_frames=frame_metas[1:],
        prompt_type="box",
        box_padding_ratio=0.0,
        source_frame_filename=frame_metas[0].filename,
    )

    assert len(results) == 2
    for r in results:
        assert r.object_id == "video_prop_obj_1"
        assert r.source == "sam2_box_track"
        assert r.tracking_status == "tracked"
        assert len(r.points) >= 3
        assert r.bounding_box != (0.0, 0.0, 0.0, 0.0)
        assert r.confidence is not None and r.confidence > 0.5

    # Check that release clears predictor
    adapter.release()
    assert adapter._video_predictor is None

