"""Unit tests for customizable frame propagation features."""

from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

from sam2_annotator.config.config import AppConfig, ModelConfig
from sam2_annotator.ui.properties_panel import PropertiesPanel
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation


@pytest.fixture
def sample_frames():
    return [
        FrameMetadata(frame_id=1, source_frame_index=0, timestamp_seconds=0.0, filename="f1.jpg", width=640, height=480, is_keyframe=True),
        FrameMetadata(frame_id=2, source_frame_index=1, timestamp_seconds=0.03, filename="f2.jpg", width=640, height=480, is_keyframe=False),
        FrameMetadata(frame_id=3, source_frame_index=2, timestamp_seconds=0.06, filename="f3.jpg", width=640, height=480, is_keyframe=False),
        FrameMetadata(frame_id=4, source_frame_index=3, timestamp_seconds=0.09, filename="f4.jpg", width=640, height=480, is_keyframe=False),
        FrameMetadata(frame_id=5, source_frame_index=4, timestamp_seconds=0.12, filename="f5.jpg", width=640, height=480, is_keyframe=True),
        FrameMetadata(frame_id=6, source_frame_index=5, timestamp_seconds=0.15, filename="f6.jpg", width=640, height=480, is_keyframe=False),
        FrameMetadata(frame_id=7, source_frame_index=6, timestamp_seconds=0.18, filename="f7.jpg", width=640, height=480, is_keyframe=False),
        FrameMetadata(frame_id=8, source_frame_index=7, timestamp_seconds=0.21, filename="f8.jpg", width=640, height=480, is_keyframe=False),
    ]


def test_propagation_config_loading_and_saving(tmp_path):
    cfg = AppConfig()
    assert cfg.model.propagation_frames == 30
    assert cfg.model.propagation_prompt_type == "box"
    assert cfg.model.propagation_box_padding == 0.0

    cfg.model.propagation_frames = 45
    cfg.model.propagation_prompt_type = "box"
    cfg.model.propagation_box_padding = 0.10
    test_yaml = tmp_path / "test_config.yaml"
    cfg.save(test_yaml)

    loaded = AppConfig.load(test_yaml)
    assert loaded.model.propagation_frames == 45
    assert loaded.model.propagation_prompt_type == "box"
    assert loaded.model.propagation_box_padding == 0.10


def test_properties_panel_propagation_controls():
    app = QApplication.instance() or QApplication([])

    panel = PropertiesPanel()
    assert panel.propagation_frames == 30
    assert panel.propagation_mode == "fixed"
    assert panel.propagation_prompt_type == "box"
    assert panel.prop_frames_spin.isEnabled() is True

    # Custom frame count setting
    panel.prop_frames_spin.setValue(75)
    assert panel.propagation_frames == 75

    # Preset buttons
    panel._set_preset(10)
    assert panel.propagation_frames == 10
    assert panel.propagation_mode == "fixed"

    panel._set_preset(60)
    assert panel.propagation_frames == 60
    assert panel.propagation_mode == "fixed"

    # Switch to Until Next Keyframe
    panel.prop_mode_combo.setCurrentIndex(1)
    assert panel.propagation_mode == "next_keyframe"
    assert panel.prop_frames_spin.isEnabled() is False

    # Switch to Until End of Video
    panel.prop_mode_combo.setCurrentIndex(2)
    assert panel.propagation_mode == "end_of_video"
    assert panel.prop_frames_spin.isEnabled() is False

    # Switch back to Fixed Frame Count
    panel.prop_mode_combo.setCurrentIndex(0)
    assert panel.propagation_mode == "fixed"
    assert panel.prop_frames_spin.isEnabled() is True

    # Prompt type combo
    panel.set_default_propagation_prompt_type("point")
    assert panel.propagation_prompt_type == "point"
    panel.set_default_propagation_prompt_type("box")
    assert panel.propagation_prompt_type == "box"

    # Set default from config
    panel.set_default_propagation_frames(50)
    assert panel.propagation_frames == 50


def test_properties_panel_propagate_signal_emission():
    app = QApplication.instance() or QApplication([])

    panel = PropertiesPanel()
    anno = PolygonAnnotation(
        object_id="test-obj-123",
        frame_id=1,
        source_frame_index=0,
        class_id=0,
        class_name="person",
        points=[(10, 10), (50, 10), (50, 50), (10, 50)],
    )
    panel.set_annotations([anno], selected_id="test-obj-123")
    panel.prop_frames_spin.setValue(25)

    emitted = []
    panel.propagate_requested.connect(lambda obj_id, cnt, mode, pt: emitted.append((obj_id, cnt, mode, pt)))

    # Trigger click
    panel.propagate_btn.click()

    assert len(emitted) == 1
    assert emitted[0] == ("test-obj-123", 25, "fixed", "box")

    # Change mode to Next Keyframe
    panel.prop_mode_combo.setCurrentIndex(1)
    panel.propagate_btn.click()
    assert len(emitted) == 2
    assert emitted[1] == ("test-obj-123", 25, "next_keyframe", "box")


def test_propagation_target_frame_calculation(sample_frames):
    """Test target frames slice logic for different modes."""
    total_f = len(sample_frames)  # 8 frames (IDs 1 to 8)
    fid = 1  # Active on Frame 1

    # 1. Fixed count: 3 frames (should get frames 2, 3, 4)
    frame_count = 3
    step_count = min(max(1, frame_count), total_f - fid)
    target_fixed = sample_frames[fid : fid + step_count]
    assert [f.frame_id for f in target_fixed] == [2, 3, 4]

    # 2. Fixed count: 10 frames (exceeds remaining 7, should get all remaining frames 2 to 8)
    frame_count = 10
    step_count = min(max(1, frame_count), total_f - fid)
    target_clamped = sample_frames[fid : fid + step_count]
    assert [f.frame_id for f in target_clamped] == [2, 3, 4, 5, 6, 7, 8]

    # 3. Until Next Keyframe:
    # Frame 1 is keyframe. Next keyframe is Frame 5 (index 4).
    next_kf_idx = None
    for idx in range(fid, total_f):
        if sample_frames[idx].is_keyframe:
            next_kf_idx = idx
            break
    assert next_kf_idx == 4  # Frame 5
    target_kf = sample_frames[fid : next_kf_idx + 1]
    assert [f.frame_id for f in target_kf] == [2, 3, 4, 5]

    # 4. Until End of Video:
    target_end = sample_frames[fid : total_f]
    assert [f.frame_id for f in target_end] == [2, 3, 4, 5, 6, 7, 8]


def test_video_service_box_prompt_propagation(tmp_path, sample_frames):
    """Verify that SAM2VideoService uses box prompt for propagation."""
    import numpy as np
    import cv2
    from sam2_annotator.models.sam2_adapter import MockSAM2Adapter
    from sam2_annotator.models.sam2_video_service import SAM2VideoService
    from sam2_annotator.video.frame_cache import FrameCache

    frames_dir = tmp_path / "frames"
    thumbs_dir = tmp_path / "thumbs"
    frames_dir.mkdir()
    thumbs_dir.mkdir()

    # Create dummy images for sample frames
    for f in sample_frames:
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / f.filename), dummy_img)

    frame_cache = FrameCache(frames_dir, thumbs_dir)
    adapter = MockSAM2Adapter()
    adapter.load_model()
    service = SAM2VideoService(adapter, frame_cache)

    initial_anno = PolygonAnnotation(
        object_id="obj_track_test",
        frame_id=1,
        source_frame_index=0,
        class_id=0,
        class_name="car",
        points=[(50, 50), (150, 50), (150, 150), (50, 150)],
        bounding_box=(50.0, 50.0, 150.0, 150.0),
    )

    # Propagate across 3 subsequent frames with exact box prompt (no margin)
    results = service.propagate_object(
        initial_annotation=initial_anno,
        target_frames=sample_frames[1:4],
        prompt_type="box",
        box_padding_ratio=0.0,
        source_frame_filename=sample_frames[0].filename,
    )

    assert len(results) == 3
    for r in results:
        assert r.source == "sam2_box_track"
        assert r.tracking_status == "tracked"
        assert r.object_id == "obj_track_test"
        assert r.bounding_box != (0.0, 0.0, 0.0, 0.0)
        assert len(r.points) >= 3

