"""Unit tests for Keyframe toggle, timeline markers, and visual indicators."""

from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.video.frame_cache import FrameCache
from sam2_annotator.ui.frame_timeline import FrameTimeline, KeyframeTimelineSlider
from sam2_annotator.ui.video_panel import VideoPanel


@pytest.fixture
def sample_frames():
    return [
        FrameMetadata(frame_id=1, source_frame_index=0, timestamp_seconds=0.0, filename="frame_0001.jpg", width=640, height=480, is_keyframe=True),
        FrameMetadata(frame_id=2, source_frame_index=10, timestamp_seconds=0.33, filename="frame_0002.jpg", width=640, height=480, is_keyframe=False),
        FrameMetadata(frame_id=3, source_frame_index=20, timestamp_seconds=0.66, filename="frame_0003.jpg", width=640, height=480, is_keyframe=True),
    ]


def test_timeline_keyframe_toggle_and_markers(sample_frames):
    app = QApplication.instance() or QApplication([])

    timeline = FrameTimeline()
    timeline.set_frames(sample_frames)

    # Verify slider markers populated
    assert 1 in timeline.slider.keyframe_indices
    assert 3 in timeline.slider.keyframe_indices
    assert 2 not in timeline.slider.keyframe_indices

    # On frame 1 (Keyframe)
    timeline.set_current_frame(1)
    assert timeline.keyframe_btn.isChecked() is True
    assert "[ON]" in timeline.keyframe_btn.text()
    assert "★ Keyframe" in timeline.status_badge.text()

    # Move to frame 2 (Not a keyframe)
    timeline.set_current_frame(2)
    assert timeline.keyframe_btn.isChecked() is False
    assert "☆ Keyframe" in timeline.keyframe_btn.text()
    assert "★ Keyframe" not in timeline.status_badge.text()

    # Click keyframe button on frame 2
    emitted = []
    timeline.keyframe_toggled.connect(lambda fid, kf: emitted.append((fid, kf)))

    timeline.keyframe_btn.click()

    assert len(emitted) == 1
    assert emitted[0] == (2, True)
    assert sample_frames[1].is_keyframe is True
    assert 2 in timeline.slider.keyframe_indices
    assert timeline.keyframe_btn.isChecked() is True
    assert "[ON]" in timeline.keyframe_btn.text()


def test_video_panel_keyframe_filter(tmp_path, sample_frames):
    app = QApplication.instance() or QApplication([])

    fc = FrameCache(tmp_path / "frames", tmp_path / "thumbs")
    vp = VideoPanel(fc)
    vp.set_frames(sample_frames)

    # Initial: All Frames shows 3
    assert vp.list_widget.count() == 3

    # Check keyframe item has star in text
    item0 = vp.list_widget.item(0)
    assert "★" in item0.text()

    item1 = vp.list_widget.item(1)
    assert "★" not in item1.text()

    # Filter: Keyframes Only
    idx = vp.filter_combo.findText("Keyframes Only")
    assert idx >= 0
    vp.filter_combo.setCurrentIndex(idx)

    # Should only show 2 items (frames 1 and 3)
    assert vp.list_widget.count() == 2


def test_annotation_manager_tracker_keyframe_sync(sample_frames):
    from sam2_annotator.annotation.annotation_manager import AnnotationManager

    am = AnnotationManager()
    assert hasattr(am, "tracker")
    assert am.tracker is not None

    # Test marking keyframes
    am.tracker.mark_keyframe(1, True)
    am.tracker.mark_keyframe(2, False)
    assert am.tracker.is_keyframe(1) is True
    assert am.tracker.is_keyframe(2) is False

    am.tracker.mark_keyframe(1, False)
    assert am.tracker.is_keyframe(1) is False


def test_timeline_update_keyframe_marker(sample_frames):
    app = QApplication.instance() or QApplication([])

    timeline = FrameTimeline()
    timeline.set_frames(sample_frames)
    timeline.set_current_frame(2)

    assert 2 not in timeline.slider.keyframe_indices
    assert timeline.keyframe_btn.isChecked() is False

    # External toggle via update_keyframe_marker
    timeline.update_keyframe_marker(2, True)

    assert 2 in timeline.slider.keyframe_indices
    assert timeline.keyframe_btn.isChecked() is True
    assert "[ON]" in timeline.keyframe_btn.text()

    timeline.update_keyframe_marker(2, False)
    assert 2 not in timeline.slider.keyframe_indices
    assert timeline.keyframe_btn.isChecked() is False


def test_annotation_manager_get_annotation_by_id():
    from sam2_annotator.annotation.annotation_manager import AnnotationManager
    from sam2_annotator.annotation.polygon import PolygonAnnotation

    am = AnnotationManager()
    anno1 = PolygonAnnotation(
        object_id="obj_1",
        frame_id=1,
        source_frame_index=0,
        class_id=0,
        class_name="car",
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    anno2 = PolygonAnnotation(
        object_id="obj_2",
        frame_id=1,
        source_frame_index=0,
        class_id=1,
        class_name="pedestrian",
        points=[(20, 20), (30, 20), (30, 30), (20, 30)],
    )
    am.add_annotation(anno1)
    am.add_annotation(anno2)

    assert am.get_annotation_by_id(1, "obj_1") == anno1
    assert am.get_annotation_by_id(1, "obj_2") == anno2
    assert am.get_annotation_by_id(1, "obj_3") is None
    assert am.get_annotation_by_id(2, "obj_1") is None


