"""Unit tests for video reading and frame extraction logic."""

import pytest
import numpy as np
import cv2
from pathlib import Path

from sam2_annotator.video.video_reader import VideoReader, VideoMetadata
from sam2_annotator.video.frame_extractor import FrameExtractor, sanitize_filename_stem


@pytest.fixture
def dummy_video(tmp_path):
    """Generate a synthetic test video with 30 frames."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), i * 8, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


def test_video_reader_metadata(dummy_video):
    reader = VideoReader(dummy_video)
    meta = reader.metadata
    assert meta is not None
    assert meta.total_frames == 30
    assert meta.fps == 10.0
    assert meta.width == 320
    assert meta.height == 240
    assert meta.duration_seconds == 3.0

    frame = reader.read_frame(5)
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    reader.close()


def test_frame_extractor_sampling_indices(tmp_path, dummy_video):
    extractor = FrameExtractor(dummy_video, tmp_path / "frames")

    # Every 10 frames
    indices = extractor.determine_indices(total_frames=30, fps=10.0, strategy="every_n", every_n=10)
    assert indices == [0, 10, 20]

    # Fixed count 5
    indices_fixed = extractor.determine_indices(total_frames=30, fps=10.0, strategy="fixed_count", fixed_count=5)
    assert len(indices_fixed) == 5

    # Every frame
    indices_all = extractor.determine_indices(total_frames=30, fps=10.0, strategy="every_frame")
    assert len(indices_all) == 30


def test_frame_extractor_execution(tmp_path, dummy_video):
    out_dir = tmp_path / "extracted_frames"
    extractor = FrameExtractor(dummy_video, out_dir)
    frames = extractor.extract_frames(strategy="every_n", every_n=10)

    assert len(frames) == 3
    assert frames[0].frame_id == 1
    assert frames[0].source_frame_index == 0
    assert (out_dir / frames[0].filename).exists()
    assert (out_dir / "thumbnails" / frames[0].thumbnail_filename).exists()


def test_sanitize_filename():
    assert sanitize_filename_stem("My Video (1) [HD].mp4") == "My_Video__1___HD"
