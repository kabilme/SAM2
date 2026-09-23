"""Tests for multi-video project support and unannotated null/negative frames."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from sam3_annotator.video.video_reader import VideoReader, VideoMetadata
from sam3_annotator.video.frame_extractor import FrameExtractor, FrameMetadata
from sam3_annotator.project.project_manager import ProjectManager
from sam3_annotator.project.project_schema import ClassItem
from sam3_annotator.annotation.annotation_manager import AnnotationManager
from sam3_annotator.annotation.polygon import PolygonAnnotation
from sam3_annotator.dataset.split_manager import DatasetSplitter
from sam3_annotator.dataset.yolo_exporter import YOLOExporter
from sam3_annotator.dataset.dataset_validator import DatasetValidator


def create_synthetic_video(file_path: Path, num_frames: int = 15, width: int = 320, height: int = 240, fps: float = 10.0):
    """Helper to generate a synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 15) % 255, dtype=np.uint8)
        cv2.putText(frame, f"F:{i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()


def test_multi_video_frame_extraction():
    """Verify extracting frames across multiple videos maintains continuous frame IDs and video metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid1 = tmp_path / "vid_alpha.mp4"
        vid2 = tmp_path / "vid_beta.mp4"
        create_synthetic_video(vid1, num_frames=10)
        create_synthetic_video(vid2, num_frames=12)

        frames_dir = tmp_path / "frames"
        thumbs_dir = tmp_path / "thumbs"

        # Extract from video 1
        ext1 = FrameExtractor(vid1, frames_dir, thumbs_dir)
        frames_v1 = ext1.extract_frames(strategy="every_n", every_n=2, start_frame_id=1, video_name=vid1.name)
        assert len(frames_v1) == 5
        assert frames_v1[0].frame_id == 1
        assert frames_v1[-1].frame_id == 5
        assert frames_v1[0].video_name == "vid_alpha.mp4"
        assert "vid_alpha" in frames_v1[0].filename

        # Extract from video 2 starting at frame 6
        ext2 = FrameExtractor(vid2, frames_dir, thumbs_dir)
        frames_v2 = ext2.extract_frames(strategy="every_n", every_n=2, start_frame_id=len(frames_v1) + 1, video_name=vid2.name)
        assert len(frames_v2) == 6
        assert frames_v2[0].frame_id == 6
        assert frames_v2[-1].frame_id == 11
        assert frames_v2[0].video_name == "vid_beta.mp4"
        assert "vid_beta" in frames_v2[0].filename

        # Combined checks
        all_frames = frames_v1 + frames_v2
        assert len(all_frames) == 11
        for idx, f in enumerate(all_frames):
            assert f.frame_id == idx + 1
            assert (frames_dir / f.filename).exists()
            assert (thumbs_dir / f.thumbnail_filename).exists()


def test_multi_video_project_management():
    """Verify ProjectManager handles multiple videos in project schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid1 = tmp_path / "cam_front.mp4"
        vid2 = tmp_path / "cam_rear.mp4"
        create_synthetic_video(vid1, num_frames=8)
        create_synthetic_video(vid2, num_frames=8)

        r1 = VideoReader(vid1)
        meta1 = r1.metadata
        r1.close()

        r2 = VideoReader(vid2)
        meta2 = r2.metadata
        r2.close()

        proj_mgr = ProjectManager()
        proj_dir = tmp_path / "multi_proj"
        ok = proj_mgr.create_project(
            project_dir=proj_dir,
            project_name="MultiCamDemo",
            video_metadata=[meta1, meta2],
            class_names=["car", "pedestrian"],
        )
        assert ok is True
        assert len(proj_mgr.data.videos) == 2
        assert proj_mgr.data.videos[0]["filename"] == "cam_front.mp4"
        assert proj_mgr.data.videos[1]["filename"] == "cam_rear.mp4"

        # Reload project and verify persistence
        reload_mgr = ProjectManager()
        anno_mgr = AnnotationManager()
        loaded = reload_mgr.load_project(proj_dir, anno_mgr)
        assert loaded is True
        assert len(reload_mgr.data.videos) == 2


def test_unannotated_frames_exported_as_null_frames():
    """Verify that unannotated frames are correctly exported as null frames (empty label files)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid = tmp_path / "scene.mp4"
        create_synthetic_video(vid, num_frames=10)

        frames_dir = tmp_path / "frames"
        thumbs_dir = tmp_path / "thumbs"
        ext = FrameExtractor(vid, frames_dir, thumbs_dir)
        frames = ext.extract_frames(strategy="every_n", every_n=2)  # 5 frames
        assert len(frames) == 5

        # Only annotate Frame 1 and Frame 3.
        # Frames 2, 4, 5 are left UNANNOTATED (null frames).
        anno_mgr = AnnotationManager()
        anno1 = PolygonAnnotation(
            object_id="obj_1",
            frame_id=1,
            source_frame_index=0,
            class_id=0,
            class_name="target",
            points=[(50.0, 50.0), (100.0, 50.0), (100.0, 100.0), (50.0, 100.0)],
        )
        anno3 = PolygonAnnotation(
            object_id="obj_2",
            frame_id=3,
            source_frame_index=4,
            class_id=0,
            class_name="target",
            points=[(80.0, 80.0), (120.0, 80.0), (120.0, 120.0)],
        )
        anno_mgr.add_annotation(anno1)
        anno_mgr.add_annotation(anno3)

        split_dict = DatasetSplitter.split_frames(frames, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
        export_out = tmp_path / "exported_yolo"
        classes = [ClassItem(id=0, name="target")]

        exporter = YOLOExporter(export_out, classes, frames_dir)
        res = exporter.export_dataset(
            split_dict=split_dict,
            annotations_by_frame=anno_mgr.frame_annotations,
            include_null_frames=True,
            export_masks=False,
            export_previews=False,
            create_zip=False,
        )

        assert res["status"] == "success"
        assert res["total_images"] == 5
        assert res["annotated_images"] == 2
        assert res["null_images"] == 3

        # Validate with DatasetValidator
        validator = DatasetValidator(export_out)
        val_report = validator.validate()
        assert val_report["valid"] is True
        assert val_report["stats"]["images_count"] == 5
        assert val_report["stats"]["labels_count"] == 5
        assert val_report["stats"]["annotated_images_count"] == 2
        assert val_report["stats"]["null_images_count"] == 3
        assert val_report["stats"]["objects_count"] == 2

        # Verify that unannotated frame label file is 0 bytes (empty)
        # Find an unannotated frame (e.g. frame 2)
        frame2_label = None
        for p in export_out.glob("labels/*/*.txt"):
            if "frame_00000002" in p.name:
                frame2_label = p
                break
        assert frame2_label is not None
        assert frame2_label.stat().st_size == 0  # Empty label file for null frame


def test_dynamic_add_videos_flow():
    """Simulate adding videos dynamically to an existing project (add_videos_dialog logic)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid1 = tmp_path / "initial.mp4"
        vid2 = tmp_path / "additional.mp4"
        create_synthetic_video(vid1, num_frames=6)
        create_synthetic_video(vid2, num_frames=6)

        # Initialize project with vid1
        r1 = VideoReader(vid1)
        meta1 = r1.metadata
        r1.close()

        proj_mgr = ProjectManager()
        proj_dir = tmp_path / "dynamic_proj"
        proj_mgr.create_project(
            project_dir=proj_dir,
            project_name="DynamicProj",
            video_metadata=meta1,
            class_names=["item"],
        )

        ext1 = FrameExtractor(vid1, proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
        frames1 = ext1.extract_frames(strategy="every_n", every_n=2, start_frame_id=1, video_name=vid1.name)
        proj_mgr.set_frames(frames1)
        proj_mgr.save_project()
        assert len(proj_mgr.frames) == 3

        # Now simulate add_videos_dialog with vid2
        r2 = VideoReader(vid2)
        meta2 = r2.metadata
        r2.close()

        start_id = len(proj_mgr.frames) + 1
        ext2 = FrameExtractor(vid2, proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
        frames2 = ext2.extract_frames(strategy="every_n", every_n=2, start_frame_id=start_id, video_name=vid2.name)

        proj_mgr.frames.extend(frames2)
        proj_mgr.add_video_metadata([meta2])
        proj_mgr.save_project()

        assert len(proj_mgr.frames) == 6
        assert len(proj_mgr.data.videos) == 2
        assert proj_mgr.data.videos[0]["filename"] == "initial.mp4"
        assert proj_mgr.data.videos[1]["filename"] == "additional.mp4"
        assert proj_mgr.frames[3].frame_id == 4
        assert proj_mgr.frames[3].video_name == "additional.mp4"

