"""Tests for video frame deletion and continuous re-indexing in SAM2 Video Annotator."""

import tempfile
from pathlib import Path
import cv2
import numpy as np

from sam2_annotator.project.project_manager import ProjectManager
from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.annotation.annotation_manager import AnnotationManager
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.video.frame_extractor import FrameExtractor
from sam2_annotator.video.video_reader import VideoReader
from sam2_annotator.video.frame_cache import FrameCache
from sam2_annotator.dataset.split_manager import DatasetSplitter
from sam2_annotator.dataset.yolo_exporter import YOLOExporter
from sam2_annotator.dataset.dataset_validator import DatasetValidator


def create_synthetic_video(file_path: Path, num_frames: int = 10, width: int = 320, height: int = 240) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 20) % 256, dtype=np.uint8)
        cv2.putText(frame, f"F{i}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()


def test_single_frame_deletion_with_disk_cleanup():
    """Verify deleting a single frame deletes disk files, updates annotations, and re-indexes frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid = tmp_path / "test_clip.mp4"
        create_synthetic_video(vid, num_frames=6)

        proj_mgr = ProjectManager()
        proj_dir = tmp_path / "test_project"
        reader = VideoReader(vid)
        proj_mgr.create_project(
            project_dir=proj_dir,
            project_name="DeleteTest",
            video_metadata=reader.metadata,
            class_names=["target"],
        )
        reader.close()

        ext = FrameExtractor(vid, proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
        frames = ext.extract_frames(strategy="every_n", every_n=1)  # 6 frames
        proj_mgr.set_frames(frames)

        anno_mgr = AnnotationManager()
        # Annotate Frame 2 and Frame 4
        anno2 = PolygonAnnotation(
            object_id="obj_f2",
            frame_id=2,
            source_frame_index=1,
            class_id=0,
            class_name="target",
            points=[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)],
        )
        anno4 = PolygonAnnotation(
            object_id="obj_f4",
            frame_id=4,
            source_frame_index=3,
            class_id=0,
            class_name="target",
            points=[(50.0, 50.0), (80.0, 50.0), (80.0, 80.0), (50.0, 80.0)],
        )
        anno_mgr.add_annotation(anno2)
        anno_mgr.add_annotation(anno4)
        proj_mgr.save_project(anno_mgr)

        # Check files exist before deletion
        frame2_img = proj_mgr.frames_dir / frames[1].filename
        frame2_thumb = proj_mgr.thumbnails_dir / frames[1].thumbnail_filename
        assert frame2_img.exists()
        assert frame2_thumb.exists()

        cache = FrameCache(proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
        # Pre-cache frame 2
        _ = cache.get_frame(frames[1].filename)
        assert frames[1].filename in cache._cache

        # Delete frame 2 with disk cleanup
        deleted = proj_mgr.delete_frames(
            frame_ids=[2],
            delete_files=True,
            annotation_manager=anno_mgr,
            frame_cache=cache,
        )

        assert deleted == [2]
        assert len(proj_mgr.frames) == 5

        # Check disk files were removed
        assert not frame2_img.exists()
        assert not frame2_thumb.exists()
        assert frames[1].filename not in cache._cache

        # Check continuous re-indexing (1..5)
        for idx, f in enumerate(proj_mgr.frames):
            assert f.frame_id == idx + 1

        # Check annotations:
        # Frame 2 annotations should be gone
        assert len(anno_mgr.get_annotations_for_frame(2)) == 0
        # Previous Frame 4 should now be re-indexed to Frame 3!
        annos_f3 = anno_mgr.get_annotations_for_frame(3)
        assert len(annos_f3) == 1
        assert annos_f3[0].object_id == "obj_f4"
        assert annos_f3[0].frame_id == 3


def test_frame_deletion_without_disk_cleanup():
    """Verify deleting a frame with delete_files=False keeps image files on disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid = tmp_path / "test_clip2.mp4"
        create_synthetic_video(vid, num_frames=4)

        proj_mgr = ProjectManager()
        proj_dir = tmp_path / "keep_files_project"
        reader = VideoReader(vid)
        proj_mgr.create_project(
            project_dir=proj_dir,
            project_name="KeepFilesTest",
            video_metadata=reader.metadata,
            class_names=["item"],
        )
        reader.close()

        ext = FrameExtractor(vid, proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
        frames = ext.extract_frames(strategy="every_n", every_n=1)
        proj_mgr.set_frames(frames)
        proj_mgr.save_project()

        target_file = proj_mgr.frames_dir / frames[0].filename
        assert target_file.exists()

        # Delete frame 1 without deleting files
        deleted = proj_mgr.delete_frames(frame_ids=[1], delete_files=False)
        assert deleted == [1]
        assert len(proj_mgr.frames) == 3
        # Disk file should still exist
        assert target_file.exists()
        # Remaining frame IDs should be 1..3
        assert [f.frame_id for f in proj_mgr.frames] == [1, 2, 3]


def test_batch_frame_deletion_and_yolo_export():
    """Verify deleting multiple frames and exporting the resulting YOLO dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid = tmp_path / "multi_del.mp4"
        create_synthetic_video(vid, num_frames=10)

        proj_mgr = ProjectManager()
        proj_dir = tmp_path / "batch_del_project"
        reader = VideoReader(vid)
        proj_mgr.create_project(
            project_dir=proj_dir,
            project_name="BatchDelTest",
            video_metadata=reader.metadata,
            class_names=["target"],
        )
        reader.close()

        ext = FrameExtractor(vid, proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
        frames = ext.extract_frames(strategy="every_n", every_n=1)  # 10 frames
        proj_mgr.set_frames(frames)

        anno_mgr = AnnotationManager()
        # Annotate frame 1, 3, 5
        for fid in [1, 3, 5]:
            anno = PolygonAnnotation(
                object_id=f"obj_f{fid}",
                frame_id=fid,
                source_frame_index=fid - 1,
                class_id=0,
                class_name="target",
                points=[(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)],
            )
            anno_mgr.add_annotation(anno)

        proj_mgr.save_project(anno_mgr)

        # Batch delete frames 2 and 4 (unannotated frames)
        deleted = proj_mgr.delete_frames(
            frame_ids=[2, 4],
            delete_files=True,
            annotation_manager=anno_mgr,
        )
        assert deleted == [2, 4]
        assert len(proj_mgr.frames) == 8

        # Frame IDs must be continuous 1..8
        assert [f.frame_id for f in proj_mgr.frames] == list(range(1, 9))

        # Check annotations after deletion:
        # Original f1 -> still f1
        # Original f3 -> now f2 (since f2 deleted)
        # Original f5 -> now f3 (since f2 and f4 deleted)
        assert len(anno_mgr.get_annotations_for_frame(1)) == 1
        assert len(anno_mgr.get_annotations_for_frame(2)) == 1
        assert len(anno_mgr.get_annotations_for_frame(3)) == 1
        assert len(anno_mgr.get_annotations_for_frame(4)) == 0  # unannotated

        proj_mgr.save_project(anno_mgr)

        # Export dataset
        classes = [ClassItem(id=0, name="target")]
        split_dict = DatasetSplitter.split_frames(proj_mgr.frames, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
        export_out = tmp_path / "exported_yolo"

        exporter = YOLOExporter(export_out, classes, proj_mgr.frames_dir)
        res = exporter.export_dataset(
            split_dict=split_dict,
            annotations_by_frame=anno_mgr.frame_annotations,
            include_null_frames=True,
            export_masks=False,
            export_previews=False,
            create_zip=False,
        )

        assert res["status"] == "success"
        assert res["total_images"] == 8
        assert res["annotated_images"] == 3
        assert res["null_images"] == 5

        # Validate with DatasetValidator
        validator = DatasetValidator(export_out)
        val_report = validator.validate()
        assert val_report["valid"] is True
        assert val_report["stats"]["images_count"] == 8
        assert val_report["stats"]["labels_count"] == 8
        assert val_report["stats"]["annotated_images_count"] == 3
        assert val_report["stats"]["null_images_count"] == 5
