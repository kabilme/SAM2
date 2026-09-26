"""Unit tests for project manager, schema, and recovery."""

import pytest
from pathlib import Path

from sam2_annotator.project.project_manager import ProjectManager
from sam2_annotator.annotation.annotation_manager import AnnotationManager
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.video.video_reader import VideoMetadata


def test_project_create_save_and_reload(tmp_path):
    p_dir = tmp_path / "test_proj"
    proj_mgr = ProjectManager()

    v_meta = VideoMetadata(
        filename="test.mp4",
        absolute_path=str(tmp_path / "test.mp4"),
        total_frames=100,
        fps=30.0,
        duration_seconds=3.33,
        width=1920,
        height=1080,
        codec="h264",
    )

    success = proj_mgr.create_project(
        project_dir=p_dir,
        project_name="TestProject",
        video_metadata=v_meta,
        class_names=["scooter", "person"],
    )
    assert success is True
    assert (p_dir / "project.json").exists()

    # Reopen
    reopened_mgr = ProjectManager()
    load_success = reopened_mgr.load_project(p_dir)
    assert load_success is True
    assert reopened_mgr.data.project_name == "TestProject"
    assert len(reopened_mgr.classes) == 2
    assert reopened_mgr.classes[0].name == "scooter"


def test_project_recovery_from_backup(tmp_path):
    p_dir = tmp_path / "recovery_proj"
    proj_mgr = ProjectManager()
    v_meta = VideoMetadata(
        filename="test.mp4", absolute_path="test.mp4", total_frames=10, fps=10.0,
        duration_seconds=1.0, width=100, height=100, codec="mp4v"
    )
    proj_mgr.create_project(p_dir, "RecoveryTest", v_meta, ["car"])
    proj_mgr.save_project() # Creates backup

    # Corrupt project.json
    (p_dir / "project.json").write_text("{ corrupt json: ", encoding="utf-8")

    # Load should fallback to backup.json
    reopened = ProjectManager()
    assert reopened.load_project(p_dir) is True
    assert reopened.data.project_name == "RecoveryTest"
