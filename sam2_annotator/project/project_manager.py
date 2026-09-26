"""Persistent project manager handling project lifecycles, atomic saves, and recovery."""

import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from sam2_annotator.project.project_schema import ProjectData, ClassItem
from sam2_annotator.video.video_reader import VideoMetadata
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.annotation_manager import AnnotationManager
from sam2_annotator.annotation.annotation_serializer import AnnotationSerializer
from sam2_annotator.utils.image_utils import get_deterministic_color
from sam2_annotator.utils.logging_utils import logger


class ProjectManager:
    """Manages project persistence, atomic saves, file structure, and crash recovery."""

    def __init__(self, project_dir: Optional[Path] = None):
        self.project_dir: Optional[Path] = Path(project_dir) if project_dir else None
        self.data: ProjectData = ProjectData()
        self.classes: List[ClassItem] = []
        self.frames: List[FrameMetadata] = []
        self.is_dirty: bool = False

    @property
    def frames_dir(self) -> Path:
        if not self.project_dir:
            raise ValueError("Project directory not set.")
        return self.project_dir / "frames"

    @property
    def thumbnails_dir(self) -> Path:
        if not self.project_dir:
            raise ValueError("Project directory not set.")
        return self.project_dir / "thumbnails"

    @property
    def annotations_dir(self) -> Path:
        if not self.project_dir:
            raise ValueError("Project directory not set.")
        return self.project_dir / "annotations"

    @property
    def masks_dir(self) -> Path:
        if not self.project_dir:
            raise ValueError("Project directory not set.")
        return self.project_dir / "masks"

    @property
    def export_dir(self) -> Path:
        if not self.project_dir:
            raise ValueError("Project directory not set.")
        return self.project_dir / "export"

    def create_project(
        self,
        project_dir: Path,
        project_name: str,
        video_metadata: Any,
        class_names: List[str],
        settings: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Initialize a new project folder structure and schema."""
        self.project_dir = Path(project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.classes = []
        for i, name in enumerate(class_names):
            color = list(get_deterministic_color(i))
            self.classes.append(ClassItem(id=i, name=name.strip(), color_rgb=color))

        if isinstance(video_metadata, list):
            videos_list = [v.to_dict() if hasattr(v, "to_dict") else v for v in video_metadata]
            primary_video = videos_list[0] if videos_list else None
        elif hasattr(video_metadata, "to_dict"):
            primary_video = video_metadata.to_dict()
            videos_list = [primary_video]
        else:
            primary_video = video_metadata
            videos_list = [video_metadata] if video_metadata else []

        self.data = ProjectData(
            project_name=project_name,
            video=primary_video,
            videos=videos_list,
            classes=[c.to_dict() for c in self.classes],
            frames=[],
            settings=settings or {},
        )
        self.frames = []
        self.is_dirty = True
        return self.save_project()

    def add_video_metadata(self, new_videos: Any) -> None:
        """Add metadata for one or more additional videos to the project."""
        if not isinstance(new_videos, list):
            new_videos = [new_videos]
        existing_vids = list(self.data.videos or [])
        for v in new_videos:
            v_dict = v.to_dict() if hasattr(v, "to_dict") else v
            existing_vids.append(v_dict)
        self.data.videos = existing_vids
        if not self.data.video and existing_vids:
            self.data.video = existing_vids[0]
        self.is_dirty = True

    def save_project(self, annotation_manager: Optional[AnnotationManager] = None) -> bool:
        """Atomically persist project.json with a backup file."""
        if not self.project_dir:
            return False

        try:
            self.data.updated_at = time.time()
            self.data.classes = [c.to_dict() for c in self.classes]
            self.data.frames = [f.to_dict() for f in self.frames]

            # Save individual frame annotations if manager provided
            if annotation_manager:
                for frame in self.frames:
                    annos = annotation_manager.get_annotations_for_frame(frame.frame_id)
                    AnnotationSerializer.save_frame_annotations(
                        self.annotations_dir, frame.filename, annos
                    )

            project_file = self.project_dir / "project.json"
            backup_file = self.project_dir / "project.backup.json"
            temp_file = self.project_dir / "project.json.tmp"

            # 1. Write temporary file and validate JSON
            data_dict = self.data.to_dict()
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2)

            # Verify it's readable
            with open(temp_file, "r", encoding="utf-8") as f:
                _ = json.load(f)

            # 2. Create backup if current file exists
            if project_file.exists():
                shutil.copy2(project_file, backup_file)

            # 3. Atomically replace
            temp_file.replace(project_file)
            self.is_dirty = False
            logger.info("Project saved successfully: %s", project_file)
            return True
        except Exception as e:
            logger.error("Failed to save project: %s", e)
            return False

    def load_project(
        self,
        project_dir: Path,
        annotation_manager: Optional[AnnotationManager] = None,
        use_backup: bool = False,
    ) -> bool:
        """Load an existing project, falling back to backup if corrupted."""
        self.project_dir = Path(project_dir).resolve()
        project_file = self.project_dir / "project.json"
        backup_file = self.project_dir / "project.backup.json"

        target_file = backup_file if use_backup else project_file
        if not target_file.exists():
            if not use_backup and backup_file.exists():
                logger.warning("project.json not found, falling back to project.backup.json")
                target_file = backup_file
            else:
                logger.error("No project file found in %s", self.project_dir)
                return False

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.data = ProjectData.from_dict(data)
            self.classes = [ClassItem.from_dict(c) for c in self.data.classes]
            self.frames = [FrameMetadata.from_dict(f) for f in self.data.frames]

            # Load frame annotations into annotation manager if provided
            if annotation_manager:
                annotation_manager.frame_annotations.clear()
                for frame in self.frames:
                    annos = AnnotationSerializer.load_frame_annotations(
                        self.annotations_dir, frame.filename
                    )
                    if annos:
                        annotation_manager.frame_annotations[frame.frame_id] = annos

            self.is_dirty = False
            logger.info("Loaded project '%s' with %d frames, %d classes",
                        self.data.project_name, len(self.frames), len(self.classes))
            return True
        except Exception as e:
            logger.error("Error loading project from %s: %s", target_file, e)
            # Try backup if we haven't already
            if not use_backup and backup_file.exists():
                logger.info("Attempting recovery using backup file...")
                return self.load_project(self.project_dir, annotation_manager, use_backup=True)
            return False

    def add_class(self, name: str) -> ClassItem:
        """Add a new class with auto-incremented ID."""
        new_id = len(self.classes)
        color = list(get_deterministic_color(new_id))
        item = ClassItem(id=new_id, name=name.strip(), color_rgb=color)
        self.classes.append(item)
        self.is_dirty = True
        return item

    def rename_class(self, class_id: int, new_name: str) -> bool:
        """Rename an existing class."""
        for c in self.classes:
            if c.id == class_id:
                c.name = new_name.strip()
                self.is_dirty = True
                return True
        return False

    def remove_class(self, class_id: int) -> bool:
        """Remove class if safe (re-indexing remaining classes)."""
        idx = next((i for i, c in enumerate(self.classes) if c.id == class_id), None)
        if idx is not None:
            self.classes.pop(idx)
            # Reindex classes
            for i, c in enumerate(self.classes):
                c.id = i
            self.is_dirty = True
            return True
        return False

    def set_frames(self, frames: List[FrameMetadata]) -> None:
        self.frames = frames
        self.is_dirty = True

    def delete_frames(
        self,
        frame_ids: List[int],
        delete_files: bool = True,
        annotation_manager: Optional[AnnotationManager] = None,
        frame_cache: Optional[Any] = None,
    ) -> List[int]:
        """Delete one or more frames from the project.

        Args:
            frame_ids: List of project frame IDs (1-based) to delete.
            delete_files: If True, delete extracted image, thumbnail, and annotation files from disk.
            annotation_manager: Optional AnnotationManager to purge and re-index annotations.
            frame_cache: Optional FrameCache to evict deleted images.

        Returns:
            List of successfully deleted original frame IDs.
        """
        if not frame_ids or not self.frames:
            return []

        target_set = set(frame_ids)
        deleted_frames = [f for f in self.frames if f.frame_id in target_set]
        if not deleted_frames:
            return []

        # 1. Clean up files and cache for deleted frames
        for f in deleted_frames:
            if frame_cache:
                try:
                    frame_cache.evict(f.filename, f.thumbnail_filename)
                except Exception as e:
                    logger.warning("Error evicting %s from frame cache: %s", f.filename, e)

            if delete_files:
                # Delete frame image
                if self.frames_dir:
                    img_path = self.frames_dir / f.filename
                    try:
                        img_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Could not delete frame image %s: %s", img_path, e)

                # Delete thumbnail
                if self.thumbnails_dir:
                    thumb_path = self.thumbnails_dir / f.thumbnail_filename
                    try:
                        thumb_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Could not delete thumbnail %s: %s", thumb_path, e)

                # Delete annotation json if exists
                if self.annotations_dir:
                    stem = Path(f.filename).stem
                    json_path = self.annotations_dir / f"{stem}.json"
                    try:
                        json_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Could not delete annotation file %s: %s", json_path, e)

        # 2. Build mapping from old frame_id to new continuous frame_id (1..N)
        remaining_frames: List[FrameMetadata] = []
        old_to_new: Dict[int, int] = {}
        new_id = 1
        for f in self.frames:
            if f.frame_id in target_set:
                continue
            old_to_new[f.frame_id] = new_id
            f.frame_id = new_id
            new_id += 1
            remaining_frames.append(f)

        self.frames = remaining_frames

        # 3. Update annotations in annotation_manager if provided
        if annotation_manager:
            new_frame_annotations: Dict[int, List[Any]] = {}
            for old_id, annos in list(annotation_manager.frame_annotations.items()):
                if old_id in target_set:
                    continue
                new_fid = old_to_new.get(old_id, old_id)
                for a in annos:
                    a.frame_id = new_fid
                new_frame_annotations[new_fid] = annos
            annotation_manager.frame_annotations = new_frame_annotations
            # Clear undo/redo stacks since history states map to old frame indices
            annotation_manager.undo_stack.clear()
            annotation_manager.redo_stack.clear()

        self.is_dirty = True
        logger.info(
            "Deleted %d frame(s) from project (IDs: %s). %d frames remaining.",
            len(deleted_frames),
            [f.frame_id for f in deleted_frames],
            len(self.frames),
        )
        return [f.frame_id for f in deleted_frames]

    def delete_frame(
        self,
        frame_id: int,
        delete_files: bool = True,
        annotation_manager: Optional[AnnotationManager] = None,
        frame_cache: Optional[Any] = None,
    ) -> bool:
        """Delete a single frame from the project."""
        deleted = self.delete_frames([frame_id], delete_files, annotation_manager, frame_cache)
        return len(deleted) > 0
