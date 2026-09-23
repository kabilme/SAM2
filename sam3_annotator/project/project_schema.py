"""Project metadata schema and class definitions."""

import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from sam3_annotator.video.video_reader import VideoMetadata
from sam3_annotator.video.frame_extractor import FrameMetadata
from sam3_annotator.config.config import AppConfig


@dataclass
class ClassItem:
    id: int
    name: str
    color_rgb: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassItem":
        return cls(**data)


@dataclass
class ProjectData:
    schema_version: int = 1
    application_version: str = "1.0.0"
    project_name: str = "Untitled Project"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    video: Optional[Dict[str, Any]] = None
    videos: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    frames: List[Dict[str, Any]] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectData":
        videos = data.get("videos", [])
        video = data.get("video")
        if not videos and video:
            videos = [video]
        return cls(
            schema_version=data.get("schema_version", 1),
            application_version=data.get("application_version", "1.0.0"),
            project_name=data.get("project_name", "Untitled Project"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            video=video,
            videos=videos,
            classes=data.get("classes", []),
            frames=data.get("frames", []),
            settings=data.get("settings", {}),
            dataset=data.get("dataset", {}),
        )
