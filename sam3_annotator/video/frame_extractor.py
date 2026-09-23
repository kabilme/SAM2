"""Video frame extraction module supporting multiple sampling strategies."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import cv2
import numpy as np

from sam3_annotator.video.video_reader import VideoReader, VideoMetadata
from sam3_annotator.utils.image_utils import save_image_bgr, generate_thumbnail
from sam3_annotator.utils.logging_utils import logger


@dataclass
class FrameMetadata:
    frame_id: int                    # Sequential 1-based index in project
    source_frame_index: int          # Frame index in original video (0-based)
    timestamp_seconds: float
    filename: str
    width: int
    height: int
    thumbnail_filename: str = ""
    review_status: str = "unreviewed" # "unreviewed", "annotated", "reviewed", "negative", "rejected"
    is_keyframe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameMetadata":
        return cls(**data)


def sanitize_filename_stem(name: str) -> str:
    """Sanitize video name to be safe across Windows and Linux filesystem paths."""
    stem = Path(name).stem
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', stem)
    return cleaned.strip('_') or "video"


class FrameExtractor:
    """Extracts frames from video based on sampling parameters."""

    def __init__(self, video_path: Path, output_dir: Path, thumbnails_dir: Optional[Path] = None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.thumbnails_dir = Path(thumbnails_dir) if thumbnails_dir else self.output_dir / "thumbnails"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    def determine_indices(
        self,
        total_frames: int,
        fps: float,
        strategy: str = "every_n",
        every_n: int = 10,
        interval_seconds: float = 1.0,
        fixed_count: int = 100,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> List[int]:
        """Compute the list of 0-based frame indices to extract."""
        if total_frames <= 0:
            return []

        start = max(0, start_frame)
        end = min(total_frames, end_frame if end_frame is not None else total_frames)
        if start >= end:
            return []

        if strategy == "every_frame":
            return list(range(start, end))

        elif strategy == "every_n":
            step = max(1, every_n)
            return list(range(start, end, step))

        elif strategy == "interval_seconds":
            step = max(1, int(round(fps * max(0.01, interval_seconds))))
            return list(range(start, end, step))

        elif strategy == "fixed_count":
            count = max(1, min(fixed_count, end - start))
            indices = np.linspace(start, end - 1, count, dtype=int)
            return sorted(list(set(indices.tolist())))

        elif strategy == "frame_range":
            step = max(1, every_n)
            return list(range(start, end, step))

        else:
            step = max(1, every_n)
            return list(range(start, end, step))

    def extract_frames(
        self,
        strategy: str = "every_n",
        every_n: int = 10,
        interval_seconds: float = 1.0,
        fixed_count: int = 100,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        jpeg_quality: int = 95,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[FrameMetadata]:
        """Execute frame extraction and return list of FrameMetadata."""
        video_reader = VideoReader(self.video_path)
        meta = video_reader.metadata
        if not meta or meta.total_frames <= 0:
            video_reader.close()
            raise ValueError(f"Cannot extract frames from invalid video: {self.video_path}")

        target_indices = self.determine_indices(
            total_frames=meta.total_frames,
            fps=meta.fps,
            strategy=strategy,
            every_n=every_n,
            interval_seconds=interval_seconds,
            fixed_count=fixed_count,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        stem = sanitize_filename_stem(self.video_path.name)
        extracted: List[FrameMetadata] = []
        total_targets = len(target_indices)

        logger.info("Extracting %d frames using strategy '%s' from %s",
                    total_targets, strategy, self.video_path.name)

        try:
            for idx, source_idx in enumerate(target_indices):
                if is_cancelled and is_cancelled():
                    logger.info("Frame extraction cancelled by user at index %d", idx)
                    break

                frame = video_reader.read_frame(source_idx)
                if frame is None:
                    continue

                frame_id = idx + 1
                frame_filename = f"{stem}_frame_{frame_id:08d}.jpg"
                thumb_filename = f"{stem}_thumb_{frame_id:08d}.jpg"

                frame_path = self.output_dir / frame_filename
                thumb_path = self.thumbnails_dir / thumb_filename

                # Save original-resolution frame
                save_image_bgr(frame_path, frame, quality=jpeg_quality)

                # Save thumbnail
                thumb = generate_thumbnail(frame, max_size=(160, 100))
                save_image_bgr(thumb_path, thumb, quality=85)

                timestamp = round(float(source_idx) / meta.fps, 3)
                frame_meta = FrameMetadata(
                    frame_id=frame_id,
                    source_frame_index=source_idx,
                    timestamp_seconds=timestamp,
                    filename=frame_filename,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    thumbnail_filename=thumb_filename,
                    review_status="unreviewed",
                )
                extracted.append(frame_meta)

                if progress_callback:
                    progress_callback(
                        frame_id,
                        total_targets,
                        f"Extracted frame {frame_id}/{total_targets} (source #{source_idx})",
                    )
        finally:
            video_reader.close()

        logger.info("Finished extracting %d frames to %s", len(extracted), self.output_dir)
        return extracted
