"""Video file reader and metadata extraction utility."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np

from sam2_annotator.utils.logging_utils import logger


@dataclass
class VideoMetadata:
    filename: str
    absolute_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    codec: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def formatted_duration(self) -> str:
        """Format duration as HH:MM:SS or MM:SS."""
        total_sec = int(self.duration_seconds)
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        seconds = total_sec % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


class VideoReader:
    """Manages opening and reading video files via OpenCV."""

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path).resolve()
        self.cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None
        self._init_reader()

    def _init_reader(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0  # Fallback default
        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Attempt to decode codec FourCC
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        self.metadata = VideoMetadata(
            filename=self.video_path.name,
            absolute_path=str(self.video_path),
            total_frames=total_frames,
            fps=round(fps, 2),
            duration_seconds=round(duration_sec, 3),
            width=width,
            height=height,
            codec=codec if codec else "unknown",
        )
        logger.info("Opened video: %s (%dx%d, %d frames @ %.2f fps)",
                    self.metadata.filename, width, height, total_frames, fps)

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Seek and read a specific frame index (0-based) from video."""
        if not self.cap or not self.cap.isOpened():
            self._init_reader()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame %d from %s", frame_index, self.video_path.name)
            return None
        return frame

    def close(self) -> None:
        """Release OpenCV video capture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
