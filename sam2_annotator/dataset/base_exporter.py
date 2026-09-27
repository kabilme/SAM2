"""Base dataset exporter interface and common geometric/export utilities."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import shutil
import numpy as np

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.utils.logging_utils import logger


class BaseDatasetExporter(ABC):
    """Abstract base class for all dataset and artifact exporters."""

    def __init__(
        self,
        output_dir: Path,
        classes: List[ClassItem],
        frames_dir: Path,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.classes = classes
        self.frames_dir = Path(frames_dir).resolve()
        self.class_map = {c.id: c.name for c in classes}
        self.class_color_map = {
            c.id: tuple(c.color_rgb) if c.color_rgb else (0, 255, 0)
            for c in classes
        }

    @abstractmethod
    def export(
        self,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        create_zip: bool = True,
        include_null_frames: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute full dataset export process.

        Returns:
            Dictionary containing export summary metrics, output directory, and status.
        """
        pass

    @staticmethod
    def get_polygon_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float, float, float, float, float]:
        """Calculate bounding box properties from a list of polygon vertices.

        Returns:
            (xmin, ymin, xmax, ymax, width, height, center_x, center_y)
        """
        if not points:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        return xmin, ymin, xmax, ymax, w, h, cx, cy

    @staticmethod
    def get_polygon_area(points: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using Green's theorem / shoelace formula."""
        if len(points) < 3:
            return 0.0
        x = np.array([pt[0] for pt in points])
        y = np.array([pt[1] for pt in points])
        return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    @staticmethod
    def create_zip_archive(
        source_dir: Path,
        zip_name: str = "dataset.zip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        total_items: int = 100,
    ) -> Optional[str]:
        """Create a ZIP archive of the exported directory."""
        try:
            if progress_callback:
                progress_callback(total_items, total_items, f"Creating ZIP archive ({zip_name})...")
            base_zip_name = source_dir.parent / Path(zip_name).stem
            created_zip = shutil.make_archive(str(base_zip_name), "zip", source_dir)
            logger.info("Created dataset archive: %s", created_zip)
            return str(created_zip)
        except Exception as e:
            logger.error("Failed to create ZIP archive: %s", e)
            return None
