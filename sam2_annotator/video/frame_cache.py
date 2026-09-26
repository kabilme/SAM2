"""LRU in-memory cache and lazy loader for extracted video frames."""

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict
import numpy as np

from sam2_annotator.utils.image_utils import load_image_bgr
from sam2_annotator.utils.logging_utils import logger


class FrameCache:
    """Bounded LRU memory cache for frame images to prevent RAM exhaustion."""

    def __init__(self, frames_dir: Path, thumbnails_dir: Optional[Path] = None, max_size: int = 100):
        self.frames_dir = Path(frames_dir)
        self.thumbnails_dir = Path(thumbnails_dir) if thumbnails_dir else self.frames_dir / "thumbnails"
        self.max_size = max(5, max_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._thumb_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get_frame(self, filename: str) -> Optional[np.ndarray]:
        """Fetch frame from LRU cache or lazily load from disk."""
        if filename in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(filename)
            return self._cache[filename]

        path = self.frames_dir / filename
        img = load_image_bgr(path)
        if img is not None:
            if len(self._cache) >= self.max_size:
                # Evict oldest
                self._cache.popitem(last=False)
            self._cache[filename] = img
        return img

    def get_thumbnail(self, filename: str) -> Optional[np.ndarray]:
        """Fetch thumbnail from cache or disk."""
        if filename in self._thumb_cache:
            self._thumb_cache.move_to_end(filename)
            return self._thumb_cache[filename]

        path = self.thumbnails_dir / filename
        img = load_image_bgr(path)
        if img is not None:
            if len(self._thumb_cache) >= (self.max_size * 2):
                self._thumb_cache.popitem(last=False)
            self._thumb_cache[filename] = img
        return img

    def clear(self) -> None:
        """Clear all in-memory frame caches."""
        self._cache.clear()
        self._thumb_cache.clear()

    def evict(self, filename: str, thumb_filename: Optional[str] = None) -> None:
        """Evict a specific frame and optional thumbnail from cache."""
        self._cache.pop(filename, None)
        if thumb_filename:
            self._thumb_cache.pop(thumb_filename, None)

    @property
    def current_size(self) -> int:
        return len(self._cache)
