"""Central annotation manager handling per-frame objects, selection, and undo/redo."""

from typing import Dict, List, Optional, Tuple, Any, Callable
import copy

from sam3_annotator.annotation.polygon import PolygonAnnotation
from sam3_annotator.utils.logging_utils import logger


class AnnotationManager:
    """Manages all annotation objects across video frames with undo/redo history."""

    def __init__(self, max_history: int = 50):
        # Mapping: frame_id (int) -> List[PolygonAnnotation]
        self.frame_annotations: Dict[int, List[PolygonAnnotation]] = {}
        # History for undo/redo
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []
        self.max_history = max_history

        # Selection state
        self.selected_object_id: Optional[str] = None
        self.active_frame_id: int = 1
        self.active_class_id: int = 0
        self.active_class_name: str = "default"

        # Listeners / callbacks for UI synchronization
        self.on_changed_callbacks: List[Callable[[], None]] = []

    def register_change_listener(self, cb: Callable[[], None]) -> None:
        self.on_changed_callbacks.append(cb)

    def _notify_change(self) -> None:
        for cb in self.on_changed_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error("Error in annotation change callback: %s", e)

    def _save_snapshot(self, description: str = "") -> None:
        """Save deepcopy of current active frame annotations for undo."""
        current_state = {
            "frame_id": self.active_frame_id,
            "description": description,
            "annotations": [copy.deepcopy(a) for a in self.frame_annotations.get(self.active_frame_id, [])],
        }
        self.undo_stack.append(current_state)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def get_annotations_for_frame(self, frame_id: int) -> List[PolygonAnnotation]:
        """Retrieve annotations for a specific frame."""
        return self.frame_annotations.get(frame_id, [])

    def add_annotation(self, annotation: PolygonAnnotation) -> None:
        """Add a new polygon annotation to its designated frame."""
        self._save_snapshot(f"Add object {annotation.object_id}")
        fid = annotation.frame_id
        if fid not in self.frame_annotations:
            self.frame_annotations[fid] = []
        self.frame_annotations[fid].append(annotation)
        self.selected_object_id = annotation.object_id
        self._notify_change()

    def remove_annotation(self, frame_id: int, object_id: str) -> bool:
        """Remove an annotation by frame ID and object ID."""
        if frame_id in self.frame_annotations:
            self._save_snapshot(f"Remove object {object_id}")
            before_len = len(self.frame_annotations[frame_id])
            self.frame_annotations[frame_id] = [
                a for a in self.frame_annotations[frame_id] if a.object_id != object_id
            ]
            if len(self.frame_annotations[frame_id]) < before_len:
                if self.selected_object_id == object_id:
                    self.selected_object_id = None
                self._notify_change()
                return True
        return False

    def clear_frame_annotations(self, frame_id: int) -> None:
        """Remove all annotations for a frame."""
        if frame_id in self.frame_annotations and self.frame_annotations[frame_id]:
            self._save_snapshot(f"Clear frame {frame_id}")
            self.frame_annotations[frame_id] = []
            self.selected_object_id = None
            self._notify_change()

    def get_selected_annotation(self, frame_id: int) -> Optional[PolygonAnnotation]:
        """Get currently selected annotation on frame_id."""
        if not self.selected_object_id:
            return None
        for a in self.frame_annotations.get(frame_id, []):
            if a.object_id == self.selected_object_id:
                return a
        return None

    def undo(self) -> bool:
        """Undo last annotation modification."""
        if not self.undo_stack:
            return False

        snapshot = self.undo_stack.pop()
        fid = snapshot["frame_id"]

        # Push current state to redo stack
        current_state = {
            "frame_id": fid,
            "description": snapshot.get("description", "Undo action"),
            "annotations": [copy.deepcopy(a) for a in self.frame_annotations.get(fid, [])],
        }
        self.redo_stack.append(current_state)

        # Restore snapshot
        self.frame_annotations[fid] = [copy.deepcopy(a) for a in snapshot["annotations"]]
        self._notify_change()
        return True

    def redo(self) -> bool:
        """Redo previously undone modification."""
        if not self.redo_stack:
            return False

        snapshot = self.redo_stack.pop()
        fid = snapshot["frame_id"]

        # Push current state to undo stack
        current_state = {
            "frame_id": fid,
            "description": snapshot.get("description", "Redo action"),
            "annotations": [copy.deepcopy(a) for a in self.frame_annotations.get(fid, [])],
        }
        self.undo_stack.append(current_state)

        # Restore snapshot
        self.frame_annotations[fid] = [copy.deepcopy(a) for a in snapshot["annotations"]]
        self._notify_change()
        return True

    def total_annotations_count(self) -> int:
        return sum(len(annos) for annos in self.frame_annotations.values())

    def get_all_object_ids(self) -> List[str]:
        all_ids = set()
        for annos in self.frame_annotations.values():
            for a in annos:
                all_ids.add(a.object_id)
        return sorted(list(all_ids))
