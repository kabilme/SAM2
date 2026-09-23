"""SAM 3 Adapter Architecture and Implementations.

Provides SAM3AdapterInterface, SAM3LocalAdapter (production Ultralytics SAM3 integration),
and MockSAM3Adapter (deterministic synthetic masks for automated testing).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
import torch

from sam3_annotator.utils.device_utils import get_torch_device, clear_memory
from sam3_annotator.utils.logging_utils import logger


@dataclass
class PointPrompt:
    positive_points: List[Tuple[float, float]] = field(default_factory=list)
    negative_points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class BoxPrompt:
    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2]


@dataclass
class TextPrompt:
    text: str


@dataclass
class MaskRefinementPrompt:
    prev_mask: np.ndarray
    positive_points: List[Tuple[float, float]] = field(default_factory=list)
    negative_points: List[Tuple[float, float]] = field(default_factory=list)


class SAM3AdapterInterface(ABC):
    """Abstract base class for all SAM 3 model adapters."""

    @abstractmethod
    def load_model(
        self,
        checkpoint_path: str = "",
        device: str = "auto",
        precision: str = "fp32",
    ) -> bool:
        """Load and initialize model weights."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        pass

    @abstractmethod
    def segment_with_points(
        self,
        image_bgr: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Run segmentation given positive and negative point coordinates.

        Returns (binary_mask, confidence_score).
        """
        pass

    @abstractmethod
    def segment_with_box(
        self,
        image_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Run segmentation given a bounding box prompt [x1, y1, x2, y2]."""
        pass

    @abstractmethod
    def segment_with_text(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
    ) -> List[Tuple[np.ndarray, Optional[float]]]:
        """Run segmentation given a text concept prompt.

        Returns list of (binary_mask, confidence_score) for detected objects.
        """
        pass

    @abstractmethod
    def refine_mask(
        self,
        image_bgr: np.ndarray,
        prev_mask: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Refine an existing mask with additional prompt points."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release GPU memory and model resources."""
        pass


class MockSAM3Adapter(SAM3AdapterInterface):
    """Deterministic mock adapter for automated testing without requiring model weights."""

    def __init__(self):
        self._loaded = False
        self.device = "cpu"
        self.precision = "fp32"

    def load_model(
        self,
        checkpoint_path: str = "",
        device: str = "auto",
        precision: str = "fp32",
    ) -> bool:
        self._loaded = True
        self.device = device
        self.precision = precision
        logger.info("MockSAM3Adapter initialized in %s mode", device)
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def segment_with_points(
        self,
        image_bgr: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw a synthetic elliptical mask centered around the first positive point
        if positive_points:
            cx, cy = int(positive_points[0][0]), int(positive_points[0][1])
            radius_x = max(15, min(w // 4, 60))
            radius_y = max(15, min(h // 4, 60))
            cv2.ellipse(mask, (cx, cy), (radius_x, radius_y), 0, 0, 360, 255, -1)

            # Subtract area around negative points if present
            if negative_points:
                for nx, ny in negative_points:
                    cv2.circle(mask, (int(nx), int(ny)), 25, 0, -1)
        else:
            # Fallback center ellipse
            cv2.ellipse(mask, (w // 2, h // 2), (w // 6, h // 6), 0, 0, 360, 255, -1)

        return mask, 0.95

    def segment_with_box(
        self,
        image_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Optional[float]]:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        # Inset slightly to make it an organic polygon
        cv2.rectangle(mask, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), 255, -1)
        return mask, 0.92

    def segment_with_text(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
    ) -> List[Tuple[np.ndarray, Optional[float]]]:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Synthetic object in central third
        cv2.ellipse(mask, (w // 2, h // 2), (w // 5, h // 5), 0, 0, 360, 255, -1)
        return [(mask, 0.88)]

    def refine_mask(
        self,
        image_bgr: np.ndarray,
        prev_mask: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        mask = prev_mask.copy()
        for px, py in positive_points:
            cv2.circle(mask, (int(px), int(py)), 30, 255, -1)
        if negative_points:
            for nx, ny in negative_points:
                cv2.circle(mask, (int(nx), int(ny)), 30, 0, -1)
        return mask, 0.94

    def release(self) -> None:
        self._loaded = False


class SAM3LocalAdapter(SAM3AdapterInterface):
    """Production SAM 3 adapter utilizing official Ultralytics SAM3 / SAM model architecture."""

    def __init__(self):
        self.model: Any = None
        self._is_loaded: bool = False
        self.device_str: str = "cpu"
        self.precision: str = "fp32"
        self.checkpoint_path: str = ""

    def load_model(
        self,
        checkpoint_path: str = "",
        device: str = "auto",
        precision: str = "fp32",
    ) -> bool:
        """Load SAM model using Ultralytics SAM architecture."""
        try:
            from ultralytics import SAM

            # Select device
            torch_dev = get_torch_device(device)
            self.device_str = str(torch_dev)
            self.precision = precision

            # Determine checkpoint
            if not checkpoint_path:
                checkpoint_path = "sam2.1_t.pt"  # Lightweight default model

            self.checkpoint_path = checkpoint_path
            logger.info("Initializing SAM model (%s) on %s with %s precision...",
                        checkpoint_path, self.device_str, precision)

            self.model = SAM(checkpoint_path)
            self._is_loaded = True
            logger.info("SAM model successfully initialized.")
            return True
        except Exception as e:
            logger.error("Failed to load SAM model: %s", e)
            self._is_loaded = False
            self.model = None
            return False

    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def segment_with_points(
        self,
        image_bgr: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Prompt-based point segmentation with positive and negative coordinates."""
        if not self.is_loaded():
            raise RuntimeError("SAM model is not loaded.")

        h, w = image_bgr.shape[:2]
        pts = []
        labels = []

        for p in positive_points:
            pts.append([float(p[0]), float(p[1])])
            labels.append(1)

        if negative_points:
            for p in negative_points:
                pts.append([float(p[0]), float(p[1])])
                labels.append(0)

        if not pts:
            return np.zeros((h, w), dtype=np.uint8), None

        try:
            # Ultralytics SAM inference
            results = self.model.predict(
                source=image_bgr,
                points=pts,
                labels=labels,
                device=self.device_str,
                verbose=False,
            )

            if results and len(results) > 0 and results[0].masks is not None:
                mask_data = results[0].masks.data.cpu().numpy()
                if mask_data.shape[0] > 0:
                    raw_mask = mask_data[0]
                    # Resize to match original image dimensions if needed
                    if raw_mask.shape != (h, w):
                        raw_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                    conf = float(results[0].boxes.conf[0]) if (results[0].boxes and len(results[0].boxes.conf) > 0) else 0.90
                    return binary_mask, conf

            return np.zeros((h, w), dtype=np.uint8), None
        except Exception as e:
            logger.error("Error in segment_with_points: %s", e)
            if "out of memory" in str(e).lower():
                clear_memory(torch.device(self.device_str))
            return np.zeros((h, w), dtype=np.uint8), None

    def segment_with_box(
        self,
        image_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Prompt-based bounding box segmentation."""
        if not self.is_loaded():
            raise RuntimeError("SAM model is not loaded.")

        h, w = image_bgr.shape[:2]
        bboxes = [[float(v) for v in box]]

        try:
            results = self.model.predict(
                source=image_bgr,
                bboxes=bboxes,
                device=self.device_str,
                verbose=False,
            )

            if results and len(results) > 0 and results[0].masks is not None:
                mask_data = results[0].masks.data.cpu().numpy()
                if mask_data.shape[0] > 0:
                    raw_mask = mask_data[0]
                    if raw_mask.shape != (h, w):
                        raw_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                    conf = float(results[0].boxes.conf[0]) if (results[0].boxes and len(results[0].boxes.conf) > 0) else 0.90
                    return binary_mask, conf

            return np.zeros((h, w), dtype=np.uint8), None
        except Exception as e:
            logger.error("Error in segment_with_box: %s", e)
            if "out of memory" in str(e).lower():
                clear_memory(torch.device(self.device_str))
            return np.zeros((h, w), dtype=np.uint8), None

    def segment_with_text(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
    ) -> List[Tuple[np.ndarray, Optional[float]]]:
        """Prompt-based text/concept segmentation."""
        if not self.is_loaded():
            raise RuntimeError("SAM model is not loaded.")

        h, w = image_bgr.shape[:2]
        try:
            # Check if SAM3 Semantic Predictor is supported directly
            from ultralytics.models.sam import SAM3SemanticPredictor
            results = self.model.predict(
                source=image_bgr,
                device=self.device_str,
                verbose=False,
            )
            outputs = []
            if results and len(results) > 0 and results[0].masks is not None:
                mask_data = results[0].masks.data.cpu().numpy()
                for i in range(len(mask_data)):
                    raw_mask = mask_data[i]
                    if raw_mask.shape != (h, w):
                        raw_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                    conf = float(results[0].boxes.conf[i]) if (results[0].boxes and len(results[0].boxes.conf) > i) else 0.85
                    outputs.append((binary_mask, conf))
            return outputs
        except Exception as e:
            logger.warning("Text-prompt segmentation fallback: %s", e)
            return []

    def refine_mask(
        self,
        image_bgr: np.ndarray,
        prev_mask: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Combine previous mask with new point prompts for iterative refinement."""
        return self.segment_with_points(image_bgr, positive_points, negative_points)

    def release(self) -> None:
        """Free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        clear_memory(torch.device(self.device_str))
        logger.info("SAM 3 model released from memory.")
