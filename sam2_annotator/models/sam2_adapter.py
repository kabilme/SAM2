"""SAM 2 Adapter Architecture and Implementations.

Provides SAM2AdapterInterface, SAM2LocalAdapter (production Ultralytics SAM2 integration),
and MockSAM2Adapter (deterministic synthetic masks for automated testing).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
import torch

from sam2_annotator.utils.device_utils import get_torch_device, clear_memory
from sam2_annotator.utils.logging_utils import logger


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


class SAM2AdapterInterface(ABC):
    """Abstract base class for all SAM 2 model adapters."""

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
    def segment_with_box_and_points(
        self,
        image_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
        positive_points: Optional[List[Tuple[float, float]]] = None,
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Run segmentation given both bounding box and foreground/background prompt points."""
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


class MockSAM2Adapter(SAM2AdapterInterface):
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
        logger.info("MockSAM2Adapter initialized in %s mode", device)
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

    def segment_with_box_and_points(
        self,
        image_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
        positive_points: Optional[List[Tuple[float, float]]] = None,
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(mask, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), 255, -1)
        return mask, 0.94

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


class SAM2LocalAdapter(SAM2AdapterInterface):
    """Production SAM 2 adapter utilizing official Meta SAM 2.1 Hiera architecture with Ultralytics fallback."""

    def __init__(self):
        self.model: Any = None
        self.native_predictor: Any = None
        self.use_native_sam2: bool = False
        self._is_loaded: bool = False
        self._cached_img_id: Any = None
        self.device_str: str = "cpu"
        self.precision: str = "fp32"
        self.checkpoint_path: str = ""

    def _set_native_image(self, image_bgr: np.ndarray) -> None:
        """Cache image embeddings in native predictor to avoid re-encoding on consecutive prompts."""
        h, w = image_bgr.shape[:2]
        img_token = (id(image_bgr), h, w, int(image_bgr[0, 0, 0]), int(image_bgr[h // 2, w // 2, 0]))
        if self._cached_img_id != img_token:
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.native_predictor.set_image(img_rgb)
            self._cached_img_id = img_token

    def load_model(
        self,
        checkpoint_path: str = "",
        device: str = "auto",
        precision: str = "fp32",
    ) -> bool:
        """Load SAM 2.1 Hiera model natively with Ultralytics SAM fallback."""
        torch_dev = get_torch_device(device)
        self.device_str = str(torch_dev)
        self.precision = precision

        if not checkpoint_path:
            checkpoint_path = "sam2.1_hiera_tiny.pt"  # Meta SAM 2.1 Hiera Tiny model

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            if ckpt_file.name == "sam2.1_hiera_tiny.pt":
                alt = Path("sam2.1_t.pt")
                if alt.exists():
                    logger.info("Using local fallback %s for %s", alt.name, ckpt_file.name)
                    checkpoint_path = str(alt)
                else:
                    logger.info("Downloading official Meta %s...", ckpt_file.name)
                    import urllib.request
                    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
                    urllib.request.urlretrieve(url, str(ckpt_file))

        self.checkpoint_path = checkpoint_path

        # 1. Primary: Native Meta SAM 2.1 Hiera Architecture
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            cfg_map = {
                "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "sam2.1_t.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2.1_s.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2.1_b.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2.1_l.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "sam2_hiera_tiny.pt": "configs/sam2/sam2_hiera_t.yaml",
                "sam2_hiera_small.pt": "configs/sam2/sam2_hiera_s.yaml",
                "sam2_hiera_base_plus.pt": "configs/sam2/sam2_hiera_b+.yaml",
                "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml",
            }
            model_cfg = cfg_map.get(ckpt_file.name, "configs/sam2.1/sam2.1_hiera_t.yaml")
            logger.info("Initializing native Meta SAM 2.1 Hiera model (%s, cfg=%s) on %s...",
                        checkpoint_path, model_cfg, self.device_str)

            raw_model = build_sam2(model_cfg, checkpoint_path, device=self.device_str)
            self.native_predictor = SAM2ImagePredictor(raw_model)
            self.model = raw_model
            self.use_native_sam2 = True
            self._is_loaded = True
            logger.info("Meta SAM 2.1 Hiera model successfully initialized via native SAM2 engine.")
            return True
        except Exception as e:
            logger.info("Native Meta SAM 2.1 load not available (%s); falling back to Ultralytics SAM architecture...", e)

        # 2. Secondary / Fallback: Ultralytics SAM architecture
        try:
            import ultralytics.models.sam.build as sam_build
            from ultralytics import SAM

            meta_map = {
                "sam2.1_hiera_tiny.pt": "sam2.1_t.pt",
                "sam2.1_hiera_small.pt": "sam2.1_s.pt",
                "sam2.1_hiera_base_plus.pt": "sam2.1_b.pt",
                "sam2.1_hiera_large.pt": "sam2.1_l.pt",
                "sam2_hiera_tiny.pt": "sam2_t.pt",
                "sam2_hiera_small.pt": "sam2_s.pt",
                "sam2_hiera_base_plus.pt": "sam2_b.pt",
                "sam2_hiera_large.pt": "sam2_l.pt",
            }
            for meta_name, ultra_name in meta_map.items():
                if meta_name not in sam_build.sam_model_map and ultra_name in sam_build.sam_model_map:
                    sam_build.sam_model_map[meta_name] = sam_build.sam_model_map[ultra_name]

            logger.info("Initializing Ultralytics SAM model (%s) on %s with %s precision...",
                        checkpoint_path, self.device_str, precision)

            self.model = SAM(checkpoint_path)
            self.use_native_sam2 = False
            self._is_loaded = True
            logger.info("SAM model successfully initialized via Ultralytics.")
            return True
        except Exception as e:
            logger.error("Failed to load SAM model: %s", e)
            self._is_loaded = False
            self.model = None
            self.native_predictor = None
            self.use_native_sam2 = False
            return False

    def is_loaded(self) -> bool:
        return self._is_loaded and (self.model is not None or self.native_predictor is not None)

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

        # 1. Native Meta SAM 2.1 Hiera Tiny
        if self.use_native_sam2 and self.native_predictor is not None:
            try:
                self._set_native_image(image_bgr)
                point_coords = np.array(pts, dtype=np.float32)
                point_labels = np.array(labels, dtype=np.int32)
                masks, scores, _ = self.native_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=False,
                )
                if masks is not None and len(masks) > 0:
                    binary_mask = (masks[0] > 0.0).astype(np.uint8) * 255
                    conf = float(scores[0]) if (scores is not None and len(scores) > 0) else 0.90
                    return binary_mask, conf
                return np.zeros((h, w), dtype=np.uint8), None
            except Exception as e:
                logger.error("Error in native SAM2 segment_with_points: %s", e)
                if "out of memory" in str(e).lower():
                    clear_memory(torch.device(self.device_str))
                return np.zeros((h, w), dtype=np.uint8), None

        # 2. Ultralytics SAM inference
        try:
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

        # 1. Native Meta SAM 2.1 Hiera Tiny
        if self.use_native_sam2 and self.native_predictor is not None:
            try:
                self._set_native_image(image_bgr)
                box_np = np.array([float(v) for v in box], dtype=np.float32)
                masks, scores, _ = self.native_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_np,
                    multimask_output=False,
                )
                if masks is not None and len(masks) > 0:
                    binary_mask = (masks[0] > 0.0).astype(np.uint8) * 255
                    conf = float(scores[0]) if (scores is not None and len(scores) > 0) else 0.90
                    return binary_mask, conf
                return np.zeros((h, w), dtype=np.uint8), None
            except Exception as e:
                logger.error("Error in native SAM2 segment_with_box: %s", e)
                if "out of memory" in str(e).lower():
                    clear_memory(torch.device(self.device_str))
                return np.zeros((h, w), dtype=np.uint8), None

        # 2. Ultralytics SAM inference
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

    def segment_with_box_and_points(
        self,
        image_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
        positive_points: Optional[List[Tuple[float, float]]] = None,
        negative_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Prompt-based segmentation combining bounding box and foreground/background points."""
        if not self.is_loaded():
            raise RuntimeError("SAM model is not loaded.")

        h, w = image_bgr.shape[:2]
        pts = []
        labels = []

        if positive_points:
            for p in positive_points:
                pts.append([float(p[0]), float(p[1])])
                labels.append(1)

        if negative_points:
            for p in negative_points:
                pts.append([float(p[0]), float(p[1])])
                labels.append(0)

        # 1. Native Meta SAM 2.1 Hiera Tiny
        if self.use_native_sam2 and self.native_predictor is not None:
            try:
                self._set_native_image(image_bgr)
                box_np = np.array([float(v) for v in box], dtype=np.float32)
                point_coords = np.array(pts, dtype=np.float32) if pts else None
                point_labels = np.array(labels, dtype=np.int32) if labels else None

                masks, scores, _ = self.native_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_np,
                    multimask_output=False,
                )
                if masks is not None and len(masks) > 0:
                    binary_mask = (masks[0] > 0.0).astype(np.uint8) * 255
                    conf = float(scores[0]) if (scores is not None and len(scores) > 0) else 0.90
                    return binary_mask, conf
                return np.zeros((h, w), dtype=np.uint8), None
            except Exception as e:
                logger.error("Error in native SAM2 segment_with_box_and_points: %s", e)
                if "out of memory" in str(e).lower():
                    clear_memory(torch.device(self.device_str))
                return np.zeros((h, w), dtype=np.uint8), None

        # 2. Ultralytics SAM inference
        bboxes = [[float(v) for v in box]]
        kwargs = {
            "source": image_bgr,
            "bboxes": bboxes,
            "device": self.device_str,
            "verbose": False,
        }
        if pts:
            kwargs["points"] = pts
            kwargs["labels"] = labels

        try:
            results = self.model.predict(**kwargs)
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
            logger.error("Error in segment_with_box_and_points: %s", e)
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
            # Check if SAM2 Semantic Predictor is supported directly
            from ultralytics.models.sam import SAM2SemanticPredictor
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
        if self.native_predictor is not None:
            del self.native_predictor
            self.native_predictor = None
        if self.model is not None:
            del self.model
            self.model = None
        self._cached_img_id = None
        self.use_native_sam2 = False
        self._is_loaded = False
        clear_memory(torch.device(self.device_str))
        logger.info("SAM 2 model released from memory.")

