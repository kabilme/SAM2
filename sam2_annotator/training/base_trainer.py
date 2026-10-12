"""Base definitions and interfaces for deep learning model training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import time


@dataclass
class TrainingConfig:
    """Unified configuration for training deep learning models."""

    engine_id: str = "yolo"                      # "yolo", "maskrcnn", "sam2_decoder"
    model_name: str = "yolo11n-seg.pt"           # Base pretrained model checkpoint
    task: str = "segmentation"                   # "segmentation" or "detection"
    epochs: int = 50
    batch_size: int = 8
    image_size: int = 640
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    device: str = "auto"                         # "auto", "cuda:0", "cpu"
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    split_strategy: str = "sequential"           # "sequential", "random", "grouped"
    output_dir: Path = field(default_factory=lambda: Path("runs/train"))
    project_name: str = "custom_model"
    patience: int = 50
    workers: int = 2
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Snapshot of metrics at a training epoch or step."""

    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    box_loss: float = 0.0
    seg_loss: float = 0.0
    cls_loss: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    learning_rate: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "step": self.step,
            "total_steps": self.total_steps,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "box_loss": self.box_loss,
            "seg_loss": self.seg_loss,
            "cls_loss": self.cls_loss,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            **self.custom_metrics,
        }


@dataclass
class TrainingResult:
    """Final output result of a training session."""

    status: str = "completed"                    # "completed", "cancelled", "error"
    best_model_path: Optional[Path] = None
    last_model_path: Optional[Path] = None
    dataset_yaml_path: Optional[Path] = None
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    output_dir: Optional[Path] = None


class BaseModelTrainer(ABC):
    """Abstract base class for all training engines."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.stop_requested = False
        self.start_time: float = 0.0

    def request_stop(self) -> None:
        """Cooperatively signal the training loop to stop and save checkpoint."""
        self.stop_requested = True

    @abstractmethod
    def prepare_dataset(
        self,
        project_manager: Any,
        annotation_manager: Any,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """Export and format project annotations into trainer-specific dataset format."""
        pass

    @abstractmethod
    def train(
        self,
        dataset_path: Path,
        on_epoch_start: Optional[Callable[[int, int], None]] = None,
        on_epoch_end: Optional[Callable[[TrainingMetrics], None]] = None,
        on_step: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        """Execute the training loop."""
        pass

    @abstractmethod
    def export_model(self, model_path: Path, export_format: str = "onnx") -> Path:
        """Export trained weights to target deployment format (e.g. ONNX)."""
        pass
