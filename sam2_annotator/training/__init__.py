"""Training module for deep learning vision models."""

from sam2_annotator.training.base_trainer import (
    BaseModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)
from sam2_annotator.training.training_worker import TrainingWorker
from sam2_annotator.training.yolo_trainer import YOLOTrainer
from sam2_annotator.training.trainer_factory import (
    TRAINING_ENGINES,
    get_available_engines,
    create_trainer,
)

__all__ = [
    "BaseModelTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingResult",
    "TrainingWorker",
    "YOLOTrainer",
    "TRAINING_ENGINES",
    "get_available_engines",
    "create_trainer",
]
