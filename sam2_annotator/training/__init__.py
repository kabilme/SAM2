"""Training module for deep learning vision models."""

from sam2_annotator.training.base_trainer import (
    BaseModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)
from sam2_annotator.training.training_worker import TrainingWorker
from sam2_annotator.training.yolo_trainer import YOLOTrainer
from sam2_annotator.training.torchvision_trainer import TorchVisionTrainer
from sam2_annotator.training.sam2_decoder_trainer import SAM2DecoderTrainer
from sam2_annotator.training.rtdetr_trainer import RTDETRTrainer
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
    "TorchVisionTrainer",
    "SAM2DecoderTrainer",
    "RTDETRTrainer",
    "TRAINING_ENGINES",
    "get_available_engines",
    "create_trainer",
]
