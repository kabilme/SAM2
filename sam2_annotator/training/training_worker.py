"""PySide6 QThread worker for non-blocking asynchronous model training."""

from typing import Dict, Any, Optional
from pathlib import Path
from PySide6.QtCore import QThread, Signal

from sam2_annotator.training.base_trainer import (
    BaseModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)
from sam2_annotator.utils.logging_utils import logger


class TrainingWorker(QThread):
    """Executes model dataset preparation and training loop in a background thread."""

    dataset_prep_progress = Signal(int, int, str)
    epoch_started = Signal(int, int)
    epoch_completed = Signal(int, int, dict)
    step_progress = Signal(int, int, dict)
    log_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        trainer: BaseModelTrainer,
        project_manager: Any,
        annotation_manager: Any,
        dataset_yaml_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.project_manager = project_manager
        self.annotation_manager = annotation_manager
        self.dataset_yaml_path = dataset_yaml_path
        self._is_cancelled = False

    def cancel(self) -> None:
        """Cooperative cancel method callable from UI thread."""
        self._is_cancelled = True
        if self.trainer:
            self.trainer.request_stop()
        self.log_message.emit("Cancelling training session...")

    def run(self) -> None:
        """Main background execution thread."""
        try:
            # 1. Dataset preparation if not already provided
            if not self.dataset_yaml_path or not self.dataset_yaml_path.exists():
                self.log_message.emit("Preparing and validating training dataset...")
                self.dataset_yaml_path = self.trainer.prepare_dataset(
                    project_manager=self.project_manager,
                    annotation_manager=self.annotation_manager,
                    progress_callback=lambda cur, tot, msg: self.dataset_prep_progress.emit(cur, tot, msg),
                )

            if self._is_cancelled:
                self.finished.emit(TrainingResult(status="cancelled"))
                return

            # 2. Execute training loop
            self.log_message.emit(f"Launching training using dataset '{self.dataset_yaml_path}'...")
            result = self.trainer.train(
                dataset_path=self.dataset_yaml_path,
                on_epoch_start=lambda ep, tot: self.epoch_started.emit(ep, tot),
                on_epoch_end=lambda metrics: self.epoch_completed.emit(metrics.epoch, metrics.total_epochs, metrics.to_dict()),
                on_step=lambda st, tot, ldict: self.step_progress.emit(st, tot, ldict),
                on_log=lambda msg: self.log_message.emit(msg),
            )

            self.finished.emit(result)

        except Exception as e:
            logger.error("Unhandled error in TrainingWorker: %s", e, exc_info=True)
            self.error.emit(str(e))
