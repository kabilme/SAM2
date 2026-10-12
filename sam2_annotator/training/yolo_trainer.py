"""Ultralytics YOLO training engine for segmentation and detection."""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import torch

from sam2_annotator.training.base_trainer import (
    BaseModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)
from sam2_annotator.dataset.split_manager import DatasetSplitter
from sam2_annotator.dataset.yolo_exporter import YOLOExporter
from sam2_annotator.utils.logging_utils import logger


class YOLOTrainer(BaseModelTrainer):
    """Integrates Ultralytics YOLOv8/11 instance segmentation and object detection training."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)
        self.active_yolo_trainer: Any = None

    def prepare_dataset(
        self,
        project_manager: Any,
        annotation_manager: Any,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """Export project annotations into an Ultralytics-compatible dataset structure with data.yaml."""
        dataset_dir = self.config.output_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = dataset_dir / "data.yaml"
        train_img_dir = dataset_dir / "images" / "train"

        reuse_dataset = self.config.extra_params.get("reuse_dataset", True)
        if reuse_dataset and data_yaml.exists() and train_img_dir.exists() and any(train_img_dir.iterdir()):
            logger.info("Reusing existing prepared YOLO dataset at %s", data_yaml)
            if progress_callback:
                progress_callback(100, 100, "Using existing prepared dataset (re-export skipped)...")
            return data_yaml

        logger.info("Preparing YOLO dataset in '%s' for task '%s'...", dataset_dir, self.config.task)
        if progress_callback:
            progress_callback(0, 100, "Partitioning video frames for training split...")

        # Partition frames according to config
        split_dict = DatasetSplitter.split_frames(
            frames=project_manager.frames,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            strategy=self.config.split_strategy,
        )

        mode = "segmentation" if self.config.task == "segmentation" else "detection"
        exporter = YOLOExporter(
            output_dir=dataset_dir,
            classes=project_manager.classes,
            frames_dir=project_manager.frames_dir,
            mode=mode,
        )

        export_result = exporter.export_dataset(
            split_dict=split_dict,
            annotations_by_frame=annotation_manager.frame_annotations,
            export_masks=False,
            export_previews=False,
            create_zip=False,
            include_null_frames=True,
            progress_callback=lambda cur, tot, msg: progress_callback(
                int((cur / max(1, tot)) * 100), 100, f"Exporting dataset: {msg}"
            ) if progress_callback else None,
        )

        yaml_path = Path(export_result.get("yaml_path", dataset_dir / "data.yaml"))
        if not yaml_path.exists():
            raise FileNotFoundError(f"Expected data.yaml was not created at {yaml_path}")

        logger.info("YOLO dataset prepared successfully at %s", yaml_path)
        return yaml_path

    def _resolve_device(self) -> str:
        """Resolve device string to Ultralytics-compatible format."""
        dev = self.config.device.strip().lower()
        if dev == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        if "cuda" in dev:
            # Convert "cuda:0" -> "0"
            if ":" in dev:
                return dev.split(":")[-1]
            return "0" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def train(
        self,
        dataset_path: Path,
        on_epoch_start: Optional[Callable[[int, int], None]] = None,
        on_epoch_end: Optional[Callable[[TrainingMetrics], None]] = None,
        on_step: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        """Run Ultralytics YOLO training loop with real-time callback hooks."""
        from ultralytics import YOLO

        self.start_time = time.time()
        self.stop_requested = False
        device_str = self._resolve_device()

        save_project_dir = self.config.output_dir / "weights"
        save_project_dir.mkdir(parents=True, exist_ok=True)

        # Ensure data.yaml is portable and clean
        try:
            import yaml
            if dataset_path.exists():
                with open(dataset_path, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f) or {}
                # If path is "." or nonexistent on the current system, remove it so Ultralytics automatically anchors to the directory of data.yaml
                if "path" in ydata:
                    cur_path = ydata["path"]
                    if not cur_path or cur_path == "." or not Path(cur_path).exists():
                        del ydata["path"]
                        with open(dataset_path, "w", encoding="utf-8") as f:
                            yaml.dump(ydata, f, default_flow_style=False, sort_keys=False)
                        logger.info("Cleaned data.yaml: removed non-portable path key (Ultralytics anchors to data.yaml directory)")
        except Exception as e:
            logger.warning("Could not verify/clean data.yaml: %s", e)

        if on_log:
            on_log(f"Initializing YOLO model '{self.config.model_name}' on device '{device_str}'...")

        try:
            model = YOLO(self.config.model_name)
        except Exception as e:
            logger.error("Failed to load YOLO base model '%s': %s", self.config.model_name, e)
            return TrainingResult(
                status="error",
                error_message=f"Could not load base model {self.config.model_name}: {e}",
                duration_seconds=time.time() - self.start_time,
            )

        metrics_history: List[Dict[str, Any]] = []
        total_epochs = self.config.epochs

        # Hook Ultralytics callbacks
        def cb_on_train_epoch_start(trainer: Any) -> None:
            self.active_yolo_trainer = trainer
            current_epoch = getattr(trainer, "epoch", 0) + 1
            if on_epoch_start:
                on_epoch_start(current_epoch, total_epochs)
            if on_log:
                on_log(f"--- Starting Epoch {current_epoch}/{total_epochs} ---")

        def _parse_losses(source: Any) -> Dict[str, float]:
            """Safely extract named and numeric loss items from dictionary, tensor, or list."""
            res: Dict[str, float] = {}
            if source is None:
                return res

            if isinstance(source, dict):
                for k, v in source.items():
                    try:
                        clean_k = str(k).replace("train/", "")
                        res[clean_k] = float(v.item() if hasattr(v, "item") else v)
                    except Exception:
                        pass
            elif isinstance(source, (list, tuple)):
                names = ["box_loss", "seg_loss" if self.config.task == "segmentation" else "dfl_loss", "cls_loss"]
                for i, v in enumerate(source):
                    try:
                        key = names[i] if i < len(names) else f"loss_{i}"
                        res[key] = float(v.item() if hasattr(v, "item") else v)
                    except Exception:
                        pass
            elif hasattr(source, "tolist"):
                try:
                    vals = source.tolist()
                    if isinstance(vals, list):
                        names = ["box_loss", "seg_loss" if self.config.task == "segmentation" else "dfl_loss", "cls_loss"]
                        for i, v in enumerate(vals):
                            key = names[i] if i < len(names) else f"loss_{i}"
                            res[key] = float(v)
                    else:
                        res["loss_0"] = float(vals)
                except Exception:
                    pass

            return res

        def cb_on_train_batch_end(trainer: Any) -> None:
            if self.stop_requested:
                trainer.stop = True
                if on_log:
                    on_log("Stop requested by user. Finishing current step and saving model...")

            step = getattr(trainer, "batch", 0)
            total_steps = getattr(trainer, "num_batches", 1)
            losses = _parse_losses(getattr(trainer, "loss_items", None))
            if losses:
                losses["train_loss"] = float(sum(losses.values()))

            if on_step and step % 5 == 0:
                on_step(step, total_steps, losses)

        def cb_on_fit_epoch_end(trainer: Any) -> None:
            current_epoch = getattr(trainer, "epoch", 0) + 1
            elapsed = time.time() - self.start_time
            avg_epoch_time = elapsed / max(1, current_epoch)
            remaining_epochs = max(0, total_epochs - current_epoch)
            eta = avg_epoch_time * remaining_epochs

            metrics = TrainingMetrics(
                epoch=current_epoch,
                total_epochs=total_epochs,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            )

            # Extract losses from trainer.tloss, fallback to loss_items or loss
            parsed_losses = _parse_losses(getattr(trainer, "tloss", None))
            if not parsed_losses:
                parsed_losses = _parse_losses(getattr(trainer, "loss_items", None))

            if parsed_losses:
                metrics.train_loss = float(sum(parsed_losses.values()))
                metrics.box_loss = parsed_losses.get("box_loss", 0.0)
                metrics.seg_loss = parsed_losses.get("seg_loss", 0.0)
                metrics.cls_loss = parsed_losses.get("cls_loss", 0.0)
            elif hasattr(trainer, "loss") and trainer.loss is not None:
                try:
                    metrics.train_loss = float(trainer.loss.item() if hasattr(trainer.loss, "item") else trainer.loss)
                except Exception:
                    pass

            # Extract validation metrics (mAP50, mAP50-95)
            val_metrics = getattr(trainer, "metrics", {})
            if isinstance(val_metrics, dict):
                # Try mask metrics first for segmentation, fallback to box metrics
                if self.config.task == "segmentation":
                    metrics.map50 = float(val_metrics.get("metrics/mAP50(M)", val_metrics.get("metrics/mAP50(B)", 0.0)))
                    metrics.map50_95 = float(val_metrics.get("metrics/mAP50-95(M)", val_metrics.get("metrics/mAP50-95(B)", 0.0)))
                else:
                    metrics.map50 = float(val_metrics.get("metrics/mAP50(B)", 0.0))
                    metrics.map50_95 = float(val_metrics.get("metrics/mAP50-95(B)", 0.0))

            metrics_history.append(metrics.to_dict())

            if on_epoch_end:
                on_epoch_end(metrics)

            if on_log:
                m_str = f"mAP50: {metrics.map50:.3f}" if metrics.map50 > 0 else f"loss: {metrics.train_loss:.4f}"
                on_log(f"Epoch {current_epoch}/{total_epochs} finished | {m_str} | ETA: {int(eta)}s")

            if self.stop_requested:
                trainer.stop = True

        # Register callbacks
        def cb_on_model_save(trainer: Any) -> None:
            # Preserve a copy of last.pt before final_eval strips optimizer state and resets epoch
            try:
                if hasattr(trainer, "last") and trainer.last and Path(trainer.last).exists():
                    resumable_ckpt = Path(trainer.last).parent / "resumable_last.pt"
                    shutil.copy2(str(trainer.last), str(resumable_ckpt))
            except Exception:
                pass

        model.add_callback("on_train_epoch_start", cb_on_train_epoch_start)
        model.add_callback("on_train_batch_end", cb_on_train_batch_end)
        model.add_callback("on_fit_epoch_end", cb_on_fit_epoch_end)
        model.add_callback("on_model_save", cb_on_model_save)

        try:
            if on_log:
                on_log(f"Starting training for {total_epochs} epochs with batch={self.config.batch_size}, imgsz={self.config.image_size}...")

            train_args = {
                "data": str(dataset_path),
                "epochs": self.config.epochs,
                "batch": self.config.batch_size,
                "imgsz": self.config.image_size,
                "device": device_str,
                "lr0": self.config.learning_rate,
                "patience": self.config.patience,
                "workers": self.config.workers,
                "project": str(save_project_dir),
                "name": self.config.project_name,
                "exist_ok": True,
                "save": True,
                "verbose": False,
            }
            extra_clean = {k: v for k, v in self.config.extra_params.items() if k not in ("reuse_dataset", "force_reexport")}
            train_args.update(extra_clean)

            model.train(**train_args)

            duration = time.time() - self.start_time
            status = "cancelled" if self.stop_requested else "completed"

            # Locate output weights
            run_dir = save_project_dir / self.config.project_name
            weights_dir = run_dir / "weights"
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"

            best_model_path = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)
            last_model_path = last_pt if last_pt.exists() else None

            if on_log:
                if status == "completed":
                    on_log(f"Training completed successfully in {int(duration)}s! Best model saved at: {best_model_path}")
                else:
                    on_log(f"Training stopped by user. Checkpoint saved at: {last_model_path}")

            return TrainingResult(
                status=status,
                best_model_path=best_model_path,
                last_model_path=last_model_path,
                dataset_yaml_path=dataset_path,
                metrics_history=metrics_history,
                duration_seconds=duration,
                output_dir=run_dir if run_dir.exists() else save_project_dir,
            )

        except Exception as e:
            logger.error("Error during YOLO training: %s", e)
            if on_log:
                on_log(f"Training error: {e}")
            return TrainingResult(
                status="error",
                error_message=str(e),
                metrics_history=metrics_history,
                duration_seconds=time.time() - self.start_time,
            )

    def export_model(self, model_path: Path, export_format: str = "onnx") -> Path:
        """Export a trained PyTorch model (.pt) to ONNX or other supported format."""
        from ultralytics import YOLO

        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        logger.info("Exporting model %s to %s format...", model_path, export_format)
        model = YOLO(str(model_path))
        exported_path_str = model.export(format=export_format)
        exported_path = Path(exported_path_str)
        if not exported_path.exists():
            # Fallback check next to original file
            candidate = model_path.with_suffix(f".{export_format}")
            if candidate.exists():
                return candidate
        return exported_path
