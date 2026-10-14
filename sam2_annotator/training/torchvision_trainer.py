"""TorchVision Mask R-CNN & Faster R-CNN training engine (Phase 3).

Pure PyTorch baseline (BSD-3-Clause) with ResNet-50 FPN and MobileNet architectures
for instance segmentation and object detection.
"""

import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sam2_annotator.training.base_trainer import (
    BaseModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)
from sam2_annotator.dataset.split_manager import DatasetSplitter
from sam2_annotator.dataset.coco_exporter import COCOExporter
from sam2_annotator.utils.device_utils import get_torch_device
from sam2_annotator.utils.logging_utils import logger


class TorchVisionDetectionDataset(Dataset):
    """Custom PyTorch dataset loading COCO formatted images, bounding boxes, and polygon masks."""

    def __init__(self, images_dir: Path, annotation_file: Path, is_segmentation: bool = True) -> None:
        self.images_dir = images_dir
        self.is_segmentation = is_segmentation
        self.items: List[Dict[str, Any]] = []

        if not annotation_file.exists():
            return

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        images_by_id = {img["id"]: img for img in data.get("images", [])}
        annos_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for anno in data.get("annotations", []):
            img_id = anno["image_id"]
            annos_by_img.setdefault(img_id, []).append(anno)

        for img_id, img_info in images_by_id.items():
            self.items.append({
                "info": img_info,
                "annos": annos_by_img.get(img_id, []),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item = self.items[idx]
        img_info = item["info"]
        annos = item["annos"]

        img_path = self.images_dir / img_info["file_name"]
        if not img_path.exists():
            # Try finding without subfolder if needed
            alt = self.images_dir.parent / img_info["file_name"]
            img_path = alt if alt.exists() else img_path

        # Load image via cv2
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Fallback black image
            h, w = img_info.get("height", 640), img_info.get("width", 640)
            img_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Convert HWC uint8 -> CHW float32 [0.0, 1.0]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        masks_list: List[np.ndarray] = []
        areas_list: List[float] = []

        for anno in annos:
            bbox = anno.get("bbox", [])  # [x, y, w, h] in pixels
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                x1, y1, bw, bh = bbox
                x2, y2 = min(float(w), x1 + bw), min(float(h), y1 + bh)
                if x2 > x1 and y2 > y1:
                    boxes_list.append([x1, y1, x2, y2])
                    labels_list.append(int(anno.get("category_id", 1)))
                    areas_list.append(float(anno.get("area", bw * bh)))

                    if self.is_segmentation:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        segs = anno.get("segmentation", [])
                        for poly in segs:
                            if len(poly) >= 6:
                                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(mask, [pts], 1)
                        masks_list.append(mask)

        num_objs = len(boxes_list)
        if num_objs == 0:
            # Degenerate empty frame handling for TorchVision models
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks_list), dtype=torch.uint8) if self.is_segmentation else torch.zeros((0, h, w), dtype=torch.uint8)
            areas = torch.as_tensor(areas_list, dtype=torch.float32)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
        }
        if self.is_segmentation:
            target["masks"] = masks

        return img_tensor, target


def _collate_fn(batch: Any) -> Tuple[Any, ...]:
    return tuple(zip(*batch))


class TorchVisionTrainer(BaseModelTrainer):
    """Integrates TorchVision Mask R-CNN and Faster R-CNN detection architectures."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)
        self.device = get_torch_device(self.config.device)
        self.num_classes = 2  # Background (0) + 1 class default

    def prepare_dataset(
        self,
        project_manager: Any,
        annotation_manager: Any,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """Export project annotations into standard COCO format for TorchVision training."""
        dataset_dir = self.config.output_dir / "coco_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        anno_dir = dataset_dir / "annotations"
        train_json = anno_dir / "instances_train.json"

        # Check if existing prepared dataset can be reused
        reuse = self.config.extra_params.get("reuse_dataset", True)
        if reuse and train_json.exists():
            logger.info("Reusing existing prepared TorchVision COCO dataset at %s", train_json)
            if progress_callback:
                progress_callback(100, 100, "Using existing prepared COCO dataset...")
            return train_json

        logger.info("Exporting COCO dataset for TorchVision training to %s...", dataset_dir)
        if progress_callback:
            progress_callback(0, 100, "Splitting video frames for TorchVision dataset...")

        split_dict = DatasetSplitter.split_frames(
            frames=project_manager.frames,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            strategy=self.config.split_strategy,
        )

        exporter = COCOExporter(
            output_dir=dataset_dir,
            classes=project_manager.classes,
            frames_dir=project_manager.frames_dir,
        )

        exporter.export(
            split_dict=split_dict,
            annotations_by_frame=annotation_manager.frame_annotations,
            create_zip=False,
            include_null_frames=True,
            progress_callback=lambda cur, tot, msg: progress_callback(
                int((cur / max(1, tot)) * 100), 100, f"Preparing COCO: {msg}"
            ) if progress_callback else None,
        )

        if not train_json.exists():
            raise FileNotFoundError(f"Expected instances_train.json was not generated at {train_json}")

        logger.info("TorchVision COCO dataset ready at %s", train_json)
        return train_json

    def _build_model(self, num_classes: int) -> nn.Module:
        """Instantiate TorchVision model and swap predictors with custom class count."""
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        model_name = self.config.model_name.lower()
        is_seg = self.config.task == "segmentation"

        if is_seg:
            # Mask R-CNN
            if "fpn_v2" in model_name or "v2" in model_name:
                model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
            else:
                model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

            # Replace box predictor
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # Replace mask predictor
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)

        else:
            # Faster R-CNN
            if "mobilenet" in model_name:
                model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
            elif "fpn_v2" in model_name or "v2" in model_name:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
            else:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def train(
        self,
        dataset_path: Path,
        on_epoch_start: Optional[Callable[[int, int], None]] = None,
        on_epoch_end: Optional[Callable[[TrainingMetrics], None]] = None,
        on_step: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        """Run TorchVision training loop with optimizer, scheduler, and validation."""
        self.start_time = time.time()
        self.stop_requested = False

        save_weights_dir = self.config.output_dir / "weights" / self.config.project_name
        save_weights_dir.mkdir(parents=True, exist_ok=True)

        dataset_root = dataset_path.parent.parent
        train_img_dir = dataset_root / "images" / "train"
        val_img_dir = dataset_root / "images" / "val"
        val_json = dataset_path.parent / "instances_val.json"

        is_seg = self.config.task == "segmentation"

        # Determine number of classes from JSON
        num_classes = 2
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                num_classes = len(d.get("categories", [])) + 1  # 0 is background
        except Exception:
            pass
        self.num_classes = num_classes

        if on_log:
            on_log(f"Initializing TorchVision '{self.config.model_name}' on device '{self.device}' with {num_classes} classes...")

        # Load datasets
        train_ds = TorchVisionDetectionDataset(train_img_dir, dataset_path, is_segmentation=is_seg)
        val_ds = TorchVisionDetectionDataset(val_img_dir, val_json, is_segmentation=is_seg) if val_json.exists() else None

        batch_size = max(1, self.config.batch_size)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=0,  # Safe on Windows
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=0,
        ) if val_ds and len(val_ds) > 0 else None

        # Build model and move to target device
        model = self._build_model(num_classes)
        model.to(self.device)

        # Optimizer & Scheduler
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.config.learning_rate or 0.005,
            momentum=0.9,
            weight_decay=self.config.weight_decay or 0.0005,
        )
        total_epochs = self.config.epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))

        best_score = 0.0
        best_model_path: Optional[Path] = None
        last_model_path: Optional[Path] = None
        metrics_history: List[Dict[str, Any]] = []

        if on_log:
            on_log(f"Starting TorchVision training ({len(train_ds)} train, {len(val_ds) if val_ds else 0} val images) for {total_epochs} epochs...")

        for epoch in range(1, total_epochs + 1):
            if self.stop_requested:
                if on_log:
                    on_log("Stop requested by user. Finishing current epoch...")
                break

            if on_epoch_start:
                on_epoch_start(epoch, total_epochs)
            if on_log:
                on_log(f"--- Starting TorchVision Epoch {epoch}/{total_epochs} ---")

            model.train()
            epoch_loss = 0.0
            epoch_box_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_mask_loss = 0.0
            num_batches = len(train_loader)

            for step, (images, targets) in enumerate(train_loader):
                if self.stop_requested:
                    break

                # Filter out images with 0 targets if needed or keep for background
                imgs_dev = [img.to(self.device) for img in images]
                tgts_dev = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = model(imgs_dev, tgts_dev)
                losses = sum(loss for loss in loss_dict.values())
                loss_val = float(losses.item())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()

                epoch_loss += loss_val
                # Component losses
                b_loss = float(loss_dict.get("loss_box_reg", 0.0).item()) if "loss_box_reg" in loss_dict else 0.0
                c_loss = float(loss_dict.get("loss_classifier", 0.0).item()) if "loss_classifier" in loss_dict else 0.0
                m_loss = float(loss_dict.get("loss_mask", 0.0).item()) if "loss_mask" in loss_dict else 0.0
                epoch_box_loss += b_loss
                epoch_cls_loss += c_loss
                epoch_mask_loss += m_loss

                if on_step and step % max(1, num_batches // 10) == 0:
                    step_losses = {
                        "train_loss": loss_val,
                        "box_loss": b_loss,
                        "cls_loss": c_loss,
                        "seg_loss": m_loss,
                    }
                    on_step(step + 1, num_batches, step_losses)

            lr_scheduler.step()

            # Compute average epoch losses
            avg_loss = epoch_loss / max(1, num_batches)
            avg_box_loss = epoch_box_loss / max(1, num_batches)
            avg_cls_loss = epoch_cls_loss / max(1, num_batches)
            avg_mask_loss = epoch_mask_loss / max(1, num_batches)

            # Evaluate on validation set
            val_map = 0.0
            if val_loader:
                model.eval()
                val_ious: List[float] = []
                with torch.no_grad():
                    for images, targets in val_loader:
                        imgs_dev = [img.to(self.device) for img in images]
                        predictions = model(imgs_dev)
                        for pred, tgt in zip(predictions, targets):
                            tgt_boxes = tgt["boxes"].to(self.device)
                            pred_boxes = pred["boxes"]
                            if len(tgt_boxes) > 0 and len(pred_boxes) > 0:
                                # Quick Box IoU calculation
                                from torchvision.ops import box_iou
                                ious = box_iou(pred_boxes, tgt_boxes)
                                max_iou = float(ious.max().item())
                                val_ious.append(max_iou)
                if val_ious:
                    val_map = float(np.mean(val_ious))

            elapsed = time.time() - self.start_time
            avg_epoch_time = elapsed / epoch
            eta = avg_epoch_time * (total_epochs - epoch)

            metrics = TrainingMetrics(
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=avg_loss,
                box_loss=avg_box_loss,
                cls_loss=avg_cls_loss,
                seg_loss=avg_mask_loss,
                map50=val_map,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            )
            metrics_history.append(metrics.to_dict())

            if on_epoch_end:
                on_epoch_end(metrics)

            if on_log:
                m_str = f"Val Box IoU: {val_map:.3f}" if val_map > 0 else f"Loss: {avg_loss:.4f}"
                on_log(f"Epoch {epoch}/{total_epochs} finished | Loss: {avg_loss:.4f} | {m_str} | ETA: {int(eta)}s")

            # Save checkpoints
            last_pt_path = save_weights_dir / "last.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "metrics": metrics.to_dict(),
                },
                last_pt_path,
            )
            last_model_path = last_pt_path

            if val_map >= best_score:
                best_score = val_map
                best_pt_path = save_weights_dir / "best.pth"
                shutil.copy2(str(last_pt_path), str(best_pt_path))
                best_model_path = best_pt_path

        duration = time.time() - self.start_time
        status = "cancelled" if self.stop_requested else "completed"

        if on_log:
            on_log(f"TorchVision training finished in {int(duration)}s with status '{status}'")

        return TrainingResult(
            status=status,
            output_dir=save_weights_dir,
            best_model_path=best_model_path or last_model_path,
            last_model_path=last_model_path,
            duration_seconds=duration,
            metrics_history=metrics_history,
        )

    def export_model(self, model_path: Path, export_format: str = "onnx") -> Path:
        """Export trained TorchVision checkpoint to ONNX."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        export_dir = model_path.parent
        onnx_path = export_dir / f"{model_path.stem}.onnx"

        logger.info("Exporting TorchVision model %s to ONNX (%s)...", model_path, onnx_path)
        checkpoint = torch.load(str(model_path), map_location="cpu")
        num_classes = checkpoint.get("num_classes", self.num_classes)

        model = self._build_model(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dummy_input = [torch.randn(3, 640, 640)]
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["images"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
        )

        logger.info("Exported TorchVision model successfully to %s", onnx_path)
        return onnx_path
