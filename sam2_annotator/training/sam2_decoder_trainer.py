"""Meta SAM 2.1 Mask Decoder Domain Adaptation Fine-Tuner (Phase 4).

Fine-tunes SAM 2's prompt encoder and lightweight mask decoder on annotated keyframes
for enhanced video tracking domain adaptation while keeping the heavy image backbone frozen.
"""

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
import torch.nn.functional as F

from sam2_annotator.training.base_trainer import (
    BaseModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)
from sam2_annotator.dataset.split_manager import DatasetSplitter
from sam2_annotator.utils.device_utils import get_torch_device
from sam2_annotator.utils.logging_utils import logger


def _compute_dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute soft Dice loss for binary mask prediction."""
    pred_prob = torch.sigmoid(pred_logits)
    intersection = 2.0 * (pred_prob * target).sum() + 1e-5
    union = pred_prob.sum() + target.sum() + 1e-5
    return 1.0 - (intersection / union)


class SAM2DecoderTrainer(BaseModelTrainer):
    """Fine-tunes Meta SAM 2.1 Prompt Encoder and Mask Decoder for domain adaptation."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)
        self.device = get_torch_device(self.config.device)
        self.device_str = str(self.device)

    def prepare_dataset(
        self,
        project_manager: Any,
        annotation_manager: Any,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """Create keyframe dataset manifest with polygon masks and bounding boxes."""
        dataset_dir = self.config.output_dir / "sam2_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / "sam2_manifest.json"

        reuse = self.config.extra_params.get("reuse_dataset", True)
        if reuse and manifest_path.exists():
            logger.info("Reusing existing SAM 2 dataset manifest at %s", manifest_path)
            if progress_callback:
                progress_callback(100, 100, "Using existing SAM 2 dataset manifest...")
            return manifest_path

        logger.info("Preparing SAM 2 fine-tuning manifest in %s...", dataset_dir)
        if progress_callback:
            progress_callback(0, 100, "Extracting annotated keyframes for SAM 2 fine-tuning...")

        split_dict = DatasetSplitter.split_frames(
            frames=project_manager.frames,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            strategy=self.config.split_strategy,
        )

        manifest: Dict[str, Any] = {
            "created_at": time.time(),
            "train": [],
            "val": [],
        }

        total_frames = len(split_dict.get("train", [])) + len(split_dict.get("val", []))
        processed = 0

        for split_name in ["train", "val"]:
            for frame in split_dict.get(split_name, []):
                annos = annotation_manager.get_annotations_for_frame(frame.frame_id)
                if not annos:
                    continue  # Only include frames with ground truth annotations

                img_path = Path(project_manager.frames_dir) / frame.filename
                anno_records = []
                for a in annos:
                    anno_records.append({
                        "object_id": a.object_id,
                        "class_id": a.class_id,
                        "class_name": a.class_name,
                        "points": a.points,
                        "bounding_box": a.bounding_box,
                    })

                manifest[split_name].append({
                    "frame_id": frame.frame_id,
                    "filename": frame.filename,
                    "image_path": str(img_path.resolve()),
                    "width": frame.width,
                    "height": frame.height,
                    "annotations": anno_records,
                })
                processed += 1
                if progress_callback:
                    progress_callback(
                        int((processed / max(1, total_frames)) * 100),
                        100,
                        f"Processed frame {frame.frame_id}",
                    )

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "SAM 2 manifest created with %d train and %d val annotated frames at %s",
            len(manifest["train"]),
            len(manifest["val"]),
            manifest_path,
        )
        return manifest_path

    def train(
        self,
        dataset_path: Path,
        on_epoch_start: Optional[Callable[[int, int], None]] = None,
        on_epoch_end: Optional[Callable[[TrainingMetrics], None]] = None,
        on_step: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        """Run SAM 2 prompt and mask decoder domain adaptation fine-tuning loop."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.start_time = time.time()
        self.stop_requested = False

        save_weights_dir = self.config.output_dir / "weights" / self.config.project_name
        save_weights_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(dataset_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        train_records = manifest.get("train", [])
        val_records = manifest.get("val", [])

        if not train_records:
            return TrainingResult(
                status="error",
                error_message="No annotated frames available in training split for SAM 2 fine-tuning.",
                duration_seconds=time.time() - self.start_time,
            )

        # 1. Locate SAM 2 model checkpoint & config
        ckpt_name = self.config.model_name
        if not ckpt_name.endswith(".pt"):
            ckpt_name = "sam2.1_hiera_tiny.pt"

        cfg_map = {
            "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam2.1_t.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_s.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
        }
        model_cfg = cfg_map.get(Path(ckpt_name).name, "configs/sam2.1/sam2.1_hiera_t.yaml")

        if on_log:
            on_log(f"Loading base Meta SAM 2.1 model '{ckpt_name}' on device '{self.device_str}'...")

        raw_model = build_sam2(model_cfg, ckpt_name, device=self.device_str)

        # 2. Freeze Image Encoder & Memory Attention (Keep heavy backbone static)
        for param in raw_model.image_encoder.parameters():
            param.requires_grad = False
        if hasattr(raw_model, "memory_attention"):
            for param in raw_model.memory_attention.parameters():
                param.requires_grad = False
        if hasattr(raw_model, "memory_encoder"):
            for param in raw_model.memory_encoder.parameters():
                param.requires_grad = False

        # 3. Enable Gradients on Mask Decoder & Prompt Encoder
        trainable_params: List[torch.Tensor] = []
        for param in raw_model.sam_mask_decoder.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        for param in raw_model.sam_prompt_encoder.parameters():
            param.requires_grad = True
            trainable_params.append(param)

        lr = self.config.learning_rate or 1e-4
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        total_epochs = self.config.epochs

        predictor = SAM2ImagePredictor(raw_model)

        if on_log:
            on_log(f"SAM 2.1 backbone frozen. Fine-tuning {len(trainable_params)} decoder parameter tensors for {total_epochs} epochs...")

        best_val_iou = 0.0
        best_model_path: Optional[Path] = None
        last_model_path: Optional[Path] = None
        metrics_history: List[Dict[str, Any]] = []

        for epoch in range(1, total_epochs + 1):
            if self.stop_requested:
                if on_log:
                    on_log("Stop requested by user. Finishing current epoch...")
                break

            if on_epoch_start:
                on_epoch_start(epoch, total_epochs)
            if on_log:
                on_log(f"--- Starting SAM 2 Fine-Tuning Epoch {epoch}/{total_epochs} ---")

            raw_model.sam_mask_decoder.train()
            raw_model.sam_prompt_encoder.train()

            epoch_losses: List[float] = []
            total_items = len(train_records)

            for step, record in enumerate(train_records):
                if self.stop_requested:
                    break

                img_path = record["image_path"]
                if not Path(img_path).exists():
                    continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                h, w = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Set image on predictor (computes image embeddings with frozen encoder)
                with torch.no_grad():
                    predictor.set_image(img_rgb)
                    img_embed = predictor._features["image_embed"][0].unsqueeze(0)
                    high_res_feats = [feat[0].unsqueeze(0) for feat in predictor._features["high_res_feats"]]

                # Process each annotated object in frame
                for anno in record.get("annotations", []):
                    bbox = anno.get("bounding_box")  # (x1, y1, x2, y2)
                    if not bbox or bbox == (0.0, 0.0, 0.0, 0.0):
                        continue

                    # Create ground truth mask [1, 1, 256, 256] matching SAM low-res logits
                    gt_mask_full = np.zeros((h, w), dtype=np.uint8)
                    poly_pts = np.array(anno["points"], dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(gt_mask_full, [poly_pts], 1)

                    # Downsample to 256x256 for SAM decoder alignment
                    gt_mask_lowres = cv2.resize(gt_mask_full, (256, 256), interpolation=cv2.INTER_NEAREST)
                    gt_tensor = torch.from_numpy(gt_mask_lowres).float().unsqueeze(0).unsqueeze(0).to(self.device)

                    # Prepare box prompt
                    x1, y1, x2, y2 = bbox
                    box_np = np.array([[x1, y1, x2, y2]])
                    _, _, _, unnorm_box = predictor._prep_prompts(None, None, box_np, None, True, 0)
                    box_coords = unnorm_box.reshape(-1, 2, 2)
                    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)

                    # Prompt Encoder
                    sparse_emb, dense_emb = raw_model.sam_prompt_encoder(
                        points=(box_coords, box_labels),
                        boxes=None,
                        masks=None,
                    )

                    # Mask Decoder forward pass
                    low_res_masks, iou_preds, _, _ = raw_model.sam_mask_decoder(
                        image_embeddings=img_embed,
                        image_pe=raw_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )

                    # Compute BCE + Soft Dice Loss
                    bce_loss = F.binary_cross_entropy_with_logits(low_res_masks, gt_tensor)
                    dice_loss = _compute_dice_loss(low_res_masks, gt_tensor)
                    total_loss = 0.5 * bce_loss + 0.5 * dice_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    loss_val = float(total_loss.item())
                    epoch_losses.append(loss_val)

                if on_step and step % max(1, total_items // 5) == 0:
                    cur_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                    on_step(step + 1, total_items, {"train_loss": cur_loss})

            avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

            # 4. Evaluate Validation Mask IoU
            val_ious: List[float] = []
            if val_records:
                raw_model.sam_mask_decoder.eval()
                with torch.no_grad():
                    for v_rec in val_records:
                        v_path = v_rec["image_path"]
                        if not Path(v_path).exists():
                            continue
                        v_bgr = cv2.imread(str(v_path))
                        if v_bgr is None:
                            continue
                        v_h, v_w = v_bgr.shape[:2]
                        v_rgb = cv2.cvtColor(v_bgr, cv2.COLOR_BGR2RGB)
                        predictor.set_image(v_rgb)

                        for anno in v_rec.get("annotations", []):
                            bbox = anno.get("bounding_box")
                            if not bbox or bbox == (0.0, 0.0, 0.0, 0.0):
                                continue

                            box_arr = np.array(bbox)
                            pred_masks, _, _ = predictor.predict(box=box_arr, multimask_output=False)
                            pred_binary = (pred_masks[0] > 0.0).astype(np.uint8)

                            gt_mask_full = np.zeros((v_h, v_w), dtype=np.uint8)
                            poly_pts = np.array(anno["points"], dtype=np.float32).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(gt_mask_full, [poly_pts], 1)

                            intersection = np.logical_and(pred_binary, gt_mask_full).sum()
                            union = np.logical_or(pred_binary, gt_mask_full).sum()
                            iou = float(intersection / max(1, union))
                            val_ious.append(iou)

            mean_val_iou = float(np.mean(val_ious)) if val_ious else 0.0

            elapsed = time.time() - self.start_time
            avg_time = elapsed / epoch
            eta = avg_time * (total_epochs - epoch)

            metrics = TrainingMetrics(
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=avg_epoch_loss,
                seg_loss=avg_epoch_loss,
                map50=mean_val_iou,  # Validation Mask IoU
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            )
            metrics_history.append(metrics.to_dict())

            if on_epoch_end:
                on_epoch_end(metrics)

            if on_log:
                iou_str = f"Val Mask IoU: {mean_val_iou:.3f}" if mean_val_iou > 0 else f"Loss: {avg_epoch_loss:.4f}"
                on_log(f"SAM 2 Epoch {epoch}/{total_epochs} | Loss: {avg_epoch_loss:.4f} | {iou_str} | ETA: {int(eta)}s")

            # Save checkpoint
            last_pt = save_weights_dir / "last_decoder.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "sam_mask_decoder": raw_model.sam_mask_decoder.state_dict(),
                    "sam_prompt_encoder": raw_model.sam_prompt_encoder.state_dict(),
                    "metrics": metrics.to_dict(),
                },
                last_pt,
            )
            last_model_path = last_pt

            if mean_val_iou >= best_val_iou:
                best_val_iou = mean_val_iou
                best_pt = save_weights_dir / "best_decoder.pt"
                shutil.copy2(str(last_pt), str(best_pt))
                best_model_path = best_pt

        duration = time.time() - self.start_time
        status = "cancelled" if self.stop_requested else "completed"

        if on_log:
            on_log(f"SAM 2.1 fine-tuning completed in {int(duration)}s with status '{status}'")

        return TrainingResult(
            status=status,
            output_dir=save_weights_dir,
            best_model_path=best_model_path or last_model_path,
            last_model_path=last_model_path,
            duration_seconds=duration,
            metrics_history=metrics_history,
        )

    def export_model(self, model_path: Path, export_format: str = "onnx") -> Path:
        """Export fine-tuned SAM 2 mask decoder weights."""
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        # The checkpoint is saved as standard PyTorch state dict compatible directly with SAM 2 loader
        logger.info("Fine-tuned SAM 2 weights saved at %s", model_path)
        return model_path
