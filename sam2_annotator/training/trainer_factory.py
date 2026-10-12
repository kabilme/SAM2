"""Registry and factory for deep learning training engines."""

from typing import Dict, Any, List, Type, Optional

from sam2_annotator.training.base_trainer import BaseModelTrainer, TrainingConfig
from sam2_annotator.training.yolo_trainer import YOLOTrainer


TRAINING_ENGINES: Dict[str, Dict[str, Any]] = {
    "yolo": {
        "name": "⚡ Ultralytics YOLO (Fast, Segmentation & Detection)",
        "description": "Production-ready real-time vision models with high mAP, fast convergence, and direct ONNX export.",
        "class": YOLOTrainer,
        "is_available": True,
        "tasks": ["segmentation", "detection"],
        "default_models": {
            "segmentation": [
                {"id": "yolo11n-seg.pt", "name": "YOLO11 Nano-Seg (Fastest, ~6MB)"},
                {"id": "yolo11s-seg.pt", "name": "YOLO11 Small-Seg (Balanced, ~18MB)"},
                {"id": "yolo11m-seg.pt", "name": "YOLO11 Medium-Seg (High Accuracy, ~40MB)"},
                {"id": "yolov8n-seg.pt", "name": "YOLOv8 Nano-Seg (Legacy Stable, ~6MB)"},
                {"id": "yolov8s-seg.pt", "name": "YOLOv8 Small-Seg (~22MB)"},
            ],
            "detection": [
                {"id": "yolo11n.pt", "name": "YOLO11 Nano Detection (Fastest, ~5MB)"},
                {"id": "yolo11s.pt", "name": "YOLO11 Small Detection (~19MB)"},
                {"id": "yolo11m.pt", "name": "YOLO11 Medium Detection (~42MB)"},
                {"id": "yolov8n.pt", "name": "YOLOv8 Nano Detection (~6MB)"},
                {"id": "yolov8s.pt", "name": "YOLOv8 Small Detection (~22MB)"},
            ],
        },
    },
    "maskrcnn": {
        "name": "🔬 TorchVision Mask R-CNN (Pure PyTorch Baseline)",
        "description": "Permissive BSD-3-Clause standard PyTorch baseline with ResNet-50 FPN backbone.",
        "class": None,  # Phase 3
        "is_available": False,
        "tasks": ["segmentation", "detection"],
        "default_models": {
            "segmentation": [
                {"id": "maskrcnn_resnet50_fpn_v2", "name": "Mask R-CNN ResNet-50 FPN V2"},
            ],
            "detection": [
                {"id": "fasterrcnn_resnet50_fpn_v2", "name": "Faster R-CNN ResNet-50 FPN V2"},
            ],
        },
    },
    "sam2_decoder": {
        "name": "🎯 Meta SAM 2.1 Mask Decoder (Domain Adaptation)",
        "description": "Fine-tunes SAM 2 prompt & mask decoder on annotated keyframes for enhanced video propagation.",
        "class": None,  # Phase 4
        "is_available": False,
        "tasks": ["segmentation"],
        "default_models": {
            "segmentation": [
                {"id": "sam2.1_hiera_tiny.pt", "name": "SAM 2.1 Hiera Tiny (Current In-App Model)"},
                {"id": "sam2.1_hiera_small.pt", "name": "SAM 2.1 Hiera Small"},
            ],
        },
    },
}


def get_available_engines() -> List[Dict[str, Any]]:
    """Return list of registered training engines."""
    engines = []
    for engine_id, info in TRAINING_ENGINES.items():
        engines.append({
            "id": engine_id,
            "name": info["name"],
            "description": info["description"],
            "is_available": info["is_available"],
            "tasks": info["tasks"],
            "default_models": info["default_models"],
        })
    return engines


def create_trainer(config: TrainingConfig) -> BaseModelTrainer:
    """Factory function to instantiate trainer engine for given config."""
    engine_info = TRAINING_ENGINES.get(config.engine_id)
    if not engine_info:
        raise ValueError(f"Unknown training engine '{config.engine_id}'")

    trainer_cls = engine_info.get("class")
    if not trainer_cls or not engine_info.get("is_available"):
        raise NotImplementedError(
            f"Training engine '{engine_info['name']}' is not yet implemented. Please select an available engine."
        )

    return trainer_cls(config)
