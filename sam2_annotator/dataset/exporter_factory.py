"""Factory and registry for dataset and artifact exporters."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Type

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.dataset.base_exporter import BaseDatasetExporter
from sam2_annotator.dataset.yolo_exporter import YOLOExporter
from sam2_annotator.dataset.coco_exporter import COCOExporter
from sam2_annotator.dataset.pascal_voc_exporter import PascalVOCExporter
from sam2_annotator.dataset.labelme_exporter import LabelMeExporter
from sam2_annotator.dataset.mot_exporter import MOTExporter
from sam2_annotator.dataset.rendered_video_exporter import RenderedVideoExporter


EXPORT_FORMATS: Dict[str, Dict[str, Any]] = {
    "yolo_segmentation": {
        "name": "YOLOv8 Instance Segmentation (Polygon .txt)",
        "description": "Normalized polygon coordinates (class_id x1 y1 x2 y2...), data.yaml, and optional masks.",
        "class": YOLOExporter,
        "init_kwargs": {"mode": "segmentation"},
        "supports_splits": True,
        "supports_masks": True,
        "supports_previews": True,
        "supports_zip": True,
        "is_dataset": True,
    },
    "yolo_detection": {
        "name": "YOLOv8 Object Detection (Bounding Box .txt)",
        "description": "Normalized bounding box coordinates (class_id cx cy w h) and data.yaml.",
        "class": YOLOExporter,
        "init_kwargs": {"mode": "detection"},
        "supports_splits": True,
        "supports_masks": False,
        "supports_previews": True,
        "supports_zip": True,
        "is_dataset": True,
    },
    "coco": {
        "name": "COCO 1.0 (instances_*.json)",
        "description": "Standard COCO JSON format for instance segmentation (Detectron2, MMDetection, Hugging Face).",
        "class": COCOExporter,
        "init_kwargs": {},
        "supports_splits": True,
        "supports_masks": False,
        "supports_previews": False,
        "supports_zip": True,
        "is_dataset": True,
    },
    "pascal_voc": {
        "name": "Pascal VOC & Semantic Masks (.xml + .png)",
        "description": "Pascal VOC XML bounding box & polygon tags plus 8-bit indexed palette PNG segmentation masks.",
        "class": PascalVOCExporter,
        "init_kwargs": {},
        "supports_splits": True,
        "supports_masks": False,
        "supports_previews": False,
        "supports_zip": True,
        "is_dataset": True,
    },
    "labelme": {
        "name": "LabelMe JSON (.json alongside each image)",
        "description": "Standard LabelMe shape format for desktop tool interoperability.",
        "class": LabelMeExporter,
        "init_kwargs": {},
        "supports_splits": True,
        "supports_masks": False,
        "supports_previews": False,
        "supports_zip": True,
        "is_dataset": True,
    },
    "mot": {
        "name": "MOT / MOTChallenge Tracking (gt.txt + seqinfo.ini)",
        "description": "Multi-Object Tracking sequence format with persistent object IDs (ByteTrack, DeepSORT).",
        "class": MOTExporter,
        "init_kwargs": {},
        "supports_splits": False,
        "supports_masks": False,
        "supports_previews": False,
        "supports_zip": True,
        "is_dataset": True,
    },
    "rendered_video": {
        "name": "Annotated Video with Overlays (.mp4)",
        "description": "Self-contained MP4 video with color-coded polygon fills, outlines, labels, and tracking IDs.",
        "class": RenderedVideoExporter,
        "init_kwargs": {},
        "supports_splits": False,
        "supports_masks": False,
        "supports_previews": False,
        "supports_zip": False,
        "is_dataset": False,
    },
}


def get_available_formats() -> List[Dict[str, Any]]:
    """Return list of available export formats with their metadata."""
    formats = []
    for fmt_id, info in EXPORT_FORMATS.items():
        formats.append({
            "id": fmt_id,
            "name": info["name"],
            "description": info["description"],
            "supports_splits": info["supports_splits"],
            "supports_masks": info["supports_masks"],
            "supports_previews": info["supports_previews"],
            "supports_zip": info["supports_zip"],
            "is_dataset": info["is_dataset"],
        })
    return formats


def create_exporter(
    format_id: str,
    output_dir: Path,
    classes: List[ClassItem],
    frames_dir: Path,
    **kwargs: Any,
) -> BaseDatasetExporter:
    """Instantiate and return an exporter for the specified format."""
    fmt_key = format_id.lower()
    if fmt_key not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format '{format_id}'. Available: {list(EXPORT_FORMATS.keys())}")

    info = EXPORT_FORMATS[fmt_key]
    exporter_cls: Type[BaseDatasetExporter] = info["class"]
    init_kwargs = dict(info["init_kwargs"])
    init_kwargs.update(kwargs)

    return exporter_cls(
        output_dir=output_dir,
        classes=classes,
        frames_dir=frames_dir,
        **init_kwargs,
    )
