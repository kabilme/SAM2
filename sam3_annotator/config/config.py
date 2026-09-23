"""Application configuration manager."""

import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import yaml

from sam3_annotator.utils.logging_utils import logger


@dataclass
class ModelConfig:
    device: str = "auto"
    precision: str = "fp32"
    checkpoint_path: str = ""
    default_model_type: str = "sam2.1_t.pt"


@dataclass
class FrameConfig:
    default_sampling: str = "every_n"
    every_n: int = 10
    interval_seconds: float = 1.0
    fixed_count: int = 100
    cache_size_limit: int = 100
    jpeg_quality: int = 95
    image_format: str = "jpg"


@dataclass
class PolygonConfig:
    simplify_tolerance: float = 0.005
    min_area: float = 20.0
    morphology_cleanup: bool = True
    remove_duplicate_vertices: bool = True


@dataclass
class DatasetConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    split_strategy: str = "sequential"
    export_masks: bool = True
    export_previews: bool = True
    create_zip: bool = True


@dataclass
class UIConfig:
    mask_opacity: float = 0.45
    vertex_radius: int = 5
    line_width: int = 2
    autosave_interval_seconds: int = 30
    dark_theme: bool = True


@dataclass
class AppConfig:
    name: str = "SAM3 Video Polygon Annotator"
    version: str = "1.0.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from YAML file or return defaults."""
        if path is None:
            path = Path(__file__).parent / "defaults.yaml"

        if not path.exists():
            logger.info("Configuration file %s not found. Using default values.", path)
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            cfg = cls()
            if "app" in data:
                cfg.name = data["app"].get("name", cfg.name)
                cfg.version = data["app"].get("version", cfg.version)

            if "model" in data:
                m = data["model"]
                cfg.model = ModelConfig(
                    device=m.get("device", cfg.model.device),
                    precision=m.get("precision", cfg.model.precision),
                    checkpoint_path=m.get("checkpoint_path", cfg.model.checkpoint_path),
                    default_model_type=m.get("default_model_type", cfg.model.default_model_type),
                )

            if "frame" in data:
                fr = data["frame"]
                cfg.frame = FrameConfig(
                    default_sampling=fr.get("default_sampling", cfg.frame.default_sampling),
                    every_n=fr.get("every_n", cfg.frame.every_n),
                    interval_seconds=float(fr.get("interval_seconds", cfg.frame.interval_seconds)),
                    fixed_count=fr.get("fixed_count", cfg.frame.fixed_count),
                    cache_size_limit=fr.get("cache_size_limit", cfg.frame.cache_size_limit),
                    jpeg_quality=fr.get("jpeg_quality", cfg.frame.jpeg_quality),
                    image_format=fr.get("image_format", cfg.frame.image_format),
                )

            if "polygon" in data:
                p = data["polygon"]
                cfg.polygon = PolygonConfig(
                    simplify_tolerance=float(p.get("simplify_tolerance", cfg.polygon.simplify_tolerance)),
                    min_area=float(p.get("min_area", cfg.polygon.min_area)),
                    morphology_cleanup=bool(p.get("morphology_cleanup", cfg.polygon.morphology_cleanup)),
                    remove_duplicate_vertices=bool(p.get("remove_duplicate_vertices", cfg.polygon.remove_duplicate_vertices)),
                )

            if "dataset" in data:
                d = data["dataset"]
                cfg.dataset = DatasetConfig(
                    train_ratio=float(d.get("train_ratio", cfg.dataset.train_ratio)),
                    val_ratio=float(d.get("val_ratio", cfg.dataset.val_ratio)),
                    test_ratio=float(d.get("test_ratio", cfg.dataset.test_ratio)),
                    split_strategy=d.get("split_strategy", cfg.dataset.split_strategy),
                    export_masks=bool(d.get("export_masks", cfg.dataset.export_masks)),
                    export_previews=bool(d.get("export_previews", cfg.dataset.export_previews)),
                    create_zip=bool(d.get("create_zip", cfg.dataset.create_zip)),
                )

            if "ui" in data:
                u = data["ui"]
                cfg.ui = UIConfig(
                    mask_opacity=float(u.get("mask_opacity", cfg.ui.mask_opacity)),
                    vertex_radius=int(u.get("vertex_radius", cfg.ui.vertex_radius)),
                    line_width=int(u.get("line_width", cfg.ui.line_width)),
                    autosave_interval_seconds=int(u.get("autosave_interval_seconds", cfg.ui.autosave_interval_seconds)),
                    dark_theme=bool(u.get("dark_theme", cfg.ui.dark_theme)),
                )

            return cfg
        except Exception as e:
            logger.error("Error loading config from %s: %s", path, e)
            return cls()

    def save(self, path: Path) -> None:
        """Save current configuration to a YAML file."""
        data = {
            "app": {"name": self.name, "version": self.version},
            "model": asdict(self.model),
            "frame": asdict(self.frame),
            "polygon": asdict(self.polygon),
            "dataset": asdict(self.dataset),
            "ui": asdict(self.ui),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
