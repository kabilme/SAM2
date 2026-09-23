"""Command line interface for YOLOv8 dataset export."""

import argparse
import sys
from pathlib import Path

from sam3_annotator.project.project_manager import ProjectManager
from sam3_annotator.annotation.annotation_manager import AnnotationManager
from sam3_annotator.dataset.split_manager import DatasetSplitter
from sam3_annotator.dataset.yolo_exporter import YOLOExporter
from sam3_annotator.dataset.dataset_validator import DatasetValidator
from sam3_annotator.utils.logging_utils import logger


def main():
    parser = argparse.ArgumentParser(description="Export SAM 3 project to YOLOv8 instance segmentation dataset.")
    parser.add_argument("--project", type=str, required=True, help="Path to project directory")
    parser.add_argument("--output", type=str, default="exported_dataset", help="Output dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--split-strategy", type=str, default="sequential", choices=["sequential", "grouped", "random"], help="Dataset splitting strategy")
    parser.add_argument("--no-masks", action="store_true", help="Do not export binary mask PNGs")
    parser.add_argument("--no-previews", action="store_true", help="Do not export visual preview images")
    parser.add_argument("--zip", action="store_true", default=True, help="Create dataset ZIP archive")

    args = parser.parse_args()

    project_dir = Path(args.project)
    if not project_dir.exists():
        logger.error("Project directory does not exist: %s", project_dir)
        sys.exit(1)

    proj_mgr = ProjectManager()
    anno_mgr = AnnotationManager()

    if not proj_mgr.load_project(project_dir, anno_mgr):
        logger.error("Failed to load project from %s", project_dir)
        sys.exit(1)

    # Split frames
    split_dict = DatasetSplitter.split_frames(
        frames=proj_mgr.frames,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        strategy=args.split_strategy,
    )

    out_dir = Path(args.output)
    exporter = YOLOExporter(
        output_dir=out_dir,
        classes=proj_mgr.classes,
        frames_dir=proj_mgr.frames_dir,
    )

    def print_progress(cur, tot, msg):
        print(f"[{cur}/{tot}] {msg}")

    result = exporter.export_dataset(
        split_dict=split_dict,
        annotations_by_frame=anno_mgr.frame_annotations,
        export_masks=not args.no_masks,
        export_previews=not args.no_previews,
        create_zip=args.zip,
        progress_callback=print_progress,
    )

    print("\nDataset export finished! Running validation...")
    validator = DatasetValidator(out_dir)
    report = validator.validate()

    if report["valid"]:
        print("Dataset validation: PASSED!")
    else:
        print(f"Dataset validation: FAILED with {len(report['errors'])} errors.")
        for err in report["errors"][:5]:
            print(f" - {err}")


if __name__ == "__main__":
    main()
