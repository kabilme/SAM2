"""Root entry point for SAM3 Video Polygon Annotator (CLI & GUI)."""

import argparse
import sys
from pathlib import Path

# Ensure sam3_annotator package is in sys.path
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sam3_annotator.utils.logging_utils import logger


def main():
    parser = argparse.ArgumentParser(description="SAM3 Video Polygon Annotator")
    parser.add_argument("--project", type=str, default=None, help="Path to project directory to open")
    parser.add_argument("--extract-frames", type=str, default=None, help="Extract frames from video (CLI mode)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for extraction or export")
    parser.add_argument("--every-n", type=int, default=10, help="Sampling step for frame extraction")
    parser.add_argument("--export", type=str, default=None, help="Path to project directory to export to YOLOv8 dataset")
    parser.add_argument("--validate-dataset", type=str, default=None, help="Path to dataset directory to validate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")

    args = parser.parse_args()

    # 1. Dataset validation mode
    if args.validate_dataset:
        from sam3_annotator.dataset.dataset_validator import DatasetValidator
        validator = DatasetValidator(Path(args.validate_dataset))
        report = validator.validate()
        print(f"Validation Result: {'PASSED' if report['valid'] else 'FAILED'}")
        print(f"Images: {report['stats']['images_count']}, Labels: {report['stats']['labels_count']}, Objects: {report['stats']['objects_count']}")
        if not report["valid"]:
            for err in report["errors"][:10]:
                print(f"  Error: {err}")
            sys.exit(1)
        sys.exit(0)

    # 2. CLI extraction mode
    if args.extract_frames:
        from sam3_annotator.video.frame_extractor import FrameExtractor
        out_dir = Path(args.output or "extracted_frames")
        extractor = FrameExtractor(Path(args.extract_frames), out_dir)
        frames = extractor.extract_frames(strategy="every_n", every_n=args.every_n)
        print(f"Extracted {len(frames)} frames to {out_dir}")
        sys.exit(0)

    # 3. CLI export mode
    if args.export:
        from sam3_annotator.project.project_manager import ProjectManager
        from sam3_annotator.annotation.annotation_manager import AnnotationManager
        from sam3_annotator.dataset.split_manager import DatasetSplitter
        from sam3_annotator.dataset.yolo_exporter import YOLOExporter
        from sam3_annotator.dataset.dataset_validator import DatasetValidator

        proj_mgr = ProjectManager()
        anno_mgr = AnnotationManager()
        if not proj_mgr.load_project(Path(args.export), anno_mgr):
            logger.error("Failed to load project from %s", args.export)
            sys.exit(1)

        split_dict = DatasetSplitter.split_frames(proj_mgr.frames, strategy="sequential")
        out_dir = Path(args.output or "exported_dataset")
        exporter = YOLOExporter(out_dir, proj_mgr.classes, proj_mgr.frames_dir)
        exporter.export_dataset(split_dict, anno_mgr.frame_annotations)

        validator = DatasetValidator(out_dir)
        validator.validate()
        print(f"Exported dataset to {out_dir}")
        sys.exit(0)

    # 4. Default: Launch GUI application
    from sam3_annotator.app import run_app
    run_app(project_path=args.project, device=args.device)


if __name__ == "__main__":
    main()
