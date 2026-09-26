from pathlib import Path
import shutil
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam2_annotator.video.video_reader import VideoReader
from sam2_annotator.video.frame_extractor import FrameExtractor
from sam2_annotator.project.project_manager import ProjectManager
from sam2_annotator.annotation.annotation_manager import AnnotationManager
from sam2_annotator.models.sam2_adapter import MockSAM2Adapter, SAM2LocalAdapter
from sam2_annotator.models.sam2_image_service import SAM2ImageService
from sam2_annotator.models.sam2_video_service import SAM2VideoService
from sam2_annotator.annotation.polygon_editor import PolygonEditor
from sam2_annotator.dataset.split_manager import DatasetSplitter
from sam2_annotator.dataset.yolo_exporter import YOLOExporter
from sam2_annotator.dataset.dataset_validator import DatasetValidator
from sam2_annotator.utils.logging_utils import logger


def main():
    print("==================================================")
    print("SAM2 Video Polygon Annotator: End-to-End Workflow")
    print("==================================================")

    video_path = Path("D:/BestScooter/sample_scooter.mp4")
    if not video_path.exists():
        print(f"Error: Sample video {video_path} not found.")
        return 1

    project_dir = Path("example_project").resolve()
    dataset_dir = Path("example_dataset").resolve()

    if project_dir.exists():
        shutil.rmtree(project_dir, ignore_errors=True)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)

    # 1. Video Import
    print("\n[Step 1] Reading Video Metadata...")
    reader = VideoReader(video_path)
    meta = reader.metadata
    print(f"  - File: {meta.filename}")
    print(f"  - Resolution: {meta.width}x{meta.height}")
    print(f"  - FPS: {meta.fps}")
    print(f"  - Total Frames: {meta.total_frames}")
    print(f"  - Duration: {meta.formatted_duration}")
    reader.close()

    # 2. Project Creation
    print("\n[Step 2] Initializing Project...")
    proj_mgr = ProjectManager()
    anno_mgr = AnnotationManager()
    proj_mgr.create_project(
        project_dir=project_dir,
        project_name="ScooterAnnotationDemo",
        video_metadata=meta,
        class_names=["scooter", "person", "helmet"],
    )
    print(f"  - Created project in: {project_dir}")
    print(f"  - Classes: {[c.name for c in proj_mgr.classes]}")

    # 3. Frame Extraction
    print("\n[Step 3] Extracting Frames (Sampling: every 15 frames, up to 10 frames)...")
    extractor = FrameExtractor(video_path, proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
    extracted_frames = extractor.extract_frames(
        strategy="every_n",
        every_n=15,
        end_frame=150,
        progress_callback=lambda c, t, m: print(f"    {m}"),
    )
    proj_mgr.set_frames(extracted_frames)
    proj_mgr.save_project(anno_mgr)
    print(f"  - Extracted {len(extracted_frames)} frames.")

    # 4. SAM 2 Annotation
    print("\n[Step 4] Running SAM 2 Prompt Segmentation on Frame 1...")
    adapter = MockSAM2Adapter()
    adapter.load_model(device="cpu")
    img_service = SAM2ImageService(adapter)

    frame1 = extracted_frames[0]
    img1 = proj_mgr.project_dir / "frames" / frame1.filename
    import cv2
    img_bgr = cv2.imread(str(img1))

    # Positive point prompt on object
    center_pt = [(float(frame1.width // 2), float(frame1.height // 2))]
    anno1 = img_service.segment_points_to_annotation(
        image_bgr=img_bgr,
        frame_id=frame1.frame_id,
        source_frame_index=frame1.source_frame_index,
        class_id=0,
        class_name="scooter",
        positive_points=center_pt,
    )
    assert anno1 is not None, "Failed to segment object with SAM 2"
    anno_mgr.add_annotation(anno1)
    print(f"  - Generated Object: {anno1.object_id} (Class: {anno1.class_name}, Area: {anno1.area:.1f}px, Vertices: {len(anno1.points)})")

    # 5. Manual Polygon Refinement
    print("\n[Step 5] Manually Refining Polygon Vertex...")
    first_pt = anno1.points[0]
    PolygonEditor.move_vertex(anno1, 0, (first_pt[0] + 5.0, first_pt[1] + 5.0))
    print(f"  - Vertex 0 adjusted from {first_pt} to {anno1.points[0]}")

    # 6. Video Object Tracking / Propagation
    print("\n[Step 6] Propagating Object across subsequent frames...")
    from sam2_annotator.video.frame_cache import FrameCache
    cache = FrameCache(proj_mgr.frames_dir, proj_mgr.thumbnails_dir)
    vid_service = SAM2VideoService(adapter, cache)

    target_frames = extracted_frames[1:5]
    propagated = vid_service.propagate_object(
        initial_annotation=anno1,
        target_frames=target_frames,
        progress_callback=lambda c, t, m: print(f"    {m}"),
    )
    for a in propagated:
        anno_mgr.add_annotation(a)
    print(f"  - Propagated across {len(propagated)} frames.")

    # Save Project
    proj_mgr.save_project(anno_mgr)
    print(f"  - Project persisted atomically with annotations.")

    # 7. Dataset Split & Export
    print("\n[Step 7] Exporting YOLOv8 Instance Segmentation Dataset...")
    split_dict = DatasetSplitter.split_frames(
        frames=extracted_frames,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        strategy="sequential",
    )
    exporter = YOLOExporter(
        output_dir=dataset_dir,
        classes=proj_mgr.classes,
        frames_dir=proj_mgr.frames_dir,
    )
    exp_summary = exporter.export_dataset(
        split_dict=split_dict,
        annotations_by_frame=anno_mgr.frame_annotations,
        export_masks=True,
        export_previews=True,
        create_zip=True,
        progress_callback=lambda c, t, m: print(f"    {m}"),
    )
    print(f"  - Export Summary: {exp_summary['total_images']} images, {exp_summary['total_objects']} objects.")
    print(f"  - ZIP Archive: {exp_summary['zip_path']}")

    # 8. Automated Dataset Validation
    print("\n[Step 8] Validating Exported Dataset...")
    validator = DatasetValidator(dataset_dir)
    val_report = validator.validate()
    print(f"  - Status: {'PASSED [OK]' if val_report['valid'] else 'FAILED'}")
    print(f"  - Images: {val_report['stats']['images_count']}, Labels: {val_report['stats']['labels_count']}")
    print(f"  - Errors: {len(val_report['errors'])}, Warnings: {len(val_report['warnings'])}")

    if not val_report["valid"]:
        for e in val_report["errors"]:
            print(f"    Error: {e}")
        return 1

    print("\n==================================================")
    print("ALL WORKFLOW CRITERIA MET AND VERIFIED SUCCESSFULLY!")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
