"""MOT (Multiple Object Tracking) / MOTChallenge format exporter."""

import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from sam2_annotator.project.project_schema import ClassItem
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.utils.logging_utils import logger
from sam2_annotator.dataset.base_exporter import BaseDatasetExporter


class MOTExporter(BaseDatasetExporter):
    """Exports video frames and tracked annotations into MOTChallenge tracking format."""

    def export(
        self,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        sequence_name: str = "seq01",
        fps: float = 30.0,
        create_zip: bool = True,
        include_null_frames: bool = True,
        zip_name: str = "mot_dataset.zip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute MOT dataset export."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seq_dir = self.output_dir / sequence_name
        img_dir = seq_dir / "img1"
        gt_dir = seq_dir / "gt"

        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Collect all frames sorted by frame_id
        all_frames: List[FrameMetadata] = []
        for f_list in split_dict.values():
            all_frames.extend(f_list)
        all_frames = sorted(all_frames, key=lambda f: f.frame_id)

        total_frames = len(all_frames)
        processed_count = 0
        total_objects_exported = 0
        annotated_frames_exported = 0
        null_frames_exported = 0
        class_counts: Dict[str, int] = {c.name: 0 for c in self.classes}
        warnings: List[str] = []

        # Map string object_ids (e.g. uuid) to unique integer track_ids (1..N)
        track_id_map: Dict[str, int] = {}
        next_track_id = 1

        gt_lines: List[str] = []
        im_width = 1920
        im_height = 1080

        logger.info("Beginning MOT dataset export to %s", seq_dir)

        for idx, frame in enumerate(all_frames, start=1):
            if is_cancelled and is_cancelled():
                logger.info("MOT export cancelled by user.")
                return {"status": "cancelled"}

            im_width = frame.width
            im_height = frame.height

            src_image_path = self.frames_dir / frame.filename
            if not src_image_path.exists():
                warnings.append(f"Source frame missing: {frame.filename}")
                continue

            # Standard MOT image naming: 000001.jpg, 000002.jpg ...
            dest_image_name = f"{idx:06d}.jpg"
            dest_image_path = img_dir / dest_image_name
            shutil.copy2(src_image_path, dest_image_path)

            annos = annotations_by_frame.get(frame.frame_id, [])
            if not annos:
                null_frames_exported += 1
            else:
                annotated_frames_exported += 1

            for anno in annos:
                if len(anno.points) < 3:
                    continue

                obj_key = anno.object_id if anno.object_id else f"obj_f{frame.frame_id}_{anno.class_id}"
                if obj_key not in track_id_map:
                    track_id_map[obj_key] = next_track_id
                    next_track_id += 1
                int_track_id = track_id_map[obj_key]

                xmin, ymin, xmax, ymax, w, h, _, _ = self.get_polygon_bbox(anno.points)

                # MOT gt format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class_id>, <visibility>
                line = (
                    f"{idx},{int_track_id},{xmin:.2f},{ymin:.2f},{w:.2f},{h:.2f},"
                    f"1,{anno.class_id + 1},1.0"
                )
                gt_lines.append(line)
                total_objects_exported += 1
                c_name = self.class_map.get(anno.class_id, "object")
                class_counts[c_name] = class_counts.get(c_name, 0) + 1

            processed_count += 1
            if progress_callback:
                progress_callback(
                    processed_count,
                    total_frames,
                    f"MOT export: {processed_count}/{total_frames} frames",
                )

        # Write gt/gt.txt
        with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(gt_lines) + ("\n" if gt_lines else ""))

        # Write seqinfo.ini
        seqinfo_content = (
            f"[Sequence]\n"
            f"name={sequence_name}\n"
            f"imDir=img1\n"
            f"frameRate={fps:.1f}\n"
            f"seqLength={processed_count}\n"
            f"imWidth={im_width}\n"
            f"imHeight={im_height}\n"
            f"imExt=.jpg\n"
        )
        with open(seq_dir / "seqinfo.ini", "w", encoding="utf-8") as f:
            f.write(seqinfo_content)

        # Dataset README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# MOT (Multiple Object Tracking) Dataset\n\n")
            f.write(f"- Sequence Name: `{sequence_name}`\n")
            f.write(f"- Total Frames: {processed_count}\n")
            f.write(f"- Unique Tracked Objects: {len(track_id_map)}\n")
            f.write(f"- Total Detections: {total_objects_exported}\n")
            f.write(f"- Frame Resolution: {im_width}x{im_height}\n\n")
            f.write("## Structure:\n")
            f.write("- `img1/`: Sequential frames named `000001.jpg`, `000002.jpg`, etc.\n")
            f.write("- `gt/gt.txt`: Ground truth tracking annotations in MOTChallenge CSV format.\n")
            f.write("- `seqinfo.ini`: Sequence metadata configuration.\n\n")
            f.write("## Usage:\n")
            f.write("Compatible with TrackEval, ByteTrack, DeepSORT, and OC-SORT evaluation benchmarks.\n")

        # Create ZIP archive if requested
        zip_path = None
        if create_zip:
            zip_path = self.create_zip_archive(self.output_dir, zip_name, progress_callback, total_frames)

        if progress_callback:
            progress_callback(processed_count, total_frames, "MOT dataset export complete!")

        summary = {
            "status": "success",
            "format": "mot",
            "output_dir": str(self.output_dir),
            "sequence_name": sequence_name,
            "total_images": processed_count,
            "total_tracks": len(track_id_map),
            "total_objects": total_objects_exported,
            "warnings": warnings,
            "zip_path": zip_path,
        }
        logger.info("MOT dataset export completed successfully: %s", summary)
        return summary
