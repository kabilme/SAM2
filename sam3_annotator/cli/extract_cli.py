"""Command line interface for automated video frame extraction."""

import argparse
import sys
from pathlib import Path

from sam3_annotator.video.frame_extractor import FrameExtractor
from sam3_annotator.utils.logging_utils import logger


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for SAM 3 annotation.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="extracted_frames", help="Output directory for frames")
    parser.add_argument("--strategy", type=str, default="every_n", choices=["every_n", "every_frame", "fixed_count", "interval_seconds"], help="Extraction strategy")
    parser.add_argument("--every-n", type=int, default=10, help="Extract every Nth frame")
    parser.add_argument("--interval-seconds", type=float, default=1.0, help="Extract interval in seconds")
    parser.add_argument("--fixed-count", type=int, default=100, help="Fixed count of frames to extract")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to extract (for testing)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality [1-100]")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video file does not exist: %s", video_path)
        sys.exit(1)

    out_dir = Path(args.output)
    extractor = FrameExtractor(video_path=video_path, output_dir=out_dir)

    def print_progress(cur, tot, msg):
        print(f"[{cur}/{tot}] {msg}")

    try:
        frames = extractor.extract_frames(
            strategy=args.strategy,
            every_n=args.every_n,
            interval_seconds=args.interval_seconds,
            fixed_count=args.fixed_count,
            end_frame=args.max_frames,
            jpeg_quality=args.jpeg_quality,
            progress_callback=print_progress,
        )
        print(f"\nSuccessfully extracted {len(frames)} frames into {out_dir}")
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
