"""Dataset splitting strategies (sequential, grouped, and random)."""

import math
import random
from typing import List, Dict, Tuple
from sam2_annotator.video.frame_extractor import FrameMetadata
from sam2_annotator.utils.logging_utils import logger


class DatasetSplitter:
    """Assigns frames to train, val, and test splits."""

    @staticmethod
    def split_frames(
        frames: List[FrameMetadata],
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        strategy: str = "sequential",
        seed: int = 42,
    ) -> Dict[str, List[FrameMetadata]]:
        """Split frames according to chosen strategy to minimize video temporal leakage."""
        if not frames:
            return {"train": [], "val": [], "test": []}

        # Normalize ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio <= 0:
            train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
            total_ratio = 1.0

        t_r = train_ratio / total_ratio
        v_r = val_ratio / total_ratio

        n = len(frames)
        train_count = int(round(n * t_r))
        val_count = int(round(n * v_r))
        # Ensure at least 1 in val/test if enough frames exist
        if val_ratio > 0 and val_count == 0 and n >= 2:
            val_count = 1
            train_count = max(1, train_count - 1)

        test_count = max(0, n - train_count - val_count)

        if strategy == "random":
            shuffled = list(frames)
            rng = random.Random(seed)
            rng.shuffle(shuffled)
            train_frames = shuffled[:train_count]
            val_frames = shuffled[train_count : train_count + val_count]
            test_frames = shuffled[train_count + val_count :]

        elif strategy == "grouped":
            # Group into contiguous blocks of ~10 frames to avoid adjacent frame leakage
            block_size = max(5, min(30, n // 10 if n >= 50 else 5))
            blocks = [frames[i : i + block_size] for i in range(0, n, block_size)]
            rng = random.Random(seed)
            rng.shuffle(blocks)

            train_frames, val_frames, test_frames = [], [], []
            for b in blocks:
                if len(train_frames) < train_count:
                    train_frames.extend(b)
                elif len(val_frames) < val_count:
                    val_frames.extend(b)
                else:
                    test_frames.extend(b)

        else: # "sequential" (preferred for video datasets)
            train_frames = frames[:train_count]
            val_frames = frames[train_count : train_count + val_count]
            test_frames = frames[train_count + val_count :]

        logger.info(
            "Split %d frames (%s strategy): train=%d, val=%d, test=%d",
            n, strategy, len(train_frames), len(val_frames), len(test_frames)
        )
        return {
            "train": train_frames,
            "val": val_frames,
            "test": test_frames,
        }
