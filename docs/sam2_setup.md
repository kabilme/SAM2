# SAM 2 Model Setup & Architecture

The **SAM2 Video Polygon Annotator** utilizes an isolated adapter pattern to interface with Segment Anything models (including SAM 2.1 and SAM 2 architectures) via official local Python APIs.

---

## 1. Supported Checkpoints

The adapter layer supports official checkpoint weights:

| Model Checkpoint | Parameters | Typical Speed | Recommended Hardware |
| :--- | :--- | :--- | :--- |
| `sam2.1_t.pt` | Tiny (~38M) | Fast (~15-30ms) | CPU or 4 GB GPU (Default) |
| `sam2.1_s.pt` | Small (~46M) | Fast (~25-45ms) | 4 GB - 6 GB GPU |
| `sam2.1_b.pt` | Base (~80M) | Balanced (~40-70ms) | 6 GB - 8 GB GPU |
| `sam2.1_l.pt` | Large (~224M) | High Accuracy (~90ms) | 8 GB+ GPU |

---

## 2. Checkpoint Placement

Place downloaded model weights in the project root directory or specify a custom path in the Settings dialog:

```
D:\SAM2\
  ├── sam2.1_t.pt   <- Default lightweight checkpoint
  ├── main.py
  └── ...
```

If the checkpoint is not already present, Ultralytics will automatically download the checkpoint upon first initialization when internet access is available. For offline environments, pre-download the file and save it locally.

---

## 3. Hardware Acceleration & Device Configuration

The application automatically checks device availability on startup:

- **CUDA**: Preferred when NVIDIA GPU and CUDA runtime are present.
- **CPU**: Supported across all platforms as an offline fallback.
- **Precision**:
  - `fp32`: Standard 32-bit single precision for CPU and GPUs.
  - `fp16`: 16-bit half-precision for modern GPUs (saves ~50% VRAM and accelerates inference).

You can switch the active device dynamically through the GUI (**Tools > Hardware Diagnostics** or **Settings > Model Settings**), or via the CLI flag:
```bash
python main.py --device cuda
# or
python main.py --device cpu
```

---

## 4. OOM (Out-of-Memory) Recovery

If the model encounters an out-of-memory condition during interactive segmentation or batch tracking:
1. The error is safely caught without crashing the application.
2. `sam2_annotator.utils.device_utils.clear_memory()` flushes CUDA caches.
3. Temporary tensor buffers are released.
4. The user is prompted to reduce image resolution or switch precision to `fp16` or device to `cpu`.
5. Annotations and project JSON remain completely intact.

---

## 5. Mock Adapter for Testing

For CI/CD and headless unit testing without requiring GPU hardware or large model weights, the test suite leverages `MockSAM2Adapter` (`sam2_annotator/models/sam2_adapter.py`). The mock adapter produces synthetic circular and geometric masks matching the prompt points.
