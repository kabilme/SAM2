# Troubleshooting Guide

This guide covers common issues, debugging steps, performance optimizations, and error resolution.

---

## 1. Application Log Files

All internal logs, warnings, and errors are written in real-time to:
```
logs/app.log
```
To monitor application logs live in PowerShell:
```powershell
Get-Content logs/app.log -Wait -Tail 50
```
On Linux / macOS:
```bash
tail -f logs/app.log
```

---

## 2. Common Issues & Solutions

### A. Missing `hydra` or `tqdm` Dependencies
- **Symptom**: `ModuleNotFoundError: No module named 'hydra'` or `No module named 'tqdm'` upon launching the app or initiating video tracking.
- **Cause**: Meta SAM 2.1 video predictor uses Hydra for configuration management and tqdm for internal progress dispatch.
- **Resolution**:
  ```bash
  pip install hydra-core>=1.3.2 tqdm>=4.66.1
  # or re-sync all dependencies:
  pip install -r requirements.txt
  ```

### B. CUDA Out of Memory (OOM)
- **Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory` during segmentation or batch propagation.
- **Cause**: Very high video resolution (e.g., 4K/60fps) or long sequences overwhelming GPU VRAM.
- **Resolution**:
  1. The tracking engine automatically segments long sequences into **60-frame memory chunks** (`MAX_PROPAGATION_CHUNK = 60`) to cap memory token accumulation.
  2. Automatic memory recovery calls `device_utils.clear_memory()` to flush CUDA caches and Python garbage collectors.
  3. Open **Settings > Model Settings** and enable `fp16` half-precision.
  4. If working on low-VRAM GPUs (< 4 GB), run on CPU:
     ```bash
     python main.py --device cpu
     ```

### C. Windows CPU Post-Processing Warning (`_C` Extension)
- **Symptom**: Terminal output displays a warning regarding missing `_C` extension or MSVC compiler.
- **Explanation**: Meta's default repository includes an optional C++ postprocessing kernel. In this application, `build_sam2_video_predictor` is initialized with `apply_postprocessing=False`.
- **Resolution**: No action required. Mask vectorization, contour extraction, and polygon simplification are handled natively by OpenCV topological contours and Ramer-Douglas-Peucker (RDP) algorithms, completely bypassing the need for C++ compiler tools on Windows.

### D. SAM Model Checkpoint Fails to Load
- **Symptom**: Error: `SAM model is not loaded` or weights file not found.
- **Cause**: Missing `sam2.1_hiera_tiny.pt` in an offline environment.
- **Resolution**: Pre-download the checkpoint and place it in the `checkpoints/` directory or project root:
  ```powershell
  # Windows PowerShell
  Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" -OutFile "sam2.1_hiera_tiny.pt"
  ```
  ```bash
  # Linux / macOS
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
  ```

### E. Black or Missing Video Frame Previews
- **Symptom**: Frame images fail to display on canvas or thumbnails are blank.
- **Cause**: OpenCV `VideoCapture` failed to decode a proprietary or unsupported video container/codec (e.g. ProRes, HEVC/H.265 without OS codecs).
- **Resolution**: Transcode the video to standard H.264 MP4 with YUV420p pixel format:
  ```bash
  ffmpeg -i input_video.mov -vcodec libx264 -pix_fmt yuv420p output_video.mp4
  ```

### F. UI Latency on Completion of Long Video Propagation
- **Symptom**: Previous versions took several seconds to display the completion alert after propagating across 50+ frames.
- **Resolution**: Current version integrates two optimizations:
  1. `AnnotationManager.add_annotations()` applies all tracked frames in a single batch undo snapshot, preventing hundreds of redundant UI thumbnail filter redraws.
  2. `VideoPanel` caches thumbnail icons in memory (`_icon_cache`), eliminating repeat disk I/O. Ensure you are on the latest repository commit (`git pull origin master`).

### G. Dataset Validation Errors During Export
- **Symptom**: Export dialog reports validation failure (e.g. coordinates outside $[0, 1]$ or degenerate polygon).
- **Cause**: Vertex dragged outside image boundaries or overlapping vertices.
- **Resolution**: The exporter clamps boundary coordinates within $[0.0, 1.0]$ and prunes zero-area contours. Review `validation_report.txt` in the export folder to identify the specific frame and object ID.

### H. High-DPI Display Scaling on Windows
- **Symptom**: Fonts or canvas cursor appear misaligned or blurry on 4K / High-DPI displays.
- **Resolution**: The application enables Qt High-DPI scaling automatically (`QT_ENABLE_HIGHDPI_SCALING=1`). Ensure your Windows display scale is set to standard 100%, 125%, 150%, or 200%.

---

## 3. Reference Guides

- [Installation Guide](installation.md) — Dependencies and virtual environment setup.
- [SAM 2 Setup & Architecture](sam2_setup.md) — Checkpoint downloads and memory configuration.
- [User Guide](user_guide.md) — Interface operations and keyboard shortcuts.
- [Annotation & Tracking Workflow](annotation_workflow.md) — Prompting and video propagation.


