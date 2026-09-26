# SAM 2.1 Video Polygon Annotator

> [!IMPORTANT]
> **Active Model Specification**: This application exclusively uses **Meta Segment Anything 2.1 (SAM 2.1)** for all segmentation tasks (default checkpoint: `sam2.1_hiera_tiny.pt`). This project does **not** use SAM 3 (which has not been released). While the internal project repository and package name are labeled `sam3_annotator`, the actual model loaded, executed, and integrated throughout this application is **SAM 2.1**.

A high-performance, local-first Python desktop application for interactive **SAM 2.1 assisted polygon annotation** on video frames and automated **YOLOv8 instance-segmentation dataset export**, inspired by modern computer-vision platforms like Roboflow.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Actual Model Used: SAM 2.1](#2-actual-model-used-sam-21)
3. [Key Features](#3-key-features)
4. [System Requirements](#4-system-requirements)
5. [Python Version & Dependencies](#5-python-version--dependencies)
6. [GPU Requirements & Acceleration](#6-gpu-requirements--acceleration)
7. [SAM 2.1 Architecture & Supported Checkpoints](#7-sam-21-architecture--supported-checkpoints)
8. [Installation Guide](#8-installation-guide)
9. [Running the Application](#9-running-the-application)
10. [Multi-Video Project Workflow](#10-multi-video-project-workflow)
11. [Video Frame Sampling & Extraction](#11-video-frame-sampling--extraction)
12. [Frame Deletion & Continuous Re-indexing](#12-frame-deletion--continuous-re-indexing)
13. [SAM 2.1 Assisted Annotation Workflow](#13-sam-21-assisted-annotation-workflow)
14. [Object Tracking & Multi-Frame Propagation](#14-object-tracking--multi-frame-propagation)
15. [Interactive Polygon Editing](#15-interactive-polygon-editing)
16. [Null Frames & Background Negative Samples](#16-null-frames--background-negative-samples)
17. [YOLOv8 Dataset Export](#17-yolov8-dataset-export)
18. [Built-In Dataset Validator](#18-built-in-dataset-validator)
19. [Headless CLI Pipelines](#19-headless-cli-pipelines)
20. [Configuration (`defaults.yaml`)](#20-configuration-defaultsyaml)
21. [Troubleshooting & FAQ](#21-troubleshooting--faq)
22. [Repository Structure](#22-repository-structure)
23. [License Notices](#23-license-notices)

---

## 1. Project Overview

The **SAM 2.1 Video Polygon Annotator** bridges the gap between raw video footage and production-ready YOLOv8 instance segmentation datasets. By utilizing zero-shot Segment Anything 2.1 prompting (positive/negative points and bounding boxes), annotators can generate pixel-accurate polygon outlines in milliseconds without manual point-by-point drawing.

Annotations can be propagated forward across subsequent video frames using object tracking, refined using an interactive vertex editor, and exported directly into a validated YOLOv8 instance segmentation dataset complete with `data.yaml`, normalized label files, binary masks, overlay previews, and atomic ZIP packaging.

---

## 2. Actual Model Used: SAM 2.1

This project uses **Meta Segment Anything 2.1 (SAM 2.1)**. 

- **No SAM 3**: Meta AI has not released a SAM 3 model. The application does not use or simulate any fictitious SAM 3 architecture.
- **Production Engine**: The application integrates the official **SAM 2.1** architecture using PyTorch and the Ultralytics SAM engine (`sam3_annotator/models/sam3_adapter.py`).
- **Default Checkpoint**: The project runs Meta's official **`sam2.1_hiera_tiny.pt`** (156 MB) by default, with automatic fallback support for `sam2.1_t.pt` (78 MB).
- **Dynamic Model Title**: The application UI dynamically inspects the loaded model and displays the active checkpoint name (e.g., `sam2.1_hiera_tiny.pt`) in the window title bar and toolbar.

---

## 3. Key Features

- **Local-First & Completely Private**: Runs entirely on your local workstation; never uploads videos, frames, or annotations to external cloud services.
- **SAM 2.1 Zero-Shot Segmentation**: Click positive (`+`) points to segment objects, negative (`-`) points to exclude background bleed, or draw bounding boxes for rapid object isolation.
- **Dynamic Active Model Indicator**: Displays the active SAM 2.1 model name in the main window title bar and toolbar prompt label.
- **Multi-Video Support**: Ingest multiple video files during project creation or add additional videos to an active project at any time via **File > Add Video(s) to Project...**.
- **Contour Vectorization & Simplification**: Converts binary raster masks into simplified, organic polygons using Ramer-Douglas-Peucker (RDP) contour reduction.
- **Interactive Polygon Vertex Editor**: Drag vertices, click polygon edges to insert new control points, right-click to delete vertices, and enforce image boundary clamping.
- **Multi-Frame Propagation & Tracking**: Propagate object masks forward across frame sequences using non-blocking background workers with live progress and cancelability.
- **Video Frame Deletion & Continuous Re-indexing**: Delete single frames, multiple selections, or batch empty/unreviewed frames with optional disk cleanup (images, thumbnails, annotations) and automatic re-indexing (1..N) to prevent timeline gaps.
- **Null / Background Negative Sample Support**: Mark unannotated frames as negative background samples. YOLO exporter automatically produces empty label files (`.txt`) according to Ultralytics training best practices to drastically suppress false-positive detections.
- **High-Performance Timeline & Canvas**: Hardware-accelerated `QGraphicsView` canvas with smooth mouse-centered zoom, panning, timeline scrubbing, and review status filtering (*All Frames*, *Unreviewed*, *Annotated*, *Reviewed*, *Null / Negative Frames*).
- **Comprehensive YOLOv8 Exporter**: Supports Sequential (anti-leakage), Grouped, and Random dataset splits; normalized polygon coordinates (`class_id x1 y1 x2 y2 ...`); binary mask PNGs; visual preview overlays; and ZIP packaging.
- **Automated Dataset Validator**: Verifies image-label coordinate normalization, bounding ranges $[0.0, 1.0]$, minimum vertex counts, non-empty classes, and `data.yaml` validity.
- **Safe Persistence**: Atomic JSON file writes with automatic backup restoration (`project.backup.json`) to guarantee zero data loss.

---

## 4. System Requirements

- **Operating System**: Windows 10/11 (64-bit), Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+), or macOS 12+
- **Processor**: Intel Core i5 / AMD Ryzen 5 or higher (x86_64 or Apple Silicon ARM64)
- **Memory**: Minimum 8 GB RAM (16 GB+ recommended for 4K video datasets)
- **Storage**: At least 3 GB free disk space for application dependencies and model weights

---

## 5. Python Version & Dependencies

- **Supported Python Versions**: Python 3.10, 3.11, 3.12, 3.13
- **Core Dependencies**:
  - `PySide6` (GUI framework)
  - `ultralytics` (SAM 2.1 inference engine & model loader)
  - `torch`, `torchvision` (Deep learning inference backend)
  - `opencv-python` (Video decoding and contour extraction)
  - `numpy`, `pillow` (Numerical array and image transformations)
  - `pyyaml` (Configuration management)

---

## 6. GPU Requirements & Acceleration

- **NVIDIA GPU**: Recommended for instant sub-50ms interactive segmentation (GTX 1660, RTX 2060, RTX 3060, RTX 4070 or higher).
- **VRAM Requirements**:
  - Tiny (`sam2.1_hiera_tiny.pt` / `sam2.1_t.pt`): 2–4 GB VRAM
  - Large (`sam2.1_hiera_large.pt` / `sam2.1_l.pt`): 6–8 GB+ VRAM
- **CUDA Runtime**: CUDA 11.8 or CUDA 12.x supported out of the box.
- **CPU Fallback**: 100% functional on CPU for systems without dedicated NVIDIA GPUs.

---

## 7. SAM 2.1 Architecture & Supported Checkpoints

The application uses an isolated model adapter layer (`sam3_annotator/models/sam3_adapter.py`) interfacing with the official Ultralytics SAM 2.1 implementation.

### Default Checkpoint
The default model is Meta's **`sam2.1_hiera_tiny.pt`** (156 MB), offering the optimal balance between interactive latency and crisp object boundary accuracy.

### Supported SAM 2.1 Checkpoint Types
The adapter natively recognizes and aliases both official Meta AI checkpoints and Ultralytics release weights:

| Model Checkpoint | Architecture | Weights Size | Typical Speed (GPU) | Recommended Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **`sam2.1_hiera_tiny.pt`** (Default) | Meta SAM 2.1 Hiera-Tiny | ~156 MB | 15–30 ms | CPU or 4 GB GPU |
| **`sam2.1_t.pt`** | Ultralytics SAM 2.1 Tiny | ~78 MB | 15–25 ms | CPU or 2 GB GPU |
| **`sam2.1_hiera_small.pt` / `sam2.1_s.pt`** | SAM 2.1 Hiera-Small | ~185 MB | 25–45 ms | 4–6 GB GPU |
| **`sam2.1_hiera_base_plus.pt` / `sam2.1_b.pt`**| SAM 2.1 Hiera-Base+ | ~320 MB | 40–70 ms | 6–8 GB GPU |
| **`sam2.1_hiera_large.pt` / `sam2.1_l.pt`** | SAM 2.1 Hiera-Large | ~449 MB | 70–120 ms | 8 GB+ GPU |

You can switch models dynamically in the application via **Edit > Settings** (`Ctrl+,`) or by setting `model.checkpoint_path` in `config/defaults.yaml`.

---

## 8. Installation Guide

```bash
# 1. Clone or navigate to the repository
cd D:/SAM3

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Optional: Install package in editable development mode
pip install -e .
```

---

## 9. Running the Application

### Launch Graphical Interface:
```bash
python main.py
```

### Launch GUI with a specific project or device override:
```bash
# Open an existing project on CUDA
python main.py --project projects/my_project --device cuda

# Force CPU inference
python main.py --device cpu
```

---

## 10. Multi-Video Project Workflow

The application supports annotating frames extracted from multiple video sources inside a single unified project:

### 1. Creating a Multi-Video Project
1. Press `Ctrl+N` or click **New Project**.
2. Set **Project Name** and choose an **Output Directory** (e.g. `projects/dataset_project`).
3. Click **Browse** next to Input Video Files and select one or multiple video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
4. Set extraction sampling settings (applied uniformly across all selected videos).
5. Define your target annotation classes (e.g. `car`, `pedestrian`, `traffic_sign`).
6. Click **Create Project**. The background worker extracts frames from each video sequentially, assigning continuous frame IDs (`#1`, `#2`, `#3`...) while preserving each frame's source video filename in metadata.

### 2. Adding Videos to an Existing Project
1. Open your project.
2. Select **File > Add Video(s) to Project...** from the menu bar.
3. Choose one or more additional video files.
4. Frames from the new videos are extracted in the background, appended to the existing timeline with continuous IDs, and saved into project metadata automatically.

---

## 11. Video Frame Sampling & Extraction

Frames can be sampled during project creation or via headless CLI:

- **Every N Frames** (`every_n`): Uniform interval sampling (e.g., extract every 10th or 15th frame).
- **Interval in Seconds** (`interval_seconds`): Extract 1 frame every $S$ seconds of video playback.
- **Fixed Count** (`fixed_count`): Evenly distributes a target total number of frames across the entire video duration.
- **Every Frame** (`every_frame`): Sequential extraction of every single frame.

---

## 12. Frame Deletion & Continuous Re-indexing

Extracted video footage frequently contains blurry, redundant, or empty frames. The application provides dedicated frame deletion tools to keep projects clean:

### How to Delete Frames:
1. **Single Active Frame**: Press `Ctrl+Delete`, click the **Delete Frame** button in the video toolbar, or select **Edit > Delete Current Frame**.
2. **Multiple Selected Frames**: In the Video Panel thumbnail list, hold `Ctrl` or `Shift` to select multiple frames, then press the `🗑 Delete` button.
3. **Right-Click Context Menu**: Right-click any thumbnail and choose **Delete Selected Frame(s)...**.

### Deletion Features:
- **Confirmation Dialog**: Inspects frame ID, timestamp, source video, filename, and existing annotation count before deletion.
- **Disk Cleanup Option**: Check **"Delete image and thumbnail files from disk"** to permanently remove the corresponding `.jpg` frame, thumbnail, and `.json` annotation files from storage.
- **Continuous Re-indexing**: Remaining frames are automatically re-indexed continuously from `1..N` (no missing ID gaps), and all annotations and timeline positions are updated atomically.

---

## 13. SAM 2.1 Assisted Annotation Workflow

1. In the **Class Panel** (right dock), select the class label for the object.
2. Click the **Positive Point** tool (`2` on keyboard) in the toolbar.
3. Left-click inside the target object on the canvas. SAM 2.1 generates a segmentation mask.
4. If parts of the object are omitted, click additional positive points.
5. If background pixels are included, select the **Negative Point** tool (`3` on keyboard) and click the background region to subtract it.
6. Alternatively, select **Bounding Box** (`4` on keyboard) and drag a box around the object.
7. The mask converts automatically into a simplified, editable polygon outline.

---

## 14. Object Tracking & Multi-Frame Propagation

1. Select the completed polygon in the current frame.
2. Click **Propagate Object** (`Ctrl+P` or toolbar icon).
3. Specify the number of forward frames to track (e.g. next 10, 30, or all remaining frames).
4. The background propagation worker tracks the object across subsequent frames, generating connected polygon annotations.
5. Scrub through the timeline (`Left` / `Right` arrow keys) to review and refine the tracked polygons.

---

## 15. Interactive Polygon Editing

Switch to **Edit Mode** (`E` on keyboard) or double-click an existing polygon:

- **Drag Vertices**: Left-click and drag any vertex control point.
- **Add Vertices**: Click directly on an edge between two vertices to insert a new control point.
- **Delete Vertices**: Right-click on a vertex handle and choose **Delete Vertex**.
- **Delete Annotation**: Select the polygon and press `Delete` or `Backspace`.
- **Change Class**: Right-click the polygon or change the active class in the Class Panel.
- **Undo / Redo**: Use `Ctrl+Z` and `Ctrl+Y` to undo or redo vertex movements and deletions.

---

## 16. Null Frames & Background Negative Samples

In real-world object detection and instance segmentation, training models exclusively on images containing objects leads to high false-positive rates on empty backgrounds.

### Handling Null / Background Frames:
- **Marking Negative Frames**: Frames without objects can be marked as **Null / Negative Frame** via right-click in the timeline or Video Panel.
- **Timeline Review Filtering**: Filter frames by status:
  - ⚪ **Unreviewed**
  - 🟢 **Annotated**
  - 🔵 **Reviewed**
  - ⬛ **Null / Negative Frame**
- **YOLO Export Integration**: During YOLO dataset export, checking **"Export unannotated frames as null / negative frames"** copies the background images into the dataset and generates corresponding **empty `.txt` label files** (0 bytes), following official Ultralytics guidelines for negative background samples.

---

## 17. YOLOv8 Dataset Export

1. Click **Export Dataset** (`Ctrl+E` or **File > Export Dataset...**).
2. Set train / validation / test split ratios (default: `70% train`, `20% val`, `10% test`).
3. Select the split strategy:
   - **Sequential**: Prevents temporal data leakage across adjacent video frames.
   - **Grouped**: Splits continuous frame segments.
   - **Random**: Randomly distributes frames across splits.
4. Select export artifacts:
   - **Labels**: Normalized polygon coordinates (`class_id x1 y1 x2 y2 ...`).
   - **Null / Negative Frames**: Empty label files for background training.
   - **Binary Masks**: 8-bit single-channel PNG masks per frame.
   - **Preview Images**: Color-coded overlay preview images for rapid visual auditing.
   - **Create ZIP Archive**: Automatically bundles the dataset into `dataset.zip`.
5. Click **Start Export**.

---

## 18. Built-In Dataset Validator

Every exported dataset is automatically validated upon export. You can also validate any existing dataset directory using **File > Validate Dataset...** or the CLI.

The validator checks:
- **Image-to-Label Parity**: Verifies every image has an associated label file (or is a valid empty null label).
- **Coordinate Normalization**: Enforces that all polygon coordinates fall strictly within $[0.0, 1.0]$.
- **Valid Polygon Topology**: Ensures polygons contain at least 3 vertices and non-zero enclosed area.
- **Class Map Integrity**: Checks `data.yaml` class names and contiguous integer indexing.

---

## 19. Headless CLI Pipelines

Headless CLI commands enable integration into automated video processing scripts:

### 1. Extract Video Frames:
```bash
python main.py --extract-frames path/to/video.mp4 --output extracted_frames --every-n 10
# or via CLI module with custom strategy:
python -m sam3_annotator.cli.extract_cli --video path/to/video.mp4 --output frames/ --strategy interval_seconds --interval-seconds 0.5
```

### 2. Export Project to YOLO Dataset:
```bash
python main.py --export projects/my_project --output yolo_dataset
# or via CLI module with custom split ratios:
python -m sam3_annotator.cli.export_cli --project projects/my_project --output yolo_dataset --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05 --split-strategy sequential
```

### 3. Validate Dataset:
```bash
python main.py --validate-dataset yolo_dataset
```

---

## 20. Configuration (`defaults.yaml`)

Application defaults are configured in `config/defaults.yaml`:

```yaml
app:
  name: SAM3 Video Polygon Annotator
  version: 1.0.0
model:
  device: auto
  precision: fp32
  checkpoint_path: sam2.1_hiera_tiny.pt
  default_model_type: sam2.1_hiera_tiny.pt
frame:
  default_sampling: every_n
  every_n: 10
  interval_seconds: 1.0
  fixed_count: 100
  cache_size_limit: 100
  jpeg_quality: 95
  image_format: jpg
polygon:
  simplify_tolerance: 0.005
  min_area: 20.0
  morphology_cleanup: true
  remove_duplicate_vertices: true
dataset:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  split_strategy: sequential
  export_masks: true
  export_previews: true
  create_zip: true
ui:
  mask_opacity: 0.45
  vertex_radius: 5
  line_width: 2
  autosave_interval_seconds: 30
  dark_theme: true
```

---

## 21. Troubleshooting & FAQ

- **Does this app use SAM 3?**: No. Meta has not released SAM 3. The application exclusively uses **SAM 2.1** (specifically the `sam2.1_hiera_tiny.pt` checkpoint by default).
- **CUDA Out of Memory**: In **Settings** (`Ctrl+,`), switch precision to `fp16` or launch with `--device cpu`.
- **Checkpoint Not Found**: Place `sam2.1_hiera_tiny.pt` or `sam2.1_t.pt` in the project root directory, or select your downloaded checkpoint in the Settings dialog.
- **Corrupt Video Codec**: Re-encode unsupported video formats to standard H.264 MP4 using FFmpeg:
  ```bash
  ffmpeg -i input.mov -c:v libx264 -crf 20 output.mp4
  ```
- **Application Logs**: Detailed runtime logs are saved to `logs/app.log`.
- For more troubleshooting scenarios, consult [docs/troubleshooting.md](docs/troubleshooting.md).

---

## 22. Repository Structure

```
SAM3/
├── main.py                     <- Unified application entry point (CLI & GUI)
├── requirements.txt            <- Python dependencies
├── pyproject.toml              <- Build & packaging configuration
├── pytest.ini                  <- Automated testing configuration
├── sam2.1_hiera_tiny.pt        <- Active SAM 2.1 model checkpoint (156 MB)
├── config/
│   └── defaults.yaml           <- Global default configurations
├── docs/                       <- Comprehensive guides & documentation
│   ├── annotation_workflow.md  <- Interactive prompting & editing guide
│   ├── installation.md         <- Detailed installation walkthrough
│   ├── sam3_setup.md           <- SAM model configuration & benchmarks
│   ├── troubleshooting.md      <- Common issues and recovery steps
│   ├── user_guide.md           <- End-to-end user manual
│   └── yolo_export.md          <- YOLOv8 export specification
├── sam3_annotator/             <- Main application package
│   ├── annotation/             <- Polygon math, editing, tracking & serialization
│   ├── cli/                    <- Standalone CLI modules (extract & export)
│   ├── config/                 <- App configuration loader & schemas
│   ├── dataset/                <- Splitting, YOLOv8 exporter & dataset validator
│   ├── models/                 <- SAM 2.1 adapter layer & services
│   ├── project/                <- Project manager, schemas & persistence
│   ├── ui/                     <- PySide6 GUI windows, canvas, panels & dialogs
│   ├── utils/                  <- Image, geometry, device & logging helpers
│   └── video/                  <- Video decoding, frame extraction & LRU caching
├── scripts/
│   └── demo_end_to_end.py      <- Headless verification workflow
└── tests/                      <- Comprehensive automated test suite
```

---

## 23. License Notices

- **SAM 2.1 Video Polygon Annotator**: Licensed under the MIT License.
- **Segment Anything 2.1 (SAM 2.1)**: Developed by Meta AI Research, licensed under Apache 2.0.
- **Ultralytics YOLO**: Developed by Ultralytics, licensed under AGPL-3.0 / Enterprise.
- **OpenCV**: Licensed under Apache 2.0.
- **PySide6**: Licensed under LGPL-3.0.
