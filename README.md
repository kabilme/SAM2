# SAM2 Video Polygon Annotator

A high-performance, local-first Python desktop application for interactive **Segment Anything 2 (SAM 2 / SAM 2.1)** assisted polygon annotation on video frames and automated **YOLOv8 instance-segmentation dataset export**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Requirements](#3-system-requirements)
4. [Python Version & Dependencies](#4-python-version--dependencies)
5. [GPU Requirements & Acceleration](#5-gpu-requirements--acceleration)
6. [SAM 2 Architecture & Checkpoints](#6-sam-2-architecture--checkpoints)
7. [Installation Guide](#7-installation-guide)
8. [Running the Application](#8-running-the-application)
9. [Multi-Video Project Workflow](#9-multi-video-project-workflow)
10. [Video Frame Sampling & Extraction](#10-video-frame-sampling--extraction)
11. [Frame Deletion & Continuous Re-indexing](#11-frame-deletion--continuous-re-indexing)
12. [SAM 2 Assisted Annotation Workflow](#12-sam-2-assisted-annotation-workflow)
13. [Object Tracking & Multi-Frame Propagation](#13-object-tracking--multi-frame-propagation)
14. [Interactive Polygon Editing](#14-interactive-polygon-editing)
15. [Null Frames & Background Negative Samples](#15-null-frames--background-negative-samples)
16. [Multiple Dataset & Video Export Formats](#16-multiple-dataset--video-export-formats)
17. [Built-In Dataset Validator](#17-built-in-dataset-validator)
18. [Headless CLI Pipelines](#18-headless-cli-pipelines)
19. [Configuration (`defaults.yaml`)](#19-configuration-defaultsyaml)
20. [Troubleshooting & FAQ](#20-troubleshooting--faq)
21. [Repository Structure](#21-repository-structure)
22. [License Notices](#22-license-notices)

---

## 1. Project Overview

The **SAM2 Video Polygon Annotator** bridges the gap between raw video footage and production-ready YOLOv8 instance segmentation datasets. By utilizing zero-shot Segment Anything 2 (SAM 2 / SAM 2.1) prompting (positive/negative points and bounding boxes), annotators can generate pixel-accurate polygon outlines in milliseconds without manual point-by-point drawing.

Annotations can be propagated forward across subsequent video frames using multi-frame object tracking, refined using an interactive vertex editor, and exported directly into a validated YOLOv8 instance segmentation dataset complete with `data.yaml`, normalized label files, binary masks, overlay previews, and atomic ZIP packaging.

---

## 2. Key Features

- **Local-First & Completely Private**: Runs entirely on your local workstation; never uploads videos, frames, or annotations to external cloud services.
- **Segment Anything 2 (SAM 2 / SAM 2.1) Engine**: Zero-shot interactive segmentation using Meta's official SAM 2 Hiera architecture loaded via PyTorch and the Ultralytics engine.
- **Dynamic Active Model Indicator**: Displays the active model checkpoint name (e.g. `sam2.1_hiera_tiny.pt`) in the main window title bar and toolbar prompt label.
- **Multi-Video Support**: Ingest multiple video files during project creation or add additional videos to an active project at any time via **File > Add Video(s) to Project...**.
- **Interactive Zero-Shot Prompting**: Click positive (`+`) points to segment objects, negative (`-`) points to exclude background bleed, or draw bounding boxes for rapid object isolation.
- **Contour Vectorization & Simplification**: Converts binary raster masks into simplified, organic polygons using Ramer-Douglas-Peucker (RDP) contour reduction.
- **Interactive Polygon Vertex Editor**: Drag vertices, click polygon edges to insert new control points, right-click to delete vertices, and enforce image boundary clamping.
- **Multi-Frame Propagation & Tracking**: Propagate object masks forward across frame sequences using non-blocking background workers with live progress and cancelability.
- **Video Frame Deletion & Continuous Re-indexing**: Delete single frames, multiple selections, or batch empty/unreviewed frames with optional disk cleanup (images, thumbnails, annotations) and automatic re-indexing (`1..N`) to prevent timeline gaps.
- **Null / Background Negative Sample Support**: Mark unannotated frames as negative background samples. YOLO exporter automatically produces empty label files (`.txt`) according to Ultralytics training best practices to drastically suppress false-positive detections.
- **High-Performance Timeline & Canvas**: Hardware-accelerated `QGraphicsView` canvas with smooth mouse-centered zoom, panning, timeline scrubbing, and review status filtering (*All Frames*, *Unreviewed*, *Annotated*, *Reviewed*, *Null / Negative Frames*).
- **7 Industry-Standard Export Formats**: Complete multi-format exporter supporting:
  1. **YOLOv8 Instance Segmentation** (normalized polygons + `data.yaml`)
  2. **YOLOv8 Object Detection** (normalized bounding boxes `class_id cx cy w h` + `data.yaml`)
  3. **COCO 1.0 JSON** (`instances_*.json` for Detectron2 / MMDetection / Hugging Face)
  4. **Pascal VOC XML & Semantic Masks** (XML `<bndbox>` & `<polygon>` + 8-bit indexed palette PNG masks)
  5. **LabelMe JSON** (per-image `.json` format for desktop tool interoperability)
  6. **MOT / MOTChallenge Tracking** (`gt.txt` + `seqinfo.ini` + `img1/` sequence structure with persistent object tracking IDs)
  7. **Rendered Video Overlays** (`.mp4` video with alpha-blended polygon fills, crisp outlines, bounding boxes, labels, and tracking badges)
- **Live Progress Dialog & Background Threading**: Fully responsive `QThread` export worker with real-time percentage progress, processed frame counters, and cancelability.
- **Automated Dataset Validator**: Verifies image-label coordinate normalization, bounding ranges $[0.0, 1.0]$, minimum vertex counts, non-empty classes, and `data.yaml` validity.
- **Safe Persistence**: Atomic JSON file writes with automatic backup restoration (`project.backup.json`) to guarantee zero data loss.

---

## 3. System Requirements

- **Operating System**: Windows 10/11 (64-bit), Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+), or macOS 12+
- **Processor**: Intel Core i5 / AMD Ryzen 5 or higher (x86_64 or Apple Silicon ARM64)
- **Memory**: Minimum 8 GB RAM (16 GB+ recommended for 4K video datasets)
- **Storage**: At least 3 GB free disk space for application dependencies and model weights

---

## 4. Python Version & Dependencies

- **Supported Python Versions**: Python 3.10, 3.11, 3.12, 3.13
- **Core Dependencies**:
  - `PySide6` (GUI framework)
  - `ultralytics` (SAM 2 inference engine & model loader)
  - `torch`, `torchvision` (Deep learning inference backend)
  - `opencv-python` (Video decoding and contour extraction)
  - `numpy`, `pillow` (Numerical array and image transformations)
  - `pyyaml` (Configuration management)

---

## 5. GPU Requirements & Acceleration

- **NVIDIA GPU**: Recommended for instant sub-50ms interactive segmentation (GTX 1660, RTX 2060, RTX 3060, RTX 4070 or higher).
- **VRAM Requirements**:
  - Tiny (`sam2.1_hiera_tiny.pt` / `sam2.1_t.pt`): 2–4 GB VRAM
  - Large (`sam2.1_hiera_large.pt` / `sam2.1_l.pt`): 6–8 GB+ VRAM
- **CUDA Runtime**: CUDA 11.8 or CUDA 12.x supported out of the box.
- **CPU Fallback**: 100% functional on CPU for systems without dedicated NVIDIA GPUs.

---

## 6. SAM 2 Architecture & Checkpoints

The application utilizes an isolated adapter architecture (`sam2_annotator/models/sam2_adapter.py`) interfacing with Meta AI's official SAM 2 / SAM 2.1 Hiera models via Ultralytics.

### Default Checkpoint
The default model is Meta's **`sam2.1_hiera_tiny.pt`** (156 MB), offering the optimal balance between interactive latency and crisp object boundary accuracy.

### Supported Checkpoint Types
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

## 7. Installation Guide

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

## 8. Running the Application

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

## 9. Multi-Video Project Workflow

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

## 10. Video Frame Sampling & Extraction

Frames can be sampled during project creation or via headless CLI:

- **Every N Frames** (`every_n`): Uniform interval sampling (e.g., extract every 10th or 15th frame).
- **Interval in Seconds** (`interval_seconds`): Extract 1 frame every $S$ seconds of video playback.
- **Fixed Count** (`fixed_count`): Evenly distributes a target total number of frames across the entire video duration.
- **Every Frame** (`every_frame`): Sequential extraction of every single frame.

---

## 11. Frame Deletion & Continuous Re-indexing

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

## 12. SAM 2 Assisted Annotation Workflow

1. In the **Class Panel** (right dock), select the class label for the object.
2. Click the **Positive Point** tool (`2` on keyboard) in the toolbar.
3. Left-click inside the target object on the canvas. SAM 2 generates a segmentation mask.
4. If parts of the object are omitted, click additional positive points.
5. If background pixels are included, select the **Negative Point** tool (`3` on keyboard) and click the background region to subtract it.
6. Alternatively, select **Bounding Box** (`4` on keyboard) and drag a box around the object.
7. The mask converts automatically into a simplified, editable polygon outline.

---

## 13. Object Tracking & Multi-Frame Propagation

1. Select the completed polygon in the current frame.
2. Click **Propagate Object** (`Ctrl+P` or toolbar icon).
3. Specify the number of forward frames to track (e.g. next 10, 30, or all remaining frames).
4. The background propagation worker tracks the object across subsequent frames, generating connected polygon annotations.
5. Scrub through the timeline (`Left` / `Right` arrow keys) to review and refine the tracked polygons.

---

## 14. Interactive Polygon Editing

Switch to **Edit Mode** (`E` on keyboard) or double-click an existing polygon:

- **Drag Vertices**: Left-click and drag any vertex control point.
- **Add Vertices**: Click directly on an edge between two vertices to insert a new control point.
- **Delete Vertices**: Right-click on a vertex handle and choose **Delete Vertex**.
- **Delete Annotation**: Select the polygon and press `Delete` or `Backspace`.
- **Change Class**: Right-click the polygon or change the active class in the Class Panel.
- **Undo / Redo**: Use `Ctrl+Z` and `Ctrl+Y` to undo or redo vertex movements and deletions.

---

## 15. Null Frames & Background Negative Samples

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

## 16. Multiple Dataset & Video Export Formats

The application provides a unified export dialog (**File > Export Dataset...** or `Ctrl+E`) supporting 7 export targets:

| Format | Target Use Case | Output Structure |
| :--- | :--- | :--- |
| **YOLOv8 Instance Segmentation** | Ultralytics YOLOv8/v9/v11-seg | Polygons (`class_id x1 y1 ...`), `data.yaml`, optional binary masks & previews |
| **YOLOv8 Object Detection** | Ultralytics YOLOv8/v9/v11-det | Normalized bounding boxes (`class_id cx cy w h`), `data.yaml` |
| **COCO 1.0 JSON** | Detectron2, MMDetection, Hugging Face | Standard COCO JSON (`annotations/instances_{train,val,test}.json`) |
| **Pascal VOC & Semantic Masks** | Classic CV pipelines, Semantic Segmentation | XML `<bndbox>` & `<polygon>` tags + 8-bit palette indexed PNG masks |
| **LabelMe JSON** | LabelMe GUI interoperability | `.json` shape files paired alongside each frame image |
| **MOT / MOTChallenge** | Tracking evaluation (ByteTrack, DeepSORT) | Sequential `img1/`, `seqinfo.ini`, and ground-truth `gt/gt.txt` with track IDs |
| **Rendered Video (.mp4)** | Visual presentation & demo review | MP4 video with alpha-blended polygon fills, outlines, labels, & track badges |

### Exporting Steps:
1. Open the dialog via **Export Dataset** (`Ctrl+E` or **File > Export Dataset...**).
2. Select your desired **Export Format** from the dropdown menu.
3. Configure format-specific options:
   - **For Datasets**: Set train / val / test ratios and split strategy (**Sequential**, **Grouped**, or **Random**).
   - **For YOLO**: Toggle binary masks, visual overlay previews, and null/negative background frames.
   - **For Rendered Video**: Configure playback FPS, polygon fill alpha opacity ($0.1–0.9$), and bounding box toggles.
   - **For Archives**: Check **Create ZIP Archive** for automatic `.zip` compression.
4. Click **Start Export**. A non-blocking progress dialog displays live frame counts, percentage progress, and a cancel button.

For complete specifications, schemas, and framework loading examples, refer to [docs/export_formats.md](docs/export_formats.md).

---

## 17. Built-In Dataset Validator

Every exported dataset is automatically validated upon export. You can also validate any existing dataset directory using **File > Validate Dataset...** or the CLI.

The validator checks:
- **Image-to-Label Parity**: Verifies every image has an associated label file (or is a valid empty null label).
- **Coordinate Normalization**: Enforces that all polygon coordinates fall strictly within $[0.0, 1.0]$.
- **Valid Polygon Topology**: Ensures polygons contain at least 3 vertices and non-zero enclosed area.
- **Class Map Integrity**: Checks `data.yaml` class names and contiguous integer indexing.

---

## 18. Headless CLI Pipelines

Headless CLI commands enable integration into automated video processing scripts:

### 1. Extract Video Frames:
```bash
python main.py --extract-frames path/to/video.mp4 --output extracted_frames --every-n 10
# or via CLI module with custom strategy:
python -m sam2_annotator.cli.extract_cli --video path/to/video.mp4 --output frames/ --strategy interval_seconds --interval-seconds 0.5
```

### 2. Export Project to YOLO Dataset:
```bash
python main.py --export projects/my_project --output yolo_dataset
# or via CLI module with custom split ratios:
python -m sam2_annotator.cli.export_cli --project projects/my_project --output yolo_dataset --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05 --split-strategy sequential
```

### 3. Validate Dataset:
```bash
python main.py --validate-dataset yolo_dataset
```

---

## 19. Configuration (`defaults.yaml`)

Application defaults are configured in `config/defaults.yaml`:

```yaml
app:
  name: SAM2 Video Polygon Annotator
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

## 20. Troubleshooting & FAQ

- **CUDA Out of Memory**: In **Settings** (`Ctrl+,`), switch precision to `fp16` or launch with `--device cpu`.
- **Checkpoint Not Found**: Place `sam2.1_hiera_tiny.pt` or `sam2.1_t.pt` in the project root directory, or select your downloaded checkpoint in the Settings dialog.
- **Corrupt Video Codec**: Re-encode unsupported video formats to standard H.264 MP4 using FFmpeg:
  ```bash
  ffmpeg -i input.mov -c:v libx264 -crf 20 output.mp4
  ```
- **Application Logs**: Detailed runtime logs are saved to `logs/app.log`.
- For more troubleshooting scenarios, consult [docs/troubleshooting.md](docs/troubleshooting.md).

---

## 21. Repository Structure

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
│   ├── export_formats.md       <- Multi-format dataset & video export specification
│   ├── installation.md         <- Detailed installation walkthrough
│   ├── sam2_setup.md           <- SAM 2 model configuration & benchmarks
│   ├── troubleshooting.md      <- Common issues and recovery steps
│   ├── user_guide.md           <- End-to-end user manual
│   └── yolo_export.md          <- YOLOv8 export specification
├── sam2_annotator/             <- Main application package
│   ├── annotation/             <- Polygon math, editing, tracking & serialization
│   ├── cli/                    <- Standalone CLI modules (extract & export)
│   ├── config/                 <- App configuration loader & schemas
│   ├── dataset/                <- Exporters (YOLO, COCO, VOC, LabelMe, MOT, MP4), splitter & validator
│   ├── models/                 <- SAM 2 adapter layer & services
│   ├── project/                <- Project manager, schemas & persistence
│   ├── ui/                     <- PySide6 GUI windows, canvas, panels & dialogs
│   ├── utils/                  <- Image, geometry, device & logging helpers
│   └── video/                  <- Video decoding, frame extraction & LRU caching
├── scripts/
│   └── demo_end_to_end.py      <- Headless verification workflow
└── tests/                      <- Comprehensive automated test suite
```

---

## 22. License Notices

- **SAM2 Video Polygon Annotator**: Licensed under the MIT License.
- **Segment Anything 2 (SAM 2 / SAM 2.1)**: Developed by Meta AI Research, licensed under Apache 2.0.
- **Ultralytics YOLO**: Developed by Ultralytics, licensed under AGPL-3.0 / Enterprise.
- **OpenCV**: Licensed under Apache 2.0.
- **PySide6**: Licensed under LGPL-3.0.
