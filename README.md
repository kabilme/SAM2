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

The **SAM2 Video Polygon Annotator** bridges the gap between raw video footage and production-ready computer vision datasets. By utilizing zero-shot Segment Anything 2 (SAM 2 / SAM 2.1) prompting (positive/negative points and bounding boxes), annotators can generate pixel-accurate polygon outlines in milliseconds without manual point-by-point drawing.

Annotations can be propagated forward across subsequent video frames using Meta's official **SAM 2.1 Video Predictor** with native spatio-temporal memory attention, refined using an interactive vertex editor, and exported directly into validated YOLOv8, COCO, Pascal VOC, LabelMe, MOT tracking, and rendered video overlay formats.

---

## 2. Key Features

- **Local-First & Completely Private**: Runs entirely on your local workstation; never uploads videos, frames, or annotations to external cloud services.
- **Native Meta SAM 2.1 Hiera Engine**: Full integration with Meta AI's official SAM 2 / SAM 2.1 Hiera architecture (`build_sam2` and `SAM2ImagePredictor`), with Ultralytics SAM as an automatic secondary fallback.
- **State-of-the-Art Spatio-Temporal Video Predictor**: Multi-frame object propagation powered by Meta's official `build_sam2_video_predictor`. Conditions on initial keyframe bounding boxes and centroid points, propagating visual memory cross-attention embeddings forward without object drift or frame drops.
- **Memory-Bounded 60-Frame Chunking**: Automatically processes long sequences in 60-frame memory chunks with seamless reference chaining, capping RAM usage under ~750 MB even on videos with thousands of frames.
- **High-Performance UI Batching**: Commitments are batched in a single transaction with in-memory `QIcon` caching, eliminating UI freezes upon propagation completion.
- **Continuous Real-Time Feedback**: Live terminal logging (frame ID, completion %, confidence score, vertex counts), main window status bar updates, and responsive modal progress dialog with cancellation support.
- **Dynamic Active Model Indicator**: Displays the active model checkpoint name (e.g. `sam2.1_hiera_tiny.pt`) in the main window title bar and toolbar prompt label.
- **Multi-Video Support**: Ingest multiple video files during project creation or add additional videos to an active project at any time via **File > Add Video(s) to Project...**.
- **Interactive Zero-Shot Prompting**: Click positive (`+`) points to segment objects, negative (`-`) points to exclude background bleed, or draw bounding boxes for rapid object isolation.
- **Contour Vectorization & Simplification**: Converts binary raster masks into simplified, organic polygons using Ramer-Douglas-Peucker (RDP) contour reduction.
- **Interactive Polygon Vertex Editor**: Drag vertices, click polygon edges to insert new control points, right-click to delete vertices, and enforce image boundary clamping.
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
  - `sam2>=1.0.0` (Meta SAM 2 / SAM 2.1 Hiera segmentation engine)
  - `hydra-core>=1.3.2` (SAM 2 hierarchical configuration management)
  - `tqdm>=4.66.1` (Batch frame loading progress utilities)
  - `PySide6>=6.6.0` (GUI framework & graphics canvas)
  - `torch>=2.1.0`, `torchvision>=0.16.0` (Deep learning inference backend)
  - `ultralytics>=8.1.0` (Ultralytics SAM fallback & YOLOv8 dataset export tools)
  - `opencv-python>=4.8.0` (Video decoding, image processing & contour extraction)
  - `numpy>=1.24.0,<3.0.0` (Multi-dimensional numerical operations & raster arrays)
  - `pillow>=10.0.0` (Image transformation and thumbnail handling)
  - `shapely>=2.0.0` (Polygon geometry analysis)
  - `PyYAML>=6.0.0` (Configuration and dataset metadata serialization)
  - `psutil>=5.9.0` (System resource management and RAM monitoring)
  - `pytest>=7.4.0` (Automated testing suite)

---

## 5. GPU Requirements & Acceleration

- **NVIDIA GPU**: Recommended for instant sub-30ms interactive segmentation (GTX 1660, RTX 2060, RTX 3060, RTX 4070 or higher).
- **VRAM Requirements**:
  - Tiny (`sam2.1_hiera_tiny.pt` / `sam2.1_t.pt`): 2–4 GB VRAM
  - Large (`sam2.1_hiera_large.pt` / `sam2.1_l.pt`): 6–8 GB+ VRAM
- **CUDA Runtime**: CUDA 11.8 or CUDA 12.x supported out of the box.
- **CPU Fallback**: 100% functional on CPU for systems without dedicated NVIDIA GPUs (native SAM 2.1 Hiera Tiny averages ~1.5–2.8s per frame during video propagation on modern CPUs).

---

## 6. SAM 2 Architecture & Checkpoints

The application utilizes an isolated adapter architecture (`sam2_annotator/models/sam2_adapter.py`) interfacing with Meta AI's official SAM 2 / SAM 2.1 Hiera models:

1. **Primary Engine**: Native Meta SAM 2.1 Hiera architecture loaded via `sam2.build_sam.build_sam2` and `build_sam2_video_predictor`.
2. **Secondary Fallback**: Ultralytics SAM architecture (`from ultralytics import SAM`) for environments lacking native SAM 2 packages.
3. **Mock Testing Adapter**: `MockSAM2Adapter` providing synthetic masks for automated unit tests without requiring weights.

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
cd D:/SAM2

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

> **Note**: The default checkpoint `sam2.1_hiera_tiny.pt` is loaded from the root directory. If absent, the application will automatically download it from Meta's official release repository upon first launch.

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

Object tracking utilizes Meta's official **SAM 2.1 Video Predictor** (`build_sam2_video_predictor` from `sam2`) featuring native **Spatio-Temporal Memory Attention**:

```
[Keyframe Annotation] ──> [Seed Memory Bank] ──> [Cross-Frame Memory Attention] ──> [Pixel-Accurate Polygons]
   (Box / Centroid)          (Frame 0)                (SAM 2.1 Video Predictor)        (Zero Lost Frames)
```

### Propagation Configuration in Properties Panel:

- **Propagation Scope**:
  - **Fixed Frame Count**: Tracks forward for a designated number of frames (preset buttons: `10`, `30`, `60`, `100` frames or spinbox).
  - **Until Next Keyframe**: Tracks forward until encountering the next keyframe in the sequence.
  - **Until End of Video**: Propagates through all subsequent frames up to the end of the video.
- **Prompt Mode**:
  - **Box Prompt** *(Default & Recommended)*: Seeds the temporal memory bank using the object's exact bounding box and foreground centroid.
  - **Point Prompt**: Seeds tracking using the object's interior centroid coordinate.
  - **Combined Prompt**: Utilizes both bounding box anchors and interior foreground points.
- **Adaptive Margin**:
  - Default is `0.0` (exact bounding box reference without artificial expansion margins, ensuring precise tracking). Configurable via `model.propagation_box_padding` in `defaults.yaml`.

### Technical Architecture Highlights:
- **Temporal Memory Attention**: Replaces naive frame-by-frame optical flow heuristics with Meta's cross-frame attention bank. Objects maintain identity through turns, accelerations, scale changes, and partial occlusions with 100% completion rate (zero dropped frames).
- **Bounded 60-Frame Chunking**: Automatically processes long video sequences in 60-frame attention chunks with reference chaining, capping RAM usage under ~750 MB regardless of video length.
- **Instant UI Batching**: Propagated annotations are committed to `AnnotationManager` in a single batch with a single undo snapshot. In-memory `QIcon` caching in `VideoPanel` prevents GUI freezing upon completion.
- **Continuous Feedback**: Outputs real-time progress logs to the console (`%`, frame ID, confidence score, vertex count), updates the main window status bar, and keeps the progress dialog responsive.

---

## 14. Interactive Polygon Editing

Switch to **Edit Mode** (`E` on keyboard) or double-click an existing polygon:

- **Drag Vertices**: Left-click and drag any vertex control point.
- **Add Vertices**: Click directly on an edge between two vertices to insert a new control point.
- **Delete Vertices**: Right-click on a vertex handle and choose **Delete Vertex**.
- **Delete Annotation**: Select the polygon and press `Delete` or `Backspace`.
- **Change Class**: Right-click the polygon or change the active class in the Class Panel.
- **Undo / Redo**: Use `Ctrl+Z` and `Ctrl+Y` to undo or redo vertex movements, additions, and deletions.

---

## 15. Null Frames & Background Negative Samples

In real-world object detection and instance segmentation, training models exclusively on images containing objects leads to high false-positive rates on empty backgrounds.

### Handling Null / Background Frames:
- **Marking Negative Frames**: Frames without objects can be marked as **Null / Negative Frame** via right-click in the timeline or Video Panel.
- **Timeline Review Filtering**: Filter frames by status:
  - ⚪ **Unreviewed**
  - 🔵 **Annotated**
  - 🟢 **Reviewed**
  - ⚫ **Null / Negative Frame**
- **Batch Marking**: Choose **Annotation > Mark All Unannotated as Null** to automatically mark all empty frames as negative samples in one click.
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
  propagation_frames: 30
  propagation_prompt_type: box
  propagation_box_padding: 0.0
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
SAM2/
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
│   ├── models/                 <- SAM 2 adapter layer & video propagation service
│   ├── project/                <- Project manager, schemas & persistence
│   ├── ui/                     <- PySide6 GUI windows, canvas, panels & dialogs
│   ├── utils/                  <- Image, geometry, device & logging helpers
│   └── video/                  <- Video decoding, frame extraction & LRU caching
├── scripts/
│   └── demo_end_to_end.py      <- Headless verification workflow
└── tests/                      <- Comprehensive automated test suite (50 tests)
```

---

## 22. License Notices

The **SAM2 Video Polygon Annotator** is released as free and open-source software under the **[GNU Affero General Public License v3.0 (GNU AGPLv3)](LICENSE)**.

This licensing guarantees that the software remains free and open-source forever, ensures 100% legal harmony with all dependencies (including Ultralytics YOLO), and prevents closed-source commercial exploitation of community contributions.

### Integrated Component Licensing Matrix

| Component / Library | Developer / Organization | License | Compatibility with AGPL-3.0 | Purpose / Role |
| :--- | :--- | :--- | :---: | :--- |
| **SAM2 Video Polygon Annotator** | Kabilarasan | **GNU AGPL-3.0** | Main License | Desktop GUI application, timeline & video annotation engine |
| **Ultralytics YOLO** | Ultralytics Inc. | **AGPL-3.0** | Direct Match | Single-frame SAM fallback & YOLO dataset exporter schemas |
| **Segment Anything 2.1 (SAM 2.1)** | Meta AI Research (FAIR) | **Apache 2.0** | Compatible | Hiera vision transformer weights & spatio-temporal video predictor |
| **OpenCV** | OpenCV Team | **Apache 2.0** | Compatible | Video decoding, frame extraction, contour hierarchy & image I/O |
| **PySide6 (Qt for Python)** | The Qt Company | **LGPL-3.0** | Compatible | Hardware-accelerated Qt GUI canvas, docks, dialogs & threads |

---

### Key Terms of the GNU AGPL-3.0 License

1. **Open Source & Copyleft**:
   - Anyone may run, modify, study, and distribute this software freely.
   - If you modify this project or incorporate it into another software package, any distributed versions—including instances made accessible over a computer network (cloud SaaS)—**must also be released under the GNU AGPL-3.0 with source code made available**.

2. **Internal Enterprise Use**:
   - Commercial companies and academic labs may freely download, run, and utilize this desktop application internally across their organization without paying license fees or publishing private internal workflows.

3. **User-Generated Datasets & Annotations (100% Yours)**:
   - The AGPL-3.0 license applies **strictly to the software source code**, not to the data you produce.
   - All polygon coordinates, bounding boxes, semantic masks, and exported training datasets generated using this tool belong **100% to you (the user)**.
   - You may use, distribute, sell, or train closed-source proprietary commercial AI models on your exported datasets with zero restrictions.

