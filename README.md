# SAM3 Video Polygon Annotator

A high-performance, local-first Python desktop application for interactive **SAM 3 / SAM 2 assisted polygon annotation** on video frames and automated **YOLOv8 instance-segmentation dataset export**, inspired by modern computer-vision platforms like Roboflow.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [System Requirements](#3-system-requirements)
4. [Python Version](#4-python-version)
5. [GPU Requirements](#5-gpu-requirements)
6. [SAM 3 Installation & Setup](#6-sam-3-installation--setup)
7. [Checkpoint Setup](#7-checkpoint-setup)
8. [Installation](#8-installation)
9. [Running the Application](#9-running-the-application)
10. [Creating a Project](#10-creating-a-project)
11. [Extracting Video Frames](#11-extracting-video-frames)
12. [SAM 3 Annotation Workflow](#12-sam-3-annotation-workflow)
13. [Tracking Workflow](#13-tracking-workflow)
14. [Polygon Editing](#14-polygon-editing)
15. [YOLO Export](#15-yolo-export)
16. [Dataset Validation](#16-dataset-validation)
17. [CLI Usage](#17-cli-usage)
18. [Troubleshooting](#18-troubleshooting)
19. [Project Directory Structure](#19-project-directory-structure)
20. [License Notices](#20-license-notices)

---

## 1. Project Overview

The **SAM3 Video Polygon Annotator** bridges the gap between raw video footage and trained YOLOv8 instance segmentation models. By utilizing zero-shot Segment Anything prompting (positive/negative points and bounding boxes), annotators can generate pixel-accurate polygon outlines in milliseconds without manual point-by-point drawing. Annotations are tracked across subsequent frames, manually refined using an interactive vertex editor, and exported directly into a validated, training-ready YOLOv8 segmentation dataset complete with `data.yaml`, images, labels, and zip packaging.

---

## 2. Features

- **Local-First & Private**: Runs entirely on your local machine; never uploads video, frames, or annotations to cloud services.
- **Interactive Prompting**: Positive (`+`) and negative (`-`) point prompts, bounding box prompts, and manual polygon tracing.
- **Real-Time Vectorization**: Converts raw binary masks into clean, simplified polygons using Ramer-Douglas-Peucker (RDP) contour simplification.
- **Interactive Polygon Editor**: Click-and-drag vertex editing, edge splitting (adding vertices), vertex deletion, and boundary validation.
- **Multi-Frame Propagation**: Forward object tracking across video frames with non-blocking background workers and cancelability.
- **High-Performance Canvas**: Hardware-accelerated `QGraphicsView` with smooth zoom-to-cursor, pan, and real-time coordinate inspection.
- **Timeline & Review Filtering**: Scrub through video frames with visual thumbnails and filter by status (*All*, *Unreviewed*, *Reviewed*, *Negative*).
- **YOLOv8 Segmentation Exporter**: Generates normalized coordinates (`class_id x1 y1 x2 y2 ...`), `data.yaml`, optional binary masks, preview overlays, and ZIP archives.
- **Built-In Dataset Validator**: Automatically verifies coordinate normalization, label-to-image parity, bounding ranges, and non-empty classes.
- **Robust Persistence**: Atomic JSON saving with automatic backup recovery to prevent data loss.

---

## 3. System Requirements

- **Operating System**: Windows 10/11 (64-bit), Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+), or macOS 12+
- **Processor**: Intel / AMD x86_64 or Apple Silicon ARM64 processor
- **Memory**: Minimum 8 GB RAM (16 GB+ recommended for 4K video)
- **Disk Space**: At least 3 GB for application, dependencies, and model weights

---

## 4. Python Version

- Supported: **Python 3.10, 3.11, 3.12, 3.13**
- Tested Environment: Python 3.13.5 (Win64)

---

## 5. GPU Requirements

- **NVIDIA GPU**: Recommended for real-time inference (GTX 1660, RTX 2060, RTX 3060, RTX 4070 or higher)
- **VRAM**:
  - Tiny (`sam2.1_t.pt`): 2-4 GB VRAM
  - Base / Large (`sam2.1_b.pt` / `sam2.1_l.pt`): 6-8 GB+ VRAM
- **CUDA Runtime**: CUDA 11.8 or CUDA 12.x
- **CPU Fallback**: 100% functional on CPU for systems without dedicated GPUs.

---

## 6. SAM 3 Installation & Setup

The application features an isolated model adapter architecture (`sam3_annotator/models/sam3_adapter.py`). The production adapter interfaces with the official Ultralytics SAM implementation:

```bash
pip install ultralytics torch torchvision
```

The adapter automatically determines whether CUDA is available and configures device memory management and precision (`fp32` / `fp16`) accordingly.

---

## 7. Checkpoint Setup

The default lightweight checkpoint is `sam2.1_t.pt` (78 MB), which balances fast interactive latency with high boundary segmentation accuracy:

```powershell
# Pre-download weights (or let the app auto-download on first launch)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt" -OutFile "sam2.1_t.pt"
```

You can select alternate weights (`sam2.1_s.pt`, `sam2.1_b.pt`, `sam2.1_l.pt`) from the **Settings Dialog** in the UI.

---

## 8. Installation

```bash
# 1. Clone or navigate to the repository
cd D:/SAM3

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows
# source .venv/bin/activate      # On Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: Install project in editable mode
pip install -e .
```

---

## 9. Running the Application

### Launch GUI:
```bash
python main.py
```

### Launch GUI with a specific project or device:
```bash
python main.py --project example_project --device cuda
python main.py --device cpu
```

---

## 10. Creating a Project

1. Launch the application and click **New Project** (`Ctrl+N`).
2. Provide a **Project Name** and select an **Output Directory**.
3. Select your **Input Video File** (`.mp4`, `.avi`, `.mov`, `.mkv`).
4. Define your annotation classes (e.g. `scooter`, `person`, `helmet`).
5. Choose your frame sampling settings (e.g. extract every 10 frames).
6. Click **Create Project**. The video extractor runs in the background and opens the annotation workspace upon completion.

---

## 11. Extracting Video Frames

Sampling options available during project creation or via CLI:
- **Every N frames**: Uniform subsampling (e.g. sample every 5th or 15th frame).
- **Interval in Seconds**: Extract 1 frame per $S$ seconds of video.
- **Fixed Count**: Uniformly distribute a target total frame count across video duration.
- **Every Frame**: Sequential extraction of all video frames.

---

## 12. SAM 3 Annotation Workflow

1. In the **Class Panel**, select the category of the object you want to annotate.
2. Select the **Positive Point** tool (`2` on keyboard) in the toolbar.
3. Click on the target object in the canvas. SAM generates a binary segmentation mask.
4. If parts of the object are missing, add additional positive points.
5. If background areas are captured, select **Negative Point** (`3` on keyboard) and click the background region.
6. The mask updates automatically and converts into an editable polygon.

---

## 13. Tracking Workflow

1. Select the completed polygon in the current frame.
2. Click **Propagate Object** (`Ctrl+P` or toolbar icon).
3. Select the number of forward frames to track (e.g. next 10, 30, or all frames).
4. The background propagation worker predicts object masks on subsequent frames and creates linked annotations.
5. Review each propagated frame and make manual adjustments if necessary.

---

## 14. Polygon Editing

Switch to **Edit Mode** (`E` on keyboard) or double-click an existing polygon:
- **Drag Vertex**: Click and drag any vertex control handle.
- **Add Vertex**: Click on an edge between two vertices to insert a new vertex point.
- **Delete Vertex**: Right-click on a vertex handle and select **Delete Vertex**.
- **Delete Annotation**: Select the polygon and press `Delete` or `Backspace`.

---

## 15. YOLO Export

1. Click **Export Dataset** (`Ctrl+E` or toolbar icon).
2. Configure train / validation / test split ratios (default: 70% train, 20% val, 10% test).
3. Choose split strategy:
   - **Sequential**: Prevents temporal leakage across consecutive video frames.
   - **Grouped**: Splits continuous segments.
   - **Random**: Randomly distributes frames.
4. Select optional outputs:
   - Export binary PNG masks.
   - Export visual preview overlays.
   - Create dataset ZIP archive (`dataset.zip`).
5. Click **Export**.

---

## 16. Dataset Validation

The exported dataset is validated against YOLOv8 instance segmentation requirements:
- Matches all images with label files.
- Ensures all polygon coordinates are normalized between $[0.0, 1.0]$.
- Confirms polygons have at least 3 points and non-zero area.
- Checks `data.yaml` class mapping and split counts.

You can also validate any existing dataset directory using the menu (**File > Validate Dataset**) or CLI.

---

## 17. CLI Usage

The package provides standalone CLI commands for automated headless pipelines:

### Extract Frames:
```bash
python main.py --extract-frames path/to/video.mp4 --output extracted_frames --every-n 10
# or
python -m sam3_annotator.cli.extract_cli --video video.mp4 --output frames/ --every-n 5
```

### Export Project to YOLO Dataset:
```bash
python main.py --export path/to/project --output my_yolo_dataset
# or
python -m sam3_annotator.cli.export_cli --project example_project --output yolo_dataset --split-strategy sequential
```

### Validate Dataset:
```bash
python main.py --validate-dataset path/to/dataset
```

---

## 18. Troubleshooting

- **CUDA Out of Memory**: Switch to `fp16` precision in **Settings** or start with `--device cpu`.
- **Corrupt Video Codec**: Re-encode video using H.264 MP4 (`ffmpeg -i input.mov -c:v libx264 output.mp4`).
- **Logs**: Inspect detailed logs at `logs/app.log`.
- For more troubleshooting scenarios, consult [docs/troubleshooting.md](docs/troubleshooting.md).

---

## 19. Project Directory Structure

```
SAM3/
├── main.py                     <- Application entry point (CLI & GUI)
├── requirements.txt            <- Python package dependencies
├── pyproject.toml              <- Build & packaging configuration
├── pytest.ini                  <- Pytest configuration
├── sam2.1_t.pt                 <- Default SAM model weights
├── README.md                   <- Main documentation
├── docs/                       <- Extended documentation
│   ├── installation.md
│   ├── user_guide.md
│   ├── annotation_workflow.md
│   ├── sam3_setup.md
│   ├── yolo_export.md
│   └── troubleshooting.md
├── sam3_annotator/             <- Main application package
│   ├── annotation/             <- Polygon math, editing, and tracking
│   ├── cli/                    <- Command-line interfaces
│   ├── config/                 <- App configuration and defaults
│   ├── dataset/                <- Splitting, YOLO exporter, and validator
│   ├── models/                 <- SAM 3 / SAM 2 adapter layer and services
│   ├── project/                <- Project manager and JSON schemas
│   ├── ui/                     <- PySide6 GUI windows, canvas, and panels
│   ├── utils/                  <- Image, geometry, and device helpers
│   └── video/                  <- Video reader, frame extraction, and caching
├── scripts/
│   └── demo_end_to_end.py      <- Headless end-to-end verification script
├── tests/                      <- Comprehensive automated test suite
├── example_project/            <- Sample project with annotations
└── example_dataset/            <- Exported YOLOv8 instance segmentation dataset
```

---

## 20. License Notices

- **SAM3 Video Polygon Annotator**: Licensed under the MIT License.
- **Segment Anything (SAM / SAM 2)**: Developed by Meta AI Research, licensed under the Apache 2.0 License.
- **Ultralytics YOLO**: Developed by Ultralytics, licensed under AGPL-3.0 / Enterprise.
- **OpenCV**: Licensed under the Apache 2.0 License.
- **PySide6**: Licensed under LGPL-3.0.
