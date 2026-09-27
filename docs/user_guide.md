# User Guide: SAM2 Video Polygon Annotator

This document provides a comprehensive tour of the user interface, keyboard shortcuts, project management, and daily operations.

---

## 1. Application Layout

The graphical interface is built with **PySide6** and optimized for dark-mode productivity:

- **Top Toolbar**: Quick access to file operations, interaction modes (Select, Point +, Point -, Box, Polygon, Edit), propagation, zoom controls, and export.
- **Center Canvas**: Interactive `QGraphicsView` supporting hardware-accelerated panning, smooth zooming, point placement, bounding box drawing, and polygon vertex dragging.
- **Bottom Frame Timeline**: Visual thumbnail strip showing frame numbers, timestamps, review flags (green check for reviewed, red cross for negative), and scrubbing slider.
- **Left Dock - Classes**: Create, rename, recolor, and select object categories.
- **Left Dock - Video Info**: Displays video metadata (resolution, fps, duration, extracted frame count).
- **Right Dock - Objects / Properties**: Inspect active frame annotations, polygon area, perimeter, vertex count, and confidence scores.
- **Status Bar**: Real-time cursor coordinates, active frame info, selected hardware device (CPU / CUDA), and operation status.

---

## 2. Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+N` | Create New Project |
| `Ctrl+O` | Open Existing Project |
| `Ctrl+S` | Save Project (`project.json`) |
| `Ctrl+E` | Export Dataset (YOLO, COCO, VOC, LabelMe, MOT, MP4) |
| `Right Arrow` / `D` | Advance to next frame |
| `Left Arrow` / `A` | Return to previous frame |
| `Space` + Drag | Pan across canvas |
| `Mouse Wheel` | Smooth zoom in/out under cursor |
| `F` | Fit image to view |
| `1` | Select Mode |
| `2` | Positive Point Prompt Mode (`+`) |
| `3` | Negative Point Prompt Mode (`-`) |
| `4` | Bounding Box Prompt Mode |
| `5` | Manual Polygon Mode |
| `E` | Polygon Vertex Edit Mode |
| `R` | Mark current frame as Reviewed |
| `N` | Mark current frame as Negative (Empty background) |
| `Delete` / `Backspace` | Delete selected polygon / vertex |
| `Ctrl+P` | Propagate selected object forward |
| `Ctrl+Z` | Undo last polygon vertex edit |

---

## 3. Project Management & Atomic Persistence

### Project Folder Anatomy:
```
my_project/
├── project.json           <- Main project file (atomic write)
├── project.backup.json    <- Automatic backup for crash recovery
├── frames/                <- Extracted frame images (.jpg)
├── thumbnails/            <- Scaled preview thumbnails (.jpg)
└── annotations/           <- Per-frame annotation JSON files
```

### Crash Recovery:
If power is interrupted or the process terminates abnormally, the application detects any corrupted JSON and automatically restores state from `project.backup.json`.

---

## 4. Multiple Export Options

Press `Ctrl+E` or choose **File > Export Dataset...** to open the unified export dialog. You can select between:
- **YOLOv8 Instance Segmentation**: Standard polygon labels with `data.yaml` and optional binary masks / previews.
- **YOLOv8 Object Detection**: Standard bounding box labels (`class_id cx cy w h`).
- **COCO 1.0 JSON**: Formatted for Detectron2, MMDetection, and TorchVision.
- **Pascal VOC & Semantic Masks**: Standard XML annotations + 8-bit indexed palette PNG masks.
- **LabelMe JSON**: One `.json` file per frame for LabelMe GUI interoperability.
- **MOT / MOTChallenge**: Tracking sequences with persistent track IDs in `gt.txt` and `seqinfo.ini`.
- **Rendered Video Overlays**: Standalone MP4 video with alpha-blended polygon fills, outlines, labels, and track badges.

See [docs/export_formats.md](export_formats.md) for detailed format layouts and usage instructions.

