# User Guide: SAM2 Video Polygon Annotator

This document provides a comprehensive tour of the user interface, keyboard shortcuts, project management, tracking controls, and daily operations.

---

## 1. Application Layout

The graphical interface is built with **PySide6** and optimized for dark-mode productivity:

- **Top Toolbar**: Quick access to project operations (New, Open, Save, Export), interaction modes (Select, Point +, Point -, Box, Polygon, Edit), propagation trigger (`Ctrl+P`), frame review actions (`Reviewed`, `Negative`), and zoom controls.
- **Center Canvas**: Interactive `QGraphicsView` supporting hardware-accelerated panning, smooth zooming under cursor, prompt placement (points and bounding boxes), and polygon vertex dragging.
- **Bottom Frame Timeline**: Visual thumbnail strip showing frame numbers, timestamps, review flags (green check for reviewed, red cross for negative), and an interactive scrubbing slider. Accelerated by in-memory icon caching for instant filtering.
- **Left Dock - Classes**: Create, rename, recolor, and select object categories.
- **Left Dock - Video Info**: Displays video metadata (resolution, fps, duration, extracted frame count).
- **Right Dock - Objects & Tracking**:
  - **Inspection**: View active frame annotations, polygon area, perimeter, vertex count, and prediction confidence.
  - **Tracking Scope**: Choose between *Fixed Frame Count*, *Until Next Keyframe*, or *Until End of Video*.
  - **Tracking Prompt**: Choose between *🔲 Box Prompt*, *📍 Centroid Point*, or *🔲+📍 Box + Center Point*.
  - **Interval Presets**: Quick-select buttons for `10`, `30`, `60`, or `100` frames, or fine-tune with the frame count spinbox.
  - **Actions**: Trigger background tracking (**Propagate ⏩**) or remove instances (**Delete 🗑️**).
- **Status Bar**: Real-time cursor coordinates, active frame info, selected hardware device (CPU / CUDA), and live propagation logs (`Frame X/Y, conf=0.XX, N vertices`).

---

## 2. Keyboard Shortcuts

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `Ctrl+N` | New Project | Extract frames from video and initialize annotation workspace |
| `Ctrl+O` | Open Project | Load an existing `project.json` workspace |
| `Ctrl+S` | Save Project | Atomically persist all frame annotations and project metadata |
| `Ctrl+E` | Export Dataset | Open export dialog for YOLO, COCO, VOC, LabelMe, MOT, MP4 |
| `Ctrl+P` | Propagate Object | Track selected object forward using Meta SAM 2.1 video predictor |
| `Ctrl+Z` | Undo | Undo the last annotation or vertex modification |
| `Ctrl+Y` | Redo | Redo the last undone modification |
| `Right Arrow` / `D` | Next Frame | Advance to the subsequent video frame |
| `Left Arrow` / `A` | Previous Frame | Return to the preceding video frame |
| `Space` + Drag | Pan Canvas | Pan across the canvas at any zoom level |
| `Mouse Wheel` | Smooth Zoom | Zoom centered directly under the mouse cursor |
| `F` | Fit to View | Auto-scale image to fit canvas viewport |
| `1` | Select Mode | Select and inspect polygon instances |
| `2` | Positive Point (`+`) | Place foreground prompt for SAM segmentation |
| `3` | Negative Point (`-`) | Place background exclusion prompt for SAM |
| `4` | Box Prompt Mode | Draw bounding box prompt for SAM |
| `5` | Manual Polygon | Click arbitrary boundary points to build custom polygon |
| `E` | Edit Mode | Drag, insert, or delete polygon vertices |
| `R` | Mark Reviewed | Mark active frame as reviewed and verified |
| `N` | Mark Negative | Mark active frame as negative/background (no target objects) |
| `Delete` / `Backspace`| Delete | Remove selected vertex or entire polygon annotation |

---

## 3. Video Tracking (Propagation) Step-by-Step

To propagate an object across a video sequence:

1. **Annotate Initial Frame**: Use point prompts (`+`/`-`) or a bounding box to segment the target object in your starting frame.
2. **Select the Object**: Click the polygon in the canvas or select it in the **Objects List** on the right dock.
3. **Configure Propagation**:
   - **Scope**: Choose *Fixed Frame Count*, *Until Next Keyframe*, or *Until End of Video*.
   - **Prompt**: Choose *🔲 Box Prompt* (default), *📍 Centroid Point*, or *🔲+📍 Box + Center Point*.
   - **Frames**: Select a preset (`10`, `30`, `60`, `100`) or set the exact frame count.
4. **Click Propagate ⏩** (or press `Ctrl+P`).
5. **Monitor Progress**:
   - The non-blocking progress dialog displays percentage, frame number, confidence score, and vertex count.
   - Main thread remains completely responsive; click **Cancel** at any time to preserve annotations computed so far.
6. **Automatic Batch Commit**:
   - On completion, annotations are committed in a single atomic transaction.
   - The progress dialog indicates "Applying tracked annotations to project..." before closing smoothly with zero UI freeze.

---

## 4. Project Management & Atomic Persistence

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

## 5. Multiple Export Options

Press `Ctrl+E` or choose **File > Export Dataset...** to open the unified export dialog. You can select between:
- **YOLOv8 Instance Segmentation**: Standard polygon labels with `data.yaml` and optional binary masks / previews.
- **YOLOv8 Object Detection**: Standard bounding box labels (`class_id cx cy w h`).
- **COCO 1.0 JSON**: Formatted for Detectron2, MMDetection, and TorchVision.
- **Pascal VOC & Semantic Masks**: Standard XML annotations + 8-bit indexed palette PNG masks.
- **LabelMe JSON**: One `.json` file per frame for LabelMe GUI interoperability.
- **MOT / MOTChallenge**: Tracking sequences with persistent track IDs in `gt.txt` and `seqinfo.ini`.
- **Rendered Video Overlays**: Standalone MP4 video with alpha-blended polygon fills, outlines, labels, and track badges.

See [docs/export_formats.md](export_formats.md) for detailed format layouts and usage instructions.


