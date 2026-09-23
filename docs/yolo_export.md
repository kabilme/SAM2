# YOLOv8 Instance Segmentation Dataset Export

This document details the YOLOv8 dataset export structure, formatting specifications, automated validation, and Ultralytics training instructions.

---

## 1. Directory Structure

Exporting creates the standard YOLOv8 instance segmentation folder layout:

```
exported_dataset/
├── data.yaml                  <- YOLO configuration file
├── dataset_manifest.json      <- Machine-readable metadata and hashes
├── README.md                  <- Dataset summary, license, and stats
├── validation_report.txt      <- Human-readable validation report
├── validation_report.json     <- Machine-readable validation report
├── images/
│   ├── train/                 <- Training images (JPEG)
│   ├── val/                   <- Validation images (JPEG)
│   └── test/                  <- Optional test images (JPEG)
├── labels/
│   ├── train/                 <- Training label text files
│   ├── val/                   <- Validation label text files
│   └── test/                  <- Optional test label text files
├── masks/ (Optional)
│   ├── train/                 <- Binary segmentation masks (PNG)
│   └── val/
└── previews/ (Optional)
    ├── train/                 <- Visual overlays with rendered polygons
    └── val/
```

If the **Create ZIP Archive** option is enabled, an atomic compressed file (`dataset.zip`) is created alongside the output directory.

---

## 2. Label File Format

Each `.txt` file in `labels/` corresponds to an image file with the same basename in `images/`.

### Format Specification:
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```

- `class_id`: 0-indexed integer corresponding to classes in `data.yaml`.
- All coordinates $x_i, y_i$ are **normalized floating-point numbers** in the range $[0.0, 1.0]$:
  $$x = \frac{X_{\text{pixel}}}{W}, \quad y = \frac{Y_{\text{pixel}}}{H}$$
- Minimum 3 coordinate pairs ($N \ge 3$) representing a closed polygon.
- An empty `.txt` file represents a **negative background image** (contains no target objects).

### Example:
```
0 0.452300 0.312000 0.485100 0.315000 0.512000 0.395000 0.442000 0.391000
1 0.120000 0.650000 0.180000 0.652000 0.175000 0.720000 0.115000 0.718000
```

---

## 3. Configuration File (`data.yaml`)

Generated `data.yaml` example:

```yaml
path: D:/SAM3/exported_dataset
train: images/train
val: images/val
test: images/test

names:
  0: scooter
  1: person
  2: helmet
```

---

## 4. Dataset Splitting Strategies

The application provides three splitting strategies:
1. **Sequential**: Frames are chronologically split (e.g. 70% start = train, 20% middle = val, 10% end = test). Best for video to avoid temporal leakage between consecutive frames.
2. **Grouped**: Splits continuous frame segments or video chunks into independent sets.
3. **Random**: Randomly distributes frames into splits according to user-specified ratios.

---

## 5. Automated Dataset Validation

Before dataset completion, the internal `DatasetValidator` automatically verifies:
- Pairwise matching between all images and labels.
- Image resolution validity and non-zero byte size.
- Class IDs are valid integers within the defined class list.
- All polygon vertices are finite, non-NaN, and strictly inside $[0.0, 1.0]$.
- Polygon area is greater than the zero-area noise threshold.
- Non-empty label counts, split distribution, and balance.

---

## 6. Training with Ultralytics YOLO

The exported dataset is immediately compatible with the official Ultralytics YOLOv8 segmentation pipeline:

```bash
# Command Line Interface:
yolo segment train data=exported_dataset/data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640 batch=16

# Python API:
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
results = model.train(data="exported_dataset/data.yaml", epochs=50, imgsz=640)
```
