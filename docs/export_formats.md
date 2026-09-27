# Dataset & Video Export Formats Guide

The **SAM2 Video Polygon Annotator** supports exporting projects into 7 industry-standard computer vision formats, ranging from deep learning object detection and instance segmentation datasets to multi-object tracking sequences and standalone overlay video files.

---

## Supported Formats Summary

| Format ID | Name | Primary Target | Splits | Segmentation | BBoxes | Tracking IDs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `yolo_segmentation` | YOLOv8 Instance Segmentation | Ultralytics YOLOv8/v9/v11-seg | Yes | Polygon | Derived | No |
| `yolo_detection` | YOLOv8 Object Detection | Ultralytics YOLOv8/v9/v11-det | Yes | No | Normalized BBox | No |
| `coco` | COCO 1.0 JSON | Detectron2, MMDetection, Hugging Face | Yes | Polygon | $[x, y, w, h]$ | No |
| `pascal_voc` | Pascal VOC & Semantic Masks | Classic VOC, Semantic Segmentation | Yes | Palette PNG | $[xmin, ymin, xmax, ymax]$ | No |
| `labelme` | LabelMe JSON | LabelMe GUI, CVAT, Roboflow | Yes | Polygon | Derived | Yes |
| `mot` | MOT / MOTChallenge Tracking | ByteTrack, DeepSORT, TrackEval | No | No | $[x, y, w, h]$ | Yes (1..N) |
| `rendered_video` | Annotated Video Overlays (.mp4) | Presentations, Demos, Reviewers | No | Alpha Blend | Overlay Box & Badge | Yes (Badge) |

---

## 1. YOLOv8 Instance Segmentation (`yolo_segmentation`)

Generates normalized polygon coordinates for Ultralytics YOLO segmentation models.

### Directory Structure:
```
dataset_out/
├── data.yaml
├── images/
│   ├── train/ (frame_0001.jpg, ...)
│   └── val/
├── labels/
│   ├── train/ (frame_0001.txt, ...)
│   └── val/
├── masks/ (Optional binary PNGs)
├── previews/ (Optional overlay previews)
└── dataset.zip (Optional)
```

### Label Format (`.txt`):
```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```
*Coordinates are normalized to $[0.0, 1.0]$. Empty `.txt` files represent negative background samples.*

---

## 2. YOLOv8 Object Detection (`yolo_detection`)

Generates normalized bounding box coordinates for standard object detection training without polygon overhead.

### Directory Structure:
```
dataset_out/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.zip (Optional)
```

### Label Format (`.txt`):
```text
<class_id> <x_center> <y_center> <width> <height>
```
*Bounding box coordinates are normalized floating-point values between $0.0$ and $1.0$.*

---

## 3. COCO 1.0 Instance Segmentation (`coco`)

Standard JSON format widely accepted by PyTorch, TorchVision, Detectron2, MMDetection, and Hugging Face.

### Directory Structure:
```
coco_out/
├── README.md
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── coco_dataset.zip (Optional)
```

### JSON Structure:
```json
{
  "info": { "description": "COCO dataset exported from SAM2 Annotator", ... },
  "images": [
    { "id": 1, "file_name": "frame_0001.jpg", "width": 1920, "height": 1080 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 14250.5,
      "bbox": [xmin, ymin, width, height],
      "iscrowd": 0
    }
  ],
  "categories": [
    { "id": 1, "name": "vehicle", "supercategory": "object" }
  ]
}
```

---

## 4. Pascal VOC & Semantic Masks (`pascal_voc`)

Exports classic Pascal VOC XML annotations containing both `<bndbox>` and `<polygon>` tags, paired with 8-bit indexed palette PNG masks for semantic segmentation networks (DeepLab, UNet, FCN).

### Directory Structure:
```
pascal_voc_out/
├── Annotations/          <- XML files (frame_0001.xml)
├── JPEGImages/           <- Images (frame_0001.jpg)
├── SegmentationClass/    <- 8-bit indexed palette PNG masks (pixel value = class_id + 1)
├── ImageSets/
│   ├── Main/
│   │   ├── train.txt
│   │   └── val.txt
│   └── Segmentation/
│       ├── train.txt
│       └── val.txt
└── pascal_voc_dataset.zip (Optional)
```

### XML Annotation Example:
```xml
<annotation>
  <folder>JPEGImages</folder>
  <filename>frame_0001.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
    <depth>3</depth>
  </size>
  <segmented>1</segmented>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>320</xmax>
      <ymax>580</ymax>
    </bndbox>
    <polygon>
      <pt><x>100</x><y>150</y></pt>
      <pt><x>320</x><y>150</y></pt>
      <pt><x>320</x><y>580</y></pt>
      <pt><x>100</x><y>580</y></pt>
    </polygon>
  </object>
</annotation>
```

---

## 5. LabelMe JSON (`labelme`)

Generates one `.json` annotation file alongside each frame image. Fully compatible with the open-source LabelMe desktop annotation tool.

### Directory Structure:
```
labelme_out/
├── images/
│   ├── train/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0001.json
│   │   └── ...
│   └── val/
└── labelme_dataset.zip (Optional)
```

### File Format:
```json
{
  "version": "5.3.1",
  "flags": {},
  "shapes": [
    {
      "label": "car",
      "points": [[100.0, 150.0], [320.0, 150.0], [320.0, 580.0], [100.0, 580.0]],
      "group_id": null,
      "description": "track_id:veh_01",
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "frame_0001.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

---

## 6. MOT / MOTChallenge Video Tracking (`mot`)

Exports ground-truth sequences formatted for Multi-Object Tracking (MOT) benchmarks and tracker evaluations (e.g. ByteTrack, DeepSORT, OC-SORT, TrackEval).

### Directory Structure:
```
mot_out/
└── seq01/
    ├── seqinfo.ini       <- Sequence metadata (fps, length, resolution)
    ├── img1/             <- Sequential frames: 000001.jpg, 000002.jpg, ...
    └── gt/
        └── gt.txt        <- Tracking ground truth CSV
```

### Ground Truth CSV Format (`gt.txt`):
```text
<frame_index>,<track_id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class_id>,<visibility>
1,1,120.00,80.00,240.00,310.00,1,1,1.0
1,2,540.00,110.00,180.00,290.00,1,2,1.0
2,1,122.50,81.00,239.00,311.00,1,1,1.0
```

---

## 7. Rendered Video with Overlays (`rendered_video`)

Directly produces a high-definition MP4 video with polished visual overlays:
- **Semi-transparent color-coded polygon fills** with configurable alpha blending (default: $0.45$).
- **Anti-aliased polygon boundary outlines**.
- **Optional bounding boxes**.
- **Class label badges** with tracking ID indicators.
- **Top-left frame watermark** (`Frame: #0001 | Objects: 2`).

### Export Options:
- **Playback FPS**: Configure playback speed (e.g., 10, 24, 30, 60 fps).
- **Alpha Blending**: Control polygon opacity ($0.1$ to $0.9$).
- **Bounding Boxes**: Toggle overlay bounding boxes on/off.
- **Output Codec**: Automatically attempts `mp4v`, `avc1`, `H264`, and `XVID` with cross-platform fallback.
