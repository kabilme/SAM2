# In-App Deep Learning Model Training

The SAM 2 Video Polygon Annotator includes an integrated **Training Studio** allowing you to train computer vision models directly from your annotated project data without writing external scripts.

---

## 1. Quick Start

1. Annotate keyframes in your video project and propagate annotations.
2. Select **Dataset $\rightarrow$ Train Model...** (`Ctrl+T`) from the top menu.
3. Configure your desired engine, model architecture, and hyperparameters:
   - **Engine:** Ultralytics YOLO (Instance Segmentation & Object Detection)
   - **Base Checkpoint:** `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, etc.
   - **Epochs:** 50 (default)
   - **Batch Size:** 8 (default)
   - **Image Size:** 640 (default)
   - **Device:** Auto (detects CUDA GPU acceleration or falls back to CPU)
   - **Train / Val Split:** 80% / 20% (**Per-Video Stratified Split** for multi-video projects, or Sequential for single video)
4. Click **🚀 Start Training**.
5. Watch real-time metrics, live loss curves, and validation mAP in the telemetry dashboard.
6. When complete, click **📦 Export to ONNX...** to export ready-to-deploy weights.

---

## 2. Supported Engines & Training Modes

| Engine | Task | Backend | Status |
| :--- | :--- | :--- | :--- |
| **⚡ Ultralytics YOLO** | Instance Segmentation & Detection | `ultralytics` (YOLOv8, YOLO11) | **Available** |
| **🦅 Baidu RT-DETR** | NMS-Free Vision Transformer Detection | `ultralytics.RTDETR` | **Available** |
| **🔬 TorchVision Mask R-CNN** | Instance Segmentation & Detection | Native PyTorch & Torchvision (BSD-3) | **Available** |
| **🎯 Meta SAM 2.1 Fine-Tuner** | Video Tracking Domain Adaptation | Meta SAM 2.1 Mask Decoder | **Available** |

---

### Engine 1: Ultralytics YOLO (Production Edge Models)
- **Architectures:** YOLO11 / YOLOv8 (Nano, Small, Medium, Large)
- **Features:** High-speed inference, automatic hyperparameter scheduling, live batch loss telemetry, and one-click ONNX export.
- **Output:** `best.pt`, `last.pt`, `best.onnx`.

### Engine 2: Baidu RT-DETR (Real-Time Detection Transformer)
- **Architectures:** `rtdetr-l.pt` (Large, ~32M params), `rtdetr-x.pt` (Extra-Large, ~65M params).
- **Features:** NMS-free end-to-end detection via bipartite Hungarian matching, cross-scale feature interactions, and global self-attention for complex, occluded, or overlapping scenes.
- **Output:** `best.pt`, `last.pt`, `best.onnx`.

### Engine 3: TorchVision Mask R-CNN & Faster R-CNN (Permissive PyTorch Baseline)
- **Architectures:** `maskrcnn_resnet50_fpn_v2`, `fasterrcnn_resnet50_fpn_v2`, `fasterrcnn_mobilenet_v3_large_fpn`.
- **License:** Pure PyTorch BSD-3-Clause license for unrestricted enterprise and commercial distribution without AGPL copyleft requirements.
- **Dataset:** Formatted as standard COCO 1.0 JSON instances (`instances_train.json` / `instances_val.json`).
- **Output:** `best.pth`, `last.pth`, `best.onnx`.

### Engine 4: Meta SAM 2.1 Mask Decoder (Domain Adaptation)
- **Purpose:** Specializes Meta SAM 2.1 for your custom object textures and geometries to boost video propagation tracking accuracy.
- **Architecture:** Keeps the heavy Hiera image backbone frozen while fine-tuning the prompt encoder and lightweight mask decoder via combined Binary Cross-Entropy and Soft Dice loss.
- **Output:** `best_decoder.pt` / `last_decoder.pt` compatible directly with the in-app SAM 2 video predictor.

---

## 3. Training Architecture & Non-Blocking GUI

The training system is designed with a non-blocking `QThread` architecture:
- `BaseModelTrainer` defines a unified interface for dataset formatting, training, and model export across all engines.
- `TrainingWorker` runs training in the background while streaming progress signals to the UI thread.
- `LossChartWidget` draws smooth antialiased loss and mAP curves directly via `QPainter` without heavy extra plotting dependencies.
- Automatic checkpointing and cooperative cancellation safely preserve trained weights if stopped prematurely.

