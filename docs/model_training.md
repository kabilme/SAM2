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
   - **Train / Val Split:** 80% / 20% (Sequential split to avoid temporal video leakage)
4. Click **🚀 Start Training**.
5. Watch real-time metrics, live loss curves, and validation mAP in the telemetry dashboard.
6. When complete, click **📦 Export to ONNX...** to export ready-to-deploy weights.

---

## 2. Supported Engines & Roadmap

| Engine | Task | Backend | Status |
| :--- | :--- | :--- | :--- |
| **⚡ Ultralytics YOLO** | Instance Segmentation & Detection | `ultralytics` (YOLOv8, YOLO11) | **Available (Phase 1 & 2)** |
| **🔬 TorchVision Mask R-CNN** | Instance Segmentation & Detection | Native PyTorch & Torchvision | *Phase 3 (Next)* |
| **🎯 Meta SAM 2.1 Fine-Tuner** | Video Tracking Domain Adaptation | Meta SAM 2.1 Mask Decoder | *Phase 4 (Next)* |

---

## 3. Training Architecture & Non-Blocking GUI

The training system is designed with a non-blocking `QThread` architecture:
- `BaseModelTrainer` defines a unified interface for dataset formatting, training, and model export.
- `TrainingWorker` runs training in the background while streaming progress signals to the UI thread.
- `LossChartWidget` draws smooth antialiased loss and mAP curves directly via `QPainter` without heavy extra plotting dependencies.
- Cooperative cancellation safely saves the latest checkpoint (`last.pt`) if the user stops training prematurely.
