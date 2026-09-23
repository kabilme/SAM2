# Troubleshooting Guide

This guide covers common issues, debugging steps, and error resolution.

---

## 1. Application Log Files

All internal logs, warnings, and errors are written in real-time to:
```
logs/app.log
```
To monitor application logs live in PowerShell:
```powershell
Get-Content logs/app.log -Wait -Tail 50
```

---

## 2. Common Issues & Solutions

### A. CUDA Out of Memory (OOM)
- **Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory` during segmentation or batch propagation.
- **Cause**: Image resolution is very high (e.g. 4K) or VRAM is insufficient.
- **Resolution**:
  1. The app automatically intercepts OOM and clears CUDA cache.
  2. Open **Settings > Model Settings** and switch precision to `fp16`.
  3. If your GPU has < 4 GB VRAM, switch device to `cpu` in Settings or start with `--device cpu`.

### B. Black or Missing Frame Previews
- **Symptom**: Frame images fail to display on canvas or thumbnails are blank.
- **Cause**: OpenCV VideoCapture failed to decode a proprietary or unsupported codec.
- **Resolution**: Re-encode the video using standard H.264 MP4:
  ```bash
  ffmpeg -i input_video.mov -vcodec libx264 -pix_fmt yuv420p output_video.mp4
  ```

### C. SAM Model Fails to Load
- **Symptom**: Error: `SAM model is not loaded` or weights not found.
- **Cause**: Missing `sam2.1_t.pt` file in offline environment.
- **Resolution**: Download the weights file and place it in the application root directory:
  ```powershell
  Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt" -OutFile "sam2.1_t.pt"
  ```

### D. Dataset Validation Errors
- **Symptom**: Export dialog reports validation failure (e.g. coordinates outside $[0, 1]$ or empty polygon).
- **Cause**: Polygon was dragged outside canvas limits.
- **Resolution**: The exporter clamps boundary coordinates within $[0.0, 1.0]$. Check `validation_report.txt` in the export folder to pinpoint the frame and object ID causing the warning.

### E. High-DPI Display Scaling Issues on Windows
- **Symptom**: Interface fonts or canvas cursor appear misaligned on 4K / High-DPI screens.
- **Resolution**: The application enables Qt High-DPI scaling automatically (`QT_ENABLE_HIGHDPI_SCALING=1`). Ensure Windows display scale is set to recommended 100%, 125%, or 150%.
