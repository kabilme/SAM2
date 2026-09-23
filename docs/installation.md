# Installation Guide

This guide covers installing and setting up the **SAM3 Video Polygon Annotator** on local systems.

---

## 1. System Requirements

### Operating Systems
- **Windows**: Windows 10 / 11 (64-bit)
- **Linux**: Ubuntu 20.04+, Debian 11+, Fedora 36+, or compatible Linux distributions
- **macOS**: macOS 12+ (Apple Silicon supported via MPS / CPU)

### Hardware Requirements
- **RAM**: Minimum 8 GB (16 GB or more recommended for high-resolution video)
- **CPU**: Multi-core processor (x86_64 or ARM64)
- **GPU (Recommended)**: NVIDIA GPU with CUDA 11.8+ or 12.x support
  - Minimum VRAM: 4 GB (e.g. GTX 1650, RTX 3050)
  - Recommended VRAM: 8 GB+ (e.g. RTX 3060, RTX 4070)
- **Storage**: At least 2 GB free disk space for dependencies and model checkpoints; additional space for video datasets.

---

## 2. Python Environment Setup

The application supports Python **3.10**, **3.11**, **3.12**, and **3.13**.

### Step 1: Clone or Navigate to Repository
```bash
cd D:/SAM3
```

### Step 2: Create a Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

### Install via `requirements.txt`:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Optional: CUDA-Accelerated PyTorch
If you have an NVIDIA GPU, install PyTorch with CUDA support matching your system:
```bash
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Verification

Verify that PySide6, OpenCV, PyTorch, and Ultralytics are functioning:
```bash
python -c "import PySide6, cv2, torch, ultralytics; print('All core libraries imported successfully!')"
```

To run the automated test suite:
```bash
python -m pytest tests/ -v
```
