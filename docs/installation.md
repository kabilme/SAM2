# Installation Guide

This guide covers installing and setting up the **SAM2 Video Polygon Annotator** on local systems.

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
  - Recommended VRAM: 8 GB+ (e.g. RTX 3060, RTX 4070, RTX 4090)
- **Storage**: At least 2 GB free disk space for dependencies and model checkpoints; additional space for video datasets.

---

## 2. Python Environment Setup

The application supports Python **3.10**, **3.11**, **3.12**, and **3.13**.

### Step 1: Navigate to Repository
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

### Core Dependencies Overview:
- **GUI**: `PySide6>=6.5.0`
- **Deep Learning**: `torch>=2.3.0`, `torchvision>=0.18.0`
- **Vision & Models**: `ultralytics>=8.3.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, `pillow>=10.0.0`
- **SAM 2 Engine & Pipeline**: `hydra-core>=1.3.2`, `tqdm>=4.66.1`, `sam2` (or bundled local adapter)
- **Utilities & System**: `pyyaml>=6.0.0`, `psutil>=5.9.0`, `pytest>=7.4.0`

### Optional: CUDA-Accelerated PyTorch
If you have an NVIDIA GPU, install PyTorch with CUDA support matching your system:
```bash
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> [!NOTE]
> On Windows, native Meta SAM 2 initializes with `apply_postprocessing=False` by default, which executes pure PyTorch tensor operations and avoids requiring MSVC C++ compilers for custom `_C` CUDA extensions.

---

## 4. Verification

Verify that all core libraries and pipeline modules are functioning:
```bash
python -c "import PySide6, cv2, torch, ultralytics, hydra, tqdm; print('All core libraries imported successfully!')"
```

Check dependency consistency:
```bash
pip check
```

Run the automated test suite (50+ unit and integration tests):
```bash
python -m pytest tests/ -v
```

---

## 5. Next Steps

- Proceed to [SAM 2 Setup & Architecture](sam2_setup.md) to configure model checkpoints and GPU acceleration.
- Follow the [User Guide](user_guide.md) to launch the GUI and create your first video annotation project.
- Review [Troubleshooting](troubleshooting.md) if you encounter any environment or driver issues.


