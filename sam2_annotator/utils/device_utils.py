"""Hardware device detection and diagnostics utility."""

import platform
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

from sam2_annotator.utils.logging_utils import logger


@dataclass
class SystemDiagnostics:
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_device_count: int
    gpu_name: str
    gpu_vram_gb: Optional[float]
    cpu_info: str
    total_ram_gb: float
    available_ram_gb: float
    recommended_device: str


def get_system_diagnostics() -> SystemDiagnostics:
    """Detect local hardware, Python, PyTorch, and CUDA specifications."""
    py_ver = platform.python_version()
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0

    gpu_name = "None"
    vram_gb: Optional[float] = None

    if cuda_avail and device_count > 0:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)
        except Exception as e:
            logger.warning("Error querying CUDA device properties: %s", e)
            gpu_name = f"CUDA Device (error querying: {e})"
    else:
        # Check if Intel / AMD / other GPU controller exists
        try:
            import subprocess
            if platform.system() == "Windows":
                cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
                out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=3)
                names = [line.strip() for line in out.strip().splitlines() if line.strip()]
                if names:
                    gpu_name = ", ".join(names)
        except Exception:
            pass

    cpu_info = platform.processor() or platform.machine()
    ram = psutil.virtual_memory()
    total_ram = round(ram.total / (1024**3), 2)
    avail_ram = round(ram.available / (1024**3), 2)

    recommended = "cuda" if cuda_avail else "cpu"

    diag = SystemDiagnostics(
        python_version=py_ver,
        pytorch_version=torch_ver,
        cuda_available=cuda_avail,
        cuda_device_count=device_count,
        gpu_name=gpu_name,
        gpu_vram_gb=vram_gb,
        cpu_info=cpu_info,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        recommended_device=recommended,
    )
    logger.info("System Diagnostics: %s", diag)
    return diag


def get_torch_device(preferred: str = "auto") -> torch.device:
    """Return a safe PyTorch device based on preference and hardware availability."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preferred.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def clear_memory(device: torch.device) -> None:
    """Attempt safe garbage collection and CUDA cache clearing."""
    import gc
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning("Failed to empty CUDA cache: %s", e)
