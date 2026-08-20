"""
gpu_detector.py
===============
Hardware auto-detection module for deep learning and QLoRA fine-tuning.

Detects:
- GPU vendor, name, total & available VRAM
- CUDA availability, version, compute capability
- FP16 and BF16 hardware support
- Environment versions (PyTorch, Transformers, BitsAndBytes)
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    gpu_available: bool
    vendor: str
    gpu_name: str
    total_vram_gb: float
    free_vram_gb: float
    cuda_version: str
    compute_capability: Tuple[int, int]
    fp16_supported: bool
    bf16_supported: bool
    pytorch_version: str
    transformers_version: str
    bitsandbytes_available: bool
    bitsandbytes_version: str
    device_count: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["compute_capability"] = f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        return d


def check_bitsandbytes() -> Tuple[bool, str]:
    """Check if bitsandbytes is installed and operational."""
    try:
        import bitsandbytes as bnb
        version = getattr(bnb, "__version__", "unknown")
        return True, version
    except Exception as e:
        log.debug("bitsandbytes import error: %s", e)
        return False, "N/A"


def check_transformers() -> str:
    """Get transformers version if available."""
    try:
        import transformers
        return getattr(transformers, "__version__", "unknown")
    except ImportError:
        return "N/A"


def detect_hardware(device_idx: int = 0) -> HardwareProfile:
    """
    Auto-detect target system hardware capabilities.

    Parameters
    ----------
    device_idx : int
        CUDA device index to query (default: 0).

    Returns
    -------
    HardwareProfile
    """
    import torch

    pytorch_ver = torch.__version__
    trans_ver = check_transformers()
    bnb_avail, bnb_ver = check_bitsandbytes()

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return HardwareProfile(
            gpu_available=False,
            vendor="CPU",
            gpu_name="CPU (No CUDA GPU detected)",
            total_vram_gb=0.0,
            free_vram_gb=0.0,
            cuda_version="N/A",
            compute_capability=(0, 0),
            fp16_supported=False,
            bf16_supported=False,
            pytorch_version=pytorch_ver,
            transformers_version=trans_ver,
            bitsandbytes_available=bnb_avail,
            bitsandbytes_version=bnb_ver,
            device_count=0,
        )

    device_count = torch.cuda.device_count()
    device_idx = min(device_idx, device_count - 1)

    gpu_name = torch.cuda.get_device_name(device_idx)
    cuda_ver = torch.version.cuda or "Unknown"

    total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
    total_vram_gb = round(total_bytes / (1024**3), 2)

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
        free_vram_gb = round(free_bytes / (1024**3), 2)
    except Exception:
        free_vram_gb = total_vram_gb

    comp_cap = torch.cuda.get_device_capability(device_idx)

    # FP16 is supported on compute capability >= 7.0 (Volta, Turing, Ampere, Ada, Hopper, Blackwell)
    fp16_supported = comp_cap[0] >= 7

    # BF16 is supported natively on compute capability >= 8.0 (Ampere or newer, e.g., RTX 30xx, 40xx, 50xx, A100, H100)
    bf16_supported = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else (comp_cap[0] >= 8)

    vendor = "NVIDIA"

    return HardwareProfile(
        gpu_available=True,
        vendor=vendor,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        free_vram_gb=free_vram_gb,
        cuda_version=cuda_ver,
        compute_capability=comp_cap,
        fp16_supported=fp16_supported,
        bf16_supported=bf16_supported,
        pytorch_version=pytorch_ver,
        transformers_version=trans_ver,
        bitsandbytes_available=bnb_avail,
        bitsandbytes_version=bnb_ver,
        device_count=device_count,
    )


def print_hardware_report(profile: Optional[HardwareProfile] = None) -> HardwareProfile:
    """Print clean formatted hardware detection report."""
    if profile is None:
        profile = detect_hardware()

    comp_cap_str = f"{profile.compute_capability[0]}.{profile.compute_capability[1]}" if profile.gpu_available else "N/A"

    print("========================================")
    print(" HARDWARE DETECTION REPORT")
    print("========================================")
    print(f"  GPU Vendor       : {profile.vendor}")
    print(f"  GPU Name         : {profile.gpu_name}")
    print(f"  Total VRAM       : {profile.total_vram_gb:.2f} GB")
    print(f"  Free VRAM        : {profile.free_vram_gb:.2f} GB")
    print(f"  CUDA Available   : {'Yes' if profile.gpu_available else 'No'}")
    print(f"  CUDA Version     : {profile.cuda_version}")
    print(f"  Compute Cap.     : {comp_cap_str}")
    print(f"  FP16 Support     : {'Supported' if profile.fp16_supported else 'Unsupported'}")
    print(f"  BF16 Support     : {'Supported' if profile.bf16_supported else 'Unsupported'}")
    print(f"  PyTorch Version  : {profile.pytorch_version}")
    print(f"  Transformers Ver : {profile.transformers_version}")
    print(f"  BitsAndBytes     : {profile.bitsandbytes_version if profile.bitsandbytes_available else 'Not Available'}")
    print("========================================\n")

    return profile


if __name__ == "__main__":
    print_hardware_report()
