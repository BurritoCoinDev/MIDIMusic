"""Shared helpers for the torch-based backends.

Centralises the things that are easy to get wrong on a multi-vendor Windows
install: picking a device, guarding imports that fail at import time on ROCm
Windows builds, and never touching torch.compile (there is no Triton for
Windows, so compiling is a hard failure rather than a slow path).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "require_torch",
    "resolve_device",
    "torch_dtype_for",
    "apply_model_env",
    "free_vram_gb",
    "empty_cache",
    "can_compile",
    "seed_everything",
]


def require_torch() -> Any:
    """Import torch, or raise BackendUnavailable naming what to install."""
    from ..core.generator import BackendUnavailable

    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BackendUnavailable(
            "PyTorch is not installed. Install a compute backend from Settings.",
            missing_packages=["torch"],
        ) from exc


def resolve_device(preferred: str = "auto") -> str:
    """Choose a torch device string.

    ROCm builds report themselves as ``cuda``, which is correct -- the HIP
    runtime deliberately mimics the CUDA API -- so there is no separate
    ``rocm`` device to ask for.
    """
    torch = require_torch()
    if preferred and preferred not in ("auto", "rocm", "rocm-windows"):
        return preferred
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


def is_rocm() -> bool:
    try:
        torch = require_torch()
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def torch_dtype_for(device: str) -> Any:
    """Half precision on GPU, float32 on CPU (where fp16 is slower, not faster)."""
    torch = require_torch()
    if device == "cpu":
        return torch.float32
    if device == "mps":
        return torch.float32
    return torch.float16


def can_compile() -> bool:
    """Whether torch.compile is safe here.

    It is not on Windows: AMD publishes no Triton wheels for Windows, so the
    Inductor backend fails outright rather than falling back.
    """
    if sys.platform == "win32":
        return False
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def apply_model_env(env: dict[str, str] | None) -> None:
    """Apply a catalog entry's environment requirements before loading.

    ACE-Step, for example, must be told to use its PyTorch LM backend; its
    default is vLLM, which does not exist on AMD or on Windows.
    """
    for key, value in (env or {}).items():
        os.environ[key] = str(value)


def free_vram_gb(device: str = "cuda") -> float:
    try:
        torch = require_torch()
        if device.startswith("cuda") and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1024 ** 3
    except Exception:
        pass
    return 0.0


def empty_cache() -> None:
    try:
        torch = require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        torch = require_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
