"""GPU and compute-capability detection.

Detection has to work *before* torch is installed, because the whole point is
to choose which torch build to install.  On Windows that means asking the OS:
DXGI through ctypes gives an accurate VRAM figure, and CIM is the fallback.
(``wmic`` is deliberately not used -- it is removed in current Windows 11 --
and ``Win32_VideoController.AdapterRAM`` is a 32-bit field that saturates at
4 GB, so it is only ever a last resort.)
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum

__all__ = ["Vendor", "GPU", "SystemInfo", "detect_system", "detect_gpus", "recommended_backend"]


class Vendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


_VENDOR_IDS = {0x10DE: Vendor.NVIDIA, 0x1002: Vendor.AMD, 0x1022: Vendor.AMD,
               0x8086: Vendor.INTEL, 0x106B: Vendor.APPLE}

# RDNA3/RDNA4 discrete cards that AMD's Windows ROCm wheels target.
_ROCM_WINDOWS_GFX = {
    "7900 xtx": "gfx1100", "7900 xt": "gfx1100", "7900 gre": "gfx1100",
    "7800 xt": "gfx1101", "7700 xt": "gfx1101",
    "7600": "gfx1102", "7650": "gfx1102",
    "9070 xt": "gfx1201", "9070": "gfx1201", "9060": "gfx1200",
    "780m": "gfx1103", "880m": "gfx1150", "890m": "gfx1151",
}


@dataclass
class GPU:
    name: str = "Unknown"
    vendor: Vendor = Vendor.UNKNOWN
    vram_mb: int = 0
    driver: str = ""
    gfx_arch: str = ""  # AMD LLVM target, e.g. gfx1100

    @property
    def vram_gb(self) -> float:
        return round(self.vram_mb / 1024.0, 1)

    def supports_rocm_windows(self) -> bool:
        return self.vendor is Vendor.AMD and bool(self.gfx_arch)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vendor"] = self.vendor.value
        d["vram_gb"] = self.vram_gb
        return d


@dataclass
class SystemInfo:
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    cpu: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpus: list[GPU] = field(default_factory=list)
    torch_installed: bool = False
    torch_version: str = ""
    torch_device: str = "cpu"

    @property
    def primary_gpu(self) -> GPU | None:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: (g.vendor is not Vendor.INTEL, g.vram_mb))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpus"] = [g.to_dict() for g in self.gpus]
        return d

    def summary(self) -> str:
        gpu = self.primary_gpu
        gpu_txt = f"{gpu.name} ({gpu.vram_gb} GB)" if gpu else "no discrete GPU"
        return f"{self.cpu or 'CPU'} / {gpu_txt} / torch: {self.torch_version or 'not installed'}"


# --------------------------------------------------------------------------
# GPU enumeration
# --------------------------------------------------------------------------

def _gfx_for(name: str) -> str:
    low = name.lower()
    for key, gfx in _ROCM_WINDOWS_GFX.items():
        if key in low:
            return gfx
    return ""


def _vendor_from_name(name: str) -> Vendor:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low:
        return Vendor.NVIDIA
    if "radeon" in low or "amd" in low or "firepro" in low:
        return Vendor.AMD
    if "intel" in low or "arc" in low or "iris" in low:
        return Vendor.INTEL
    if "apple" in low:
        return Vendor.APPLE
    return Vendor.UNKNOWN


def _detect_windows_dxgi() -> list[GPU]:
    """Enumerate adapters through DXGI, which reports real VRAM."""
    gpus: list[GPU] = []
    try:
        from ctypes import POINTER, Structure, byref, c_void_p, wintypes

        class LUID(Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class DXGI_ADAPTER_DESC1(Structure):
            _fields_ = [
                ("Description", wintypes.WCHAR * 128),
                ("VendorId", wintypes.UINT),
                ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT),
                ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID),
                ("Flags", wintypes.UINT),
            ]

        dxgi = ctypes.windll.dxgi
        factory = c_void_p()
        # IID_IDXGIFactory1
        iid = (ctypes.c_byte * 16)(
            0x2A, 0xB3, 0x0A, 0x77, 0x0C, 0xD6, 0xF6, 0x47,
            0xB2, 0x2D, 0xDF, 0xF3, 0x87, 0x71, 0xC5, 0xAF,
        )
        if dxgi.CreateDXGIFactory1(byref(iid), byref(factory)) != 0:
            return gpus

        vtbl = ctypes.cast(factory, POINTER(POINTER(c_void_p))).contents
        # IDXGIFactory1::EnumAdapters1 is slot 12; Release is slot 2.
        enum_proto = ctypes.WINFUNCTYPE(
            ctypes.c_long, c_void_p, wintypes.UINT, POINTER(c_void_p)
        )
        enum_adapters = enum_proto(vtbl[12])
        release_proto = ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)

        index = 0
        while index < 16:
            adapter = c_void_p()
            if enum_adapters(factory, index, byref(adapter)) != 0:
                break
            a_vtbl = ctypes.cast(adapter, POINTER(POINTER(c_void_p))).contents
            # IDXGIAdapter1::GetDesc1 is slot 10.
            desc_proto = ctypes.WINFUNCTYPE(
                ctypes.c_long, c_void_p, POINTER(DXGI_ADAPTER_DESC1)
            )
            get_desc = desc_proto(a_vtbl[10])
            desc = DXGI_ADAPTER_DESC1()
            if get_desc(adapter, byref(desc)) == 0:
                # Flags bit 0 marks the software (WARP) adapter; skip it.
                if not (desc.Flags & 0x2):
                    name = desc.Description.strip()
                    vendor = _VENDOR_IDS.get(desc.VendorId, _vendor_from_name(name))
                    vram = int(desc.DedicatedVideoMemory // (1024 * 1024))
                    if vram > 64:
                        gpus.append(GPU(name=name, vendor=vendor, vram_mb=vram,
                                        gfx_arch=_gfx_for(name)))
            release_proto(a_vtbl[2])(adapter)
            index += 1
        release_proto(vtbl[2])(factory)
    except Exception:
        return []
    return gpus


def _detect_windows_cim() -> list[GPU]:
    """Fallback: PowerShell CIM query. AdapterRAM saturates at 4 GB."""
    gpus: list[GPU] = []
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return gpus
    cmd = [
        ps, "-NoProfile", "-NonInteractive", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        data = json.loads(out.stdout or "[]")
    except Exception:
        return gpus
    if isinstance(data, dict):
        data = [data]
    for entry in data:
        name = (entry.get("Name") or "").strip()
        if not name:
            continue
        ram = entry.get("AdapterRAM") or 0
        gpus.append(
            GPU(name=name, vendor=_vendor_from_name(name),
                vram_mb=int(ram) // (1024 * 1024) if ram else 0,
                driver=str(entry.get("DriverVersion") or ""),
                gfx_arch=_gfx_for(name))
        )
    return gpus


def _detect_linux() -> list[GPU]:
    gpus: list[GPU] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpus.append(GPU(parts[0], Vendor.NVIDIA, int(float(parts[1])),
                                    parts[2] if len(parts) > 2 else ""))
        except Exception:
            pass
    if shutil.which("rocm-smi"):
        try:
            out = subprocess.run(["rocm-smi", "--showproductname", "--json"],
                                 capture_output=True, text=True, timeout=15).stdout
            data = json.loads(out or "{}")
            for _card, info in data.items():
                name = info.get("Card Series") or info.get("Card model") or "AMD GPU"
                gpus.append(GPU(str(name), Vendor.AMD, 0, gfx_arch=_gfx_for(str(name))))
        except Exception:
            pass
    if not gpus:
        try:
            out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                if re.search(r"VGA|3D controller|Display", line):
                    name = line.split(":", 2)[-1].strip()
                    gpus.append(GPU(name, _vendor_from_name(name), 0, gfx_arch=_gfx_for(name)))
        except Exception:
            pass
    return gpus


def detect_gpus() -> list[GPU]:
    if sys.platform == "win32":
        gpus = _detect_windows_dxgi()
        if not gpus:
            gpus = _detect_windows_cim()
        return gpus
    if sys.platform == "darwin":
        return [GPU("Apple Silicon GPU", Vendor.APPLE, 0)]
    return _detect_linux()


# --------------------------------------------------------------------------
# System summary
# --------------------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "win32":
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()) or "CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "CPU"


def _ram_gb() -> float:
    try:
        if sys.platform == "win32":
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MemStatus()
            stat.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1024 ** 3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except Exception:
        return 0.0


def detect_system() -> SystemInfo:
    info = SystemInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=platform.python_version(),
        cpu=_cpu_name(),
        cpu_cores=os.cpu_count() or 0,
        ram_gb=_ram_gb(),
        gpus=detect_gpus(),
    )
    try:
        import torch

        info.torch_installed = True
        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            # torch.cuda covers ROCm builds too; version.hip distinguishes them.
            info.torch_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info.torch_device = "mps"
        else:
            info.torch_device = "cpu"
    except Exception:
        info.torch_installed = False
    return info


def recommended_backend(info: SystemInfo | None = None) -> str:
    """Which torch build this machine should install.

    Returns one of ``rocm-windows``, ``cuda``, ``xpu``, ``mps``, ``cpu``.
    """
    info = info or detect_system()
    gpu = info.primary_gpu
    if gpu is None:
        return "mps" if sys.platform == "darwin" else "cpu"
    if gpu.vendor is Vendor.NVIDIA:
        return "cuda"
    if gpu.vendor is Vendor.APPLE:
        return "mps"
    if gpu.vendor is Vendor.AMD:
        # AMD ships native Windows ROCm wheels; on Linux ROCm is also fine.
        if sys.platform == "win32":
            return "rocm-windows" if gpu.supports_rocm_windows() else "cpu"
        return "rocm"
    if gpu.vendor is Vendor.INTEL:
        return "xpu"
    return "cpu"
