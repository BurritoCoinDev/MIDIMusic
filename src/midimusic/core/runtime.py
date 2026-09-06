"""Provisioning the compute runtime.

Torch is not bundled: the right build depends on the GPU, and shipping every
variant would mean a multi-gigabyte installer that is wrong for most users.
Instead the app detects the hardware and installs the matching wheels on
demand.

The AMD case is the interesting one.  AMD now publishes native Windows ROCm
wheels with gfx1100 (RX 7900 XT/XTX) as a named target, so an AMD user on
Windows gets real GPU acceleration -- no WSL, no DirectML, no ZLUDA.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["RUNTIME_OPTIONS", "InstallHandle", "RuntimeOption", "install_runtime", "options_for"]

log = logging.getLogger(__name__)


@dataclass
class RuntimeOption:
    id: str
    name: str
    description: str
    index_url: str = ""
    packages: tuple[str, ...] = ("torch", "torchaudio")
    extra_args: tuple[str, ...] = ()
    python_requires: str = ""
    platforms: tuple[str, ...] = ("win32", "linux", "darwin")
    approx_gb: float = 3.0
    notes: str = ""
    incompatible: tuple[str, ...] = ()

    def available_here(self) -> bool:
        if sys.platform not in self.platforms:
            return False
        if self.python_requires:
            major, minor = sys.version_info[:2]
            want = tuple(int(p) for p in self.python_requires.split("."))
            if (major, minor) != want:
                return False
        return True


RUNTIME_OPTIONS: list[RuntimeOption] = [
    RuntimeOption(
        id="rocm-windows",
        name="AMD GPU (ROCm for Windows)",
        description="Native GPU acceleration for Radeon RX 7000/9000 series on Windows.",
        index_url="https://stable.repo.amd.com/rocm/whl-next/",
        packages=("torch[device-gfx1100]", "torchaudio"),
        platforms=("win32",),
        approx_gb=4.0,
        notes=(
            "Requires a current Adrenalin driver. The first generation compiles GPU "
            "kernels and can take 10-15 minutes; later runs are fast. torchao is not "
            "compatible with this build and is removed if present."
        ),
        incompatible=("torchao",),
    ),
    RuntimeOption(
        id="rocm-windows-pinned",
        name="AMD GPU (ROCm 7.2.1, pinned)",
        description="The build most projects pin today. Python 3.12 only.",
        index_url="https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/",
        packages=("torch", "torchaudio"),
        platforms=("win32",),
        python_requires="3.12",
        approx_gb=4.0,
        notes="Better field-tested than the newer channel, but no longer moving.",
        incompatible=("torchao",),
    ),
    RuntimeOption(
        id="cuda",
        name="NVIDIA GPU (CUDA)",
        description="GPU acceleration for GeForce and RTX cards.",
        index_url="https://download.pytorch.org/whl/cu126",
        approx_gb=3.5,
    ),
    RuntimeOption(
        id="rocm",
        name="AMD GPU (ROCm for Linux)",
        description="GPU acceleration for Radeon cards on Linux.",
        index_url="https://download.pytorch.org/whl/rocm6.2",
        platforms=("linux",),
        approx_gb=4.0,
    ),
    RuntimeOption(
        id="cpu",
        name="CPU only",
        description="Works everywhere. Slow for audio models, fine for MIDI models.",
        index_url="https://download.pytorch.org/whl/cpu",
        approx_gb=0.5,
    ),
]


@dataclass
class InstallHandle:
    option: RuntimeOption
    thread: threading.Thread | None = None
    lines: list[str] = field(default_factory=list)
    error: str = ""
    done: bool = False
    returncode: int | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    _process: subprocess.Popen | None = None

    def stop(self) -> None:
        self._cancel.set()
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.terminate()
            except OSError:
                pass


def options_for(vendor: str) -> list[RuntimeOption]:
    """Runtime options relevant to a GPU vendor, best first."""
    preferred = {
        "nvidia": ["cuda", "cpu"],
        "amd": ["rocm-windows", "rocm-windows-pinned", "rocm", "cpu"],
        "intel": ["cpu"],
        "apple": ["cpu"],
    }.get(vendor, ["cpu"])
    by_id = {o.id: o for o in RUNTIME_OPTIONS}
    out = [by_id[i] for i in preferred if i in by_id and by_id[i].available_here()]
    if not out:
        out = [by_id["cpu"]]
    return out


def install_runtime(
    option: RuntimeOption,
    on_line: Callable[[str], None] | None = None,
    on_finished: Callable[[InstallHandle], None] | None = None,
    python_executable: str | None = None,
) -> InstallHandle:
    """Install a torch build in the background, streaming pip output."""
    handle = InstallHandle(option=option)
    executable = python_executable or sys.executable

    def emit(line: str) -> None:
        handle.lines.append(line)
        if on_line is not None:
            try:
                on_line(line)
            except Exception:
                log.exception("install output callback failed")

    def run() -> None:
        try:
            # Some builds fail at import time when an incompatible package is
            # installed alongside them, so clear those first.
            for package in option.incompatible:
                emit(f"Removing incompatible package: {package}")
                subprocess.run(
                    [executable, "-m", "pip", "uninstall", "-y", package],
                    capture_output=True, text=True, timeout=300,
                )

            cmd = [executable, "-m", "pip", "install", "--upgrade", *option.packages]
            if option.index_url:
                cmd += ["--index-url", option.index_url]
            cmd += list(option.extra_args)
            emit("$ " + " ".join(cmd))

            handle._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for line in handle._process.stdout or []:
                if handle._cancel.is_set():
                    break
                emit(line.rstrip())
            handle.returncode = handle._process.wait()
            if handle._cancel.is_set():
                handle.error = "Cancelled"
            elif handle.returncode != 0:
                handle.error = f"pip exited with code {handle.returncode}"
        except Exception as exc:
            handle.error = f"{type(exc).__name__}: {exc}"
            log.exception("runtime install failed")
        finally:
            handle.done = True
            if on_finished is not None:
                try:
                    on_finished(handle)
                except Exception:
                    log.exception("install finished callback failed")

    handle.thread = threading.Thread(target=run, name="runtime-install", daemon=True)
    handle.thread.start()
    return handle
