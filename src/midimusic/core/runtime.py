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
    # AMD's wheel-variant channel selects the GPU build through an extra, so
    # the package name depends on the detected architecture.
    needs_gfx_extra: bool = False

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
        # A real PEP 503 index carrying win_amd64 wheels for cp310-cp314, plus
        # the amd-torch-device-gfxNNNN packages the extras resolve to.
        index_url="https://repo.amd.com/rocm/whl-multi-arch/",
        packages=("torch", "torchaudio"),
        platforms=("win32",),
        approx_gb=4.0,
        notes=(
            "Requires a current Adrenalin driver. The first generation compiles GPU "
            "kernels and can take 10-15 minutes; later runs are fast. torchao is "
            "removed if present, because it registers distributed operators these "
            "builds do not have and then fails at import."
        ),
        incompatible=("torchao",),
        needs_gfx_extra=True,
    ),
    RuntimeOption(
        id="cuda",
        name="NVIDIA GPU (CUDA)",
        description="GPU acceleration for GeForce and RTX cards.",
        index_url="https://download.pytorch.org/whl/cu130",
        approx_gb=3.5,
    ),
    RuntimeOption(
        id="rocm-linux",
        name="AMD GPU (ROCm for Linux)",
        description="GPU acceleration for Radeon cards on Linux.",
        index_url="https://download.pytorch.org/whl/rocm6.4",
        platforms=("linux",),
        approx_gb=4.0,
    ),
    RuntimeOption(
        id="xpu",
        name="Intel GPU (XPU)",
        description="GPU acceleration for Intel Arc.",
        index_url="https://download.pytorch.org/whl/xpu",
        platforms=("win32", "linux"),
        approx_gb=3.0,
    ),
    RuntimeOption(
        id="cpu",
        name="CPU only",
        description="Works everywhere. Slow for audio models, fine for MIDI models.",
        index_url="https://download.pytorch.org/whl/cpu",
        approx_gb=1.0,
    ),
]

# The application libraries every runtime needs on top of torch.
BASE_PACKAGES = ("transformers", "soundfile", "numpy")


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
        "amd": ["rocm-windows", "rocm-linux", "cpu"],
        "intel": ["xpu", "cpu"],
        "apple": ["cpu"],
    }.get(vendor, ["cpu"])
    by_id = {o.id: o for o in RUNTIME_OPTIONS}
    out = [by_id[i] for i in preferred if i in by_id and by_id[i].available_here()]
    if not out:
        out = [by_id["cpu"]]
    return out


def resolve_packages(option: RuntimeOption, gfx_arch: str = "") -> tuple[str, ...]:
    """The exact package specifiers to install for this machine.

    On AMD's wheel-variant channel the GPU build is selected by an extra, so a
    gfx1101 card must ask for torch[device-gfx1101]; sending every AMD user the
    gfx1100 build would install the wrong kernels.
    """
    packages = list(option.packages)
    if option.needs_gfx_extra and gfx_arch:
        packages = [
            f"torch[device-{gfx_arch}]" if p == "torch" else p for p in packages
        ]
    return tuple(packages)


def install_runtime(
    option: RuntimeOption,
    on_line: Callable[[str], None] | None = None,
    on_finished: Callable[[InstallHandle], None] | None = None,
    python_executable: str | None = None,
    gfx_arch: str = "",
    extra_packages: tuple[str, ...] = BASE_PACKAGES,
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

            packages = resolve_packages(option, gfx_arch)
            cmd = [executable, "-m", "pip", "install", "--upgrade", *packages]
            if option.index_url:
                cmd += ["--index-url", option.index_url]
                # Everything not on the vendor index still has to resolve.
                # Note this must be --extra-index-url, not a second
                # --index-url: pip would otherwise prefer PyPI and silently
                # install a CUDA build over the one we asked for.
                cmd += ["--extra-index-url", "https://pypi.org/simple"]
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

            # Application libraries come from PyPI, after torch, so the vendor
            # index never has to satisfy them.
            if handle.returncode == 0 and extra_packages and not handle._cancel.is_set():
                emit("\nInstalling model libraries")
                follow_up = subprocess.run(
                    [executable, "-m", "pip", "install", "--upgrade", *extra_packages],
                    capture_output=True, text=True, timeout=1800,
                )
                for line in (follow_up.stdout or "").splitlines()[-20:]:
                    emit(line)
                if follow_up.returncode != 0:
                    handle.returncode = follow_up.returncode
                    emit((follow_up.stderr or "")[-2000:])
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
