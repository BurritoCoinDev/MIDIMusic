"""Creating the compute runtime.

A packaged application cannot install its own Python packages: ``sys.executable``
is ``MIDIMusic.exe``, not an interpreter, so ``sys.executable -m pip`` would
just relaunch the GUI.  Something has to build a real Python environment first.

That something is ``uv`` -- a single static binary that can download a CPython
of the version we ask for and create a virtual environment from it, with no
Python on the machine beforehand.  It is bundled with the application, so the
app remains self-contained: everything needed to *bootstrap* is present, and
only the multi-gigabyte GPU wheels are fetched on demand.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import sys
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config.paths import get_paths

__all__ = [
    "RUNTIME_PYTHON_VERSION",
    "BootstrapHandle",
    "ensure_runtime",
    "find_uv",
    "runtime_exists",
]

log = logging.getLogger(__name__)

# ACE-Step requires <3.13, and AMD's Windows ROCm wheels cover 3.10-3.14.
# 3.12 is the version every backend agrees on.
RUNTIME_PYTHON_VERSION = "3.12"

_UV_RELEASE = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-pc-windows-msvc.zip"
)


def _exe(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def find_uv() -> Path | None:
    """Locate the uv binary: bundled first, then installed, then on PATH."""
    # 1. Shipped alongside the frozen application.
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidate = Path(bundle) / "uv" / _exe("uv")
        if candidate.exists():
            return candidate

    # 2. Downloaded by us on a previous run.
    cached = get_paths().data / "uv" / _exe("uv")
    if cached.exists():
        return cached

    # 3. Installed as a Python package in this environment (development).
    try:
        import uv as uv_package

        candidate = Path(uv_package.find_uv_bin())
        if candidate.exists():
            return candidate
    except Exception:
        pass

    # 4. Already on the user's PATH.
    found = shutil.which("uv")
    return Path(found) if found else None


def download_uv(on_progress: Callable[[float, str], None] | None = None) -> Path | None:
    """Fetch uv as a last resort, when it was not bundled.

    Only reached if the application was built without it; a normal install has
    uv already and never touches the network here.
    """
    if sys.platform != "win32":
        return None
    target_dir = get_paths().data / "uv"
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / "uv.zip"

    if on_progress:
        on_progress(0.0, "Downloading the environment builder")
    try:
        request = urllib.request.Request(_UV_RELEASE, headers={"User-Agent": "MIDIMusic"})
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            with open(archive, "wb") as fh:
                while True:
                    chunk = response.read(262144)
                    if not chunk:
                        break
                    fh.write(chunk)
                    read += len(chunk)
                    if total and on_progress:
                        on_progress(read / total, f"{read / 1e6:.0f} of {total / 1e6:.0f} MB")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
        archive.unlink(missing_ok=True)
    except Exception:
        log.exception("could not download uv")
        return None

    binary = target_dir / _exe("uv")
    if not binary.exists():
        for found in target_dir.rglob(_exe("uv")):
            binary = found
            break
    if binary.exists() and sys.platform != "win32":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary if binary.exists() else None


def runtime_python_path(runtime_dir: Path | None = None) -> Path:
    root = runtime_dir or get_paths().runtime
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def runtime_exists(runtime_dir: Path | None = None) -> bool:
    return runtime_python_path(runtime_dir).exists()


@dataclass
class BootstrapHandle:
    thread: threading.Thread | None = None
    python: Path | None = None
    error: str = ""
    done: bool = False
    lines: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.lines is None:
            self.lines = []

    def wait(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)


def create_runtime(
    runtime_dir: Path | None = None,
    python_version: str = RUNTIME_PYTHON_VERSION,
    on_line: Callable[[str], None] | None = None,
) -> Path:
    """Build the runtime environment synchronously. Returns its interpreter.

    Raises RuntimeError with something the user can act on rather than a
    subprocess error they cannot.
    """
    root = runtime_dir or get_paths().runtime
    existing = runtime_python_path(root)
    if existing.exists():
        return existing

    def emit(text: str) -> None:
        log.info("%s", text)
        if on_line is not None:
            on_line(text)

    uv = find_uv()
    if uv is None:
        emit("Environment builder not found; downloading it")
        uv = download_uv()
    if uv is None:
        raise RuntimeError(
            "Could not find or download the environment builder (uv). "
            "Check your network connection, or install Python 3.12 manually and "
            "point the app at it in Settings."
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    emit(f"Creating a Python {python_version} environment at {root}")
    result = subprocess.run(
        [str(uv), "venv", "--python", python_version, str(root)],
        capture_output=True, text=True, timeout=900,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for line in (result.stderr or result.stdout or "").splitlines():
        emit(line)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not create the Python {python_version} environment. "
            f"{(result.stderr or '').strip()[-400:]}"
        )

    python = runtime_python_path(root)
    if not python.exists():
        raise RuntimeError(f"The environment was created but {python} is missing.")
    emit("Environment ready")
    return python


def ensure_runtime(
    runtime_dir: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    on_finished: Callable[[BootstrapHandle], None] | None = None,
) -> BootstrapHandle:
    """Create the runtime in the background if it does not exist yet."""
    handle = BootstrapHandle()

    def run() -> None:
        try:
            handle.python = create_runtime(
                runtime_dir,
                on_line=lambda line: (handle.lines.append(line),
                                      on_line(line) if on_line else None),
            )
        except Exception as exc:
            handle.error = str(exc)
        finally:
            handle.done = True
            if on_finished is not None:
                try:
                    on_finished(handle)
                except Exception:
                    log.exception("bootstrap finished callback failed")

    handle.thread = threading.Thread(target=run, name="runtime-bootstrap", daemon=True)
    handle.thread.start()
    return handle


def uv_pip_install(
    packages: list[str],
    python: Path,
    index_url: str = "",
    extra_index_url: str = "https://pypi.org/simple",
    on_line: Callable[[str], None] | None = None,
    timeout: float = 3600,
) -> subprocess.Popen | None:
    """Start a package install into ``python`` using uv, streaming output.

    uv is used rather than pip because it is already present (it built the
    environment) and resolves multi-gigabyte GPU wheel sets considerably faster.
    """
    uv = find_uv()
    if uv is None:
        return None
    cmd = [str(uv), "pip", "install", "--python", str(python), *packages]
    if index_url:
        cmd += ["--index-url", index_url]
        # Must be an *extra* index: a second --index-url would let the resolver
        # prefer PyPI and silently install a CUDA build over the vendor one.
        if extra_index_url:
            cmd += ["--extra-index-url", extra_index_url]
    if on_line is not None:
        on_line("$ " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("UV_NO_PROGRESS", "1")
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
