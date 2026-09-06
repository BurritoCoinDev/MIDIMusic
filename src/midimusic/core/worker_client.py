"""Driving the out-of-process generation worker.

Finds the provisioned runtime interpreter, starts the worker in it, and
translates the newline-delimited JSON protocol into progress callbacks and a
finished audio file.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config.paths import get_paths

__all__ = [
    "WorkerCancelled",
    "WorkerError",
    "probe_runtime",
    "run_worker",
    "runtime_python",
    "runtime_ready",
    "worker_script",
]

log = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    """The worker failed. Carries the remote traceback where there is one."""

    def __init__(self, message: str, remote_traceback: str = ""):
        super().__init__(message)
        self.remote_traceback = remote_traceback


class WorkerCancelled(Exception):
    pass


def worker_script() -> Path:
    """Path to runner.py on disk.

    The worker is executed by a *different* interpreter, so it has to exist as
    a real file. In a frozen build the package itself lives in an archive, so
    the spec ships runner.py as a data file and it is found under the bundle
    root instead of next to this module.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidate = Path(bundle) / "midimusic" / "worker" / "runner.py"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent.parent / "worker" / "runner.py"


def runtime_python(runtime_dir: Path | None = None) -> Path | None:
    """The interpreter of the provisioned runtime, if there is one.

    Falls back to the interpreter running the app, which is right during
    development (where torch is in the same environment) and correctly absent
    in a frozen build (where ``sys.executable`` is the application itself).
    """
    root = runtime_dir or get_paths().runtime
    candidates = [
        root / "Scripts" / "python.exe",   # Windows venv
        root / "bin" / "python",           # POSIX venv
        root / "python.exe",               # embeddable distribution
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not getattr(sys, "frozen", False):
        return Path(sys.executable)
    return None


def runtime_ready(runtime_dir: Path | None = None) -> bool:
    return runtime_python(runtime_dir) is not None


@dataclass
class WorkerResult:
    output_path: Path | None
    meta: dict


def probe_runtime(runtime_dir: Path | None = None, timeout: float = 60.0) -> dict:
    """Ask the runtime what it has installed. Returns {} when unavailable."""
    python = runtime_python(runtime_dir)
    if python is None:
        return {}
    try:
        result = run_worker(
            {"cmd": "probe"}, python=python, timeout=timeout,
        )
        return result.meta
    except Exception as exc:
        log.info("runtime probe failed: %s", exc)
        return {}


def run_worker(
    request: dict,
    python: Path | None = None,
    on_progress: Callable[[float, str, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> WorkerResult:
    """Run one request in the worker and return its result.

    Blocks until the worker answers. Cancellation terminates the process, which
    is the only reliable way to stop a native library mid-computation.
    """
    interpreter = python or runtime_python()
    if interpreter is None:
        raise WorkerError(
            "No compute runtime is installed. Install one from the Models tab."
        )
    script = worker_script()
    if not script.exists():
        raise WorkerError(f"Worker script is missing: {script}")

    child_env = os.environ.copy()
    # Unbuffered, so progress lines arrive as they are produced rather than in
    # a block when the pipe flushes.
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if env:
        child_env.update({str(k): str(v) for k, v in env.items()})

    process = subprocess.Popen(
        [str(interpreter), str(script)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=child_env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    stderr_lines: list[str] = []

    def drain_stderr() -> None:
        for line in process.stderr or []:
            stderr_lines.append(line.rstrip())
            if len(stderr_lines) > 200:
                del stderr_lines[:100]

    threading.Thread(target=drain_stderr, daemon=True).start()

    cancel_timer: threading.Timer | None = None
    if timeout:
        cancel_timer = threading.Timer(timeout, process.kill)
        cancel_timer.start()

    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        for raw in process.stdout or []:
            raw = raw.strip()
            if not raw:
                continue
            if should_cancel is not None and should_cancel():
                process.kill()
                raise WorkerCancelled()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                # A library printing to stdout is not fatal; ignore the noise.
                log.debug("non-JSON from worker: %s", raw[:200])
                continue

            event = message.get("event")
            if event == "progress":
                if on_progress is not None:
                    on_progress(
                        float(message.get("fraction", 0.0)),
                        str(message.get("message", "")),
                        str(message.get("stage", "")),
                    )
            elif event == "result":
                path = message.get("output_path") or ""
                return WorkerResult(Path(path) if path else None,
                                    message.get("meta") or {})
            elif event == "cancelled":
                raise WorkerCancelled()
            elif event == "error":
                raise WorkerError(
                    str(message.get("message", "worker failed")),
                    str(message.get("traceback", "")),
                )

        # The pipe closed with no result: the worker died.
        code = process.poll()
        tail = "\n".join(stderr_lines[-15:])
        raise WorkerError(
            f"The worker exited unexpectedly (code {code}). "
            f"{'Details: ' + tail if tail else ''}".strip()
        )
    finally:
        if cancel_timer is not None:
            cancel_timer.cancel()
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
