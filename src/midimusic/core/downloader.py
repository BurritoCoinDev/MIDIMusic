"""Model downloads.

Wraps huggingface_hub so the UI gets progress, cancellation and a resumable
transfer without knowing anything about the hub.  Downloads land in the app's
models directory so a user can move a 50 GB collection to another drive by
changing one setting.
"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config.paths import disk_free, human_bytes
from .catalog import ModelEntry

__all__ = ["DownloadHandle", "download_model", "local_size", "model_is_present", "remove_model"]

log = logging.getLogger(__name__)


@dataclass
class DownloadHandle:
    """A running download that can be watched and cancelled."""

    entry: ModelEntry
    thread: threading.Thread | None = None
    cancel: threading.Event = None  # type: ignore[assignment]
    error: str = ""
    path: Path | None = None
    done: bool = False

    def __post_init__(self) -> None:
        if self.cancel is None:
            self.cancel = threading.Event()

    def stop(self) -> None:
        self.cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)


def _repo_dir(models_dir: Path, entry: ModelEntry) -> Path:
    safe = entry.repo.replace("/", "--") or entry.id
    return models_dir / "hub" / f"models--{safe}"


def model_is_present(entry: ModelEntry, models_dir: Path) -> bool:
    if not entry.needs_download:
        return True
    directory = _repo_dir(models_dir, entry)
    if not directory.exists():
        return False
    # A partial download leaves the blobs directory without any snapshot.
    snapshots = directory / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def local_size(entry: ModelEntry, models_dir: Path) -> int:
    directory = _repo_dir(models_dir, entry)
    if not directory.exists():
        return 0
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def remove_model(entry: ModelEntry, models_dir: Path) -> bool:
    directory = _repo_dir(models_dir, entry)
    if not directory.exists():
        return False
    try:
        shutil.rmtree(directory)
        return True
    except OSError:
        log.exception("could not remove %s", directory)
        return False


def download_model(
    entry: ModelEntry,
    models_dir: Path,
    token: str = "",
    on_progress: Callable[[float, str], None] | None = None,
    on_finished: Callable[[DownloadHandle], None] | None = None,
) -> DownloadHandle:
    """Start a background download. Returns immediately."""
    handle = DownloadHandle(entry=entry)

    def report(fraction: float, message: str) -> None:
        if on_progress is not None:
            try:
                on_progress(fraction, message)
            except Exception:
                log.exception("download progress callback failed")

    def run() -> None:
        try:
            needed = int(entry.size_gb * 1.15 * 1024 ** 3)
            free = disk_free(models_dir)
            if needed and free and free < needed:
                raise OSError(
                    f"Not enough disk space: {human_bytes(needed)} needed, "
                    f"{human_bytes(free)} free on that drive."
                )

            from huggingface_hub import snapshot_download

            report(0.02, f"Contacting Hugging Face for {entry.repo}")

            # Weights only: skip the extra formats a repo often carries so a
            # download is not two or three times larger than it needs to be.
            ignore = ["*.msgpack", "*.h5", "*.onnx_data" if entry.adapter != "onnx-midi" else ""]
            ignore = [p for p in ignore if p]

            class _Callback:
                """Adapts hub progress to a simple fraction."""

                def __init__(self) -> None:
                    self.total = 0
                    self.seen = 0

                def __call__(self, *_args, **_kwargs) -> None:
                    if handle.cancel.is_set():
                        raise KeyboardInterrupt

            path = snapshot_download(
                repo_id=entry.repo,
                revision=entry.revision or "main",
                cache_dir=str(models_dir / "hub"),
                token=token or None,
                ignore_patterns=ignore or None,
                max_workers=4,
                tqdm_class=_make_tqdm(report, handle),
            )
            handle.path = Path(path)
            report(1.0, "Download complete")
        except KeyboardInterrupt:
            handle.error = "Cancelled"
        except Exception as exc:
            handle.error = f"{type(exc).__name__}: {exc}"
            log.exception("download of %s failed", entry.repo)
        finally:
            handle.done = True
            if on_finished is not None:
                try:
                    on_finished(handle)
                except Exception:
                    log.exception("download finished callback failed")

    handle.thread = threading.Thread(target=run, name=f"download-{entry.id}", daemon=True)
    handle.thread.start()
    return handle


def _make_tqdm(report: Callable[[float, str], None], handle: DownloadHandle):
    """Build a tqdm subclass that reports progress and honours cancellation."""
    try:
        from tqdm.auto import tqdm as base
    except ImportError:  # pragma: no cover - tqdm ships with huggingface_hub
        return None

    class _Tqdm(base):  # type: ignore[misc, valid-type]
        def update(self, n=1):
            result = super().update(n)
            if handle.cancel.is_set():
                raise KeyboardInterrupt
            total = getattr(self, "total", 0) or 0
            current = getattr(self, "n", 0) or 0
            if total:
                report(min(0.99, current / total),
                       f"{human_bytes(current)} of {human_bytes(total)}")
            return result

    return _Tqdm
