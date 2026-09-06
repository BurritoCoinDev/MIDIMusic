"""Logging configuration."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from .paths import get_paths

__all__ = ["setup_logging"]


def setup_logging(level: int = logging.INFO, to_file: bool = True) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    )

    # A packaged GUI app has no console attached, so stderr may be unusable.
    if sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        root.addHandler(stream)

    if to_file:
        try:
            logs = get_paths().logs
            logs.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                logs / "midimusic.log", maxBytes=2_000_000, backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(fmt)
            root.addHandler(handler)
        except OSError:
            pass

    for noisy in ("urllib3", "filelock", "huggingface_hub", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
