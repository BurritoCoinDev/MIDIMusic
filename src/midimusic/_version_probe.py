"""Small capability checks the UI needs without importing heavy modules."""

from __future__ import annotations

import importlib.util

__all__ = ["transcription_available"]


def transcription_available() -> bool:
    """Whether audio-to-MIDI transcription can run in this process."""
    return importlib.util.find_spec("basic_pitch") is not None
