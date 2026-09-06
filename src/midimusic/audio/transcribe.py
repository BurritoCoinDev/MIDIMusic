"""Audio to MIDI transcription.

Waveform models have no notes inside them, so getting a score out of one means
estimating pitches from the audio.  This is genuinely lossy -- it works well on
sparse or monophonic material and degrades on dense mixes -- so the result is
offered as a starting point, not as the model's "real" output.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..core.models import AudioBuffer, Song
from .midi_io import midi_to_song

__all__ = ["is_available", "transcribe_audio", "transcribe_file"]

log = logging.getLogger(__name__)


def is_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("basic_pitch") is not None


def transcribe_audio(
    buffer: AudioBuffer,
    progress: Callable[[float], None] | None = None,
    title: str = "Transcription",
) -> Song | None:
    """Transcribe an in-memory buffer by way of a temporary WAV file."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "input.wav"
        samples = np.asarray(buffer.samples, dtype=np.float32)
        sf.write(str(wav), samples, buffer.sample_rate, subtype="PCM_16")
        return transcribe_file(wav, progress=progress, title=title)


def transcribe_file(
    path: str | Path,
    progress: Callable[[float], None] | None = None,
    title: str = "Transcription",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
) -> Song | None:
    """Transcribe an audio file to a Song, or return None if unavailable."""
    if not is_available():
        log.info("basic-pitch is not installed; skipping transcription")
        return None

    if progress:
        progress(0.1)
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
    except ImportError:
        log.info("basic-pitch import failed; skipping transcription")
        return None

    try:
        _model_output, midi_data, _notes = predict(
            str(path),
            ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
        )
    except Exception:
        log.exception("transcription failed")
        return None

    if progress:
        progress(0.8)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "transcribed.mid"
        midi_data.write(str(out))
        song = midi_to_song(out, title=title)

    song.meta["transcribed"] = True
    song.meta["source"] = str(path)
    if progress:
        progress(1.0)
    return song
