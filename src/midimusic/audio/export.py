"""Writing finished audio to disk, with tags.

FLAC and WAV are written through python-soundfile, whose Windows wheels bundle
libsndfile -- so there is no ffmpeg dependency and nothing to install
separately.  Vorbis comments are added with mutagen afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..core.models import AudioBuffer, OutputFormat, Song
from . import dsp

__all__ = ["ExportOptions", "TrackMetadata", "export_audio", "export_song", "safe_filename"]

_INVALID = '<>:"/\\|?*'


def safe_filename(name: str, fallback: str = "untitled", max_length: int = 120) -> str:
    """Make a string safe as a Windows filename.

    Windows forbids a set of characters, trailing dots and spaces, and a list
    of reserved device names -- all of which produce confusing failures rather
    than clean errors if ignored.
    """
    cleaned = "".join("_" if c in _INVALID else c for c in (name or ""))
    cleaned = "".join(c for c in cleaned if ord(c) >= 32).strip().rstrip(". ")
    cleaned = cleaned[:max_length].strip()
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if not cleaned or cleaned.upper() in reserved:
        return fallback
    return cleaned


@dataclass
class TrackMetadata:
    title: str = "Untitled"
    artist: str = "MIDIMusic"
    album: str = "MIDIMusic Generations"
    genre: str = ""
    comment: str = ""
    date: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class ExportOptions:
    sample_rate: int = 44100
    bit_depth: int = 24          # 16 or 24 for FLAC/WAV
    target_lufs: float | None = -14.0
    peak_ceiling_db: float = -1.0
    fade_in: float = 0.005
    fade_out: float = 0.35
    trim: bool = True
    tag: bool = True


_SUBTYPES = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


def export_audio(
    buffer: AudioBuffer,
    path: str | Path,
    fmt: OutputFormat = OutputFormat.FLAC,
    options: ExportOptions | None = None,
    metadata: TrackMetadata | None = None,
) -> Path:
    """Master and write ``buffer`` to ``path``. Returns the written path."""
    options = options or ExportOptions()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = dsp.to_stereo(np.asarray(buffer.samples, dtype=np.float32))
    if options.trim:
        x = dsp.trim_silence(x, sample_rate=buffer.sample_rate)
    if x.size == 0:
        x = np.zeros((max(1, buffer.sample_rate // 10), 2), dtype=np.float32)

    if buffer.sample_rate != options.sample_rate:
        x = dsp.resample(x, buffer.sample_rate, options.sample_rate)

    x = dsp.apply_fades(x, options.sample_rate, options.fade_in, options.fade_out)
    if options.target_lufs is not None:
        x = dsp.loudness_normalize(
            x, options.sample_rate, options.target_lufs, options.peak_ceiling_db
        )
    else:
        x = dsp.peak_normalize(x, options.peak_ceiling_db)

    depth = options.bit_depth if options.bit_depth in _SUBTYPES else 24
    subtype = _SUBTYPES[depth]
    container = "FLAC" if fmt is OutputFormat.FLAC else "WAV"
    if container == "FLAC" and depth == 32:
        # FLAC is integer-only; 32-bit float silently is not a thing.
        subtype = "PCM_24"

    sf.write(str(path), x, options.sample_rate, subtype=subtype, format=container)

    if options.tag and metadata is not None:
        _write_tags(path, fmt, metadata)
    return path


def _write_tags(path: Path, fmt: OutputFormat, meta: TrackMetadata) -> None:
    try:
        if fmt is OutputFormat.FLAC:
            from mutagen.flac import FLAC

            audio = FLAC(str(path))
        else:
            from mutagen.wave import WAVE

            audio = WAVE(str(path))
            audio.add_tags() if audio.tags is None else None
    except Exception:
        return

    try:
        fields = {
            "title": meta.title,
            "artist": meta.artist,
            "album": meta.album,
            "genre": meta.genre,
            "date": meta.date,
            "comment": meta.comment,
        }
        for key, value in fields.items():
            if value:
                audio[key] = value
        for key, value in meta.extra.items():
            if value:
                audio[key] = str(value)
        audio.save()
    except Exception:
        # Tagging is a nicety; never fail an export because of it.
        return


def export_song(
    song: Song,
    path: str | Path,
    fmt: OutputFormat = OutputFormat.MIDI,
    soundfont: str | Path | None = None,
    options: ExportOptions | None = None,
    metadata: TrackMetadata | None = None,
    progress=None,
) -> Path:
    """Export a symbolic song as MIDI, or render it and export as audio."""
    from .midi_io import write_midi

    path = Path(path)
    if fmt is OutputFormat.MIDI:
        return write_midi(song, path)

    if soundfont is None:
        from .synth_fallback import render_song_fallback

        buf = render_song_fallback(song, progress=progress)
    else:
        from .render import render_song

        buf = render_song(song, soundfont, progress=progress)

    meta = metadata or TrackMetadata(
        title=song.title,
        genre=str(song.meta.get("style", "")),
        comment=str(song.meta.get("prompt", "")),
    )
    return export_audio(buf, path, fmt, options, meta)
