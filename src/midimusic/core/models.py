"""Core data model shared by every generator and exporter.

Generators produce a :class:`Song` (symbolic) or an :class:`AudioBuffer`
(waveform), or both.  Everything downstream — export, playback, the library —
speaks these types, so adding a backend never changes the rest of the app.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "AudioBuffer",
    "GenerationRequest",
    "GenerationResult",
    "JobStatus",
    "Note",
    "OutputFormat",
    "Progress",
    "Song",
    "Track",
    "resolve_durations",
]


class OutputFormat(str, Enum):
    MIDI = "midi"
    FLAC = "flac"
    WAV = "wav"

    @property
    def extension(self) -> str:
        return {"midi": ".mid", "flac": ".flac", "wav": ".wav"}[self.value]


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Note:
    """A single note. Times are in beats from the start of the song."""

    pitch: int
    start: float
    duration: float
    velocity: int = 90
    channel: int = 0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Track:
    """One instrument's worth of notes."""

    name: str = "Track"
    program: int = 0  # General MIDI program number
    channel: int = 0
    is_drum: bool = False
    notes: list[Note] = field(default_factory=list)
    role: str = ""  # bass / chords / lead / pad / arp / drums / counter

    def add(self, pitch: int, start: float, duration: float, velocity: int = 90) -> None:
        self.notes.append(Note(pitch, start, duration, velocity, self.channel))

    @property
    def length_beats(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    def sort(self) -> None:
        self.notes.sort(key=lambda n: (n.start, n.pitch))


@dataclass
class Song:
    """A complete symbolic arrangement."""

    tracks: list[Track] = field(default_factory=list)
    tempo: float = 120.0
    beats_per_bar: int = 4
    beat_unit: int = 4
    key: str = "C major"
    title: str = "Untitled"
    sections: list[tuple[str, float, float]] = field(default_factory=list)  # name, start, end (beats)
    chords: list[tuple[str, float, float]] = field(default_factory=list)  # symbol, start, end
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length_beats(self) -> float:
        return max((t.length_beats for t in self.tracks), default=0.0)

    @property
    def duration_seconds(self) -> float:
        return self.length_beats * 60.0 / max(1e-6, self.tempo)

    @property
    def note_count(self) -> int:
        return sum(len(t.notes) for t in self.tracks)

    def non_empty_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.notes]


@dataclass
class AudioBuffer:
    """Interleaved or planar float audio plus its sample rate.

    ``samples`` is shaped ``(frames,)`` for mono or ``(frames, channels)``.
    Stored as a plain object so the model layer does not import numpy types
    into its signature, but in practice this is always an ndarray.
    """

    samples: Any
    sample_rate: int = 44100

    @property
    def channels(self) -> int:
        s = self.samples
        return 1 if getattr(s, "ndim", 1) == 1 else int(s.shape[1])

    @property
    def frames(self) -> int:
        return int(getattr(self.samples, "shape", (0,))[0])

    @property
    def duration_seconds(self) -> float:
        return self.frames / max(1, self.sample_rate)


@dataclass
class Progress:
    """A progress report from a running generator."""

    fraction: float = 0.0  # 0..1, negative means indeterminate
    message: str = ""
    stage: str = ""

    @property
    def percent(self) -> int:
        return max(0, min(100, int(round(self.fraction * 100))))


@dataclass
class GenerationRequest:
    """Everything a generator needs to make one piece of music."""

    prompt: str = ""
    backend: str = "builtin"
    output_format: OutputFormat = OutputFormat.MIDI

    # Musical intent. None means "let the style or the prompt decide".
    style: str | None = None
    key: str | None = None
    tempo: float | None = None
    structure: str | None = None

    # Length. ``duration_seconds`` is the resolved target every generator
    # reads; the range is what the user asked for, and the service draws each
    # variation's length from it before the job runs. Leaving the range unset
    # means "exactly duration_seconds".
    duration_seconds: float | None = 60.0
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    instrumental: bool = True
    lyrics: str = ""

    # Reproducibility and sampling.
    seed: int | None = None
    variations: int = 1
    temperature: float = 1.0
    guidance: float = 3.0
    negative_prompt: str = ""

    # Runtime.
    device: str = "auto"
    model_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_format"] = self.output_format.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationRequest:
        d = dict(d)
        fmt = d.pop("output_format", "midi")
        req = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        req.output_format = OutputFormat(fmt)
        return req


@dataclass
class GenerationResult:
    """What a generator hands back."""

    request_id: str = ""
    song: Song | None = None
    audio: AudioBuffer | None = None
    paths: list[Path] = field(default_factory=list)
    title: str = "Untitled"
    backend: str = ""
    seed: int | None = None
    duration_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and (self.song is not None or self.audio is not None or bool(self.paths))


def resolve_durations(
    request: GenerationRequest, count: int = 1, cap: float | None = None
) -> list[float | None]:
    """Pick a concrete length for each variation of ``request``.

    Someone who asks for tracks between 90 and 180 seconds means each one to
    land somewhere in that band -- so the lengths are spread across the range
    rather than drawn independently. Independent draws cluster, and two
    variations that come out the same length waste the point of asking for a
    range at all.

    ``cap`` is the backend's ceiling. It clamps the range rather than
    overriding it, so a model that can only manage 30 seconds quietly produces
    30 instead of failing.
    """
    count = max(1, int(count))
    low, high = request.min_duration_seconds, request.max_duration_seconds

    if low is None and high is None:
        target = request.duration_seconds
        if target is not None and cap:
            target = min(float(target), float(cap))
        return [target] * count

    fallback = request.duration_seconds
    if low is None:
        low = fallback if fallback is not None else high
    if high is None:
        high = fallback if fallback is not None else low
    low, high = float(low), float(high)
    if low > high:
        low, high = high, low
    if cap:
        low, high = min(low, float(cap)), min(high, float(cap))

    span = high - low
    if span < 1.0:
        return [round(high, 1)] * count

    rng = random.Random(request.seed) if request.seed is not None else random.Random()
    out: list[float | None] = []
    for i in range(count):
        # One slice of the range each, jittered within that slice: the lengths
        # stay distinct and none of them can leave the band.
        slice_width = span / count
        centre = low + slice_width * (i + 0.5)
        jitter = slice_width * 0.5 * (rng.random() * 2.0 - 1.0)
        out.append(round(min(high, max(low, centre + jitter)), 1))
    return out
