"""Pitch, interval and scale primitives.

Everything downstream speaks MIDI note numbers (C4 = 60).  Pitch classes are
integers 0-11 with 0 = C.  Scales are stored as ascending semitone offsets from
the tonic, which keeps degree lookup and quantisation to trivial arithmetic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "FLAT_NAMES",
    "MODE_ALIASES",
    "NOTE_NAMES",
    "SCALES",
    "Scale",
    "midi_to_freq",
    "note_name",
    "parse_key",
    "parse_note",
    "prefers_flats",
    "transpose_into_range",
]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_NAME_TO_PC = {
    "C": 0, "B#": 0, "D--": 0,
    "C#": 1, "Db": 1,
    "D": 2, "C##": 2, "Ebb": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "D##": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7, "F##": 7,
    "G#": 8, "Ab": 8,
    "A": 9, "G##": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Ascending semitone offsets from the tonic.
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "octatonic": (0, 2, 3, 5, 6, 8, 9, 11),
    "chromatic": tuple(range(12)),
    "hirajoshi": (0, 2, 3, 7, 8),
    "in_sen": (0, 1, 5, 7, 10),
    "iwato": (0, 1, 5, 6, 10),
    "double_harmonic": (0, 1, 4, 5, 7, 8, 11),
    "lydian_dominant": (0, 2, 4, 6, 7, 9, 10),
    "altered": (0, 1, 3, 4, 6, 8, 10),
}

MODE_ALIASES = {
    "ionian": "major",
    "aeolian": "minor",
    "natural_minor": "minor",
    "nat_minor": "minor",
    "maj": "major",
    "min": "minor",
    "m": "minor",
    "pentatonic": "major_pentatonic",
    "pent": "major_pentatonic",
    "minor_pent": "minor_pentatonic",
    "spanish": "phrygian_dominant",
    "byzantine": "double_harmonic",
}


def canonical_scale_name(name: str) -> str:
    """Normalise a user-supplied scale/mode name to a key of ``SCALES``."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = MODE_ALIASES.get(key, key)
    return key if key in SCALES else "major"


@dataclass(frozen=True)
class Scale:
    """A tonic plus an interval set, with helpers for degree access."""

    tonic: int  # pitch class 0-11
    name: str = "major"

    @property
    def intervals(self) -> tuple[int, ...]:
        return SCALES[canonical_scale_name(self.name)]

    @property
    def size(self) -> int:
        return len(self.intervals)

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple(sorted((self.tonic + i) % 12 for i in self.intervals))

    def is_minor_ish(self) -> bool:
        """True when the third above the tonic is minor."""
        return 3 in self.intervals and 4 not in self.intervals

    def degree_to_pitch(self, degree: int, octave: int = 4) -> int:
        """Map a zero-based scale degree to a MIDI pitch.

        Degrees outside ``0..size-1`` wrap into neighbouring octaves, so degree
        ``-1`` is the leading tone below the tonic and ``size`` is the octave.
        """
        n = self.size
        octave_shift, idx = divmod(degree, n)
        semitone = self.intervals[idx]
        return 12 * (octave + 1) + self.tonic + semitone + 12 * octave_shift

    def contains(self, pitch: int) -> bool:
        return pitch % 12 in self.pitch_classes

    def quantize(self, pitch: int, prefer: int = 1) -> int:
        """Snap ``pitch`` to the nearest scale tone.

        ``prefer`` breaks ties: 1 rounds up, -1 rounds down.
        """
        if self.contains(pitch):
            return pitch
        for dist in range(1, 7):
            up, down = pitch + dist, pitch - dist
            cands = (up, down) if prefer >= 0 else (down, up)
            for c in cands:
                if self.contains(c):
                    return c
        return pitch

    def nearest_degree(self, pitch: int) -> int:
        """Index of the scale degree closest to ``pitch`` (ignoring octave)."""
        pc = (pitch - self.tonic) % 12
        best, best_dist = 0, 99
        for i, iv in enumerate(self.intervals):
            d = min((pc - iv) % 12, (iv - pc) % 12)
            if d < best_dist:
                best, best_dist = i, d
        return best

    def __str__(self) -> str:
        table = FLAT_NAMES if prefers_flats(self) else NOTE_NAMES
        return f"{table[self.tonic]} {canonical_scale_name(self.name).replace('_', ' ')}"


def note_name(pitch: int, flats: bool = False) -> str:
    """Render a MIDI pitch as e.g. ``C4`` / ``Eb3``."""
    table = FLAT_NAMES if flats else NOTE_NAMES
    return f"{table[pitch % 12]}{pitch // 12 - 1}"


def parse_note(text: str, default_octave: int = 4) -> int | None:
    """Parse ``C#4`` / ``Bb`` / ``f2`` into a MIDI pitch."""
    m = re.fullmatch(r"\s*([A-Ga-g])([#b♯♭]{0,2})(-?\d+)?\s*", text)
    if not m:
        return None
    letter, accidental, octave = m.groups()
    accidental = accidental.replace("♯", "#").replace("♭", "b")
    pc = _NAME_TO_PC.get(letter.upper() + accidental)
    if pc is None:
        return None
    octv = int(octave) if octave is not None else default_octave
    return 12 * (octv + 1) + pc


def parse_key(text: str) -> tuple[int, str]:
    """Parse ``F# minor`` / ``Cmaj`` / ``Bb dorian`` into (pitch class, scale).

    Falls back to C major when the text cannot be understood.
    """
    t = text.strip()
    m = re.match(r"\s*([A-Ga-g][#b♯♭]{0,2})\s*(.*)", t)
    if not m:
        return 0, "major"
    root_txt, rest = m.groups()
    root_txt = root_txt.replace("♯", "#").replace("♭", "b")
    pc = _NAME_TO_PC.get(root_txt[0].upper() + root_txt[1:])
    if pc is None:
        return 0, "major"
    rest = rest.strip().lower()
    if not rest:
        return pc, "major"
    return pc, canonical_scale_name(rest)


def midi_to_freq(pitch: int, a4: float = 440.0) -> float:
    return a4 * (2.0 ** ((pitch - 69) / 12.0))


def transpose_into_range(pitch: int, low: int, high: int) -> int:
    """Octave-shift ``pitch`` until it sits inside ``[low, high]``."""
    if low > high:
        low, high = high, low
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return max(low, min(high, pitch))


# Keys whose signatures use flats, as pitch classes of the tonic.
_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1, 6}   # F Bb Eb Ab Db Gb
_FLAT_MINOR_TONICS = {2, 7, 0, 5, 10, 3}   # Dm Gm Cm Fm Bbm Ebm


def prefers_flats(scale: Scale) -> bool:
    """Whether chords in ``scale`` should be spelled with flats."""
    tonics = _FLAT_MINOR_TONICS if scale.is_minor_ish() else _FLAT_MAJOR_TONICS
    return scale.tonic in tonics
