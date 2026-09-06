"""Chord construction, roman-numeral parsing and voice leading."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pitch import FLAT_NAMES, NOTE_NAMES, Scale, transpose_into_range

__all__ = ["CHORD_QUALITIES", "Chord", "parse_roman", "realize_progression", "voice_lead"]

# Semitone offsets from the chord root.
CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "minmaj7": (0, 3, 7, 11),
    "aug7": (0, 4, 8, 10),
    "maj9": (0, 4, 7, 11, 14),
    "min9": (0, 3, 7, 10, 14),
    "dom9": (0, 4, 7, 10, 14),
    "add9": (0, 4, 7, 14),
    "madd9": (0, 3, 7, 14),
    "sus4add9": (0, 5, 7, 14),
    "dom11": (0, 4, 7, 10, 14, 17),
    "min11": (0, 3, 7, 10, 14, 17),
    "maj13": (0, 4, 7, 11, 14, 21),
    "dom13": (0, 4, 7, 10, 14, 21),
    "min13": (0, 3, 7, 10, 14, 21),
    "dom7b9": (0, 4, 7, 10, 13),
    "dom7s9": (0, 4, 7, 10, 15),
    "dom7s11": (0, 4, 7, 10, 18),
    "dom7b13": (0, 4, 7, 10, 20),
    "maj7s11": (0, 4, 7, 11, 18),
}

# Quality of the triad built on each degree of a 7-note scale, derived rather
# than tabulated so exotic modes behave correctly.
_TRIAD_BY_INTERVALS = {
    (4, 7): "maj",
    (3, 7): "min",
    (3, 6): "dim",
    (4, 8): "aug",
    (2, 7): "sus2",
    (5, 7): "sus4",
}
_SEVENTH_BY_INTERVALS = {
    (4, 7, 11): "maj7",
    (3, 7, 10): "min7",
    (4, 7, 10): "dom7",
    (3, 6, 10): "m7b5",
    (3, 6, 9): "dim7",
    (3, 7, 11): "minmaj7",
    (4, 8, 11): "maj7",
    (4, 8, 10): "aug7",
}

_ROMAN_TO_INDEX = {"i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6}


@dataclass
class Chord:
    """A chord as a root pitch class plus a quality name."""

    root: int  # pitch class 0-11
    quality: str = "maj"
    inversion: int = 0
    bass: int | None = None  # explicit slash-chord bass, pitch class
    degree: int | None = None  # scale degree it came from, when known
    roman: str = ""  # the numeral it was parsed from, for display/debug
    prefer_flat: bool = False  # spell this chord with flats (came from bVII etc.)

    @property
    def intervals(self) -> tuple[int, ...]:
        return CHORD_QUALITIES.get(self.quality, CHORD_QUALITIES["maj"])

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple(sorted({(self.root + i) % 12 for i in self.intervals}))

    def name(self, flats: bool = False) -> str:
        """Chord symbol, e.g. ``Cmaj7`` or ``Bb/D``.

        ``flats`` selects the accidental spelling; callers that know the key
        should pass ``prefers_flats(scale)`` so a Bb chart never says A#.
        """
        suffix = {
            "maj": "", "min": "m", "dom7": "7", "maj7": "maj7", "min7": "m7",
            "dim": "dim", "aug": "aug", "m7b5": "m7b5", "dim7": "dim7",
            "maj6": "6", "min6": "m6", "dom9": "9", "min9": "m9", "maj9": "maj9",
            "sus2": "sus2", "sus4": "sus4", "add9": "add9", "5": "5",
        }.get(self.quality, self.quality)
        table = FLAT_NAMES if (flats or self.prefer_flat) else NOTE_NAMES
        base = f"{table[self.root]}{suffix}"
        if self.bass is not None and self.bass != self.root:
            base += f"/{table[self.bass]}"
        return base

    def notes(self, octave: int = 3, spread: bool = False) -> list[int]:
        """Absolute MIDI pitches for the chord in close position."""
        base = 12 * (octave + 1) + self.root
        pitches = [base + i for i in self.intervals]
        for _ in range(self.inversion % max(1, len(pitches))):
            pitches.append(pitches.pop(0) + 12)
        if spread and len(pitches) >= 3:
            pitches[1] += 12
        if self.bass is not None:
            bass_pitch = transpose_into_range(12 * (octave + 1) + self.bass, base - 12, base - 1)
            pitches = [bass_pitch, *pitches]
        return sorted(pitches)

    def bass_pitch(self, octave: int = 2) -> int:
        pc = self.bass if self.bass is not None else self.root
        return 12 * (octave + 1) + pc


def diatonic_quality(scale: Scale, degree: int, sevenths: bool = False) -> str:
    """Quality of the chord stacked in thirds on ``degree`` of ``scale``."""
    if scale.size < 7:
        # Pentatonic and other gapped scales have no reliable tertian stack;
        # fall back to a triad implied by the parent major/minor colour.
        return "min" if scale.is_minor_ish() else "maj"
    root = scale.degree_to_pitch(degree, 4)
    third = scale.degree_to_pitch(degree + 2, 4)
    fifth = scale.degree_to_pitch(degree + 4, 4)
    iv = ((third - root) % 12, (fifth - root) % 12)
    if sevenths:
        seventh = scale.degree_to_pitch(degree + 6, 4)
        iv7 = (*iv, (seventh - root) % 12)
        return _SEVENTH_BY_INTERVALS.get(iv7, _TRIAD_BY_INTERVALS.get(iv, "maj"))
    return _TRIAD_BY_INTERVALS.get(iv, "maj")


_ROMAN_RE = re.compile(
    r"^(?P<flat>[b#♭♯]*)"
    r"(?P<numeral>[ivxIVX]+)"
    r"(?P<quality>maj7|maj9|maj13|m7b5|min7|dim7|add9|sus2|sus4|m7|m9|m6|7|9|11|13|6|o|\+|°|ø|dim|aug)?"
    r"(?P<ext>[b#]\d+)?"
    r"(?:/(?P<slash>[b#♭♯]*[ivxIVX]+|[A-G][#b]?))?$"
)


def parse_roman(symbol: str, scale: Scale, sevenths: bool = False) -> Chord:
    """Parse a roman-numeral symbol against ``scale``.

    Handles accidentals (``bVII``), explicit qualities (``V7``, ``iv6``),
    secondary dominants (``V/V``) and slash basses (``I/3``).  Unknown input
    degrades to the tonic rather than raising, because progressions come from
    user-editable data.
    """
    sym = symbol.strip().replace("♭", "b").replace("♯", "#")
    m = _ROMAN_RE.match(sym)
    if not m:
        return Chord(root=scale.tonic, quality="maj", roman=sym)

    flat, numeral, quality, _ext, slash = (
        m.group("flat"), m.group("numeral"), m.group("quality"),
        m.group("ext"), m.group("slash"),
    )

    idx = _ROMAN_TO_INDEX.get(numeral.lower())
    if idx is None:
        return Chord(root=scale.tonic, quality="maj", roman=sym)

    # Secondary function: V/V means "the V of the key a fifth up".
    if slash and re.fullmatch(r"[b#]*[ivxIVX]+", slash):
        target = parse_roman(slash, scale, sevenths=False)
        local = Scale(target.root, "major" if target.quality in ("maj", "dom7") else "minor")
        return parse_roman(f"{flat}{numeral}{quality or ''}", local, sevenths)

    if flat:
        # An accidental on a numeral is conventionally read against the
        # parallel MAJOR scale, which is what makes "i-bVII-bVI" resolve to
        # Am-G-F in A minor rather than Am-F#-E.  Unaccented numerals stay
        # diatonic to the prevailing scale.
        root = Scale(scale.tonic, "major").degree_to_pitch(idx, 4) % 12
        for ch in flat:
            root = (root - 1) % 12 if ch == "b" else (root + 1) % 12
    else:
        root = scale.degree_to_pitch(idx, 4) % 12

    if quality:
        q = {
            "7": "dom7", "maj7": "maj7", "m7": "min7", "min7": "min7",
            "m9": "min9", "9": "dom9", "11": "dom11", "13": "dom13",
            "6": "maj6", "m6": "min6", "maj9": "maj9", "maj13": "maj13",
            "dim": "dim", "o": "dim", "°": "dim", "dim7": "dim7",
            "ø": "m7b5", "m7b5": "m7b5", "aug": "aug", "+": "aug",
            "sus2": "sus2", "sus4": "sus4", "add9": "add9",
        }.get(quality, "maj")
        # A lowercase numeral means minor: fold that in where the suffix is
        # ambiguous (vi7 is a minor 7th, VI7 is a dominant).
        if numeral.islower() and q in ("maj", "dom7", "maj6"):
            q = {"maj": "min", "dom7": "min7", "maj6": "min6"}[q]
    elif flat:
        # The root has been chromatically altered, so the unaltered degree's
        # diatonic quality is meaningless (bVII in C is Bb major, not B dim).
        # Take the colour from the numeral's case instead.
        if numeral.islower():
            q = "min7" if sevenths else "min"
        else:
            q = "maj7" if sevenths else "maj"
    else:
        q = diatonic_quality(scale, idx, sevenths)
        if numeral.islower() and q == "maj":
            q = "min"
        elif numeral.isupper() and q == "min":
            q = "maj"

    bass = None
    if slash:
        from .pitch import parse_note

        p = parse_note(slash)
        if p is not None:
            bass = p % 12

    return Chord(
        root=root, quality=q, bass=bass, degree=idx, roman=sym,
        prefer_flat="b" in flat,
    )


def realize_progression(
    symbols: list[str], scale: Scale, sevenths: bool = False
) -> list[Chord]:
    return [parse_roman(s, scale, sevenths) for s in symbols]


def voice_lead(
    chords: list[Chord],
    low: int = 52,
    high: int = 76,
    voices: int = 4,
) -> list[list[int]]:
    """Voice a chord sequence so adjacent voicings move as little as possible.

    Each chord is tried in every inversion and octave that fits the register;
    the candidate with the smallest total voice movement from the previous
    voicing wins.  This is what makes block-chord comping sound intentional
    rather than like a sequence of root-position stabs.
    """
    result: list[list[int]] = []
    prev: list[int] | None = None

    for ch in chords:
        candidates = _voicing_candidates(ch, low, high, voices)
        if not candidates:
            result.append([transpose_into_range(12 * 4 + ch.root, low, high)])
            prev = result[-1]
            continue
        if prev is None:
            # Open on something centred in the register so later chords have
            # room to move in both directions.
            centre = (low + high) / 2
            best = min(candidates, key=lambda v: abs(sum(v) / len(v) - centre))
        else:
            best = min(candidates, key=lambda v: _movement(prev, v))
        result.append(best)
        prev = best
    return result


def _voicing_candidates(ch: Chord, low: int, high: int, voices: int) -> list[list[int]]:
    pcs = list(ch.pitch_classes)
    if not pcs:
        return []
    out: list[list[int]] = []
    # Build every set of `voices` pitches drawn from the chord tones inside the
    # register, preferring one of each chord tone before doubling.
    pool: list[int] = []
    for pc in pcs:
        p = 12 * ((low // 12) + 1) + pc
        while p < low:
            p += 12
        while p <= high:
            pool.append(p)
            p += 12
    pool.sort()
    n = min(voices, max(len(pcs), 1))
    for start in range(len(pool)):
        chosen: list[int] = []
        seen: set[int] = set()
        for p in pool[start:]:
            if p % 12 in seen:
                continue
            chosen.append(p)
            seen.add(p % 12)
            if len(chosen) == n:
                break
        if len(chosen) == n and len(seen) == len(set(pcs[:n])):
            out.append(chosen)
    return out or [list(pool[:n])]


def _movement(a: list[int], b: list[int]) -> float:
    """Total semitone movement between two voicings, plus a register penalty."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    cost = sum(abs(a[i] - b[i]) for i in range(n))
    cost += abs(len(a) - len(b)) * 3
    # Discourage drifting to the extremes of the register over time.
    cost += abs(sum(b) / len(b) - sum(a) / len(a)) * 0.5
    return float(cost)
