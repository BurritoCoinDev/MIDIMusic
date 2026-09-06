"""Turning a free-text prompt into musical parameters.

The neural backends take the prompt as-is.  The built-in composer cannot, so
this module reads what it can -- genre, key, tempo, mood, instrumentation,
length -- and leaves the rest alone.  It is deliberately permissive: an
unrecognised word is ignored rather than treated as an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..theory.style import STYLES, list_styles

__all__ = ["MOOD_TERMS", "STYLE_SYNONYMS", "ParsedPrompt", "parse_prompt"]

# Words that map onto a style when the style name itself is not present.
STYLE_SYNONYMS: dict[str, str] = {
    "edm": "house", "dance": "house", "deep house": "house", "club": "house",
    "electro": "techno", "industrial": "techno", "minimal": "techno",
    "drum and bass": "dnb", "drum n bass": "dnb", "jungle": "dnb",
    "breakbeat": "dnb", "liquid": "dnb",
    "hip hop": "hiphop", "hip-hop": "hiphop", "rap": "hiphop", "boom bap": "hiphop",
    "chill": "lofi", "chillhop": "lofi", "study": "lofi", "chillout": "lofi",
    "study beats": "lofi", "sleepy": "lofi",
    "retrowave": "synthwave", "outrun": "synthwave", "vaporwave": "synthwave",
    "80s": "synthwave", "eighties": "synthwave",
    "orchestral": "cinematic", "epic": "cinematic", "trailer": "cinematic",
    "film score": "cinematic", "soundtrack": "cinematic", "score": "cinematic",
    "classic rock": "rock", "indie": "rock", "grunge": "rock", "alternative": "rock",
    "heavy metal": "metal", "djent": "metal", "doom": "metal", "thrash": "metal",
    "hardcore": "punk", "pop punk": "punk",
    "swing": "jazz", "bebop": "jazz", "smooth jazz": "jazz", "fusion": "jazz",
    "rnb": "funk", "r&b": "funk", "soul": "funk", "motown": "funk", "groove": "funk",
    "bluegrass": "country", "americana": "country", "western": "country",
    "acoustic": "folk", "singer songwriter": "folk", "celtic": "folk",
    "samba": "bossa", "bossa nova": "bossa", "brazilian": "bossa",
    "salsa": "latin", "flamenco": "latin", "tango": "latin", "mariachi": "latin",
    "dub": "reggae", "ska": "reggae", "dancehall": "reggae",
    "drone": "ambient", "atmospheric": "ambient", "meditation": "ambient",
    "new age": "ambient", "spa": "ambient", "background": "ambient",
    "baroque": "classical", "romantic": "classical", "piano solo": "classical",
    "waltz": "classical", "symphony": "classical",
    "chiptune": "chiptune", "8-bit": "chiptune", "8 bit": "chiptune",
    "video game": "chiptune", "gameboy": "chiptune",
    "boogie": "blues", "shuffle": "blues", "delta blues": "blues",
    "funky": "funk", "disco house": "disco", "nu disco": "disco",
}

# Mood words nudge brightness, energy and mode rather than picking a genre.
MOOD_TERMS: dict[str, dict[str, float | str]] = {
    "dark": {"brightness": 0.15, "mode": "minor"},
    "moody": {"brightness": 0.25, "mode": "minor"},
    "sad": {"brightness": 0.2, "energy": 0.3, "mode": "minor"},
    "melancholy": {"brightness": 0.2, "energy": 0.3, "mode": "minor"},
    "somber": {"brightness": 0.15, "energy": 0.25, "mode": "minor"},
    "haunting": {"brightness": 0.2, "mode": "minor"},
    "tense": {"brightness": 0.25, "energy": 0.7, "mode": "minor"},
    "aggressive": {"energy": 0.95, "brightness": 0.35, "mode": "minor"},
    "angry": {"energy": 0.95, "mode": "minor"},
    "heavy": {"energy": 0.9, "brightness": 0.3},
    "intense": {"energy": 0.9},
    "driving": {"energy": 0.85},
    "energetic": {"energy": 0.9, "brightness": 0.75},
    "upbeat": {"energy": 0.85, "brightness": 0.85, "mode": "major"},
    "happy": {"brightness": 0.9, "energy": 0.75, "mode": "major"},
    "bright": {"brightness": 0.9, "mode": "major"},
    "uplifting": {"brightness": 0.85, "energy": 0.8, "mode": "major"},
    "hopeful": {"brightness": 0.8, "mode": "major"},
    "warm": {"brightness": 0.6},
    "gentle": {"energy": 0.3, "brightness": 0.6},
    "calm": {"energy": 0.25, "brightness": 0.55},
    "peaceful": {"energy": 0.2, "brightness": 0.65},
    "relaxing": {"energy": 0.25, "brightness": 0.6},
    "dreamy": {"energy": 0.3, "brightness": 0.7},
    "ethereal": {"energy": 0.25, "brightness": 0.8},
    "nostalgic": {"brightness": 0.45},
    "mysterious": {"brightness": 0.3, "mode": "minor"},
    "epic": {"energy": 0.9, "brightness": 0.6},
    "triumphant": {"energy": 0.9, "brightness": 0.9, "mode": "major"},
    "romantic": {"brightness": 0.65, "energy": 0.4},
    "playful": {"brightness": 0.85, "energy": 0.7, "mode": "major"},
    "quirky": {"brightness": 0.75, "energy": 0.65},
    "cold": {"brightness": 0.2},
    "lush": {"complexity": 0.9},
    "sparse": {"complexity": 0.25, "energy": 0.35},
    "minimal": {"complexity": 0.2},
    "simple": {"complexity": 0.25},
    "complex": {"complexity": 0.95},
    "rich": {"complexity": 0.9},
    "dense": {"complexity": 0.95},
    "layered": {"complexity": 0.9},
    "orchestral": {"complexity": 0.95},
    "cinematic": {"complexity": 0.85},
}

_INSTRUMENT_ROLES = {
    "piano": "chords", "guitar": "chords", "synth": "arp", "strings": "strings",
    "drums": "drums", "bass": "bass", "pad": "pad", "brass": "brass",
    "horns": "brass", "sax": "lead", "violin": "lead", "flute": "lead",
    "organ": "chords", "vocals": "lead", "choir": "pad", "percussion": "perc",
}

_SCALE_WORDS = [
    "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian",
    "harmonic minor", "melodic minor", "blues", "pentatonic", "aeolian", "ionian",
]


@dataclass
class ParsedPrompt:
    """Musical parameters recovered from free text."""

    text: str = ""
    style: str | None = None
    key: str | None = None
    tempo: float | None = None
    duration_seconds: float | None = None
    brightness: float | None = None
    energy: float | None = None
    complexity: float | None = None
    instrumental: bool = True
    matched_terms: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = []
        if self.style:
            bits.append(self.style)
        if self.key:
            bits.append(self.key)
        if self.tempo:
            bits.append(f"{self.tempo:.0f} bpm")
        return ", ".join(bits) or "no musical hints found"


def _find_tempo(text: str) -> float | None:
    m = re.search(r"(\d{2,3})\s*(?:bpm|beats per minute)", text)
    if m:
        value = float(m.group(1))
        return value if 30 <= value <= 300 else None
    for word, value in (("very slow", 60), ("slow", 72), ("moderate", 100),
                        ("medium tempo", 100), ("upbeat", 128), ("fast", 150),
                        ("very fast", 175), ("uptempo", 138)):
        if word in text:
            return float(value)
    return None


def _find_duration(text: str) -> float | None:
    m = re.search(r"(\d+)\s*(?:min(?:ute)?s?)\s*(?:and\s*)?(\d+)?\s*(?:sec(?:ond)?s?)?", text)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2) or 0)
        total = minutes * 60 + seconds
        if 5 <= total <= 3600:
            return float(total)
    m = re.search(r"(\d{2,4})\s*(?:sec(?:ond)?s?|s)\b", text)
    if m:
        value = float(m.group(1))
        if 5 <= value <= 3600:
            return value
    return None


def _find_key(text: str) -> str | None:
    # "in F# minor", "key of Bb", "C dorian"
    pattern = (
        r"(?:in|key of)\s+([A-Ga-g][#b♯♭]?)\s*(" + "|".join(_SCALE_WORDS) + r")?"
        r"|\b([A-G][#b])\s+(" + "|".join(_SCALE_WORDS) + r")\b"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    root = m.group(1) or m.group(3)
    mode = m.group(2) or m.group(4)
    if not root:
        return None
    root = root[0].upper() + root[1:].replace("♯", "#").replace("♭", "b")
    return f"{root} {mode.lower()}" if mode else root


def parse_prompt(text: str, default_style: str = "pop") -> ParsedPrompt:
    """Extract musical parameters from a free-text prompt."""
    raw = (text or "").strip()
    low = raw.lower()
    out = ParsedPrompt(text=raw)

    # Style: prefer an exact genre name, then a synonym. Longest match wins so
    # "drum and bass" beats "bass" and "pop punk" beats "pop".
    candidates: list[tuple[int, str, str]] = []
    for name in list_styles():
        if re.search(rf"\b{re.escape(name)}\b", low):
            candidates.append((len(name), name, name))
    for phrase, target in STYLE_SYNONYMS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            candidates.append((len(phrase), target, phrase))
    if candidates:
        candidates.sort(reverse=True)
        _, out.style, term = candidates[0]
        out.matched_terms.append(term)
    else:
        out.style = default_style if default_style in STYLES else "pop"

    out.tempo = _find_tempo(low)
    out.duration_seconds = _find_duration(low)
    out.key = _find_key(raw)

    # Moods accumulate; several words average rather than the last one winning.
    sums: dict[str, list[float]] = {}
    mode_hint: str | None = None
    for term, effects in MOOD_TERMS.items():
        if not re.search(rf"\b{re.escape(term)}\b", low):
            continue
        out.matched_terms.append(term)
        for attr, value in effects.items():
            if attr == "mode":
                mode_hint = str(value)
            else:
                sums.setdefault(attr, []).append(float(value))
    for attr, values in sums.items():
        setattr(out, attr, sum(values) / len(values))

    # A bare mood-implied mode only applies when no explicit key was given.
    if out.key is None and mode_hint:
        out.key = f"C {mode_hint}"
    elif out.key and mode_hint and not any(w in out.key.lower() for w in _SCALE_WORDS):
        out.key = f"{out.key} {mode_hint}"

    for word, role in _INSTRUMENT_ROLES.items():
        if re.search(rf"\b{re.escape(word)}s?\b", low):
            out.instruments.append(role)

    if re.search(r"\b(vocal|vocals|singing|singer|lyrics|sung|voice)\b", low):
        out.instrumental = False

    return out
