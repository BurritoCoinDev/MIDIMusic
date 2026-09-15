"""Grouping General MIDI programs into the sections of an orchestra.

A transcriber hands back notes tagged with a General MIDI program. That is a
sound, not a seat in an orchestra, so something has to decide that program 71
is a woodwind and program 47 is percussion rather than a string. This module
is that decision, kept in one place because it is a musical judgement rather
than a detail of any one backend.

The granularity available is the GM *family*: a model can tell strings from
brass, but not the second violins from the firsts. Layers are named for what
they are -- a string body, a brass section -- rather than implying a precision
that is not there.
"""

from __future__ import annotations

__all__ = [
    "SCORE_ORDER",
    "SECTIONS",
    "family_name",
    "section_for",
    "sort_key",
]

# Conductor's score order, top to bottom. Writing the layers out this way
# means the exported score reads the way a printed one does.
SCORE_ORDER: tuple[str, ...] = (
    "Woodwinds",
    "Brass",
    "Percussion",
    "Keyboards",
    "Voice",
    "Guitars",
    "Synths",
    "Strings",
    "Bass",
    "Other",
)

SECTIONS: tuple[str, ...] = SCORE_ORDER

# (first program, last program, section). Ordered, and read first-match-wins.
_RANGES: tuple[tuple[int, int, str], ...] = (
    (0, 7, "Keyboards"),        # pianos
    (8, 15, "Percussion"),      # glockenspiel, vibraphone, tubular bells: tuned percussion
    (16, 23, "Keyboards"),      # organs and accordion
    (24, 31, "Guitars"),
    (32, 39, "Bass"),
    (40, 46, "Strings"),        # violin through orchestral harp
    (47, 47, "Percussion"),     # timpani sits with the percussion, not the strings
    (48, 51, "Strings"),        # string ensembles
    (52, 54, "Voice"),          # choir aahs, voice oohs, synth voice
    (55, 55, "Other"),          # orchestra hit
    (56, 63, "Brass"),
    (64, 71, "Woodwinds"),      # reeds
    (72, 79, "Woodwinds"),      # flutes and pipes
    (80, 99, "Synths"),
    # Multi-instrument transcribers reserve these two for a sung line, which
    # is not what General MIDI puts there -- they matter more than the default.
    (100, 101, "Voice"),
    (102, 103, "Synths"),
    (104, 111, "Other"),        # ethnic instruments
    (112, 119, "Percussion"),
    (120, 127, "Other"),        # sound effects
)

# Programs whose family label the range table would get wrong.
_NAMED: dict[int, str] = {
    46: "Harp", 47: "Timpani",
    52: "Choir", 53: "Voice", 54: "Voice",
    55: "Orchestra hit",
    100: "Voice", 101: "Choir",   # the transcriber's sung-line reservation
}

# The General MIDI family a program belongs to, for labelling one layer.
_FAMILIES: tuple[tuple[int, str], ...] = (
    (7, "Piano"), (15, "Tuned percussion"), (23, "Organ"), (31, "Guitar"),
    (39, "Bass"), (47, "Strings"), (55, "Ensemble"), (63, "Brass"),
    (71, "Reeds"), (79, "Flutes"), (87, "Synth lead"), (95, "Synth pad"),
    (103, "Synth effects"), (111, "Ethnic"), (119, "Percussion"),
    (127, "Sound effects"),
)


def section_for(program: int, is_drum: bool = False) -> str:
    """Which orchestral section a General MIDI program belongs to."""
    if is_drum:
        return "Percussion"
    program = int(program)
    for low, high, section in _RANGES:
        if low <= program <= high:
            return section
    return "Other"


def family_name(program: int, is_drum: bool = False) -> str:
    """A readable name for the General MIDI family of ``program``."""
    if is_drum:
        return "Drum kit"
    program = int(program)
    if program in _NAMED:
        return _NAMED[program]
    for limit, name in _FAMILIES:
        if program <= limit:
            return name
    return "Instrument"


def sort_key(section: str) -> int:
    """Position of a section in conductor's score order."""
    try:
        return SCORE_ORDER.index(section)
    except ValueError:
        return len(SCORE_ORDER)
