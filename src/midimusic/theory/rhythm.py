"""Grooves: drum patterns, bass lines and chord comping.

Patterns are written as step strings on a 16th-note grid, one character per
step, which keeps them readable and editable by hand:

    ``"x..x..x...x.x..."``

``x`` is an accented hit, ``o`` a normal hit, ``.`` a rest and ``g`` a ghost
note.  Two bars are written as a 32-character string; anything not a multiple
of 16 is looped to fill the bar.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = [
    "DRUM",
    "DRUM_PATTERNS",
    "BASS_STYLES",
    "COMP_STYLES",
    "DrumKit",
    "parse_steps",
    "build_drum_bar",
    "build_bass_bar",
    "build_comp_bar",
]

# General MIDI percussion key map (channel 10).
DRUM = {
    "kick": 36,
    "kick2": 35,
    "snare": 38,
    "snare_rim": 37,
    "clap": 39,
    "snare2": 40,
    "tom_low": 41,
    "hat_closed": 42,
    "tom_mid": 45,
    "hat_pedal": 44,
    "hat_open": 46,
    "tom_high": 48,
    "crash": 49,
    "ride": 51,
    "ride_bell": 53,
    "tambourine": 54,
    "cowbell": 56,
    "shaker": 70,
    "sidestick": 37,
}

_VELOCITY = {"x": 1.0, "o": 0.72, "g": 0.38, "X": 1.0, "O": 0.72}


def parse_steps(pattern: str, steps: int = 16) -> list[float]:
    """Expand a step string to ``steps`` velocity multipliers (0.0 = rest)."""
    cleaned = [c for c in pattern if c in "xoXOg."]
    if not cleaned:
        return [0.0] * steps
    out: list[float] = []
    while len(out) < steps:
        for c in cleaned:
            out.append(_VELOCITY.get(c, 0.0))
            if len(out) == steps:
                break
    return out[:steps]


@dataclass
class DrumKit:
    """One bar of drum programming: instrument name -> step string."""

    parts: dict[str, str] = field(default_factory=dict)
    swing: float = 0.0  # 0 = straight, 0.66 ~ triplet swing
    fill_every: int = 4  # insert a fill on the last bar of every N bars


# Each entry is a bar of 16 steps unless noted.  These are deliberately plain:
# character comes from the humanisation and arrangement layers, not from
# elaborate patterns that fight the harmony.
DRUM_PATTERNS: dict[str, DrumKit] = {
    "rock": DrumKit({
        "kick":       "x..x..x...x.....",
        "snare":      "....x.......x...",
        "hat_closed": "x.x.x.x.x.x.x.x.",
        "crash":      "x...............",
    }),
    "pop": DrumKit({
        "kick":       "x.....x...x.....",
        "snare":      "....x.......x...",
        "hat_closed": "x.x.x.x.x.x.x.x.",
        "clap":       "....x.......x...",
    }),
    "funk": DrumKit({
        "kick":       "x..x.....x..x...",
        "snare":      "....x..g....x..g",
        "hat_closed": "xgxgxgxgxgxgxgxg",
        "hat_open":   "..........x.....",
    }, swing=0.12),
    "hiphop": DrumKit({
        "kick":       "x.......x.x.....",
        "snare":      "....x.......x...",
        "hat_closed": "x.x.x.x.x.x.x.x.",
    }, swing=0.18),
    "trap": DrumKit({
        "kick":       "x.....x.....x...",
        "snare":      "........x.......",
        "hat_closed": "xxxxxxxxxxxxxxxx",
    }),
    "house": DrumKit({
        "kick":       "x...x...x...x...",
        "clap":       "....x.......x...",
        "hat_open":   "..x...x...x...x.",
        "hat_closed": "x.x.x.x.x.x.x.x.",
    }),
    "techno": DrumKit({
        "kick":       "x...x...x...x...",
        "hat_closed": "..x...x...x...x.",
        "snare":      "............x...",
        "shaker":     "xgxgxgxgxgxgxgxg",
    }),
    "dnb": DrumKit({
        "kick":       "x.......x.x.....",
        "snare":      "....x.......x...",
        "hat_closed": "xgxgxgxgxgxgxgxg",
        "ride":       "..x...x...x...x.",
    }),
    "jazz": DrumKit({
        "ride":       "x..xx..xx..xx..x",
        "hat_pedal":  "....x.......x...",
        "kick":       "x..........g....",
        "snare":      "..g....g..g....g",
    }, swing=0.62),
    "swing": DrumKit({
        "ride":       "x..xx..xx..xx..x",
        "hat_pedal":  "....x.......x...",
        "snare":      "....g.......g...",
        "kick":       "x...............",
    }, swing=0.66),
    "bossa": DrumKit({
        "sidestick":  "x..x..x...x..x..",
        "hat_closed": "x.x.x.x.x.x.x.x.",
        "kick":       "x..x..x...x..x..",
    }),
    "latin": DrumKit({
        "kick":       "x..x..x...x..x..",
        "cowbell":    "x.x.x.x.x.x.x.x.",
        "snare_rim":  "..x..x..x..x..x.",
        "shaker":     "xgxgxgxgxgxgxgxg",
    }),
    "reggae": DrumKit({
        "kick":       "........x.......",
        "snare":      "....x.......x...",
        "hat_closed": "..x...x...x...x.",
    }),
    "metal": DrumKit({
        "kick":       "x.xxx.xxx.xxx.xx",
        "snare":      "....x.......x...",
        "crash":      "x.......x.......",
        "ride":       "x.x.x.x.x.x.x.x.",
    }),
    "punk": DrumKit({
        "kick":       "x.x.x.x.x.x.x.x.",
        "snare":      "....x.......x...",
        "hat_closed": "x.x.x.x.x.x.x.x.",
    }),
    "country": DrumKit({
        "kick":       "x.......x.......",
        "snare":      "....x.......x...",
        "hat_closed": "x.x.x.x.x.x.x.x.",
    }),
    "blues": DrumKit({
        "ride":       "x..xx..xx..xx..x",
        "kick":       "x.......x.......",
        "snare":      "....x.......x...",
    }, swing=0.6),
    "waltz": DrumKit({
        "kick":       "x.....",
        "snare":      "..x..x",
        "hat_closed": "x.x.x.",
    }),
    "ambient": DrumKit({}),
    "cinematic": DrumKit({
        "kick":       "x.......x.......",
        "tom_low":    "........x...x...",
        "crash":      "x...............",
    }),
    "lofi": DrumKit({
        "kick":       "x.......x.x.....",
        "snare":      "....x.......x...",
        "hat_closed": "x.g.x.g.x.g.x.g.",
    }, swing=0.22),
    "disco": DrumKit({
        "kick":       "x...x...x...x...",
        "snare":      "....x.......x...",
        "hat_open":   "..x...x...x...x.",
        "hat_closed": "x.x.x.x.x.x.x.x.",
    }),
    "synthwave": DrumKit({
        "kick":       "x...x...x...x...",
        "snare":      "....x.......x...",
        "hat_closed": "x.x.x.x.x.x.x.x.",
        "tom_low":    "..............x.",
    }),
}

# Bass rhythm + contour recipes.  ``degrees`` indexes chord tones (0 = root).
BASS_STYLES: dict[str, dict] = {
    "roots":      {"steps": "x.......x.......", "degrees": [0]},
    "eighths":    {"steps": "x.x.x.x.x.x.x.x.", "degrees": [0]},
    "root_fifth": {"steps": "x...x...x...x...", "degrees": [0, 2]},
    "octaves":    {"steps": "x.x.x.x.x.x.x.x.", "degrees": [0, "oct"]},
    "walking":    {"steps": "x...x...x...x...", "degrees": "walk"},
    "arpeggio":   {"steps": "x.x.x.x.x.x.x.x.", "degrees": [0, 1, 2, 1]},
    "syncopated": {"steps": "x..x..x...x.x...", "degrees": [0, 0, 2, 0]},
    "funk":       {"steps": "x.gxg..x.g.x..g.", "degrees": [0, 0, 2, 0, "oct"]},
    "pedal":      {"steps": "x...............", "degrees": [0]},
    "driving":    {"steps": "xxxxxxxxxxxxxxxx", "degrees": [0]},
    "reggae":     {"steps": "..x..x....x..x..", "degrees": [0, 2, 0]},
    "house":      {"steps": "..x...x...x...x.", "degrees": [0, 0, "oct"]},
    "none":       {"steps": "................", "degrees": [0]},
}

# Chord comping recipes: when to hit, and how to voice it.
COMP_STYLES: dict[str, dict] = {
    "sustained":  {"steps": "x...............", "mode": "block", "length": 1.0},
    "half":       {"steps": "x.......x.......", "mode": "block", "length": 0.5},
    "quarters":   {"steps": "x...x...x...x...", "mode": "block", "length": 0.24},
    "offbeat":    {"steps": "..x...x...x...x.", "mode": "block", "length": 0.12},
    "charleston": {"steps": "x......x........", "mode": "block", "length": 0.2},
    "eighths":    {"steps": "x.x.x.x.x.x.x.x.", "mode": "block", "length": 0.12},
    "strum":      {"steps": "x.x.x.x.x.x.x.x.", "mode": "strum", "length": 0.2},
    "arpeggio":   {"steps": "x.x.x.x.x.x.x.x.", "mode": "arp", "length": 0.12},
    "arp_16":     {"steps": "xxxxxxxxxxxxxxxx", "mode": "arp", "length": 0.06},
    "stabs":      {"steps": "....x.......x...", "mode": "block", "length": 0.1},
    "pad":        {"steps": "x...............", "mode": "block", "length": 1.0},
    "none":       {"steps": "................", "mode": "block", "length": 0.0},
}


def build_drum_bar(
    kit: DrumKit,
    steps_per_bar: int = 16,
    intensity: float = 1.0,
    rng: random.Random | None = None,
    fill: bool = False,
) -> list[tuple[int, float, float]]:
    """Return ``(step_index, velocity_0_1, note)`` triples for one bar."""
    rng = rng or random.Random()
    events: list[tuple[int, float, float]] = []
    for part, pattern in kit.parts.items():
        note = DRUM.get(part)
        if note is None:
            continue
        for i, vel in enumerate(parse_steps(pattern, steps_per_bar)):
            if vel <= 0.0:
                continue
            v = vel * intensity
            # Thin out the busiest parts at low intensity so quiet sections
            # read as arrangement rather than as the same bar played softly.
            if intensity < 0.65 and part in ("hat_closed", "shaker", "ride") and i % 2 == 1:
                continue
            if intensity < 0.4 and part in ("crash", "hat_open"):
                continue
            events.append((i, max(0.05, min(1.0, v)), note))

    if fill:
        events = [e for e in events if e[0] < steps_per_bar - 4 or e[2] == DRUM["kick"]]
        toms = [DRUM["tom_high"], DRUM["tom_mid"], DRUM["tom_low"], DRUM["snare"]]
        for j, step in enumerate(range(steps_per_bar - 4, steps_per_bar)):
            events.append((step, 0.7 + 0.1 * j, toms[j % len(toms)]))
    return sorted(events)


def build_bass_bar(
    style: str,
    chord_tones: list[int],
    next_root: int | None,
    steps_per_bar: int = 16,
    rng: random.Random | None = None,
) -> list[tuple[int, float, int]]:
    """Return ``(step_index, velocity, pitch)`` for one bar of bass."""
    rng = rng or random.Random()
    recipe = BASS_STYLES.get(style, BASS_STYLES["roots"])
    vels = parse_steps(recipe["steps"], steps_per_bar)
    hits = [(i, v) for i, v in enumerate(vels) if v > 0]
    if not hits or not chord_tones:
        return []

    root = chord_tones[0]
    degrees = recipe["degrees"]
    out: list[tuple[int, float, int]] = []

    if degrees == "walk":
        # Walk from this chord's root toward the next, stepping mostly by
        # chord tones and filling the final beat with an approach note.
        target = next_root if next_root is not None else root
        for n, (step, vel) in enumerate(hits):
            if n == len(hits) - 1 and target != root:
                pitch = target - 1 if target > root else target + 1
            else:
                pitch = chord_tones[min(n, len(chord_tones) - 1)]
            out.append((step, vel, pitch))
        return out

    for n, (step, vel) in enumerate(hits):
        spec = degrees[n % len(degrees)]
        if spec == "oct":
            pitch = root + 12
        elif isinstance(spec, int):
            pitch = chord_tones[spec % len(chord_tones)] if chord_tones else root
        else:
            pitch = root
        out.append((step, vel, pitch))
    return out


def build_comp_bar(
    style: str,
    voicing: list[int],
    steps_per_bar: int = 16,
    rng: random.Random | None = None,
) -> list[tuple[int, float, int, float]]:
    """Return ``(step, velocity, pitch, length_in_bars)`` for one bar of comping."""
    rng = rng or random.Random()
    recipe = COMP_STYLES.get(style, COMP_STYLES["sustained"])
    vels = parse_steps(recipe["steps"], steps_per_bar)
    hits = [(i, v) for i, v in enumerate(vels) if v > 0]
    if not hits or not voicing:
        return []

    mode, length = recipe["mode"], recipe["length"]
    out: list[tuple[int, float, int, float]] = []

    if mode == "arp":
        # Up-down through the voicing, so long arpeggios do not just cycle.
        shape = voicing + voicing[-2:0:-1] if len(voicing) > 2 else voicing
        for n, (step, vel) in enumerate(hits):
            out.append((step, vel, shape[n % len(shape)], length))
        return out

    for step, vel in hits:
        if mode == "strum":
            for k, pitch in enumerate(voicing):
                out.append((step + k * 0.12, vel * (1.0 - 0.04 * k), pitch, length))
        else:
            for pitch in voicing:
                out.append((step, vel, pitch, length))
    return out
