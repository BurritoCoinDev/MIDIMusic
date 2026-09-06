"""Song form: sections, arrangement density and per-section instrumentation.

A song is a list of :class:`Section` objects.  Each section knows how many
bars it lasts, how intense it should feel (which drives drum density, voicing
width and velocity), and which roles are active.  The arranger uses the role
set to decide what actually plays, so "drop the drums in the bridge" is a data
change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Section", "SongForm", "FORMS", "build_form", "ROLES"]

# Musical roles the arranger can fill.  A style maps each role to a GM program.
# The core seven carry the song; the rest are enrichment layers that switch on
# as the complexity setting rises, so a dense mix is arrangement rather than
# the same four parts played louder.
CORE_ROLES = ("drums", "bass", "chords", "lead", "pad", "arp", "counter")
EXTRA_ROLES = ("perc", "sub", "chords2", "strings", "lead2", "brass", "texture")
ROLES = CORE_ROLES + EXTRA_ROLES


@dataclass
class Section:
    """One labelled span of a song."""

    name: str
    bars: int
    intensity: float = 0.8  # 0..1, drives density and velocity
    roles: tuple[str, ...] = ROLES
    progression: list[str] | None = None  # override the style's progression
    transpose: int = 0  # semitones, for last-chorus lifts
    fill_at_end: bool = True

    def with_roles(self, *roles: str) -> "Section":
        return Section(self.name, self.bars, self.intensity, tuple(roles),
                       self.progression, self.transpose, self.fill_at_end)


@dataclass
class SongForm:
    """An ordered list of sections plus the harmonic plan."""

    sections: list[Section] = field(default_factory=list)

    @property
    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)

    def duration_seconds(self, bpm: float, beats_per_bar: int = 4) -> float:
        return self.total_bars * beats_per_bar * 60.0 / max(1e-6, bpm)


def _s(name: str, bars: int, intensity: float, *roles: str) -> Section:
    return Section(name, bars, intensity, roles or ROLES)


# Named arrangement templates.  Bars here are a starting point; ``build_form``
# stretches or trims them to hit a requested duration.
FORMS: dict[str, list[Section]] = {
    "pop": [
        _s("intro", 4, 0.45, "drums", "chords", "pad", "texture"),
        _s("verse", 8, 0.6, "drums", "bass", "chords", "pad", "perc"),
        _s("prechorus", 4, 0.75, "drums", "bass", "chords", "arp", "strings"),
        _s("chorus", 8, 1.0, "drums", "bass", "chords", "lead", "pad",
           "lead2", "strings", "perc", "sub"),
        _s("verse", 8, 0.65, "drums", "bass", "chords", "arp", "perc", "counter"),
        _s("chorus", 8, 1.0, "drums", "bass", "chords", "lead", "pad",
           "lead2", "strings", "perc", "sub"),
        _s("bridge", 8, 0.5, "chords", "pad", "lead", "strings", "texture"),
        _s("chorus", 8, 1.0, "drums", "bass", "chords", "chords2", "lead", "pad",
           "lead2", "strings", "perc", "sub", "brass"),
        _s("outro", 4, 0.5, "chords", "pad", "texture"),
    ],
    "verse_chorus": [
        _s("intro", 4, 0.5, "drums", "chords", "texture"),
        _s("verse", 8, 0.65, "drums", "bass", "chords", "perc"),
        _s("chorus", 8, 1.0, "drums", "bass", "chords", "lead", "pad",
           "strings", "lead2", "sub"),
        _s("verse", 8, 0.7, "drums", "bass", "chords", "arp", "perc", "counter"),
        _s("chorus", 8, 1.0, "drums", "bass", "chords", "chords2", "lead", "pad",
           "strings", "lead2", "sub", "brass"),
        _s("solo", 8, 0.9, "drums", "bass", "chords", "lead", "perc"),
        _s("chorus", 8, 1.0, "drums", "bass", "chords", "chords2", "lead", "pad",
           "strings", "lead2", "sub", "brass"),
        _s("outro", 4, 0.5, "chords", "pad", "texture"),
    ],
    "aaba": [
        _s("A", 8, 0.7),
        _s("A", 8, 0.75),
        _s("B", 8, 0.85),
        _s("A", 8, 0.8),
    ],
    "twelve_bar": [
        _s("head", 12, 0.7, "drums", "bass", "chords", "lead", "counter"),
        _s("solo", 12, 0.9, "drums", "bass", "chords", "lead", "perc"),
        _s("solo", 12, 1.0, "drums", "bass", "chords", "lead", "brass", "perc"),
        _s("head", 12, 0.8, "drums", "bass", "chords", "lead", "counter", "brass"),
    ],
    "electronic": [
        _s("intro", 8, 0.35, "drums", "pad", "texture"),
        _s("build", 8, 0.6, "drums", "bass", "arp", "pad", "perc"),
        _s("drop", 16, 1.0, "drums", "bass", "chords", "lead", "arp",
           "sub", "perc", "lead2", "strings"),
        _s("breakdown", 8, 0.4, "pad", "chords", "texture", "counter"),
        _s("build", 8, 0.7, "drums", "bass", "arp", "perc", "strings"),
        _s("drop", 16, 1.0, "drums", "bass", "chords", "chords2", "lead", "arp",
           "sub", "perc", "lead2", "strings", "brass"),
        _s("outro", 8, 0.4, "drums", "pad", "texture"),
    ],
    "ambient": [
        _s("intro", 8, 0.25, "pad", "texture"),
        _s("A", 16, 0.4, "pad", "chords", "arp", "texture"),
        _s("B", 16, 0.55, "pad", "chords", "arp", "lead", "strings", "counter"),
        _s("A", 16, 0.4, "pad", "chords", "texture", "counter"),
        _s("outro", 8, 0.2, "pad", "texture"),
    ],
    "loop": [
        _s("loop", 8, 0.8),
    ],
    "cinematic": [
        _s("intro", 8, 0.25, "pad", "chords", "texture"),
        _s("build", 8, 0.5, "pad", "chords", "counter", "strings", "perc"),
        _s("theme", 16, 0.8, "drums", "bass", "chords", "lead", "pad",
           "strings", "counter", "perc"),
        _s("climax", 16, 1.0, "drums", "bass", "chords", "chords2", "lead", "pad",
           "counter", "strings", "brass", "lead2", "sub", "perc"),
        _s("outro", 8, 0.3, "pad", "strings", "texture"),
    ],
}


def apply_final_lift(sections: list["Section"], semitones: int = 2) -> None:
    """Transpose the last occurrence of the most-repeated section.

    The final-chorus key change is a cliche because it works: it makes the last
    repeat feel like an arrival instead of a third identical pass.
    """
    if semitones == 0 or not sections:
        return
    counts: dict[str, int] = {}
    for sec in sections:
        counts[sec.name] = counts.get(sec.name, 0) + 1
    repeated = [n for n, c in counts.items() if c >= 3]
    if not repeated:
        return
    target = max(repeated, key=lambda n: counts[n])
    for sec in reversed(sections):
        if sec.name == target:
            sec.transpose = semitones
            break


def _clone(sec: Section) -> Section:
    return Section(sec.name, sec.bars, sec.intensity, sec.roles,
                   sec.progression, sec.transpose, sec.fill_at_end)


def build_form(
    template: str,
    target_seconds: float | None,
    bpm: float,
    beats_per_bar: int = 4,
    max_bars: int = 512,
    min_section_bars: int = 2,
) -> SongForm:
    """Instantiate a form, resized to land near ``target_seconds``.

    Resizing happens in two passes.  A coarse pass repeats or drops whole
    sections so the arrangement keeps a sensible shape, then a proportional
    pass nudges individual section lengths in even-bar steps to converge on the
    target.  Doing only the coarse pass undershoots badly on short targets;
    doing only the proportional pass turns a 9-section pop form into nine
    two-bar fragments.
    """
    base = [_clone(s) for s in FORMS.get(template, FORMS["verse_chorus"])]
    if not target_seconds or target_seconds <= 0:
        return SongForm(base)

    bars_per_second = bpm / (60.0 * beats_per_bar)
    target_bars = max(min_section_bars, min(max_bars, round(target_seconds * bars_per_second)))

    out = _coarse_resize(base, target_bars, max_bars)
    out = _proportional_resize(out, target_bars, min_section_bars)
    return SongForm(out)


def _coarse_resize(base: list[Section], target_bars: int, max_bars: int) -> list[Section]:
    """Add or remove whole sections while that moves us closer to target."""
    out = [_clone(s) for s in base]

    # Grow: repeat body sections (never intro/outro) while it helps.
    body = [s for s in base if s.name not in ("intro", "outro")] or base
    i = 0
    while len(out) < 64:
        total = sum(s.bars for s in out)
        if total >= target_bars:
            break
        src = body[i % len(body)]
        if total + src.bars > max_bars:
            break
        # Only add if the result is not further from target than staying put.
        if abs(total + src.bars - target_bars) >= abs(total - target_bars):
            break
        out.insert(max(1, len(out) - 1), _clone(src))
        i += 1

    # Shrink: drop duplicated middle sections while it helps.
    while len(out) > 2:
        total = sum(s.bars for s in out)
        if total <= target_bars:
            break
        names = [s.name for s in out]
        victim = None
        for j in range(len(out) - 2, 0, -1):
            if names.count(out[j].name) > 1:
                victim = j
                break
        if victim is None:
            break
        if abs(total - out[victim].bars - target_bars) >= abs(total - target_bars):
            break
        out.pop(victim)
    return out


def _proportional_resize(
    sections: list[Section], target_bars: int, min_bars: int
) -> list[Section]:
    """Scale section lengths toward the target, keeping even bar counts."""
    out = [_clone(s) for s in sections]
    total = sum(s.bars for s in out)
    if total == 0 or total == target_bars:
        return out

    scale = target_bars / total
    for sec in out:
        scaled = max(min_bars, round(sec.bars * scale / 2) * 2)
        sec.bars = scaled

    # Residual correction: nudge in 2-bar steps, taking from the longest
    # section when over and giving to the longest when under, so the dominant
    # sections absorb the difference instead of the short intro.
    guard = 0
    while guard < 512:
        guard += 1
        total = sum(s.bars for s in out)
        diff = target_bars - total
        if abs(diff) < 2:
            break
        if diff > 0:
            idx = max(range(len(out)), key=lambda i: out[i].bars)
            out[idx].bars += 2
        else:
            shrinkable = [i for i, s in enumerate(out) if s.bars - 2 >= min_bars]
            if not shrinkable:
                break
            idx = max(shrinkable, key=lambda i: out[i].bars)
            out[idx].bars -= 2
    return out
