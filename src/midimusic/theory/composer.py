"""The arranger: turns a style and a form into a complete multi-track Song.

This is the zero-download generation engine.  It needs no model weights, no
GPU and no network, so the app always produces music on any machine; the
neural backends layer on top of it rather than replacing it.

Density is controlled by ``complexity``.  Raising it does not just play the
same parts louder: it activates additional instrument layers, enriches the
voicings, subdivides the harmonic rhythm, inserts approach chords and varies
each repeat of a section, which is what a fuller arrangement actually means.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..core.models import Note, Song, Track
from .chords import CHORD_QUALITIES, Chord, realize_progression, voice_lead
from .humanize import Humanizer
from .melody import PHRASE_SHAPES, Melodist
from .pitch import Scale, parse_key, prefers_flats
from .rhythm import DRUM, DRUM_PATTERNS, build_bass_bar, build_comp_bar, build_drum_bar
from .structure import CORE_ROLES, Section, apply_final_lift, build_form
from .style import Style, get_style

__all__ = ["CompositionSpec", "Composer", "compose"]

STEPS_PER_BAR = 16

# How rich the arrangement must be before each enrichment layer switches on.
LAYER_THRESHOLDS = {
    "texture": 0.30,
    "perc": 0.35,
    "strings": 0.42,
    "sub": 0.48,
    "chords2": 0.56,
    "lead2": 0.62,
    "brass": 0.72,
}

# Extensions added to a quality when the style allows richer voicings.
_EXTENDED = {
    "maj": "add9", "min": "madd9", "maj7": "maj9", "min7": "min9",
    "dom7": "dom9", "maj6": "maj6", "min6": "min6",
}


@dataclass
class CompositionSpec:
    """Fully-resolved instructions for one composition."""

    style: Style
    scale: Scale
    tempo: float = 120.0
    duration_seconds: float | None = 60.0
    seed: int = 0
    form_name: str | None = None
    title: str = "Untitled"
    beats_per_bar: int = 4
    density: float | None = None
    complexity: float = 0.65   # 0 = sparse trio, 1 = full production
    brightness: float = 0.5    # 0 dark, 1 bright: nudges register and voicing
    energy: float = 0.5        # scales velocities and drum intensity
    modulate: bool = True      # lift the final repeated section
    roles: tuple[str, ...] | None = None
    prompt: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class _ChordEvent:
    chord: Chord
    start_beats: float
    duration_beats: float


@dataclass
class _BarPlan:
    index: int
    section: Section
    events: list[_ChordEvent]
    start_beats: float
    intensity: float
    is_section_end: bool
    is_phrase_end: bool
    variation: int = 0     # which repeat of this section we are in
    transpose: int = 0

    @property
    def chord(self) -> Chord:
        return self.events[0].chord


class Composer:
    def __init__(self, spec: CompositionSpec):
        self.spec = spec
        self.style = spec.style
        self.scale = spec.scale
        self.rng = random.Random(spec.seed)
        self.hum = Humanizer(
            amount=self.style.humanize, seed=spec.seed ^ 0x5EED, swing=self.style.swing
        )
        self.steps_per_beat = STEPS_PER_BAR // max(1, spec.beats_per_bar)
        self.complexity = max(0.0, min(1.0, spec.complexity))

    # -- planning -----------------------------------------------------------

    def _active(self, role: str, section: Section) -> bool:
        if role not in section.roles:
            return False
        if self.spec.roles is not None and role not in self.spec.roles:
            return False
        threshold = LAYER_THRESHOLDS.get(role)
        return threshold is None or self.complexity >= threshold

    def _enrich(self, chord: Chord) -> Chord:
        """Add an extension when the style calls for richer harmony."""
        if not self.style.extensions or self.complexity < 0.5:
            return chord
        upgraded = _EXTENDED.get(chord.quality)
        if not upgraded or upgraded not in CHORD_QUALITIES:
            return chord
        if self.rng.random() > 0.35 + 0.4 * self.complexity:
            return chord
        return Chord(chord.root, upgraded, chord.inversion, chord.bass,
                     chord.degree, chord.roman, chord.prefer_flat)

    def _bar_events(self, chord: Chord, start: float, split: bool,
                    next_chord: Chord | None) -> list[_ChordEvent]:
        """One or two chord events for a bar, depending on harmonic rhythm."""
        bpb = self.spec.beats_per_bar
        chord = self._enrich(chord)
        if not split or next_chord is None:
            return [_ChordEvent(chord, start, bpb)]
        # Approach the next chord from a half-bar away, either by its own
        # dominant or by a chromatic neighbour, so the motion has direction.
        if self.rng.random() < 0.55:
            approach = Chord((next_chord.root + 7) % 12, "dom7",
                             prefer_flat=next_chord.prefer_flat)
        else:
            step = -1 if self.rng.random() < 0.5 else 1
            approach = Chord((next_chord.root + step) % 12, next_chord.quality,
                             prefer_flat=next_chord.prefer_flat)
        return [
            _ChordEvent(chord, start, bpb / 2),
            _ChordEvent(approach, start + bpb / 2, bpb / 2),
        ]

    def _plan(self) -> list[_BarPlan]:
        form = build_form(
            self.spec.form_name or self.style.form,
            self.spec.duration_seconds,
            self.spec.tempo,
            self.spec.beats_per_bar,
            max_bars=2048,
        )
        if self.spec.modulate:
            apply_final_lift(form.sections, 2)

        bank = list(self.style.progressions) or [("I", "V", "vi", "IV")]
        plans: list[_BarPlan] = []
        bar_index = 0
        per_section: dict[str, list[Chord]] = {}
        seen_count: dict[str, int] = {}

        # Probability that a bar splits into two chords, rising with complexity.
        split_chance = max(0.0, (self.complexity - 0.5)) * 0.34

        for section in form.sections:
            if section.progression:
                chords = realize_progression(
                    list(section.progression), self.scale, self.style.sevenths
                )
            else:
                if section.name not in per_section:
                    symbols = list(bank[len(per_section) % len(bank)])
                    per_section[section.name] = realize_progression(
                        symbols, self.scale, self.style.sevenths
                    )
                chords = per_section[section.name]

            variation = seen_count.get(section.name, 0)
            seen_count[section.name] = variation + 1

            for b in range(section.bars):
                chord = chords[b % len(chords)]
                nxt = chords[(b + 1) % len(chords)]
                start = bar_index * self.spec.beats_per_bar
                # Split only where it will not muddy a cadence or an opening.
                split = (
                    b not in (0, section.bars - 1)
                    and self.rng.random() < split_chance
                    and section.intensity > 0.55
                )
                plans.append(
                    _BarPlan(
                        index=bar_index,
                        section=section,
                        events=self._bar_events(chord, start, split, nxt),
                        start_beats=start,
                        intensity=min(1.0, section.intensity * (0.6 + 0.8 * self.spec.energy)),
                        is_section_end=(b == section.bars - 1),
                        is_phrase_end=((b + 1) % 4 == 0),
                        variation=variation,
                        transpose=section.transpose,
                    )
                )
                bar_index += 1
        return plans

    # -- helpers ------------------------------------------------------------

    def _voicings(self, plans: list[_BarPlan], low: int, high: int, voices: int
                  ) -> list[list[list[int]]]:
        """Voice-lead across every chord event, returned grouped per bar."""
        flat: list[Chord] = []
        spans: list[int] = []
        for plan in plans:
            spans.append(len(plan.events))
            flat.extend(
                Chord(
                    (e.chord.root + plan.transpose) % 12, e.chord.quality,
                    e.chord.inversion, e.chord.bass, e.chord.degree,
                    e.chord.roman, e.chord.prefer_flat,
                )
                for e in plan.events
            )
        led = voice_lead(flat, low=low, high=high, voices=voices)
        out: list[list[list[int]]] = []
        i = 0
        for n in spans:
            out.append(led[i:i + n])
            i += n
        return out

    def _emit(self, track: Track, step: float, plan: _BarPlan, pitch: int,
              velocity_base: float, duration: float, laid_back: float = 0.0,
              accent: bool = True) -> None:
        t = step + self.hum.time(step, self.steps_per_beat, laid_back=laid_back)
        start = plan.start_beats + t / self.steps_per_beat
        velocity = self.hum.velocity(
            velocity_base, int(step), STEPS_PER_BAR, self.steps_per_beat, accent=accent
        )
        track.notes.append(Note(pitch, max(0.0, start), duration, velocity, track.channel))

    # -- percussion ---------------------------------------------------------

    def _drums(self, plans: list[_BarPlan]) -> Track:
        kit = DRUM_PATTERNS.get(self.style.drums, DRUM_PATTERNS["pop"])
        track = Track(name="Drums", program=0, channel=9, is_drum=True, role="drums")
        for plan in plans:
            if not self._active("drums", plan.section) or not kit.parts:
                continue
            fill = plan.is_section_end and plan.section.fill_at_end and plan.section.bars >= 4
            # Later repeats of a section get an extra mid-section fill.
            if not fill and plan.variation > 0 and plan.is_phrase_end:
                fill = self.rng.random() < 0.25 * self.complexity
            for step, vel, note in build_drum_bar(
                kit, STEPS_PER_BAR, plan.intensity, self.rng, fill=fill
            ):
                self._emit(track, step, plan, note, vel, 0.12)
        return track

    def _perc(self, plans: list[_BarPlan]) -> Track:
        """Auxiliary percussion, layered on the drum channel."""
        track = Track(name="Percussion", program=0, channel=9, is_drum=True, role="perc")
        patterns = [
            (DRUM["shaker"], "..x...x...x...x."),
            (DRUM["tambourine"], "....x.......x..."),
            (DRUM["cowbell"], "x.....x...x....."),
            (DRUM["woodblock"] if "woodblock" in DRUM else DRUM["sidestick"], "..x..x..x..x..x."),
        ]
        choice = patterns[self.rng.randrange(len(patterns))]
        note, pattern = choice
        from .rhythm import parse_steps

        vels = parse_steps(pattern, STEPS_PER_BAR)
        for plan in plans:
            if not self._active("perc", plan.section):
                continue
            for step, vel in enumerate(vels):
                if vel <= 0:
                    continue
                self._emit(track, step, plan, note, vel * 0.42 * plan.intensity, 0.1)
        return track

    # -- low end ------------------------------------------------------------

    def _bass(self, plans: list[_BarPlan], channel: int, sub: bool = False) -> Track:
        role = "sub" if sub else "bass"
        track = Track(
            name="Sub Bass" if sub else "Bass",
            program=self.style.program_for(role), channel=channel, role=role,
        )
        octave = self.style.bass_octave - (1 if sub else 0)
        for i, plan in enumerate(plans):
            if not self._active(role, plan.section):
                continue
            event = plan.events[0]
            chord = event.chord
            root_pc = (chord.root + plan.transpose) % 12
            tones = sorted(
                12 * (octave + 1) + ((pc + plan.transpose) % 12)
                for pc in chord.pitch_classes
            )
            nxt = plans[i + 1].chord if i + 1 < len(plans) else None
            next_root = (
                12 * (octave + 1) + ((nxt.root + plans[i + 1].transpose) % 12) if nxt else None
            )
            # The sub layer just reinforces roots; it must not play lines.
            style_name = "pedal" if sub else self.style.bass
            for step, vel, pitch in build_bass_bar(
                style_name, tones if not sub else [12 * (octave + 1) + root_pc],
                next_root, STEPS_PER_BAR, self.rng,
            ):
                dur = self.hum.duration(
                    self.spec.beats_per_bar / (1.2 if sub else 4.0), 0.85
                )
                self._emit(track, step, plan, pitch,
                           vel * (0.5 if sub else 0.78) * (0.75 + 0.3 * plan.intensity),
                           dur, laid_back=0.15)
        return track

    # -- harmony ------------------------------------------------------------

    def _comp(self, plans: list[_BarPlan], channel: int, role: str = "chords") -> Track:
        track = Track(
            name="Chords" if role == "chords" else "Chords 2",
            program=self.style.program_for(role), channel=channel, role=role,
        )
        octave = self.style.chord_octave + (1 if role == "chords2" else 0)
        low = 12 * (octave + 1)
        high = low + 22 + int(6 * self.spec.brightness)
        bars = self._voicings(plans, low, high, 4 if role == "chords" else 3)

        # The second comping voice deliberately plays a different rhythm.
        alt_styles = ["offbeat", "charleston", "stabs", "arpeggio", "eighths"]

        for plan, voicings in zip(plans, bars, strict=False):
            if not self._active(role, plan.section):
                continue
            comp_style = self.style.comp
            if role == "chords2":
                comp_style = alt_styles[(plan.index // 4) % len(alt_styles)]
            elif plan.variation > 0 and self.complexity > 0.5:
                # Vary the comping on later repeats rather than repeating verbatim.
                comp_style = {"quarters": "eighths", "sustained": "half",
                              "half": "quarters", "eighths": "offbeat",
                              "offbeat": "stabs"}.get(comp_style, comp_style)

            for event, voicing in zip(plan.events, voicings, strict=False):
                span = event.duration_beats / self.spec.beats_per_bar
                offset = (event.start_beats - plan.start_beats) * self.steps_per_beat
                for step, vel, pitch, length in build_comp_bar(
                    comp_style, voicing, STEPS_PER_BAR, self.rng
                ):
                    if step >= STEPS_PER_BAR * span:
                        continue
                    self._emit(
                        track, step + offset, plan, pitch,
                        vel * (0.45 if role == "chords2" else 0.6) * (0.6 + 0.4 * plan.intensity),
                        max(0.1, length * self.spec.beats_per_bar * span),
                    )
        return track

    def _sustained(self, plans: list[_BarPlan], channel: int, role: str) -> Track:
        """Pad, string section or texture: long tones under everything else."""
        gains = {"pad": 0.42, "strings": 0.38, "texture": 0.26}
        offsets = {"pad": 1, "strings": 1, "texture": 2}
        track = Track(
            name=role.capitalize(), program=self.style.program_for(role),
            channel=channel, role=role,
        )
        octave = self.style.chord_octave + offsets.get(role, 1)
        low = 12 * (octave + 1)
        voices = 3 if role != "texture" else 2
        bars = self._voicings(plans, low, low + 20, voices)

        for plan, voicings in zip(plans, bars, strict=False):
            if not self._active(role, plan.section):
                continue
            for event, voicing in zip(plan.events, voicings, strict=False):
                velocity = self.hum.velocity(
                    gains.get(role, 0.35) * (0.6 + 0.5 * plan.intensity), 0,
                    STEPS_PER_BAR, self.steps_per_beat, accent=False,
                )
                for pitch in voicing:
                    track.notes.append(
                        Note(pitch, event.start_beats, event.duration_beats * 0.98,
                             velocity, channel)
                    )
        return track

    def _brass(self, plans: list[_BarPlan], channel: int) -> Track:
        """Short section hits that punctuate rather than sustain."""
        track = Track(
            name="Brass", program=self.style.program_for("brass"),
            channel=channel, role="brass",
        )
        low = 12 * (self.style.chord_octave + 2)
        bars = self._voicings(plans, low, low + 18, 3)
        for plan, voicings in zip(plans, bars, strict=False):
            if not self._active("brass", plan.section):
                continue
            if not (plan.is_phrase_end or plan.index % 4 == 0):
                continue
            voicing = voicings[0]
            for step in (0, 6) if plan.is_phrase_end else (0,):
                for pitch in voicing:
                    self._emit(track, step, plan, pitch, 0.7 * plan.intensity, 0.45)
        return track

    def _arp(self, plans: list[_BarPlan], channel: int) -> Track:
        track = Track(
            name="Arp", program=self.style.program_for("arp"), channel=channel, role="arp",
        )
        low = 12 * (self.style.chord_octave + 2)
        bars = self._voicings(plans, low, low + 24, 4)
        for plan, voicings in zip(plans, bars, strict=False):
            if not self._active("arp", plan.section):
                continue
            style = "arp_16" if plan.intensity > 0.8 else "arpeggio"
            for event, voicing in zip(plan.events, voicings, strict=False):
                span = event.duration_beats / self.spec.beats_per_bar
                offset = (event.start_beats - plan.start_beats) * self.steps_per_beat
                for step, vel, pitch, length in build_comp_bar(
                    style, voicing, STEPS_PER_BAR, self.rng
                ):
                    if step >= STEPS_PER_BAR * span:
                        continue
                    self._emit(track, step + offset, plan, pitch,
                               vel * 0.55 * (0.6 + 0.5 * plan.intensity),
                               max(0.1, length * self.spec.beats_per_bar))
        return track

    # -- melody -------------------------------------------------------------

    def _melody(self, plans: list[_BarPlan], channel: int, role: str = "lead",
                octave_shift: int = 0) -> Track:
        track = Track(
            name={"lead": "Lead", "lead2": "Lead Double", "counter": "Counter"}[role],
            program=self.style.program_for(role), channel=channel, role=role,
        )
        lo, hi = self.style.melody_range
        if role == "counter":
            lo, hi = lo - 12, hi - 12
        shift = int((self.spec.brightness - 0.5) * 8) + octave_shift * 12
        density = self.spec.density if self.spec.density is not None else self.style.melody_density
        if role == "counter":
            density *= 0.6

        mel = Melodist(
            self.scale, self.rng, low=lo + shift, high=hi + shift,
            density=density, leapiness=self.style.melody_leapiness,
        )

        motifs: dict[str, object] = {}
        variant_counter: dict[str, int] = {}
        current: Section | None = None

        for plan in plans:
            if not self._active(role, plan.section):
                continue
            if plan.section is not current:
                current = plan.section
                mel.reset()
            key = plan.section.name
            if key not in motifs:
                motifs[key] = mel.make_motif()
                variant_counter[key] = 0

            base = motifs[key]
            bar_in_phrase = plan.index % 4
            if bar_in_phrase == 0 and plan.variation == 0:
                motif = base
            else:
                variant_counter[key] += 1
                # Later repeats develop the motif further from the original.
                motif = mel.develop(base, variant_counter[key] + plan.variation)

            shape = PHRASE_SHAPES["arch" if role != "counter" else "wave"]
            target = shape[bar_in_phrase % len(shape)]
            if role == "counter":
                target = 1.0 - target

            chord = Chord(
                (plan.chord.root + plan.transpose) % 12, plan.chord.quality,
                prefer_flat=plan.chord.prefer_flat,
            )
            for start, dur, pitch in mel.realize_bar(
                motif, chord, plan.start_beats, STEPS_PER_BAR, self.spec.beats_per_bar,
                shape_target=target, cadence=plan.is_phrase_end,
            ):
                step = (start - plan.start_beats) * self.steps_per_beat
                base_vel = {"lead": 0.8, "lead2": 0.5, "counter": 0.62}[role]
                self._emit(track, step, plan, pitch,
                           base_vel * (0.55 + 0.5 * plan.intensity),
                           self.hum.duration(dur, 0.9))
        return track

    # -- assembly -----------------------------------------------------------

    def compose(self) -> Song:
        plans = self._plan()
        if not plans:
            return Song(tempo=self.spec.tempo, title=self.spec.title)

        tracks: list[Track] = []
        channels = [ch for ch in range(16) if ch != 9]
        ci = 0

        def nxt() -> int:
            nonlocal ci
            ch = channels[ci % len(channels)]
            ci += 1
            return ch

        tracks.append(self._drums(plans))
        tracks.append(self._perc(plans))
        tracks.append(self._bass(plans, nxt()))
        tracks.append(self._bass(plans, nxt(), sub=True))
        tracks.append(self._comp(plans, nxt(), "chords"))
        tracks.append(self._comp(plans, nxt(), "chords2"))
        tracks.append(self._sustained(plans, nxt(), "pad"))
        tracks.append(self._sustained(plans, nxt(), "strings"))
        tracks.append(self._sustained(plans, nxt(), "texture"))
        tracks.append(self._arp(plans, nxt()))
        tracks.append(self._brass(plans, nxt()))
        tracks.append(self._melody(plans, nxt(), "lead"))
        tracks.append(self._melody(plans, nxt(), "lead2", octave_shift=1))
        tracks.append(self._melody(plans, nxt(), "counter"))

        for t in tracks:
            t.sort()

        song = Song(
            tracks=[t for t in tracks if t.notes],
            tempo=self.spec.tempo,
            beats_per_bar=self.spec.beats_per_bar,
            key=str(self.scale),
            title=self.spec.title,
        )

        bpb = self.spec.beats_per_bar
        flats = prefers_flats(self.scale)
        cursor = 0.0
        for name, bars in _section_runs(plans):
            song.sections.append((name, cursor, cursor + bars * bpb))
            cursor += bars * bpb

        last: tuple[str, float] | None = None
        for plan in plans:
            for event in plan.events:
                shifted = Chord(
                    (event.chord.root + plan.transpose) % 12, event.chord.quality,
                    prefer_flat=event.chord.prefer_flat,
                )
                sym = shifted.name(flats)
                if last is None:
                    last = (sym, event.start_beats)
                elif last[0] != sym:
                    song.chords.append((last[0], last[1], event.start_beats))
                    last = (sym, event.start_beats)
        if last is not None:
            song.chords.append((last[0], last[1], plans[-1].start_beats + bpb))

        song.meta.update({
            "style": self.style.name,
            "seed": self.spec.seed,
            "bars": len(plans),
            "complexity": self.complexity,
            "engine": "builtin-composer",
            "prompt": self.spec.prompt,
        })
        return song


def _section_runs(plans: list[_BarPlan]) -> list[tuple[str, int]]:
    """Collapse the bar plan into (section name, bar count) runs."""
    result: list[tuple[str, int]] = []
    current: Section | None = None
    count = 0
    for plan in plans:
        if plan.section is not current:
            if current is not None:
                result.append((current.name, count))
            current = plan.section
            count = 0
        count += 1
    if current is not None:
        result.append((current.name, count))
    return result


def compose(
    style_name: str = "pop",
    key: str = "C major",
    tempo: float | None = None,
    duration_seconds: float | None = 60.0,
    seed: int | None = None,
    **kw,
) -> Song:
    """Convenience wrapper used by the builtin generator and by tests."""
    style = get_style(style_name)
    rng = random.Random(seed)
    if tempo is None:
        tempo = rng.uniform(*style.tempo)
    spec = CompositionSpec(
        style=style,
        scale=Scale(*parse_key(key)),
        tempo=float(tempo),
        duration_seconds=duration_seconds,
        seed=seed if seed is not None else rng.randrange(1 << 30),
        **kw,
    )
    return Composer(spec).compose()
