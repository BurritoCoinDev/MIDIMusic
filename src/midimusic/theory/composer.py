"""The arranger: turns a style and a form into a complete multi-track Song.

This is the zero-download generation engine.  It needs no model weights, no
GPU and no network, so the app always produces music on any machine; the
neural backends layer on top of it rather than replacing it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..core.models import Note, Song, Track
from .chords import Chord, realize_progression, voice_lead
from .humanize import Humanizer, crescendo
from .melody import PHRASE_SHAPES, Melodist
from .pitch import Scale, parse_key, prefers_flats
from .rhythm import DRUM_PATTERNS, build_bass_bar, build_comp_bar, build_drum_bar
from .structure import Section, build_form
from .style import Style, get_style

__all__ = ["CompositionSpec", "Composer", "compose"]

STEPS_PER_BAR = 16


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
    brightness: float = 0.5   # 0 dark, 1 bright: nudges register and voicing
    energy: float = 0.5       # scales velocities and drum intensity
    roles: tuple[str, ...] | None = None  # force a role set
    prompt: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class _BarPlan:
    """What happens in one bar."""

    index: int
    section: Section
    chord: Chord
    start_beats: float
    intensity: float
    is_section_end: bool
    is_phrase_end: bool


class Composer:
    def __init__(self, spec: CompositionSpec):
        self.spec = spec
        self.style = spec.style
        self.scale = spec.scale
        self.rng = random.Random(spec.seed)
        self.hum = Humanizer(
            amount=self.style.humanize,
            seed=spec.seed ^ 0x5EED,
            swing=self.style.swing,
        )
        self.steps_per_beat = STEPS_PER_BAR // max(1, spec.beats_per_bar)

    # -- planning -----------------------------------------------------------

    def _plan(self) -> list[_BarPlan]:
        form = build_form(
            self.spec.form_name or self.style.form,
            self.spec.duration_seconds,
            self.spec.tempo,
            self.spec.beats_per_bar,
        )
        bank = list(self.style.progressions) or [("I", "V", "vi", "IV")]

        plans: list[_BarPlan] = []
        bar_index = 0
        # Give each distinct section name its own progression so the verse and
        # chorus differ harmonically, but a repeated chorus stays the same.
        per_section: dict[str, list[Chord]] = {}

        for section in form.sections:
            if section.progression:
                symbols = list(section.progression)
                chords = realize_progression(symbols, self.scale, self.style.sevenths)
            else:
                if section.name not in per_section:
                    symbols = list(bank[len(per_section) % len(bank)])
                    per_section[section.name] = realize_progression(
                        symbols, self.scale, self.style.sevenths
                    )
                chords = per_section[section.name]

            for b in range(section.bars):
                chord = chords[b % len(chords)]
                plans.append(
                    _BarPlan(
                        index=bar_index,
                        section=section,
                        chord=chord,
                        start_beats=bar_index * self.spec.beats_per_bar,
                        intensity=min(1.0, section.intensity * (0.6 + 0.8 * self.spec.energy)),
                        is_section_end=(b == section.bars - 1),
                        is_phrase_end=((b + 1) % 4 == 0),
                    )
                )
                bar_index += 1
        return plans

    # -- track builders -----------------------------------------------------

    def _drums(self, plans: list[_BarPlan]) -> Track:
        kit = DRUM_PATTERNS.get(self.style.drums, DRUM_PATTERNS["pop"])
        track = Track(name="Drums", program=0, channel=9, is_drum=True, role="drums")
        for plan in plans:
            if "drums" not in plan.section.roles or not kit.parts:
                continue
            fill = plan.is_section_end and plan.section.fill_at_end and plan.section.bars >= 4
            for step, vel, note in build_drum_bar(
                kit, STEPS_PER_BAR, plan.intensity, self.rng, fill=fill
            ):
                t = step + self.hum.time(step, self.steps_per_beat)
                start = plan.start_beats + t / self.steps_per_beat
                velocity = self.hum.velocity(vel, step, STEPS_PER_BAR, self.steps_per_beat)
                track.notes.append(Note(note, max(0.0, start), 0.12, velocity, 9))
        return track

    def _bass(self, plans: list[_BarPlan], channel: int) -> Track:
        track = Track(
            name="Bass", program=self.style.programs.get("bass", 33),
            channel=channel, role="bass",
        )
        octave = self.style.bass_octave
        for i, plan in enumerate(plans):
            if "bass" not in plan.section.roles:
                continue
            chord = plan.chord
            tones = [12 * (octave + 1) + pc for pc in chord.pitch_classes]
            tones.sort()
            nxt = plans[i + 1].chord if i + 1 < len(plans) else None
            next_root = 12 * (octave + 1) + nxt.root if nxt else None

            for step, vel, pitch in build_bass_bar(
                self.style.bass, tones, next_root, STEPS_PER_BAR, self.rng
            ):
                t = step + self.hum.time(step, self.steps_per_beat, laid_back=0.15)
                start = plan.start_beats + t / self.steps_per_beat
                velocity = self.hum.velocity(
                    vel * (0.75 + 0.3 * plan.intensity), step, STEPS_PER_BAR, self.steps_per_beat
                )
                dur = self.hum.duration(self.spec.beats_per_bar / 4.0, 0.85)
                track.notes.append(Note(pitch, max(0.0, start), dur, velocity, channel))
        return track

    def _chords(self, plans: list[_BarPlan], channel: int) -> Track:
        track = Track(
            name="Chords", program=self.style.programs.get("chords", 0),
            channel=channel, role="chords",
        )
        octave = self.style.chord_octave
        low = 12 * (octave + 1)
        high = low + 22 + int(6 * self.spec.brightness)
        voicings = voice_lead([p.chord for p in plans], low=low, high=high, voices=4)

        for plan, voicing in zip(plans, voicings, strict=False):
            if "chords" not in plan.section.roles:
                continue
            for step, vel, pitch, length in build_comp_bar(
                self.style.comp, voicing, STEPS_PER_BAR, self.rng
            ):
                t = step + self.hum.time(step, self.steps_per_beat)
                start = plan.start_beats + t / self.steps_per_beat
                velocity = self.hum.velocity(
                    vel * (0.6 + 0.4 * plan.intensity), int(step), STEPS_PER_BAR,
                    self.steps_per_beat,
                )
                dur = max(0.1, length * self.spec.beats_per_bar)
                track.notes.append(Note(pitch, max(0.0, start), dur, velocity, channel))
        return track

    def _pad(self, plans: list[_BarPlan], channel: int) -> Track:
        track = Track(
            name="Pad", program=self.style.programs.get("pad", 89),
            channel=channel, role="pad",
        )
        octave = self.style.chord_octave + 1
        low = 12 * (octave + 1)
        voicings = voice_lead([p.chord for p in plans], low=low, high=low + 20, voices=3)
        for plan, voicing in zip(plans, voicings, strict=False):
            if "pad" not in plan.section.roles:
                continue
            velocity = self.hum.velocity(0.42 * (0.6 + 0.5 * plan.intensity), 0,
                                         STEPS_PER_BAR, self.steps_per_beat, accent=False)
            for pitch in voicing:
                track.notes.append(
                    Note(pitch, plan.start_beats, self.spec.beats_per_bar * 0.98,
                         velocity, channel)
                )
        return track

    def _arp(self, plans: list[_BarPlan], channel: int) -> Track:
        track = Track(
            name="Arp", program=self.style.programs.get("arp", 80),
            channel=channel, role="arp",
        )
        octave = self.style.chord_octave + 1
        low = 12 * (octave + 1)
        voicings = voice_lead([p.chord for p in plans], low=low, high=low + 24, voices=4)
        for plan, voicing in zip(plans, voicings, strict=False):
            if "arp" not in plan.section.roles:
                continue
            style = "arp_16" if plan.intensity > 0.8 else "arpeggio"
            for step, vel, pitch, length in build_comp_bar(
                style, voicing, STEPS_PER_BAR, self.rng
            ):
                t = step + self.hum.time(step, self.steps_per_beat)
                start = plan.start_beats + t / self.steps_per_beat
                velocity = self.hum.velocity(
                    vel * 0.55 * (0.6 + 0.5 * plan.intensity), int(step),
                    STEPS_PER_BAR, self.steps_per_beat,
                )
                track.notes.append(
                    Note(pitch, max(0.0, start), max(0.1, length * self.spec.beats_per_bar),
                         velocity, channel)
                )
        return track

    def _lead(self, plans: list[_BarPlan], channel: int, role: str = "lead") -> Track:
        program = self.style.programs.get(role, 81)
        track = Track(name=role.capitalize(), program=program, channel=channel, role=role)

        lo, hi = self.style.melody_range
        if role == "counter":
            lo, hi = lo - 12, hi - 12
        shift = int((self.spec.brightness - 0.5) * 8)
        density = self.spec.density if self.spec.density is not None else self.style.melody_density
        if role == "counter":
            density *= 0.6

        mel = Melodist(
            self.scale, self.rng, low=lo + shift, high=hi + shift,
            density=density, leapiness=self.style.melody_leapiness,
        )

        # One motif per distinct section name, developed across its bars, so a
        # returning chorus brings its tune back rather than a new one.
        motifs: dict[str, object] = {}
        variant_counter: dict[str, int] = {}
        current_section: Section | None = None

        for plan in plans:
            if role not in plan.section.roles:
                continue
            if plan.section is not current_section:
                current_section = plan.section
                mel.reset()
            key = plan.section.name
            if key not in motifs:
                motifs[key] = mel.make_motif()
                variant_counter[key] = 0

            base = motifs[key]
            bar_in_phrase = plan.index % 4
            if bar_in_phrase == 0:
                motif = base
            else:
                variant_counter[key] += 1
                motif = mel.develop(base, variant_counter[key])

            shape = PHRASE_SHAPES["arch" if role == "lead" else "wave"]
            target = shape[bar_in_phrase % len(shape)]
            if role == "counter":
                target = 1.0 - target

            for start, dur, pitch in mel.realize_bar(
                motif, plan.chord, plan.start_beats, STEPS_PER_BAR,
                self.spec.beats_per_bar, shape_target=target,
                cadence=plan.is_phrase_end,
            ):
                jitter = self.hum.time(int((start - plan.start_beats) * self.steps_per_beat),
                                       self.steps_per_beat)
                s = start + jitter / self.steps_per_beat
                base_vel = (0.62 if role == "counter" else 0.8) * (0.55 + 0.5 * plan.intensity)
                velocity = self.hum.velocity(
                    base_vel, int((start - plan.start_beats) * self.steps_per_beat),
                    STEPS_PER_BAR, self.steps_per_beat,
                )
                track.notes.append(
                    Note(pitch, max(0.0, s), self.hum.duration(dur, 0.9), velocity, channel)
                )
        return track

    # -- assembly -----------------------------------------------------------

    def compose(self) -> Song:
        plans = self._plan()
        if not plans:
            return Song(tempo=self.spec.tempo, title=self.spec.title)

        allowed = set(self.spec.roles) if self.spec.roles else None

        def wanted(role: str) -> bool:
            return allowed is None or role in allowed

        tracks: list[Track] = []
        # Channel 9 is reserved for percussion in General MIDI.
        channels = [ch for ch in range(16) if ch != 9]
        ci = 0

        def next_channel() -> int:
            nonlocal ci
            ch = channels[ci % len(channels)]
            ci += 1
            return ch

        if wanted("drums"):
            tracks.append(self._drums(plans))
        if wanted("bass"):
            tracks.append(self._bass(plans, next_channel()))
        if wanted("chords"):
            tracks.append(self._chords(plans, next_channel()))
        if wanted("pad"):
            tracks.append(self._pad(plans, next_channel()))
        if wanted("arp"):
            tracks.append(self._arp(plans, next_channel()))
        if wanted("lead"):
            tracks.append(self._lead(plans, next_channel(), "lead"))
        if wanted("counter"):
            tracks.append(self._lead(plans, next_channel(), "counter"))

        for t in tracks:
            t.sort()

        song = Song(
            tracks=[t for t in tracks if t.notes],
            tempo=self.spec.tempo,
            beats_per_bar=self.spec.beats_per_bar,
            key=str(self.scale),
            title=self.spec.title,
        )

        # Annotate sections and chords for the UI and for MIDI markers.
        bpb = self.spec.beats_per_bar
        flats = prefers_flats(self.scale)
        cursor = 0.0
        for section in _dedupe_sections(plans):
            name, bars = section
            song.sections.append((name, cursor, cursor + bars * bpb))
            cursor += bars * bpb
        last: tuple[str, float] | None = None
        for plan in plans:
            sym = plan.chord.name(flats)
            if last is None or last[0] != sym:
                if last is not None:
                    song.chords.append((last[0], last[1], plan.start_beats))
                last = (sym, plan.start_beats)
        if last is not None:
            song.chords.append((last[0], last[1], plans[-1].start_beats + bpb))

        song.meta.update(
            {
                "style": self.style.name,
                "seed": self.spec.seed,
                "bars": len(plans),
                "engine": "builtin-composer",
                "prompt": self.spec.prompt,
            }
        )
        return song


def _dedupe_sections(plans: list[_BarPlan]) -> list[tuple[str, int]]:
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


def compose(self) -> Song:
        plans = self._plan()
        if not plans:
            return Song(tempo=self.spec.tempo, title=self.spec.title)

        allowed = set(self.spec.roles) if self.spec.roles else None

        def wanted(role: str) -> bool:
            return allowed is None or role in allowed

        tracks: list[Track] = []
        # Channel 9 is reserved for percussion in General MIDI.
        channels = [ch for ch in range(16) if ch != 9]
        ci = 0

        def next_channel() -> int:
            nonlocal ci
            ch = channels[ci % len(channels)]
            ci += 1
            return ch

        if wanted("drums"):
            tracks.append(self._drums(plans))
        if wanted("bass"):
            tracks.append(self._bass(plans, next_channel()))
        if wanted("chords"):
            tracks.append(self._chords(plans, next_channel()))
        if wanted("pad"):
            tracks.append(self._pad(plans, next_channel()))
        if wanted("arp"):
            tracks.append(self._arp(plans, next_channel()))
        if wanted("lead"):
            tracks.append(self._lead(plans, next_channel(), "lead"))
        if wanted("counter"):
            tracks.append(self._lead(plans, next_channel(), "counter"))

        for t in tracks:
            t.sort()

        song = Song(
            tracks=[t for t in tracks if t.notes],
            tempo=self.spec.tempo,
            beats_per_bar=self.spec.beats_per_bar,
            key=str(self.scale),
            title=self.spec.title,
        )

        # Annotate sections and chords for the UI and for MIDI markers.
        bpb = self.spec.beats_per_bar
        flats = prefers_flats(self.scale)
        cursor = 0.0
        for section in _dedupe_sections(plans):
            name, bars = section
            song.sections.append((name, cursor, cursor + bars * bpb))
            cursor += bars * bpb
        last: tuple[str, float] | None = None
        for plan in plans:
            sym = plan.chord.name(flats)
            if last is None or last[0] != sym:
                if last is not None:
                    song.chords.append((last[0], last[1], plan.start_beats))
                last = (sym, plan.start_beats)
        if last is not None:
            song.chords.append((last[0], last[1], plans[-1].start_beats + bpb))

        song.meta.update(
            {
                "style": self.style.name,
                "seed": self.spec.seed,
                "bars": len(plans),
                "engine": "builtin-composer",
                "prompt": self.spec.prompt,
            }
        )
        return song


def _dedupe_sections(plans: list[_BarPlan]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for plan in plans:
        if out and out[-1][0] == plan.section.name and not plan.index == 0:
            # Only merge when it is genuinely the same Section object run.
            pass
        if not out or plan.is_section_end is None:
            pass
    # Rebuild by walking section identity.
    result: list[tuple[str, int]] = []
    current = None
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
