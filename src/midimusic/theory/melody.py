"""Melody writing by motif and development.

Sampling notes one at a time from a scale produces noodling, not tunes.  What
makes a melody sound composed is that a short idea comes back changed: moved to
fit the new chord, turned upside down, stretched, or answered.  So this module
generates a motif once per section and then develops it, rather than making
independent choices bar by bar.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .chords import Chord
from .pitch import Scale

__all__ = ["PHRASE_SHAPES", "Melodist", "Motif"]

# Rhythmic cells on a 16th grid, as (step, length_in_steps) pairs for one bar.
_RHYTHM_CELLS: list[list[tuple[int, int]]] = [
    [(0, 4), (4, 4), (8, 4), (12, 4)],
    [(0, 8), (8, 8)],
    [(0, 4), (4, 2), (6, 2), (8, 4), (12, 4)],
    [(0, 6), (6, 2), (8, 6), (14, 2)],
    [(0, 2), (2, 2), (4, 4), (8, 2), (10, 2), (12, 4)],
    [(0, 3), (3, 3), (6, 2), (8, 4), (12, 4)],
    [(0, 12), (12, 4)],
    [(0, 4), (6, 2), (8, 4), (12, 4)],
    [(2, 2), (4, 4), (8, 4), (12, 4)],
    [(0, 16)],
    [(0, 2), (4, 2), (8, 2), (12, 2)],
    [(0, 4), (4, 4), (8, 8)],
]

# Contour templates over a phrase: relative shape the melody should trace.
PHRASE_SHAPES = {
    "arch": (0.0, 0.5, 1.0, 0.6),
    "descend": (1.0, 0.7, 0.4, 0.0),
    "ascend": (0.0, 0.35, 0.7, 1.0),
    "wave": (0.3, 0.8, 0.2, 0.6),
    "static": (0.4, 0.5, 0.45, 0.5),
    "call": (0.2, 0.6, 0.3, 0.75),
}


@dataclass
class Motif:
    """A rhythm plus a contour in scale degrees, relative to a chord root."""

    rhythm: list[tuple[int, int]]      # (step, length) in 16ths within a bar
    degrees: list[int]                 # offsets in scale steps from an anchor
    anchor: int = 0                    # scale-degree anchor

    def transposed(self, delta: int) -> Motif:
        return Motif(list(self.rhythm), [d + delta for d in self.degrees], self.anchor)

    def inverted(self) -> Motif:
        if not self.degrees:
            return self
        pivot = self.degrees[0]
        return Motif(list(self.rhythm), [2 * pivot - d for d in self.degrees], self.anchor)

    def retrograde(self) -> Motif:
        return Motif(list(self.rhythm), list(reversed(self.degrees)), self.anchor)

    def augmented(self, steps_per_bar: int = 16) -> Motif:
        """Stretch the rhythm to twice the length, clipped to the bar."""
        out = [
            (step * 2, min(length * 2, steps_per_bar))
            for step, length in self.rhythm
            if step * 2 < steps_per_bar
        ]
        return Motif(out or list(self.rhythm), list(self.degrees), self.anchor)

    def sparser(self, keep: float, rng: random.Random) -> Motif:
        """Thin the motif out, always keeping the first note."""
        if len(self.rhythm) <= 1:
            return self
        kept = [self.rhythm[0]] + [c for c in self.rhythm[1:] if rng.random() < keep]
        return Motif(kept, list(self.degrees), self.anchor)


class Melodist:
    """Generates and develops melodies over a chord plan."""

    def __init__(
        self,
        scale: Scale,
        rng: random.Random,
        low: int = 60,
        high: int = 84,
        density: float = 0.55,
        leapiness: float = 0.3,
    ):
        self.scale = scale
        self.rng = rng
        self.low = low
        self.high = high
        self.density = max(0.05, min(1.0, density))
        self.leapiness = max(0.0, min(1.0, leapiness))
        self._last_pitch: int | None = None

    # -- motif construction -------------------------------------------------

    def make_motif(self, length_bars: int = 1) -> Motif:
        cell = list(self.rng.choice(_RHYTHM_CELLS))
        # Thin toward the requested density.
        if self.density < 0.9 and len(cell) > 2:
            keep = 0.35 + 0.65 * self.density
            cell = [cell[0]] + [c for c in cell[1:] if self.rng.random() < keep]

        degrees: list[int] = [0]
        for _ in range(len(cell) - 1):
            if self.rng.random() < self.leapiness:
                step = self.rng.choice([-4, -3, -2, 2, 3, 4])
            else:
                step = self.rng.choice([-2, -1, -1, 1, 1, 2])
            nxt = degrees[-1] + step
            # Keep the motif inside a singable span of about an octave and a half.
            if abs(nxt) > 6:
                nxt = degrees[-1] - step
            degrees.append(nxt)
        return Motif(cell, degrees)

    def develop(self, motif: Motif, variant: int) -> Motif:
        """Return a recognisable variation of ``motif``."""
        choices = [
            lambda m: m,
            lambda m: m.transposed(self.rng.choice([-2, -1, 1, 2])),
            lambda m: m.inverted(),
            lambda m: m.transposed(self.rng.choice([2, 3, -3])),
            lambda m: m.retrograde(),
            lambda m: m.sparser(0.65, self.rng),
            lambda m: m.augmented(),
        ]
        # Early variants stay close to the original; later ones range further.
        pool = choices[: 3 + min(4, variant)]
        return self.rng.choice(pool)(motif)

    # -- realisation --------------------------------------------------------

    def realize_bar(
        self,
        motif: Motif,
        chord: Chord,
        bar_start_beats: float,
        steps_per_bar: int = 16,
        beats_per_bar: int = 4,
        shape_target: float = 0.5,
        cadence: bool = False,
    ) -> list[tuple[float, float, int]]:
        """Render one bar of the motif against ``chord``.

        Returns ``(start_beats, duration_beats, pitch)``.  Notes landing on
        strong beats are pulled onto chord tones so the line agrees with the
        harmony; weaker positions may sit on passing scale tones.
        """
        out: list[tuple[float, float, int]] = []
        if not motif.rhythm:
            return out

        step_beats = beats_per_bar / steps_per_bar
        # Anchor the phrase in the register according to the contour target.
        span = self.high - self.low
        centre = self.low + span * (0.25 + 0.5 * shape_target)
        anchor_pitch = self._nearest_chord_tone(int(centre), chord)

        for i, (step, length) in enumerate(motif.rhythm):
            deg = motif.degrees[i % len(motif.degrees)] if motif.degrees else 0
            pitch = self._degree_from(anchor_pitch, deg)

            strong = (step % (steps_per_bar // max(1, beats_per_bar))) == 0
            is_last = i == len(motif.rhythm) - 1

            if strong or (cadence and is_last):
                pitch = self._nearest_chord_tone(pitch, chord)
            else:
                pitch = self.scale.quantize(pitch, prefer=1 if deg >= 0 else -1)

            if cadence and is_last:
                # Land on the root or third for a sense of arrival.
                target_pcs = [chord.root, (chord.root + chord.intervals[1]) % 12
                              if len(chord.intervals) > 1 else chord.root]
                pitch = self._nearest_pc(pitch, target_pcs)

            pitch = self._fit_range(pitch)
            pitch = self._smooth(pitch)

            start = bar_start_beats + step * step_beats
            dur = max(step_beats * 0.5, length * step_beats * 0.92)
            out.append((start, dur, pitch))
            self._last_pitch = pitch
        return out

    # -- helpers ------------------------------------------------------------

    def _degree_from(self, pitch: int, degree_offset: int) -> int:
        """Move ``degree_offset`` scale steps from ``pitch``."""
        idx = self.scale.nearest_degree(pitch)
        octave = (pitch - self.scale.tonic) // 12 - 1
        return self.scale.degree_to_pitch(idx + degree_offset, octave)

    def _nearest_chord_tone(self, pitch: int, chord: Chord) -> int:
        return self._nearest_pc(pitch, list(chord.pitch_classes))

    def _nearest_pc(self, pitch: int, pcs: list[int]) -> int:
        if not pcs:
            return pitch
        best, best_d = pitch, 99
        for pc in pcs:
            base = pitch - (pitch % 12) + pc
            for cand in (base - 12, base, base + 12):
                d = abs(cand - pitch)
                if d < best_d:
                    best, best_d = cand, d
        return best

    def _fit_range(self, pitch: int) -> int:
        while pitch < self.low:
            pitch += 12
        while pitch > self.high:
            pitch -= 12
        return pitch

    def _smooth(self, pitch: int) -> int:
        """Fold octave-wide jumps back in, unless a leap was intended."""
        if self._last_pitch is None:
            return pitch
        gap = pitch - self._last_pitch
        if abs(gap) > 12 and self.rng.random() > self.leapiness:
            adjusted = pitch - 12 if gap > 0 else pitch + 12
            if self.low <= adjusted <= self.high:
                return adjusted
        return pitch

    def reset(self) -> None:
        self._last_pitch = None
