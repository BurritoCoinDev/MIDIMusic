"""Humanisation: the difference between a MIDI file and a performance.

Quantised MIDI sounds like a machine because every note starts exactly on the
grid at exactly the same velocity.  These helpers reintroduce the small,
correlated deviations a player produces: swing, push and pull against the beat,
accent patterns tied to metrical position, and note lengths that vary with
articulation.
"""

from __future__ import annotations

import math
import random

__all__ = [
    "swing_offset",
    "metric_accent",
    "humanize_time",
    "humanize_velocity",
    "humanize_duration",
    "Humanizer",
    "crescendo",
]


def swing_offset(step: int, steps_per_beat: int, amount: float) -> float:
    """Delay for a swung off-beat, in steps.

    ``amount`` 0 is straight; 0.5 is a dotted-eighth/sixteenth triplet feel;
    0.66 approximates jazz swing.  Only odd subdivisions move.
    """
    if amount <= 0.0 or steps_per_beat < 2:
        return 0.0
    half = steps_per_beat // 2
    if half and step % steps_per_beat == half:
        # Push the "and" of the beat later by a fraction of the gap.
        return half * amount * 0.5
    if steps_per_beat >= 4 and step % 2 == 1:
        return 0.5 * amount * 0.25
    return 0.0


def metric_accent(step: int, steps_per_bar: int, steps_per_beat: int) -> float:
    """Velocity multiplier from metrical position (downbeats are loudest)."""
    if step % steps_per_bar == 0:
        return 1.0
    if steps_per_beat and step % steps_per_beat == 0:
        beat = step // steps_per_beat
        # Backbeat emphasis: beats 2 and 4 in common time.
        return 0.94 if beat % 2 == 1 else 0.88
    if steps_per_beat >= 4 and step % (steps_per_beat // 2) == 0:
        return 0.78
    return 0.7


class Humanizer:
    """Stateful humaniser so drift is correlated rather than white noise."""

    def __init__(self, amount: float = 0.5, seed: int | None = None, swing: float = 0.0):
        self.amount = max(0.0, min(1.0, amount))
        self.swing = swing
        self.rng = random.Random(seed)
        self._drift = 0.0
        self._vel_drift = 0.0

    def _step_drift(self) -> float:
        """A slow random walk, so timing wanders instead of jittering."""
        self._drift = self._drift * 0.82 + self.rng.gauss(0.0, 1.0) * 0.18
        return self._drift

    def _step_vel_drift(self) -> float:
        self._vel_drift = self._vel_drift * 0.75 + self.rng.gauss(0.0, 1.0) * 0.25
        return self._vel_drift

    def time(
        self,
        step: float,
        steps_per_beat: int,
        laid_back: float = 0.0,
    ) -> float:
        """Return an offset in steps to add to ``step``."""
        off = swing_offset(int(step), steps_per_beat, self.swing)
        # Up to ~7% of a step of random-walk drift at full humanisation.
        off += self._step_drift() * 0.07 * self.amount * steps_per_beat * 0.5
        off += laid_back * 0.06 * steps_per_beat
        return off

    def velocity(
        self,
        base: float,
        step: int,
        steps_per_bar: int,
        steps_per_beat: int,
        accent: bool = True,
    ) -> int:
        """Map a 0..1 velocity to a MIDI 1..127 value with accents and drift."""
        v = base
        if accent:
            v *= metric_accent(step, steps_per_bar, steps_per_beat)
        v *= 1.0 + self._step_vel_drift() * 0.13 * self.amount
        return max(1, min(127, int(round(v * 127))))

    def duration(self, nominal: float, articulation: float = 0.92) -> float:
        """Vary note length around ``nominal`` (in beats)."""
        jitter = 1.0 + self.rng.gauss(0.0, 0.045) * self.amount
        return max(0.02, nominal * articulation * jitter)

    def maybe(self, probability: float) -> bool:
        return self.rng.random() < probability


def humanize_time(step: float, amount: float, rng: random.Random, steps_per_beat: int = 4) -> float:
    return step + rng.gauss(0.0, 0.05) * amount * steps_per_beat


def humanize_velocity(v: float, amount: float, rng: random.Random) -> int:
    return max(1, min(127, int(round(v * 127 * (1.0 + rng.gauss(0.0, 0.09) * amount)))))


def humanize_duration(d: float, amount: float, rng: random.Random) -> float:
    return max(0.02, d * (1.0 + rng.gauss(0.0, 0.05) * amount))


def crescendo(index: int, total: int, start: float = 0.7, end: float = 1.0) -> float:
    """Smooth velocity ramp across ``total`` events, for builds."""
    if total <= 1:
        return end
    t = index / (total - 1)
    return start + (end - start) * (0.5 - 0.5 * math.cos(math.pi * t))
