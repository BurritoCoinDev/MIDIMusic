"""A small numpy synthesiser used when no SoundFont is installed.

This exists so the app can always produce audio -- on a fresh install, offline,
with no downloads at all.  It is a subtractive/additive hybrid: harmonically
shaped oscillators per instrument family, an ADSR envelope, a one-pole filter
and synthesised percussion.  It will not be mistaken for a sampled orchestra,
but it is musical, and it makes the FLAC path work out of the box.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.models import AudioBuffer, Song, Track
from ..theory.pitch import midi_to_freq

__all__ = ["render_song_fallback", "VOICES"]


# Harmonic recipes: (harmonic number, amplitude) plus envelope and character.
VOICES: dict[str, dict] = {
    "piano":  {"harmonics": [(1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (6, 0.05)],
               "attack": 0.004, "decay": 0.55, "sustain": 0.28, "release": 0.25,
               "detune": 0.0, "brightness": 0.7},
    "epiano": {"harmonics": [(1, 1.0), (2, 0.3), (3, 0.5), (5, 0.12)],
               "attack": 0.006, "decay": 0.8, "sustain": 0.35, "release": 0.35,
               "detune": 0.002, "brightness": 0.6},
    "organ":  {"harmonics": [(1, 1.0), (2, 0.7), (3, 0.5), (4, 0.35), (6, 0.2), (8, 0.12)],
               "attack": 0.02, "decay": 0.05, "sustain": 0.95, "release": 0.08,
               "detune": 0.001, "brightness": 0.8},
    "guitar": {"harmonics": [(1, 1.0), (2, 0.55), (3, 0.4), (4, 0.25), (5, 0.15), (7, 0.08)],
               "attack": 0.005, "decay": 0.5, "sustain": 0.3, "release": 0.3,
               "detune": 0.003, "brightness": 0.75},
    "bass":   {"harmonics": [(1, 1.0), (2, 0.35), (3, 0.12), (4, 0.05)],
               "attack": 0.008, "decay": 0.4, "sustain": 0.55, "release": 0.15,
               "detune": 0.0, "brightness": 0.35},
    "synth":  {"harmonics": [(1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25), (5, 0.2), (6, 0.16)],
               "attack": 0.01, "decay": 0.3, "sustain": 0.7, "release": 0.2,
               "detune": 0.004, "brightness": 0.85},
    "pad":    {"harmonics": [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.18), (5, 0.1)],
               "attack": 0.35, "decay": 0.6, "sustain": 0.8, "release": 0.9,
               "detune": 0.006, "brightness": 0.5},
    "strings": {"harmonics": [(1, 1.0), (2, 0.6), (3, 0.4), (4, 0.28), (5, 0.18), (6, 0.1)],
                "attack": 0.12, "decay": 0.4, "sustain": 0.85, "release": 0.45,
                "detune": 0.005, "brightness": 0.65},
    "brass":  {"harmonics": [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.42), (5, 0.3), (6, 0.2)],
               "attack": 0.045, "decay": 0.25, "sustain": 0.8, "release": 0.18,
               "detune": 0.002, "brightness": 0.9},
    "reed":   {"harmonics": [(1, 1.0), (3, 0.5), (5, 0.3), (7, 0.16), (9, 0.08)],
               "attack": 0.03, "decay": 0.3, "sustain": 0.78, "release": 0.2,
               "detune": 0.001, "brightness": 0.7},
    "flute":  {"harmonics": [(1, 1.0), (2, 0.18), (3, 0.06)],
               "attack": 0.06, "decay": 0.2, "sustain": 0.9, "release": 0.2,
               "detune": 0.0, "brightness": 0.45},
    "bell":   {"harmonics": [(1, 1.0), (2.76, 0.5), (5.4, 0.25), (8.9, 0.12)],
               "attack": 0.002, "decay": 1.2, "sustain": 0.05, "release": 0.8,
               "detune": 0.0, "brightness": 0.95},
    "pluck":  {"harmonics": [(1, 1.0), (2, 0.4), (3, 0.2), (4, 0.1)],
               "attack": 0.002, "decay": 0.35, "sustain": 0.1, "release": 0.2,
               "detune": 0.0, "brightness": 0.6},
}


def _voice_for_program(program: int, role: str) -> str:
    """Pick a synth voice from a General MIDI program number."""
    if role == "bass":
        return "bass"
    if role == "pad":
        return "pad"
    table = [
        (7, "piano"), (15, "bell"), (23, "organ"), (31, "guitar"), (39, "bass"),
        (47, "strings"), (55, "strings"), (63, "brass"), (71, "reed"),
        (79, "flute"), (87, "synth"), (95, "pad"), (103, "synth"),
        (111, "pluck"), (119, "bell"), (127, "synth"),
    ]
    for limit, name in table:
        if program <= limit:
            return name
    return "synth"


def _adsr(n: int, sr: int, spec: dict, sustain_frames: int) -> np.ndarray:
    a = max(1, int(spec["attack"] * sr))
    d = max(1, int(spec["decay"] * sr))
    r = max(1, int(spec["release"] * sr))
    s_level = float(spec["sustain"])

    env = np.zeros(n, dtype=np.float32)
    i = 0
    a = min(a, n)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    i = a
    if i < n:
        d_end = min(n, i + d)
        env[i:d_end] = np.linspace(1.0, s_level, d_end - i, dtype=np.float32)
        i = d_end
    hold_end = min(n, max(i, sustain_frames))
    if i < hold_end:
        env[i:hold_end] = s_level
        i = hold_end
    if i < n:
        start_level = env[i - 1] if i > 0 else s_level
        env[i:] = np.linspace(float(start_level), 0.0, n - i, dtype=np.float32)
    return env


def _render_note(freq: float, frames: int, sr: int, spec: dict,
                 velocity: float, sustain_frames: int) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / sr
    out = np.zeros(frames, dtype=np.float32)
    detune = spec["detune"]

    for harmonic, amp in spec["harmonics"]:
        f = freq * harmonic
        if f >= sr * 0.45:  # stay below Nyquist to avoid aliasing
            continue
        # Roll off high harmonics on quiet notes, as a real instrument does.
        h_amp = amp * (0.35 + 0.65 * velocity) ** (0.4 * harmonic)
        out += h_amp * np.sin(2 * np.pi * f * t, dtype=np.float32)
        if detune > 0:
            out += 0.5 * h_amp * np.sin(2 * np.pi * f * (1.0 + detune) * t, dtype=np.float32)

    norm = sum(a for _, a in spec["harmonics"]) * (1.5 if detune > 0 else 1.0)
    out /= max(1e-6, norm)
    out *= _adsr(frames, sr, spec, sustain_frames)
    return out * velocity


def _render_drum(note: int, frames: int, sr: int, velocity: float,
                 rng: np.random.Generator) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / sr
    if note in (35, 36):  # kick
        f = 110.0 * np.exp(-t * 28.0) + 42.0
        env = np.exp(-t * 16.0, dtype=np.float32)
        sig = np.sin(2 * np.pi * np.cumsum(f) / sr, dtype=np.float32) * env
    elif note in (38, 40, 37, 39):  # snare / clap / rim
        noise = rng.standard_normal(frames).astype(np.float32)
        env = np.exp(-t * (26.0 if note != 39 else 18.0), dtype=np.float32)
        tone = np.sin(2 * np.pi * 190.0 * t, dtype=np.float32) * np.exp(-t * 34.0)
        sig = (noise * 0.75 + tone * 0.45) * env
    elif note in (42, 44, 46, 51, 53, 59):  # hats and cymbals
        noise = rng.standard_normal(frames).astype(np.float32)
        decay = 55.0 if note in (42, 44) else 8.0
        env = np.exp(-t * decay, dtype=np.float32)
        # Crude high-pass: subtract a smoothed copy.
        hp = noise - np.convolve(noise, np.ones(12, dtype=np.float32) / 12, mode="same")
        sig = hp * env * 0.6
    elif note in (49, 57, 55):  # crash
        noise = rng.standard_normal(frames).astype(np.float32)
        env = np.exp(-t * 3.2, dtype=np.float32)
        hp = noise - np.convolve(noise, np.ones(8, dtype=np.float32) / 8, mode="same")
        sig = hp * env * 0.55
    elif 41 <= note <= 48:  # toms
        base = 220.0 - (note - 41) * 14.0
        f = base * np.exp(-t * 9.0) + base * 0.55
        env = np.exp(-t * 11.0, dtype=np.float32)
        sig = np.sin(2 * np.pi * np.cumsum(f) / sr, dtype=np.float32) * env
    else:  # shakers, cowbell, misc percussion
        noise = rng.standard_normal(frames).astype(np.float32)
        env = np.exp(-t * 40.0, dtype=np.float32)
        sig = noise * env * 0.4
    return (sig * velocity).astype(np.float32)


def _pan_for(track: Track, index: int) -> float:
    """Spread instruments across the stereo field, keeping the core centred."""
    if track.is_drum or track.role in ("bass", "chords"):
        return 0.0
    spread = {"lead": 0.12, "pad": -0.3, "arp": 0.34, "counter": -0.22}
    return spread.get(track.role, ((index % 3) - 1) * 0.18)


def render_song_fallback(
    song: Song,
    sample_rate: int = 44100,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    seed: int = 0,
) -> AudioBuffer:
    """Render a song with the built-in synth. Always available."""
    rng = np.random.default_rng(seed)
    spb = 60.0 / max(1e-6, song.tempo)
    total_seconds = song.duration_seconds + 2.0
    n = max(1, int(total_seconds * sample_rate))
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    tracks = song.non_empty_tracks()
    total_notes = max(1, sum(len(t.notes) for t in tracks))
    done = 0

    for ti, track in enumerate(tracks):
        voice_name = _voice_for_program(track.program, track.role)
        spec = VOICES[voice_name]
        pan = _pan_for(track, ti)
        gain = {"drums": 0.85, "bass": 0.9, "chords": 0.5, "pad": 0.36,
                "arp": 0.4, "lead": 0.62, "counter": 0.4}.get(track.role, 0.5)
        l_gain = gain * float(np.cos((pan + 1) * np.pi / 4))
        r_gain = gain * float(np.sin((pan + 1) * np.pi / 4))

        for note in track.notes:
            if should_cancel is not None and should_cancel() and done % 64 == 0:
                break
            start = int(note.start * spb * sample_rate)
            if start >= n:
                continue
            sustain_frames = max(1, int(note.duration * spb * sample_rate))
            tail = int((spec["release"] + 0.05) * sample_rate)
            frames = min(sustain_frames + tail, n - start)
            if frames <= 0:
                continue
            vel = max(0.0, min(1.0, note.velocity / 127.0)) ** 1.2

            if track.is_drum:
                sig = _render_drum(note.pitch, frames, sample_rate, vel, rng)
            else:
                sig = _render_note(
                    midi_to_freq(note.pitch), frames, sample_rate, spec, vel, sustain_frames
                )

            left[start:start + frames] += sig * l_gain
            right[start:start + frames] += sig * r_gain
            done += 1
            if progress is not None and done % 32 == 0:
                progress(min(1.0, done / total_notes))

    out = np.stack([left, right], axis=1)
    peak = float(np.abs(out).max())
    if peak > 0.95:
        out = np.tanh(out / peak * 1.1) * 0.95
    if progress is not None:
        progress(1.0)
    return AudioBuffer(out.astype(np.float32), sample_rate)
