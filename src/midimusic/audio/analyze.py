"""Estimating tempo, key and structure from recorded audio.

Implemented in numpy rather than pulling in librosa: the algorithms here are
small and well understood, and the core install is deliberately wheel-only.
Everything returns a confidence alongside its answer, because these are
estimates and the UI should say so rather than presenting a guess as a fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..theory.pitch import NOTE_NAMES

__all__ = ["AudioAnalysis", "analyze_audio", "estimate_key", "estimate_tempo"]

# Krumhansl-Kessler key profiles: how strongly each pitch class is expected to
# feature in a major or minor key. Correlating a piece's chroma against all 24
# rotations of these is the standard approach to key finding.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


@dataclass
class AudioAnalysis:
    duration: float = 0.0
    tempo: float = 0.0
    tempo_confidence: float = 0.0
    key: str = ""
    key_confidence: float = 0.0
    chroma: list[float] = field(default_factory=list)
    key_alternatives: list[tuple[str, float]] = field(default_factory=list)
    peak_db: float = 0.0
    rms_db: float = 0.0
    lufs: float = 0.0

    def describe(self) -> str:
        bits = []
        if self.tempo:
            bits.append(f"{self.tempo:.0f} bpm")
        if self.key:
            # Say when the key is a close call rather than implying certainty.
            if self.key_confidence < 0.4 and len(self.key_alternatives) > 1:
                bits.append(f"{self.key} (or {self.key_alternatives[1][0]})")
            else:
                bits.append(self.key)
        if self.duration:
            bits.append(f"{int(self.duration // 60)}:{int(self.duration % 60):02d}")
        return "  |  ".join(bits) or "no analysis"


def _to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def _onset_envelope(mono: np.ndarray, rate: int, hop: int = 512,
                    frame: int = 2048) -> tuple[np.ndarray, float]:
    """Spectral flux: how much the spectrum brightens frame to frame.

    Peaks in this signal line up with note and drum onsets, which is what the
    tempo estimate is built on.
    """
    if mono.size < frame:
        return np.zeros(0, dtype=np.float32), rate / hop
    window = np.hanning(frame).astype(np.float32)
    n_frames = 1 + (mono.size - frame) // hop
    n_frames = min(n_frames, 20000)  # cap the work on very long files
    prev = None
    flux = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = mono[i * hop: i * hop + frame] * window
        mag = np.abs(np.fft.rfft(seg))
        if prev is not None:
            # Half-wave rectified: only increases in energy count as onsets.
            flux[i] = float(np.sum(np.maximum(0.0, mag - prev)))
        prev = mag
    if flux.size:
        flux -= flux.mean()
        peak = np.abs(flux).max()
        if peak > 0:
            flux /= peak
    return flux, rate / hop


def estimate_tempo(samples: np.ndarray, rate: int,
                   low_bpm: float = 60.0, high_bpm: float = 200.0) -> tuple[float, float]:
    """Estimate tempo in BPM, with a 0-1 confidence.

    Autocorrelates the onset envelope and takes the strongest lag inside the
    plausible range. Octave errors (half or double time) are the usual failure,
    so the result is folded into a musically sensible band.
    """
    mono = _to_mono(samples)
    flux, fps = _onset_envelope(mono, rate)
    if flux.size < 16:
        return 0.0, 0.0

    corr = np.correlate(flux, flux, mode="full")[flux.size - 1:]
    if corr.size < 4 or corr[0] <= 0:
        return 0.0, 0.0
    corr = corr / corr[0]

    min_lag = max(1, int(fps * 60.0 / high_bpm))
    max_lag = min(corr.size - 1, int(fps * 60.0 / low_bpm))
    if max_lag <= min_lag:
        return 0.0, 0.0

    window = corr[min_lag:max_lag]
    best = int(np.argmax(window)) + min_lag
    bpm = 60.0 * fps / best
    confidence = float(np.clip(window.max(), 0.0, 1.0))

    # Fold obvious octave errors into a range people actually count in.
    while bpm < 70 and bpm > 0:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return round(float(bpm), 1), round(confidence, 3)


def _bass_chroma(mono: np.ndarray, rate: int) -> np.ndarray:
    """Pitch-class energy in the bass register.

    Full-spectrum chroma cannot separate a key from its relative minor: C major
    and A minor contain exactly the same notes. What distinguishes them is
    which note behaves as the tonic, and the bass line is the strongest
    available evidence for that.
    """
    frame, hop = 8192, 4096  # long frames: bass needs frequency resolution
    if mono.size < frame:
        return np.zeros(12)
    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, np.nan) / 440.0)
    valid = np.isfinite(midi) & (freqs > 40) & (freqs < 260)
    if not valid.any():
        return np.zeros(12)
    pitch_class = np.zeros(freqs.shape, dtype=np.int64)
    pitch_class[valid] = np.rint(midi[valid]).astype(np.int64) % 12

    out = np.zeros(12, dtype=np.float64)
    n_frames = max(1, min(1 + (mono.size - frame) // hop, 2000))
    for i in range(n_frames):
        seg = mono[i * hop: i * hop + frame]
        if seg.size < frame:
            break
        mag = np.abs(np.fft.rfft(seg * window))
        # Only the strongest bass partial per frame counts, which approximates
        # "what note is the bass playing right now".
        band = np.where(valid, mag, 0.0)
        if band.max() <= 0:
            continue
        out[pitch_class[int(np.argmax(band))]] += 1.0
    total = out.sum()
    return out / total if total > 0 else out


def estimate_key(
    samples: np.ndarray, rate: int
) -> tuple[str, float, list[float], list[tuple[str, float]]]:
    """Estimate the key from chroma, disambiguated by the bass register."""
    mono = _to_mono(samples)
    if mono.size < 4096:
        return "", 0.0, [0.0] * 12, []

    frame, hop = 4096, 2048
    n_frames = max(1, min(1 + (mono.size - frame) // hop, 4000))
    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)

    # Map every bin to a pitch class once, rather than per frame.
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, np.nan) / 440.0)
    valid = np.isfinite(midi) & (freqs > 80) & (freqs < 2100)
    pitch_class = np.zeros(freqs.shape, dtype=np.int64)
    pitch_class[valid] = np.rint(midi[valid]).astype(np.int64) % 12

    chroma = np.zeros(12, dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop: i * hop + frame]
        if seg.size < frame:
            break
        mag = np.abs(np.fft.rfft(seg * window))
        # Log compression stops loud bass fundamentals from dominating the
        # profile. Without it the tonic is swamped by whatever the bass player
        # happens to be sitting on, and the estimate lands a fifth away.
        mag = np.log1p(mag * 8.0)
        frame_chroma = np.zeros(12, dtype=np.float64)
        np.add.at(frame_chroma, pitch_class[valid], mag[valid])
        # Normalise per frame so a loud chorus does not outvote a whole verse.
        total_frame = frame_chroma.sum()
        if total_frame > 0:
            chroma += frame_chroma / total_frame

    if chroma.sum() <= 0:
        return "", 0.0, [0.0] * 12, []
    chroma /= chroma.sum()

    # Sharpen. Harmonic leakage and spectral spread put energy on every pitch
    # class, including notes the piece never plays, so the measured profile is
    # far flatter than the music's true note histogram. A flat profile
    # correlates about equally with all 24 keys, which is how the estimate
    # collapses toward one answer. Subtracting the noise floor and squaring
    # restores the zeros the music actually has.
    chroma = np.maximum(0.0, chroma - np.median(chroma))
    if chroma.sum() > 0:
        chroma = chroma**2
        chroma /= chroma.sum()

    bass = _bass_chroma(mono, rate)

    scored: list[tuple[float, str]] = []
    for tonic in range(12):
        for profile, quality in ((_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")):
            rotated = np.roll(profile, tonic)
            score = float(np.corrcoef(chroma, rotated)[0, 1])
            if not np.isfinite(score):
                continue
            # Reward candidates whose tonic the bass actually emphasises. This
            # is what breaks the tie between a key and its relative.
            score += 0.55 * float(bass[tonic])
            # The fifth is the next most common bass note; a little credit for
            # it stops a dominant-heavy passage pulling the answer a fifth up.
            score += 0.15 * float(bass[(tonic + 7) % 12])
            scored.append((score, f"{NOTE_NAMES[tonic]} {quality}"))

    if not scored:
        return "", 0.0, [round(float(c), 4) for c in chroma], []
    scored.sort(reverse=True)
    best_score, best_name = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else -2.0

    # Confidence is how far clear the winner is, not its raw correlation: a
    # piece that fits two keys almost equally well is genuinely ambiguous.
    margin = max(0.0, best_score - runner_up)
    confidence = float(np.clip(margin * 3.0, 0.0, 1.0)) if best_name else 0.0

    # Report the runners-up too. A key and its relative share every note, so on
    # genuinely ambiguous material the honest answer is a shortlist, not one
    # name presented as fact.
    span = scored[0][0] - scored[min(len(scored) - 1, 5)][0] or 1.0
    alternatives = [
        (name, round(float(np.clip((score - scored[5][0]) / span, 0.0, 1.0)), 3))
        for score, name in scored[:3]
    ]
    return best_name, round(confidence, 3), [round(float(c), 4) for c in chroma], alternatives


def analyze_audio(samples: np.ndarray, rate: int) -> AudioAnalysis:
    """Full analysis of one buffer."""
    from .dsp import measure

    mono = _to_mono(samples)
    stats = measure(np.asarray(samples), rate)
    tempo, tempo_conf = estimate_tempo(mono, rate)
    key, key_conf, chroma, alternatives = estimate_key(mono, rate)
    return AudioAnalysis(
        duration=mono.size / max(1, rate),
        tempo=tempo,
        tempo_confidence=tempo_conf,
        key=key,
        key_confidence=key_conf,
        chroma=chroma,
        key_alternatives=alternatives,
        peak_db=round(float(stats["peak_db"]), 2),
        rms_db=round(float(stats["rms_db"]), 2),
        lufs=round(float(stats["lufs"]), 2) if np.isfinite(stats["lufs"]) else 0.0,
    )
