"""Signal processing between a generated buffer and a delivered file.

Kept to numpy alone in the core path so it works from a plain pip install on
Windows with no compiler and no ffmpeg, and so nothing here carries a
statically-linked LGPL component.  Better resamplers are used when present.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_CROSSFADE",
    "apply_fades",
    "crossfade_frames",
    "dither_to_int",
    "fit_length",
    "loudness_normalize",
    "match_loudness",
    "measure",
    "mix",
    "mix_layers",
    "peak_envelope",
    "peak_normalize",
    "resample",
    "soft_clip",
    "tiles_needed",
    "to_stereo",
    "trim_silence",
]


def to_stereo(x: np.ndarray) -> np.ndarray:
    """Coerce any shape to float32 ``(frames, 2)``."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] > 2:
        # Downmix extra channels rather than dropping them.
        left = x[:, 0::2].mean(axis=1)
        right = x[:, 1::2].mean(axis=1)
        return np.stack([left, right], axis=1).astype(np.float32)
    return x


def resample(x: np.ndarray, src_rate: int, dst_rate: int, quality: str = "HQ") -> np.ndarray:
    """Sample-rate convert, using the best resampler available.

    Preference order is soxr, then scipy's polyphase filter, then a windowed-
    sinc implementation in numpy.  The numpy path exists so the core install
    needs neither of the others: linear interpolation would audibly alias on a
    32 kHz to 44.1 kHz conversion, which is exactly the case the audio models
    produce.
    """
    if src_rate == dst_rate or x.size == 0:
        return x

    try:
        import soxr

        return np.asarray(soxr.resample(x, src_rate, dst_rate, quality=quality),
                          dtype=np.float32)
    except ImportError:
        pass

    from math import gcd

    divisor = gcd(int(src_rate), int(dst_rate))
    up, down = int(dst_rate) // divisor, int(src_rate) // divisor

    try:
        from scipy.signal import resample_poly

        return np.asarray(resample_poly(x, up, down, axis=0), dtype=np.float32)
    except ImportError:
        pass

    return _resample_sinc(x, up, down)


def _resample_sinc(x: np.ndarray, up: int, down: int, half_width: int = 16) -> np.ndarray:
    """Polyphase resampling with a Kaiser-windowed sinc, in numpy only.

    Upsamples by ``up`` and decimates by ``down`` in one convolution, with the
    anti-imaging and anti-aliasing filter combined into a single kernel cut off
    at the lower of the two Nyquist limits.
    """
    x = np.asarray(x, dtype=np.float32)
    single = x.ndim == 1
    if single:
        x = x[:, None]

    # Guard against a pathological ratio producing an enormous kernel.
    if up > 512 or down > 512:
        n_out = int(round(x.shape[0] * up / down))
        idx = np.linspace(0, x.shape[0] - 1, n_out)
        out = np.stack(
            [np.interp(idx, np.arange(x.shape[0]), x[:, c]) for c in range(x.shape[1])],
            axis=1,
        ).astype(np.float32)
        return out[:, 0] if single else out

    cutoff = 1.0 / max(up, down)
    taps = 2 * half_width * max(up, down) + 1
    n = np.arange(taps) - (taps - 1) / 2.0
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n)
    kernel *= np.kaiser(taps, 8.0)
    kernel = (kernel / kernel.sum()) * up

    frames, channels = x.shape
    upsampled = np.zeros((frames * up, channels), dtype=np.float64)
    upsampled[::up] = x

    pad = taps // 2
    out_len = int(np.ceil(frames * up / down))
    out = np.empty((out_len, channels), dtype=np.float32)
    for c in range(channels):
        padded = np.pad(upsampled[:, c], (pad, pad), mode="constant")
        filtered = np.convolve(padded, kernel, mode="valid")
        out[:, c] = filtered[::down][:out_len]
    return out[:, 0] if single else out


def peak_normalize(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak <= 1e-9:
        return x
    target = 10.0 ** (target_db / 20.0)
    return (x * (target / peak)).astype(np.float32)


def loudness_normalize(
    x: np.ndarray, sample_rate: int, target_lufs: float = -14.0, ceiling_db: float = -1.0
) -> np.ndarray:
    """Normalise to a target integrated loudness (ITU-R BS.1770).

    Falls back to peak normalisation when pyloudnorm is unavailable or the
    signal is too short for the 400 ms measurement block.
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        return peak_normalize(x, ceiling_db)

    if x.shape[0] < int(sample_rate * 0.5):
        return peak_normalize(x, ceiling_db)

    try:
        meter = pyln.Meter(sample_rate)
        current = meter.integrated_loudness(x)
        if not np.isfinite(current):
            return peak_normalize(x, ceiling_db)
        gain = 10.0 ** ((target_lufs - current) / 20.0)
        y = (x * gain).astype(np.float32)
    except Exception:
        return peak_normalize(x, ceiling_db)

    # Never let loudness matching push us into clipping.
    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > ceiling:
        y = soft_clip(y * (ceiling / peak) * 1.02, ceiling)
    return y.astype(np.float32)


def soft_clip(x: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Gentle saturation instead of hard digital clipping."""
    return (np.tanh(x / max(1e-6, ceiling)) * ceiling).astype(np.float32)


def apply_fades(x: np.ndarray, sample_rate: int, fade_in: float = 0.01,
                fade_out: float = 0.25) -> np.ndarray:
    """Equal-power fades to stop clicks at the boundaries."""
    y = x.copy()
    n = y.shape[0]
    fi = min(int(fade_in * sample_rate), n // 2)
    fo = min(int(fade_out * sample_rate), n // 2)
    if fi > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, fi, dtype=np.float32)) ** 2
        y[:fi] *= ramp[:, None] if y.ndim == 2 else ramp
    if fo > 0:
        ramp = np.cos(np.linspace(0, np.pi / 2, fo, dtype=np.float32)) ** 2
        y[-fo:] *= ramp[:, None] if y.ndim == 2 else ramp
    return y


def trim_silence(x: np.ndarray, threshold_db: float = -60.0,
                 keep_head: float = 0.0, sample_rate: int = 44100) -> np.ndarray:
    """Trim leading/trailing near-silence, keeping a little head room."""
    if x.size == 0:
        return x
    mag = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    thresh = 10.0 ** (threshold_db / 20.0)
    above = np.flatnonzero(mag > thresh)
    if above.size == 0:
        return x
    start = max(0, above[0] - int(keep_head * sample_rate))
    end = min(x.shape[0], above[-1] + int(0.1 * sample_rate))
    return x[start:end]


def dither_to_int(x: np.ndarray, bits: int = 24) -> np.ndarray:
    """Convert float audio to integer PCM with TPDF dither.

    Dither matters most at 16-bit; at 24-bit it is inaudible but harmless, and
    applying it uniformly keeps the export path simple.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    max_val = float(2 ** (bits - 1) - 1)
    scaled = x * max_val
    if bits < 32:
        rng = np.random.default_rng(0)
        # Triangular PDF dither: the sum of two uniform sources, 1 LSB wide.
        tpdf = rng.random(scaled.shape) - rng.random(scaled.shape)
        scaled = scaled + tpdf
    scaled = np.clip(np.round(scaled), -max_val - 1, max_val)
    return scaled.astype(np.int32)


def peak_envelope(x: np.ndarray, buckets: int = 1200) -> np.ndarray:
    """Min/max envelope for waveform drawing, shaped ``(buckets, 2)``."""
    if x.size == 0:
        return np.zeros((buckets, 2), dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = mono.shape[0]
    buckets = max(1, min(buckets, n))
    edges = np.linspace(0, n, buckets + 1, dtype=np.int64)
    out = np.zeros((buckets, 2), dtype=np.float32)
    for i in range(buckets):
        seg = mono[edges[i]:edges[i + 1]]
        if seg.size:
            out[i, 0] = float(seg.min())
            out[i, 1] = float(seg.max())
    return out


def measure(x: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Quick quality metrics used by the UI and by tests."""
    if x.size == 0:
        return {"peak_db": -np.inf, "rms_db": -np.inf, "lufs": -np.inf, "clipped": 0.0}
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    stats = {
        "peak_db": 20 * np.log10(max(peak, 1e-12)),
        "rms_db": 20 * np.log10(max(rms, 1e-12)),
        "clipped": float(np.mean(np.abs(x) >= 0.999)),
    }
    try:
        import pyloudnorm as pyln

        if x.shape[0] > sample_rate * 0.5:
            stats["lufs"] = float(pyln.Meter(sample_rate).integrated_loudness(x))
        else:
            stats["lufs"] = -np.inf
    except Exception:
        stats["lufs"] = -np.inf
    return stats


# How much of a clip a tiling join overlaps. Short on purpose: a crossfade
# between a clip's tail and its own head has no material to borrow from, so
# every join costs this much musical time. At 30 ms that is a third of a
# hundredth of a bar and inaudible as drift; at a second -- which this used to
# use -- it is a beat and a half, and a looped backing walks off the beat.
DEFAULT_CROSSFADE = 0.03


def crossfade_frames(source_frames: int, sample_rate: int,
                     crossfade: float = DEFAULT_CROSSFADE) -> int:
    """How many frames a tiling join will overlap, given the clip length."""
    if source_frames <= 0 or sample_rate <= 0:
        return 0
    return int(max(0, min(max(0.0, crossfade) * sample_rate, source_frames // 4)))


def tiles_needed(source_frames: int, frames: int, sample_rate: int,
                 crossfade: float = DEFAULT_CROSSFADE) -> int:
    """How many copies of a clip :func:`fit_length` would lay end to end.

    Each join after the first advances the timeline by ``len - overlap``, not
    ``len``, so counting copies as ``ceil(frames / len)`` reports fewer than
    are really used. Callers that tell the user how often the material repeats
    need the real number.
    """
    source_frames, frames = int(source_frames), int(frames)
    if source_frames <= 0 or frames <= source_frames:
        return 1
    overlap = crossfade_frames(source_frames, sample_rate, crossfade)
    stride = max(1, source_frames - overlap)
    return 1 + int(np.ceil((frames - source_frames) / stride))


def fit_length(x: np.ndarray, frames: int, sample_rate: int,
               crossfade: float = DEFAULT_CROSSFADE) -> np.ndarray:
    """Trim or repeat ``x`` until it is exactly ``frames`` long.

    Repeating is what makes a thirty-second generated bed usable under a
    four-minute song. The joins are crossfaded rather than butt spliced: a hard
    cut between two takes of the same material clicks, and the click is more
    noticeable than the repetition.

    The crossfade is deliberately short. A clip has no material beyond its own
    end, so the only thing a join can fade into is the clip's head -- which
    means every join consumes ``overlap`` frames of musical time. That is
    unavoidable; keeping it to a few tens of milliseconds is what stops it
    mattering. :func:`tiles_needed` reproduces the arithmetic for callers that
    have to report how often the material repeats.
    """
    x = np.asarray(x, dtype=np.float32)
    frames = max(0, int(frames))
    if frames == 0 or x.size == 0:
        return np.zeros((frames, *x.shape[1:]), dtype=np.float32)
    if x.shape[0] >= frames:
        return x[:frames]

    overlap = crossfade_frames(x.shape[0], sample_rate, crossfade)
    out = x.copy()
    if overlap <= 0:
        while out.shape[0] < frames:
            out = np.concatenate([out, x], axis=0)
        return out[:frames]

    # Equal power is right for a join between unrelated material, which is the
    # usual case: a clip's tail and its head are different music. It is wrong
    # when they happen to correlate -- a sustained tone splices into itself
    # either in phase, swelling by up to 3 dB, or in antiphase, dropping out --
    # so the blended block is held to the level of the louder of its two
    # sources afterwards. That covers both directions without having to decide
    # which case this is.
    ramp = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
    fade_out, fade_in = np.cos(ramp), np.sin(ramp)
    if x.ndim == 2:
        fade_out, fade_in = fade_out[:, None], fade_in[:, None]

    while out.shape[0] < frames:
        tail, head = out[-overlap:], x[:overlap]
        joined = tail * fade_out + head * fade_in
        ceiling = max(float(np.abs(tail).max()), float(np.abs(head).max()))
        peak = float(np.abs(joined).max())
        if ceiling > 0 and peak > ceiling:
            joined = joined * (ceiling / peak)
        out = np.concatenate([out[:-overlap], joined, x[overlap:]], axis=0)
    return out[:frames].astype(np.float32)


def match_loudness(x: np.ndarray, reference: np.ndarray, sample_rate: int,
                   max_gain_db: float = 24.0) -> np.ndarray:
    """Scale ``x`` to sit at the same loudness as ``reference``.

    Used when one layer of a mix is replaced: the new part has to arrive at the
    level of the parts it stands in for, or whatever was kept -- a vocal, say --
    is left either buried or naked. Absolute normalisation cannot do this,
    because the target is whatever the rest of the mix happens to be.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or np.asarray(reference).size == 0:
        return x

    here = measure(x, sample_rate)
    there = measure(np.asarray(reference, dtype=np.float32), sample_rate)
    # LUFS where it is measurable, RMS where the material is too short or too
    # quiet for a loudness meter to return anything finite.
    if np.isfinite(here.get("lufs", -np.inf)) and np.isfinite(there.get("lufs", -np.inf)):
        delta = there["lufs"] - here["lufs"]
    elif np.isfinite(here["rms_db"]) and np.isfinite(there["rms_db"]):
        delta = there["rms_db"] - here["rms_db"]
    else:
        return x

    delta = float(np.clip(delta, -max_gain_db, max_gain_db))
    return (x * (10.0 ** (delta / 20.0))).astype(np.float32)


def mix(layers: list[np.ndarray], ceiling_db: float = -1.0) -> np.ndarray:
    """Sum layers, holding the result under a peak ceiling."""
    return mix_layers(layers, ceiling_db)[0]


def mix_layers(layers: list[np.ndarray],
               ceiling_db: float = -1.0) -> tuple[np.ndarray, float]:
    """Sum layers, and report the gain applied to hold the peak ceiling.

    Gain riding rather than clipping: summing four stems routinely exceeds full
    scale, and turning the sum down preserves the balance between the layers
    where limiting each one would not.

    The gain comes back because a caller that also writes the layers out
    separately has to apply the same number to them. Without it the parts are
    louder than the mix they came from and no longer add up to it, which
    defeats the point of saving them.
    """
    usable = [np.asarray(layer, dtype=np.float32) for layer in layers if np.asarray(layer).size]
    if not usable:
        return np.zeros((0, 2), dtype=np.float32), 1.0
    frames = max(layer.shape[0] for layer in usable)
    channels = max(layer.shape[1] if layer.ndim == 2 else 1 for layer in usable)

    total = np.zeros((frames, channels), dtype=np.float32)
    for layer in usable:
        if layer.ndim == 1:
            # One channel means the same signal everywhere, so spreading it is
            # right. Anything wider is real, distinct content: place it in the
            # channels it belongs to and leave the rest alone rather than
            # copying channel one over the others and losing the difference.
            layer = np.repeat(layer[:, None], channels, axis=1)
        elif layer.shape[1] == 1:
            layer = np.repeat(layer, channels, axis=1)
        clipped = layer[:frames]
        total[: clipped.shape[0], : clipped.shape[1]] += clipped

    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = float(np.abs(total).max())
    gain = 1.0
    if peak > ceiling:
        gain = ceiling / peak
        total *= gain
    return total, gain
