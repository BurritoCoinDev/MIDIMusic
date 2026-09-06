"""Signal processing between a generated buffer and a delivered file.

Kept to numpy alone in the core path so it works from a plain pip install on
Windows with no compiler and no ffmpeg, and so nothing here carries a
statically-linked LGPL component.  Better resamplers are used when present.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_fades",
    "dither_to_int",
    "loudness_normalize",
    "measure",
    "peak_envelope",
    "peak_normalize",
    "resample",
    "soft_clip",
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
