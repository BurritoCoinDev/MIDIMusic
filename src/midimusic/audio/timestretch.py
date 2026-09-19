"""Changing how long audio lasts without changing its pitch.

Genre is partly tempo. Turning a slow record into a dance track means moving it
to dance tempo, and everything kept from the original has to come with it --
played faster by resampling it would rise in pitch and stop agreeing with the
new backing's key, so the length has to change while the frequencies stay put.

This is a phase vocoder: take the signal apart into overlapping short-time
spectra, step through them at a different rate than they were taken, and put
them back. Each bin's phase is advanced by what its frequency implies rather
than by what the original frame held, which is what stops the output turning
into a stutter of repeated grains.

Numpy only, like the rest of the core audio path. The alternatives are a
compiled resampling library or a command-line tool, and neither belongs in an
install that has to work on a Windows machine with no compiler.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MAX_STRETCH", "MIN_STRETCH", "stretch_to", "time_stretch"]

# Beyond about a third either way a phase vocoder stops sounding like the same
# performance: transients smear on the way out and the tail of every note gets
# chopped on the way in. Refusing is better than delivering mush.
MIN_STRETCH = 0.5
MAX_STRETCH = 2.0

_FRAME = 2048
_HOP = 512


def time_stretch(x: np.ndarray, factor: float, frame: int = _FRAME,
                 hop: int = _HOP) -> np.ndarray:
    """Return ``x`` lasting ``factor`` times as long, at the same pitch.

    ``factor`` above one stretches (slower), below one compresses (faster).
    """
    x = np.asarray(x, dtype=np.float32)
    factor = float(factor)
    if x.size == 0 or abs(factor - 1.0) < 1e-3:
        return x
    factor = float(np.clip(factor, MIN_STRETCH, MAX_STRETCH))

    if x.ndim == 1:
        return _stretch_mono(x, factor, frame, hop)
    # Channel by channel. They share an analysis grid and a phase origin, so
    # the stereo image survives rather than wandering between them.
    return np.stack(
        [_stretch_mono(x[:, c], factor, frame, hop) for c in range(x.shape[1])],
        axis=1,
    ).astype(np.float32)


def _stretch_mono(x: np.ndarray, factor: float, frame: int, hop: int) -> np.ndarray:
    if x.size < frame * 2:
        return x

    window = np.hanning(frame).astype(np.float32)
    # Hann at 75% overlap sums to a constant, so the output needs no
    # normalisation pass beyond dividing by that constant.
    padded = np.concatenate([np.zeros(frame, dtype=np.float32), x,
                             np.zeros(frame * 2, dtype=np.float32)])

    analysis = np.arange(0, padded.size - frame, hop)
    spectra = np.stack([
        np.fft.rfft(padded[i:i + frame] * window) for i in analysis
    ])
    magnitude = np.abs(spectra)
    phase = np.angle(spectra)

    bins = np.arange(spectra.shape[1])
    expected = 2.0 * np.pi * hop * bins / frame

    # How far each bin's phase really moved between frames, minus how far a
    # bin at its centre frequency would have moved, wrapped into +/- pi. That
    # residual is the bin's true frequency offset.
    delta = np.diff(phase, axis=0, prepend=phase[:1])
    delta = delta - expected
    delta = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
    true_freq = expected + delta

    out_hop = hop
    steps = np.arange(0, spectra.shape[0] - 1, 1.0 / factor)
    out = np.zeros(int(len(steps) * out_hop) + frame, dtype=np.float32)
    weight = np.zeros_like(out)

    running = phase[0].copy()
    for n, position in enumerate(steps):
        lower = int(position)
        fraction = position - lower
        # Interpolate the magnitude between the two frames this output frame
        # sits between; a nearest-frame copy audibly stutters at slow rates.
        mag = (1.0 - fraction) * magnitude[lower] + fraction * magnitude[lower + 1]

        # Advance each bin's phase on its own and consecutive output frames
        # stop agreeing with each other: the overlap-add then sums partly out
        # of phase, which both hollows the sound out and costs several dB.
        # Locking every bin to the peak it belongs to -- keeping the phase
        # relationship the analysis found within each peak's skirt -- keeps
        # the frames coherent. This is what separates a usable phase vocoder
        # from one that sounds underwater.
        advanced = running + true_freq[min(lower + 1, len(true_freq) - 1)]
        owner = _peak_regions(mag)
        running = advanced[owner] + (phase[lower] - phase[lower][owner])

        grain = np.fft.irfft(mag * np.exp(1j * running), n=frame).astype(np.float32)
        start = n * out_hop
        out[start:start + frame] += grain * window
        weight[start:start + frame] += window ** 2

    busy = weight > 1e-6
    out[busy] /= weight[busy]
    # Drop the lead-in padding, and take exactly as much as was asked for.
    wanted = int(round(x.size * factor))
    out = out[frame:frame + wanted]
    if out.size < wanted:
        out = np.concatenate([out, np.zeros(wanted - out.size, dtype=np.float32)])
    return out.astype(np.float32)


def _peak_regions(mag: np.ndarray) -> np.ndarray:
    """For each bin, the index of the spectral peak it belongs to.

    A sinusoid does not occupy one bin: it spreads across several, and those
    neighbours only reconstruct it if they keep the phase relationship the
    analysis found. Grouping every bin with its nearest peak is what lets the
    synthesis preserve that.
    """
    if mag.size < 3:
        return np.arange(mag.size)
    higher_than_neighbours = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    peaks = np.flatnonzero(higher_than_neighbours) + 1
    if peaks.size == 0:
        return np.full(mag.size, int(np.argmax(mag)))
    # Each bin joins whichever peak is nearer, so the boundaries fall halfway
    # between adjacent peaks.
    midpoints = (peaks[:-1] + peaks[1:] + 1) // 2
    return peaks[np.searchsorted(midpoints, np.arange(mag.size))]


def stretch_to(x: np.ndarray, source_tempo: float, target_tempo: float) -> np.ndarray:
    """Re-time audio from one tempo to another, keeping its pitch."""
    if source_tempo <= 0 or target_tempo <= 0:
        return np.asarray(x, dtype=np.float32)
    return time_stretch(x, source_tempo / target_tempo)
