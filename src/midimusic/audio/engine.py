"""Choosing how a score becomes sound.

Three callers used to make this decision separately -- the exporter, the
service and the remix -- and each wrote the same "use the SoundFont if there
is one, otherwise the built-in synth" two-branch choice, with only two of the
three remembering to fall back when the SoundFont failed to load.

There are three engines now rather than two, so the choice lives here once.
The new one matters most: a sampled General MIDI set has no sound for dance
music. Its "saw lead" is one thin sawtooth and its pads are preset strings,
which is why an electronic arrangement rendered through even a good SoundFont
comes out sounding like a 1990s MIDI file. Those styles want synthesis, not
samples, so they get a synthesiser.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..core.models import AudioBuffer, Song

__all__ = ["ELECTRONIC_STYLES", "is_electronic", "render_audio"]

log = logging.getLogger(__name__)

# Styles whose sound is synthesised rather than played. A sampled instrument
# set is the wrong tool for these however good the samples are.
ELECTRONIC_STYLES = frozenset({
    "house", "techno", "trance", "dnb", "trap", "synthwave", "disco",
})


def is_electronic(song: Song) -> bool:
    """Whether this arrangement wants a synthesiser rather than samples."""
    return str((song.meta or {}).get("style", "")).lower() in ELECTRONIC_STYLES


def render_audio(
    song: Song,
    soundfont: str | Path | None = None,
    sample_rate: int = 44100,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> AudioBuffer:
    """Render ``song`` with whichever engine suits it.

    A SoundFont that fails to load is a reason to fall back, not a reason to
    fail: the built-in synth always works, and silence helps nobody.
    """
    if is_electronic(song):
        try:
            from .edm_synth import render_song_edm

            return render_song_edm(
                song, sample_rate=sample_rate, progress=progress,
                should_cancel=should_cancel,
            )
        except ImportError:
            # Not built into this install; the other engines still work.
            log.debug("no dance synth available")
        except Exception:
            log.exception("the dance synth failed; falling back")

    if soundfont is not None:
        try:
            from .render import render_song

            return render_song(
                song, soundfont, progress=progress, should_cancel=should_cancel
            )
        except Exception:
            log.exception("SoundFont render failed; falling back to the built-in synth")

    from .synth_fallback import render_song_fallback

    return render_song_fallback(
        song, sample_rate=sample_rate, progress=progress, should_cancel=should_cancel
    )
