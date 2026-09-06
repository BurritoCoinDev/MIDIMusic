"""Offline MIDI-to-audio rendering.

Uses TinySoundFont (MIT, prebuilt wheels on Windows, no external DLL) to play
a :class:`Song` through a SoundFont into a numpy buffer.  Rendering is done by
stepping through note events in time order and asking the synth for the audio
between them, so it is fully offline and deterministic -- no audio device is
opened and it is not real-time bound.

If no SoundFont is available the caller can fall back to
:mod:`midimusic.audio.synth_fallback`, which needs nothing at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.models import AudioBuffer, Song

__all__ = ["RenderOptions", "SoundFontRenderer", "is_available", "render_song"]


@dataclass
class RenderOptions:
    sample_rate: int = 44100
    gain_db: float = -3.0
    tail_seconds: float = 2.0  # let reverb/release ring out
    max_voices: int = 256
    block_size: int = 512


def is_available() -> bool:
    try:
        import tinysoundfont  # noqa: F401
    except ImportError:
        return False
    return True


class SoundFontRenderer:
    """Renders songs through a SoundFont."""

    def __init__(self, soundfont: str | Path, options: RenderOptions | None = None):
        import tinysoundfont

        self.options = options or RenderOptions()
        self.path = Path(soundfont)
        if not self.path.exists():
            raise FileNotFoundError(f"SoundFont not found: {self.path}")
        self._synth = tinysoundfont.Synth(
            samplerate=self.options.sample_rate, gain=self.options.gain_db
        )
        self._sfid = self._synth.sfload(str(self.path), gain=self.options.gain_db)

    def render(
        self,
        song: Song,
        progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AudioBuffer:
        sr = self.options.sample_rate
        spb = 60.0 / max(1e-6, song.tempo)  # seconds per beat

        # Assign each track a MIDI channel and select its program.
        for i, track in enumerate(song.tracks):
            channel = 9 if track.is_drum else _safe_channel(track.channel, i)
            try:
                self._synth.program_select(
                    channel, self._sfid, 128 if track.is_drum else 0,
                    0 if track.is_drum else max(0, min(127, track.program)),
                    is_drums=track.is_drum,
                )
            except Exception:
                # A SoundFont may not have every GM preset; fall back to the
                # first available program rather than dropping the track.
                try:
                    self._synth.program_change(channel, 0, is_drums=track.is_drum)
                except Exception:
                    pass

        events = _collect_events(song, spb)
        if not events:
            return AudioBuffer(np.zeros((1, 2), dtype=np.float32), sr)

        total_seconds = events[-1][0] + self.options.tail_seconds
        total_frames = int(total_seconds * sr) + 1
        out = np.zeros((total_frames, 2), dtype=np.float32)

        cursor_frame = 0
        idx = 0
        n_events = len(events)

        while idx < n_events or cursor_frame < total_frames:
            if should_cancel is not None and should_cancel():
                break

            # Apply every event due at or before the current position.
            while idx < n_events and int(events[idx][0] * sr) <= cursor_frame:
                _, kind, channel, pitch, velocity = events[idx]
                try:
                    if kind == 1:
                        self._synth.noteon(channel, pitch, velocity)
                    else:
                        self._synth.noteoff(channel, pitch)
                except Exception:
                    pass
                idx += 1

            # Render up to the next event, or to the end.
            if idx < n_events:
                next_frame = min(total_frames, int(events[idx][0] * sr))
            else:
                next_frame = total_frames
            n = max(1, min(next_frame - cursor_frame, total_frames - cursor_frame))
            if n <= 0:
                break

            block = self._render_block(n)
            end = min(total_frames, cursor_frame + block.shape[0])
            out[cursor_frame:end] = block[: end - cursor_frame]
            cursor_frame = end

            if progress is not None and total_frames:
                progress(min(1.0, cursor_frame / total_frames))
            if cursor_frame >= total_frames:
                break

        return AudioBuffer(out, sr)

    def _render_block(self, frames: int) -> np.ndarray:
        buf = self._synth.generate(frames)
        arr = np.frombuffer(bytes(buf), dtype=np.float32)
        if arr.size < frames * 2:
            arr = np.pad(arr, (0, frames * 2 - arr.size))
        return arr[: frames * 2].reshape(-1, 2)

    def close(self) -> None:
        try:
            self._synth.sfunload(self._sfid)
        except Exception:
            pass


def _safe_channel(channel: int, fallback_index: int) -> int:
    ch = channel if 0 <= channel <= 15 else fallback_index
    if ch == 9:
        ch = 10 if fallback_index != 10 else 11
    return max(0, min(15, ch))


def _collect_events(song: Song, seconds_per_beat: float) -> list[tuple[float, int, int, int, int]]:
    """Flatten a song to ``(time_s, on/off, channel, pitch, velocity)``, sorted."""
    events: list[tuple[float, int, int, int, int]] = []
    for i, track in enumerate(song.tracks):
        channel = 9 if track.is_drum else _safe_channel(track.channel, i)
        for n in track.notes:
            start = max(0.0, n.start * seconds_per_beat)
            end = max(start + 0.01, n.end * seconds_per_beat)
            pitch = max(0, min(127, int(n.pitch)))
            vel = max(1, min(127, int(n.velocity)))
            events.append((start, 1, channel, pitch, vel))
            events.append((end, 0, channel, pitch, 0))
    # Note-offs sort before note-ons at the same instant so repeated pitches
    # retrigger instead of being cut off by the previous note's release.
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def render_song(
    song: Song,
    soundfont: str | Path,
    options: RenderOptions | None = None,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> AudioBuffer:
    renderer = SoundFontRenderer(soundfont, options)
    try:
        return renderer.render(song, progress=progress, should_cancel=should_cancel)
    finally:
        renderer.close()
