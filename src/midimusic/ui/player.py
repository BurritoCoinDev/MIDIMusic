"""Audio playback for the library.

Plays numpy buffers directly through PortAudio rather than going via a media
framework, because the app already has decoded audio in memory and only needs
transport controls.  Playback is optional: if no output device is available the
UI still works, it just cannot preview.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["AudioPlayer", "playback_available"]


def playback_available() -> bool:
    try:
        import sounddevice  # noqa: F401

        return True
    except Exception:
        return False


class AudioPlayer:
    """A small transport: load, play, pause, seek, stop."""

    def __init__(self, on_position: Callable[[float, float], None] | None = None,
                 on_finished: Callable[[], None] | None = None):
        self._samples: np.ndarray | None = None
        self._rate = 44100
        self._frame = 0
        self._stream = None
        self._lock = threading.RLock()
        self._playing = False
        self.on_position = on_position
        self.on_finished = on_finished
        self.volume = 1.0
        self.path: Path | None = None

    # -- state --------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._samples is not None

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def duration(self) -> float:
        if self._samples is None:
            return 0.0
        return self._samples.shape[0] / max(1, self._rate)

    @property
    def position(self) -> float:
        return self._frame / max(1, self._rate)

    # -- loading ------------------------------------------------------------

    def load_file(self, path: str | Path) -> bool:
        try:
            import soundfile as sf

            data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            log.exception("could not read %s", path)
            return False
        self.load_samples(data, int(rate))
        self.path = Path(path)
        return True

    def load_samples(self, samples: np.ndarray, rate: int) -> None:
        self.stop()
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        with self._lock:
            self._samples = np.ascontiguousarray(data)
            self._rate = int(rate)
            self._frame = 0

    # -- transport ----------------------------------------------------------

    def play(self) -> bool:
        if self._samples is None:
            return False
        if self._playing:
            return True
        try:
            import sounddevice as sd
        except Exception:
            log.info("playback unavailable: no audio backend")
            return False

        channels = self._samples.shape[1]

        def callback(outdata, frames, _time, status):
            if status:
                log.debug("audio status: %s", status)
            with self._lock:
                if self._samples is None:
                    outdata[:] = 0
                    raise sd.CallbackStop
                end = min(self._frame + frames, self._samples.shape[0])
                chunk = self._samples[self._frame:end]
                n = chunk.shape[0]
                outdata[:n] = chunk * self.volume
                if n < frames:
                    outdata[n:] = 0
                self._frame = end
                finished = end >= self._samples.shape[0]
            if self.on_position is not None:
                self.on_position(self.position, self.duration)
            if finished:
                raise sd.CallbackStop

        try:
            self._stream = sd.OutputStream(
                samplerate=self._rate, channels=channels, dtype="float32",
                callback=callback, finished_callback=self._on_stream_end,
                blocksize=1024,
            )
            self._stream.start()
            self._playing = True
            return True
        except Exception:
            log.exception("could not open an audio output stream")
            self._stream = None
            return False

    def _on_stream_end(self) -> None:
        self._playing = False
        if self.on_finished is not None:
            try:
                self.on_finished()
            except Exception:
                log.exception("finished callback failed")

    def pause(self) -> None:
        self._close_stream()

    def toggle(self) -> bool:
        if self._playing:
            self.pause()
        else:
            self.play()
        return self._playing

    def stop(self) -> None:
        self._close_stream()
        with self._lock:
            self._frame = 0

    def seek(self, seconds: float) -> None:
        with self._lock:
            if self._samples is None:
                return
            frame = int(max(0.0, seconds) * self._rate)
            self._frame = min(frame, self._samples.shape[0] - 1)

    def set_volume(self, value: float) -> None:
        self.volume = max(0.0, min(1.0, value))

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        self._playing = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
