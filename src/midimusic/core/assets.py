"""Optional assets fetched after install.

The installer stays small by not embedding a SoundFont; the app offers to
fetch one on first run.  Until then the built-in numpy synth handles audio, so
there is never a state where the app cannot produce sound.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SOUNDFONTS", "SoundFontOption", "download_soundfont", "installed_soundfonts"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoundFontOption:
    id: str
    name: str
    url: str
    filename: str
    approx_mb: int
    license: str
    description: str


# Only permissively-licensed fonts, so the app can point users at them without
# a redistribution question.
SOUNDFONTS: list[SoundFontOption] = [
    SoundFontOption(
        id="musescore-general",
        name="MuseScore General",
        url="https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/"
            "MuseScore_General.sf3",
        filename="MuseScore_General.sf3",
        approx_mb=38,
        license="MIT",
        description="A well-rounded General MIDI set. The recommended default.",
    ),
]


def installed_soundfonts(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in (".sf2", ".sf3"))


def download_soundfont(
    option: SoundFontOption,
    directory: Path,
    on_progress: Callable[[float, str], None] | None = None,
    on_finished: Callable[[Path | None, str], None] | None = None,
) -> threading.Event:
    """Fetch a SoundFont in the background. Returns a cancel event."""
    cancel = threading.Event()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / option.filename

    def report(fraction: float, message: str) -> None:
        if on_progress is not None:
            try:
                on_progress(fraction, message)
            except Exception:
                log.exception("soundfont progress callback failed")

    def run() -> None:
        # Download to a temporary name so a cancelled or failed transfer can
        # never be mistaken for a complete font on the next launch.
        partial = destination.with_suffix(destination.suffix + ".part")
        error = ""
        try:
            report(0.0, f"Downloading {option.name}")
            request = urllib.request.Request(
                option.url, headers={"User-Agent": "MIDIMusic"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                read = 0
                with open(partial, "wb") as fh:
                    while not cancel.is_set():
                        chunk = response.read(262144)
                        if not chunk:
                            break
                        fh.write(chunk)
                        read += len(chunk)
                        if total:
                            report(read / total,
                                   f"{read / 1e6:.0f} of {total / 1e6:.0f} MB")
            if cancel.is_set():
                partial.unlink(missing_ok=True)
                error = "Cancelled"
            else:
                partial.replace(destination)
                report(1.0, "Done")
        except Exception as exc:
            partial.unlink(missing_ok=True)
            error = f"{type(exc).__name__}: {exc}"
            log.exception("soundfont download failed")
        finally:
            if on_finished is not None:
                try:
                    on_finished(None if error else destination, error)
                except Exception:
                    log.exception("soundfont finished callback failed")

    threading.Thread(target=run, name="soundfont-download", daemon=True).start()
    return cancel
