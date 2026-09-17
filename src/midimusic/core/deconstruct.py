"""Taking a finished recording apart.

Separation gives you the layers as audio. Transcription then turns each layer
into notes -- and doing that per stem rather than on the mix is the whole
point: pitch estimation on an isolated bass line or vocal is a far easier
problem than on everything at once.

Vocals deserve a word, because MIDI has no concept of a voice. A vocal layer
yields three different things: the isolated audio, the sung melody as notes,
and (where lyrics are known) MIDI lyric meta-events aligned to those notes.
The audio is the voice; the MIDI is what it sang.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio.analyze import AudioAnalysis, analyze_audio
from ..audio.export import ExportOptions, TrackMetadata, export_audio, safe_filename
from ..audio.midi_io import write_midi
from ..core.generator import (
    BackendUnavailable,
    Capabilities,
    GenerationCancelled,
    Generator,
    GeneratorContext,
)
from ..core.models import AudioBuffer, GenerationRequest, GenerationResult, OutputFormat, Song
from ..core.worker_client import WorkerCancelled, WorkerError, run_worker

__all__ = ["GM_FOR_STEM", "DeconstructGenerator", "DeconstructResult", "StemResult"]

log = logging.getLogger(__name__)

# A sensible General MIDI voice per stem, so a transcribed layer plays back as
# something resembling itself rather than all six landing on piano.
GM_FOR_STEM = {
    "vocals": 53,   # Voice Oohs -- the nearest GM has to a sung line
    "bass": 33,     # Finger bass
    "drums": 0,     # Ignored: drums go to channel 10
    "guitar": 27,   # Clean guitar
    "piano": 0,     # Acoustic grand
    "other": 48,    # Strings, as a neutral catch-all
}

# Transcribing a stem is only worth doing where pitch is meaningful.
_PITCHED_STEMS = ("vocals", "bass", "guitar", "piano", "other")


@dataclass
class StemResult:
    name: str
    audio_path: Path | None = None
    midi_path: Path | None = None
    duration: float = 0.0
    peak_db: float = 0.0
    note_count: int = 0
    transcription_skipped: str = ""


@dataclass
class DeconstructResult:
    source: Path
    stems: list[StemResult] = field(default_factory=list)
    analysis: AudioAnalysis | None = None
    model: str = ""

    def paths(self) -> list[Path]:
        out: list[Path] = []
        for stem in self.stems:
            if stem.audio_path:
                out.append(stem.audio_path)
            if stem.midi_path:
                out.append(stem.midi_path)
        return out


class DeconstructGenerator(Generator):
    """Separates a recording into layers, and optionally transcribes each.

    Implemented as a Generator so it reuses the existing queue, progress
    reporting and cancellation rather than growing a parallel job system. The
    input arrives as ``request.extra["input_path"]``.
    """

    id = "deconstruct"
    name = "Deconstruct"
    kind = "separator"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            outputs=("flac", "wav", "midi"),
            max_duration=float(getattr(self.entry, "max_duration", 1800.0)),
            supports_seed=False,
            needs_gpu=False,
            sample_rate=int(getattr(self.entry, "sample_rate", 44100)),
        )

    def required_packages(self) -> list[str]:
        return ["torch", "demucs"]

    def missing_packages(self) -> list[str]:
        from ..core.worker_client import probe_runtime, runtime_python

        if runtime_python() is None:
            return ["compute runtime"]
        info = probe_runtime()
        if not info:
            return ["compute runtime"]
        return [p for p in ("torch", "demucs") if not info.get(p)]

    def is_ready(self, ctx: GeneratorContext | None = None) -> bool:
        return not self.missing_packages()

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        source = request.extra.get("input_path")
        duration = _probe_duration(source) if source else 180.0
        limit = float(request.extra.get("max_seconds") or 0)
        if limit > 0:
            duration = min(duration, limit)
        info = _probe()
        on_gpu = bool(info.get("cuda"))
        # Separation is roughly real-time on a GPU and several times slower on
        # CPU; transcription adds a little per pitched stem.
        factor = 0.6 if on_gpu else 4.0
        estimate = duration * factor
        if request.extra.get("transcribe", True):
            estimate += duration * (0.2 if on_gpu else 0.6)
        return max(20.0, estimate)

    # -- work ---------------------------------------------------------------

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        source = Path(str(request.extra.get("input_path", "")))
        if not source.exists():
            raise BackendUnavailable(f"No such audio file: {source}")

        model = getattr(self.entry, "repo", None) or "htdemucs"
        out_root = Path(str(request.extra.get("output_dir") or source.parent))
        want_midi = bool(request.extra.get("transcribe", True))
        want_analysis = bool(request.extra.get("analyse", True))
        fmt = request.output_format if request.output_format is not OutputFormat.MIDI \
            else OutputFormat.FLAC

        folder = out_root / safe_filename(f"{source.stem} stems", fallback="stems")
        folder.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="midimusic-stems-") as tmp:
            ctx.report(0.02, f"Separating {source.name}", "separate")
            # Separating six minutes of audio on a CPU is a long wait, so let
            # the caller ask for only the opening of a long recording.
            separate_from = _trim_source(source, request.extra.get("max_seconds"), Path(tmp))
            try:
                worker = run_worker(
                    {
                        "cmd": "separate",
                        "model": model,
                        "input_path": str(separate_from),
                        "output_dir": tmp,
                        "device": ctx.device,
                    },
                    on_progress=lambda f, m, s: ctx.report(f * 0.6, m, s),
                    should_cancel=ctx.cancelled,
                )
            except WorkerCancelled as exc:
                raise GenerationCancelled() from exc
            except WorkerError as exc:
                if exc.remote_traceback:
                    log.error("separation traceback:\n%s", exc.remote_traceback)
                raise

            stem_files: dict[str, str] = dict(worker.meta.get("stems") or {})
            if not stem_files:
                raise WorkerError("Separation produced no stems.")

            result = DeconstructResult(source=source, model=model)
            options = ExportOptions(
                sample_rate=int(request.extra.get("sample_rate", 44100)),
                bit_depth=int(request.extra.get("bit_depth", 24)),
                # Stems are components of a mix, not masters: rescaling each
                # one would destroy their relative levels and make them
                # impossible to recombine. That means switching normalisation
                # off outright -- leaving target_lufs as None only swaps the
                # loudness target for peak normalisation, which rescales them
                # just as thoroughly.
                target_lufs=None,
                normalize=False,
                trim=False,
                fade_in=0.0,
                fade_out=0.0,
            )

            names = sorted(stem_files)
            for index, name in enumerate(names):
                ctx.check_cancelled()
                share = 0.6 + 0.35 * (index / max(1, len(names)))
                ctx.report(share, f"Writing {name}", "export")

                data, rate = sf.read(stem_files[name], dtype="float32", always_2d=True)
                buffer = AudioBuffer(np.ascontiguousarray(data), int(rate))
                audio_path = export_audio(
                    buffer, folder / f"{name}{fmt.extension}", fmt, options,
                    TrackMetadata(
                        title=f"{source.stem} - {name}",
                        album=source.stem,
                        comment=f"{name} stem, separated with {model}",
                    ),
                )

                stem = StemResult(
                    name=name,
                    audio_path=audio_path,
                    duration=buffer.duration_seconds,
                    peak_db=float(20 * np.log10(max(float(np.abs(data).max()), 1e-9))),
                )

                if want_midi:
                    stem.midi_path, stem.note_count, stem.transcription_skipped = (
                        self._transcribe_stem(name, stem_files[name], folder, ctx)
                    )
                result.stems.append(stem)

            if want_analysis:
                ctx.report(0.96, "Analysing tempo and key", "analyse")
                mix, mix_rate = sf.read(str(separate_from), dtype="float32", always_2d=True)
                result.analysis = analyze_audio(mix, int(mix_rate))

        ctx.report(1.0, "Done", "deconstruct")
        paths = result.paths()
        meta = {
            "model": model,
            "stems": [s.name for s in result.stems],
            "source": str(source),
            "folder": str(folder),
        }
        if result.analysis:
            meta.update(
                tempo=result.analysis.tempo,
                key=result.analysis.key,
                key_confidence=result.analysis.key_confidence,
                analysis=result.analysis.describe(),
            )

        out = self._result(
            request, started,
            title=f"{source.stem} (stems)",
            duration_seconds=max((s.duration for s in result.stems), default=0.0),
            meta=meta,
        )
        out.paths = paths
        return out

    def _transcribe_stem(
        self, name: str, wav_path: str, folder: Path, ctx: GeneratorContext
    ) -> tuple[Path | None, int, str]:
        """Turn one stem into notes, where that makes sense."""
        if name == "drums":
            # Pitch tracking on drums produces nonsense. A drum transcriber is
            # a different model, not a different threshold.
            return None, 0, "drums need a percussion transcriber, not pitch tracking"
        if name not in _PITCHED_STEMS:
            return None, 0, "not a pitched layer"

        from ..audio.transcribe import is_available, transcribe_file

        if not is_available():
            return None, 0, "basic-pitch is not installed"

        song: Song | None = transcribe_file(wav_path, title=f"{name} transcription")
        if song is None or not song.tracks:
            return None, 0, "nothing pitched was detected"

        program = GM_FOR_STEM.get(name, 0)
        for track in song.tracks:
            track.name = name.capitalize()
            track.program = program
            track.role = "lead" if name == "vocals" else name
        song.title = f"{name} transcription"
        song.meta["stem"] = name

        path = write_midi(song, folder / f"{name}.mid")
        return path, song.note_count, ""


def _probe() -> dict:
    from ..core.worker_client import probe_runtime

    try:
        return probe_runtime()
    except Exception:
        return {}


def _trim_source(source: Path, max_seconds, folder: Path) -> Path:
    """The recording, or its opening, as a file the worker can read."""
    try:
        limit = float(max_seconds or 0)
    except (TypeError, ValueError):
        return source
    if limit <= 0:
        return source
    try:
        info = sf.info(str(source))
        if info.frames <= limit * info.samplerate:
            return source
        data, rate = sf.read(str(source), frames=int(limit * info.samplerate),
                             dtype="float32", always_2d=True)
        clipped = folder / f"{source.stem}-trimmed.wav"
        sf.write(str(clipped), data, int(rate), subtype="FLOAT")
        return clipped
    except Exception:
        log.exception("could not trim %s; separating the whole recording", source)
        return source


def _probe_duration(path: str | Path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames) / max(1, info.samplerate)
    except Exception:
        return 180.0
