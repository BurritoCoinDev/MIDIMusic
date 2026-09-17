"""Taking a recording apart into the sections of an orchestra.

Separation and transcription answer different questions. Demucs asks "which
audio belongs to the singer", which is the right question for a band and the
wrong one for an orchestra: a film cue has no drum kit, no bass guitar and no
vocal, and every desk in the room lands in the same "other" stem.

A multi-instrument transcriber asks "what was played, and by what kind of
instrument", and that is the question a symphonic score answers. The result is
notes rather than audio -- strings as a body, brass as a section, percussion on
its own -- which is exactly the set of layers a score is written in.

What it cannot do is tell the second violins from the firsts. The available
granularity is the instrument family, so layers are named for the family and
the estimate is presented as one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from ..audio.analyze import AudioAnalysis, analyze_audio
from ..audio.export import safe_filename
from ..audio.midi_io import write_midi
from ..theory.orchestra import family_name, section_for, sort_key
from .generator import (
    BackendUnavailable,
    Capabilities,
    GenerationCancelled,
    Generator,
    GeneratorContext,
)
from .models import GenerationRequest, GenerationResult, Note, Song, Track
from .worker_client import WorkerCancelled, WorkerError, run_worker

__all__ = ["ScoreGenerator", "ScoreResult", "SectionLayer", "build_sections"]

log = logging.getLogger(__name__)

# MIDI channel 10 (index 9) is percussion by definition; everything else gets
# one of the remaining fifteen.
_DRUM_CHANNEL = 9


@dataclass
class SectionLayer:
    name: str
    song: Song
    path: Path | None = None
    instruments: list[str] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return self.song.note_count


@dataclass
class ScoreResult:
    source: Path
    layers: list[SectionLayer] = field(default_factory=list)
    full_score: Song | None = None
    full_score_path: Path | None = None
    analysis: AudioAnalysis | None = None
    model: str = ""

    def paths(self) -> list[Path]:
        out = [self.full_score_path] if self.full_score_path else []
        return out + [layer.path for layer in self.layers if layer.path]


# A sung line has no General MIDI program, so a transcriber writing standard
# MIDI has to substitute one -- YourMT3 files the lead under 65 (Alto Sax) and
# a chorus under 53. Its own track label still says what it heard, so the label
# is better evidence than the program number it was forced to pick.
_SUNG_LABELS = ("singing", "vocal")

# What a restored vocal line is written as, so it plays back as a voice rather
# than as the effects patch that sits at program 100 in General MIDI.
_VOICE_PROGRAM = 53  # Voice Oohs


def _is_sung(label: str) -> bool:
    lowered = (label or "").lower()
    return any(word in lowered for word in _SUNG_LABELS)


def _channel_for(index: int, is_drum: bool) -> int:
    """A MIDI channel for the ``index``-th *pitched* track.

    Percussion has channel 10 to itself by definition, so the caller numbers
    the pitched tracks without it. That keeps the numbering independent of
    where the drum track happens to sit among them, and uses the low channels
    first. Fifteen is all there are once percussion has taken its own; beyond
    that two parts must share a channel, and therefore a program.
    """
    if is_drum:
        return _DRUM_CHANNEL
    channel = index % 15
    return channel if channel < _DRUM_CHANNEL else channel + 1


def build_sections(tracks: list[dict], tempo: float = 120.0,
                   title: str = "Score") -> tuple[list[SectionLayer], Song]:
    """Group transcribed instrument tracks into orchestral sections.

    Returns one layer per section plus a single full score holding every
    instrument, laid out in conductor's order.
    """
    beats_per_second = max(1e-6, tempo) / 60.0
    grouped: dict[str, list[Track]] = {}
    ordered: list[Track] = []

    pitched = 0
    for raw in tracks:
        program = int(raw.get("program", 0))
        is_drum = bool(raw.get("is_drum"))
        sung = not is_drum and _is_sung(str(raw.get("name") or ""))
        if sung:
            # Trust what the transcriber said it heard over the program it had
            # to invent, or every vocal melody is filed as a woodwind.
            program = _VOICE_PROGRAM
        section = "Voice" if sung else section_for(program, is_drum)
        label = "Voice" if sung else family_name(program, is_drum)

        track = Track(
            name=label,
            program=0 if is_drum else program,
            channel=_channel_for(pitched, is_drum),
            is_drum=is_drum,
            role=section.lower(),
        )
        if not is_drum:
            pitched += 1
        for pitch, start, duration, velocity in raw.get("notes", []):
            track.notes.append(
                Note(
                    pitch=int(pitch),
                    start=float(start) * beats_per_second,
                    duration=max(0.01, float(duration) * beats_per_second),
                    velocity=int(velocity),
                    channel=track.channel,
                )
            )
        if not track.notes:
            continue
        track.sort()
        grouped.setdefault(section, []).append(track)
        ordered.append(track)

    layers: list[SectionLayer] = []
    for section in sorted(grouped, key=sort_key):
        section_tracks = grouped[section]
        song = Song(
            tracks=list(section_tracks),
            tempo=tempo,
            title=f"{title} - {section}",
            meta={"section": section},
        )
        layers.append(
            SectionLayer(
                name=section,
                song=song,
                instruments=sorted({t.name for t in section_tracks}),
            )
        )

    full = Song(
        # Conductor's order, and within a section the busiest part first.
        tracks=sorted(ordered, key=lambda t: (sort_key(t.role.title()), -len(t.notes))),
        tempo=tempo,
        title=title,
        meta={"sections": [layer.name for layer in layers]},
    )
    return layers, full


class ScoreGenerator(Generator):
    """Transcribes a recording into per-section MIDI layers.

    A Generator like any other, so it reuses the queue, the progress reporting
    and the cancellation the rest of the app already has. The recording arrives
    as ``request.extra["input_path"]``.
    """

    id = "score"
    name = "Orchestral score"
    kind = "transcriber"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            outputs=("midi",),
            max_duration=float(getattr(self.entry, "max_duration", 1800.0)),
            supports_seed=False,
            needs_gpu=False,
        )

    def required_packages(self) -> list[str]:
        return ["torch", "mt3-infer", "transformers<5"]

    def missing_packages(self) -> list[str]:
        from .worker_client import probe_runtime, runtime_python

        if runtime_python() is None:
            return ["compute runtime"]
        info = probe_runtime()
        if not info:
            return ["compute runtime"]
        missing = [p for p in ("torch", "mt3_infer") if not info.get(p)]
        # The vendored transcriber reaches into transformers internals that 5.x
        # removed, so a too-new transformers is as blocking as a missing one --
        # and so is no transformers at all, which an empty version means.
        version = str(info.get("transformers_version") or "")
        major = version.split(".")[0]
        if not info.get("transformers") or not major.isdigit() or int(major) >= 5:
            missing.append("transformers<5")
        return [p.replace("_", "-") for p in missing]

    def is_ready(self, ctx: GeneratorContext | None = None) -> bool:
        return not self.missing_packages()

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        source = request.extra.get("input_path")
        duration = _probe_duration(source) if source else 180.0
        limit = float(request.extra.get("max_seconds") or 0)
        if limit > 0:
            duration = min(duration, limit)
        from .worker_client import probe_runtime

        try:
            on_gpu = bool(probe_runtime().get("cuda"))
        except Exception:
            on_gpu = False
        # Measured at roughly half real time on four CPU cores; a GPU is
        # several times quicker. Loading the checkpoint costs a flat few
        # seconds on top.
        return max(15.0, 8.0 + duration * (0.12 if on_gpu else 0.6))

    # -- work ---------------------------------------------------------------

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        source = Path(str(request.extra.get("input_path", "")))
        if not source.exists():
            raise BackendUnavailable(f"No such audio file: {source}")

        out_root = Path(str(request.extra.get("output_dir") or source.parent))
        folder = out_root / safe_filename(f"{source.stem} score", fallback="score")
        folder.mkdir(parents=True, exist_ok=True)

        entry = self.entry
        payload = {
            "cmd": "transcribe-score",
            "input_path": str(source),
            "repo": getattr(entry, "repo", "") or "mimbres/YourMT3",
            "revision": getattr(entry, "revision", "main"),
            "files": list(getattr(entry, "files", ()) or []),
            "model": str(request.extra.get("backend_key") or "yourmt3"),
            "cache_dir": str(ctx.models_dir / "hub") if ctx.models_dir else "",
            "device": ctx.device,
            "offline": ctx.offline,
            "token": ctx.hf_token,
            "window_seconds": float(request.extra.get("window_seconds", 45.0)),
            "max_seconds": float(request.extra.get("max_seconds") or 0.0),
        }

        try:
            worker = run_worker(
                payload,
                on_progress=lambda f, m, s: ctx.report(f * 0.9, m, s),
                should_cancel=ctx.cancelled,
            )
        except WorkerCancelled as exc:
            raise GenerationCancelled() from exc
        except WorkerError as exc:
            if exc.remote_traceback:
                log.error("transcription traceback:\n%s", exc.remote_traceback)
            raise

        tracks = list(worker.meta.get("tracks") or [])
        if not tracks:
            raise WorkerError("The transcriber found no notes in this recording.")

        analysis: AudioAnalysis | None = None
        tempo = 120.0
        if request.extra.get("analyse", True):
            ctx.report(0.92, "Estimating tempo and key", "analyse")
            samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
            analysis = analyze_audio(samples, int(rate))
            if analysis.tempo:
                # Timing the score to the recording's own tempo means the bars
                # line up when it is opened in a DAW.
                tempo = analysis.tempo

        ctx.check_cancelled()
        ctx.report(0.95, "Writing the score", "export")
        layers, full = build_sections(tracks, tempo=tempo, title=source.stem)
        if analysis and analysis.key:
            full.key = analysis.key
            for layer in layers:
                layer.song.key = analysis.key

        result = ScoreResult(
            source=source, layers=layers, full_score=full,
            analysis=analysis, model=str(payload["repo"]),
        )
        result.full_score_path = write_midi(full, folder / "full score.mid")
        for layer in layers:
            layer.path = write_midi(
                layer.song, folder / f"{safe_filename(layer.name, fallback='layer')}.mid"
            )

        ctx.report(1.0, "Done", "score")
        meta = {
            "model": result.model,
            "folder": str(folder),
            "source": str(source),
            "sections": [layer.name for layer in layers],
            "instruments": {layer.name: layer.instruments for layer in layers},
            "note_counts": {layer.name: layer.note_count for layer in layers},
            "tempo": tempo,
        }
        if analysis:
            meta.update(key=analysis.key, key_confidence=analysis.key_confidence,
                        analysis=analysis.describe())

        out = self._result(
            request, started,
            title=f"{source.stem} (score)",
            duration_seconds=float(worker.meta.get("duration") or full.duration_seconds),
            meta=meta,
        )
        out.paths = result.paths()
        return out


def _probe_duration(path: str | Path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames) / max(1, info.samplerate)
    except Exception:
        return 180.0
