"""Keeping part of a recording and generating the rest.

The idea is narrow and useful: separation can isolate a real vocal, and a
generator can write a new backing for it. Keep the singer, replace the band,
and a film cue becomes a dance track with the original performance still on
top of it.

Two things make or break the result, and both are handled here rather than
left to the model:

* **Level.** The new backing has to arrive at the loudness of the parts it
  replaced, or the kept vocal ends up either buried or naked. That is a
  relative measurement against the discarded stems, not a normalisation target.
* **Timing.** Nothing here can make a waveform model land on the original's
  downbeats. What it can do is tell every backend the tempo and key the
  recording was actually in, and prefer a backend that honours them -- the
  built-in composer writes at exactly the tempo it is given, so on material
  with a steady pulse the new backing stays with the vocal. A neural audio
  model will drift, and the app says so rather than pretending otherwise.
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import dsp
from ..audio.analyze import AudioAnalysis, analyze_audio, beat_phase
from ..audio.export import ExportOptions, TrackMetadata, export_audio, safe_filename
from ..audio.timestretch import MAX_STRETCH, MIN_STRETCH, time_stretch
from .catalog import load_catalog
from .deconstruct import _trim_source
from .generator import (
    BackendUnavailable,
    Capabilities,
    GenerationCancelled,
    Generator,
    GeneratorContext,
)
from .models import AudioBuffer, GenerationRequest, GenerationResult, OutputFormat
from .registry import create_generator
from .worker_client import WorkerCancelled, WorkerError, run_worker

__all__ = ["DEFAULT_BED_MODEL", "DEFAULT_KEEP", "RemixGenerator", "condition_prompt"]

log = logging.getLogger(__name__)

# The vocal is the part worth keeping in nearly every case: it carries the
# performance, and it is the one layer a generator cannot replace convincingly.
DEFAULT_KEEP: tuple[str, ...] = ("vocals",)

# The built-in composer, because it is the only backend that writes at exactly
# the tempo it is handed -- which is what keeps the new backing under the kept
# vocal rather than drifting away from it.
DEFAULT_BED_MODEL = "builtin-composer"


@dataclass
class RemixPlan:
    kept: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    analysis: AudioAnalysis | None = None
    repeats: int = 1
    bed_model: str = ""
    drifts: bool = False
    beat_offset: float = 0.0
    source_tempo: float = 0.0
    tempo: float = 0.0
    stretch: float = 1.0
    silent_kept: list[str] = field(default_factory=list)


def condition_prompt(prompt: str, analysis: AudioAnalysis | None,
                     tempo: float = 0.0) -> str:
    """Add the tempo and key the backing must be written at to a prompt.

    Text is the only handle a waveform model offers on either, so a prompt that
    does not mention them is asking the model to guess. ``tempo`` overrides the
    recording's own, for a remix that is moving it.
    """
    prompt = (prompt or "").strip() or "instrumental backing"
    if analysis is None and not tempo:
        return prompt
    bits = []
    beats = tempo or (analysis.tempo if analysis else 0.0)
    if beats:
        bits.append(f"{beats:.0f} bpm")
    if analysis is not None and analysis.key:
        bits.append(f"in {analysis.key}")
    return f"{prompt}, {', '.join(bits)}" if bits else prompt


def _target_tempo(wanted, source_tempo: float) -> float:
    """The tempo a remix should actually run at, or 0 to leave it alone.

    A phase vocoder holds up over a moderate change and falls apart past one,
    so an impossible ask is met with the closest tempo that still sounds like
    the same performance rather than with mush -- and the result records both
    numbers so the difference is visible.
    """
    try:
        target = float(wanted or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if target <= 0 or source_tempo <= 0 or abs(target - source_tempo) < 0.5:
        return 0.0
    lowest = source_tempo / MAX_STRETCH
    highest = source_tempo / MIN_STRETCH
    return round(float(min(max(target, lowest), highest)), 2)


class RemixGenerator(Generator):
    """Separates a recording, keeps some layers, and generates the rest.

    A Generator like the others, so the queue, progress and cancellation all
    work unchanged. The recording arrives as ``request.extra["input_path"]``.
    """

    id = "remix"
    name = "Remix"
    kind = "remix"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            outputs=("flac", "wav"),
            max_duration=float(getattr(self.entry, "max_duration", 1800.0)),
            supports_seed=True,
            needs_gpu=False,
            sample_rate=int(getattr(self.entry, "sample_rate", 44100)),
        )

    def required_packages(self) -> list[str]:
        return ["torch", "demucs"]

    def missing_packages(self) -> list[str]:
        from .worker_client import probe_runtime, runtime_python

        if runtime_python() is None:
            return ["compute runtime"]
        info = probe_runtime()
        if not info:
            return ["compute runtime"]
        return [p for p in ("torch", "demucs") if not info.get(p)]

    def is_ready(self, ctx: GeneratorContext | None = None) -> bool:
        return not self.missing_packages()

    # -- planning -----------------------------------------------------------

    def _bed_generator(self, request: GenerationRequest):
        model_id = str(request.extra.get("bed_model") or DEFAULT_BED_MODEL)
        entry = load_catalog().get(model_id)
        generator = create_generator(entry) if entry else None
        if generator is None:
            raise BackendUnavailable(f"No backing generator called {model_id!r}.")
        return entry, generator

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        from .deconstruct import _probe_duration

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
        separation = duration * (0.6 if on_gpu else 4.0)

        try:
            _entry, bed = self._bed_generator(request)
        except BackendUnavailable:
            return max(20.0, separation)
        probe = GenerationRequest.from_dict(request.to_dict())
        probe.duration_seconds = min(duration, bed.capabilities().max_duration)
        return max(20.0, separation + bed.estimated_seconds(probe, ctx))

    # -- work ---------------------------------------------------------------

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        source = Path(str(request.extra.get("input_path", "")))
        if not source.exists():
            raise BackendUnavailable(f"No such audio file: {source}")

        # `or DEFAULT_KEEP` would be wrong here: it cannot tell a missing key
        # from a user who deliberately cleared every box, and clearing every
        # box is the natural way to ask for the whole thing to be rewritten.
        wanted = request.extra.get("keep_stems")
        keep = tuple(DEFAULT_KEEP if wanted is None else wanted)
        separator_id = str(request.extra.get("separator") or "demucs-htdemucs")
        separator = load_catalog().get(separator_id)
        model = getattr(separator, "repo", "") or "htdemucs"
        bed_entry, bed_generator = self._bed_generator(request)

        with tempfile.TemporaryDirectory(prefix="midimusic-remix-") as tmp:
            folder = Path(tmp)
            clip = _trim_source(source, request.extra.get("max_seconds"), folder)

            ctx.report(0.02, f"Separating {source.name}", "separate")
            try:
                worker = run_worker(
                    {"cmd": "separate", "model": model, "input_path": str(clip),
                     "output_dir": str(folder / "stems"), "device": ctx.device},
                    on_progress=lambda f, m, s: ctx.report(0.02 + f * 0.36, m, s),
                    should_cancel=ctx.cancelled,
                )
            except WorkerCancelled as exc:
                raise GenerationCancelled() from exc
            except WorkerError as exc:
                if exc.remote_traceback:
                    log.error("separation traceback:\n%s", exc.remote_traceback)
                raise

            stem_files = dict(worker.meta.get("stems") or {})
            if not stem_files:
                raise WorkerError("Separation produced no stems.")

            kept_names = [n for n in sorted(stem_files) if n in keep]
            replaced_names = [n for n in sorted(stem_files) if n not in keep]
            if not replaced_names:
                raise BackendUnavailable(
                    "Every layer was kept, so there is nothing to regenerate. "
                    "Clear at least one layer."
                )

            rate = int(worker.meta.get("sample_rate") or 44100)
            kept = [_read(stem_files[n]) for n in kept_names]
            replaced = [_read(stem_files[n]) for n in replaced_names]
            frames = max((layer.shape[0] for layer in kept + replaced), default=0)

            ctx.check_cancelled()
            ctx.report(0.40, "Listening for tempo and key", "analyse")
            mix_samples, mix_rate = sf.read(str(clip), dtype="float32", always_2d=True)
            analysis = analyze_audio(mix_samples, int(mix_rate))

            plan = RemixPlan(
                kept=kept_names, replaced=replaced_names, analysis=analysis,
                bed_model=getattr(bed_entry, "id", ""),
                source_tempo=analysis.tempo, tempo=analysis.tempo,
            )
            # A kept layer that turned out to be silence is worth saying out
            # loud: an instrumental still produces a vocals stem, and keeping
            # it looks like keeping something.
            plan.silent_kept = [
                name for name, layer in zip(kept_names, kept, strict=True)
                if float(np.abs(layer).max() if layer.size else 0.0) < 1e-3
            ]

            # Where the recording's pulse actually falls. A generated backing
            # starts its first beat at zero and a performance almost never
            # does, so without this the two are permanently out of phase.
            plan.beat_offset = beat_phase(mix_samples, int(mix_rate), analysis.tempo)

            target = _target_tempo(request.extra.get("target_tempo"), analysis.tempo)
            if target:
                ctx.report(0.42, f"Re-timing to {target:.0f} bpm", "stretch")
                plan.stretch = analysis.tempo / target
                plan.tempo = target
                # Everything kept has to move with the new tempo. Resampling
                # would do it by playing faster, which raises the pitch and
                # puts the kept layer in a different key from the backing --
                # so the length changes and the frequencies do not.
                kept = [time_stretch(layer, plan.stretch) for layer in kept]
                frames = max((layer.shape[0] for layer in kept), default=frames)
                plan.beat_offset *= plan.stretch

            bed, plan.repeats, plan.drifts = self._make_bed(
                request, ctx, bed_generator, analysis, frames, rate,
                plan.beat_offset, plan.tempo,
            )

            ctx.check_cancelled()
            ctx.report(0.92, "Balancing and mixing", "mix")
            reference = dsp.mix([dsp.to_stereo(layer) for layer in replaced]) \
                if replaced else np.zeros((frames, 2), dtype=np.float32)
            bed = dsp.match_loudness(bed, reference, rate)

            layers = [dsp.to_stereo(layer)[:frames] for layer in kept]
            mixed, gain = dsp.mix_layers([*layers, bed[:frames]])
            buffer = AudioBuffer(mixed, rate)

            extras: list[Path] = []
            if request.extra.get("save_stems"):
                # The same gain the mix took to stay under the ceiling, so the
                # saved parts still add up to the mix they came from.
                extras = self._save_stems(
                    request, ctx, source, kept_names,
                    [layer * gain for layer in layers], bed[:frames] * gain, rate,
                )

        ctx.report(1.0, "Done", "remix")
        out = self._result(
            request, started,
            audio=buffer,
            title=f"{source.stem} ({_verb(request.prompt)})",
            duration_seconds=buffer.duration_seconds,
            meta={
                "source": str(source),
                "kept": plan.kept,
                "replaced": plan.replaced,
                "separator": separator_id,
                "bed_model": plan.bed_model,
                "bed_repeats": plan.repeats,
                "beat_locked": not plan.drifts,
                "beat_offset": plan.beat_offset,
                "tempo": plan.tempo,
                "source_tempo": plan.source_tempo,
                "time_stretched": plan.stretch != 1.0,
                "silent_kept": plan.silent_kept,
                "key": analysis.key,
                "analysis": analysis.describe(),
            },
        )
        out.paths = extras
        return out

    def _make_bed(self, request: GenerationRequest, ctx: GeneratorContext,
                  generator: Generator, analysis: AudioAnalysis, frames: int,
                  rate: int, beat_offset: float = 0.0,
                  tempo: float = 0.0) -> tuple[np.ndarray, int, bool]:
        """Generate the replacement backing and fit it to the kept material."""
        caps = generator.capabilities()
        wanted = frames / max(1, rate) + max(0.0, beat_offset)
        # Ask for a little more than is needed. Backends that quantise to whole
        # bars land a few per cent short, and trimming an overshoot is silent
        # where looping to cover a shortfall is not.
        length = wanted * 1.1
        if caps.max_duration:
            length = min(length, caps.max_duration)

        bed_request = GenerationRequest.from_dict(request.to_dict())
        bed_request.model_id = getattr(generator, "model_id", "") or None
        bed_request.output_format = OutputFormat.WAV
        bed_request.duration_seconds = length
        bed_request.min_duration_seconds = None
        bed_request.max_duration_seconds = None
        bed_request.variations = 1
        bed_request.instrumental = True
        bed_request.lyrics = ""
        # The backing is written at the tempo the finished remix runs at,
        # which is the target when one was asked for and the recording's own
        # otherwise.
        wanted = float(tempo or analysis.tempo or 0.0)
        bed_request.prompt = condition_prompt(request.prompt, analysis, wanted)
        # Explicit fields beat the prompt in every backend that reads them, so
        # a backend that can be held to the tempo and key is held to them.
        if wanted:
            bed_request.tempo = wanted
        if analysis.key:
            bed_request.key = analysis.key
        bed_request.extra = {
            k: v for k, v in (request.extra or {}).items()
            if k in ("complexity", "energy", "brightness", "steps", "roles")
        }

        child = GeneratorContext(
            device=ctx.device, models_dir=ctx.models_dir, soundfont=ctx.soundfont,
            hf_token=ctx.hf_token, offline=ctx.offline, cancelled=ctx.cancelled,
            progress=lambda p: ctx.report(0.45 + 0.45 * max(0.0, p.fraction),
                                          p.message or "Writing the new backing",
                                          p.stage or "backing"),
        )
        ctx.report(0.45, "Writing the new backing", "backing")
        result = generator.generate(bed_request, child)

        samples, bed_rate = _result_audio(result, generator, child)
        samples = dsp.to_stereo(samples)
        if bed_rate != rate:
            samples = dsp.resample(samples, bed_rate, rate)

        # Tile first, then place. Baking the lead-in into the clip before
        # tiling would make the repeated unit `lead + length` long, so every
        # repetition would start early by the lead and the backing would walk
        # off the beat it was aligned to.
        lead = min(int(max(0.0, beat_offset) * rate), max(0, frames - 1))
        body = dsp.fit_length(samples, max(1, frames - lead), rate)
        repeats = dsp.tiles_needed(samples.shape[0], max(1, frames - lead), rate)
        if lead:
            body = np.concatenate(
                [np.zeros((lead, *body.shape[1:]), dtype=np.float32), body]
            )
        fitted = body[:frames]

        # Only a backend that was given the tempo and writes to it stays with
        # the kept layer. Tiling loses a crossfade's worth of time at each
        # join, so a looped backing does not stay locked either however good
        # the backend is -- say so rather than reporting the backend's
        # capability as though it were the outcome.
        drifts = not (caps.honours_tempo and bool(tempo or analysis.tempo)) or repeats > 1
        return fitted, repeats, drifts

    def _save_stems(self, request: GenerationRequest, ctx: GeneratorContext,
                    source: Path, kept_names: list[str], kept: list[np.ndarray],
                    bed: np.ndarray, rate: int) -> list[Path]:
        """Write the parts as well as the mix, for anyone who wants to re-balance."""
        out_root = Path(str(request.extra.get("output_dir") or source.parent))
        folder = _unique_folder(out_root, f"{source.stem} remix parts")
        options = ExportOptions(
            sample_rate=int(request.extra.get("sample_rate", rate)),
            bit_depth=int(request.extra.get("bit_depth", 24)),
            # Parts of a mix: their levels relative to each other are the
            # information, so nothing may rescale them, and a fade would move
            # the ends. They have to sum back to the mix that was delivered.
            target_lufs=None, normalize=False, trim=False,
            fade_in=0.0, fade_out=0.0,
        )
        written: list[Path] = []
        parts = [*zip(kept_names, kept, strict=True), ("new backing", bed)]
        try:
            for name, layer in parts:
                ctx.check_cancelled()
                written.append(
                    export_audio(
                        AudioBuffer(dsp.to_stereo(layer)[: bed.shape[0]], rate),
                        folder / f"{safe_filename(name, fallback='layer')}.flac",
                        OutputFormat.FLAC, options,
                        TrackMetadata(title=f"{source.stem} - {name}", album=source.stem),
                    )
                )
        except BaseException:
            # These go straight into the user's output folder, and a cancelled
            # job never reaches the library -- so anything already written
            # would sit there unreferenced with no way to clear it from the
            # app. Take it back out on the way past.
            for path in written:
                path.unlink(missing_ok=True)
            with suppress(OSError):
                folder.rmdir()
            raise
        return written


def _unique_folder(parent: Path, name: str) -> Path:
    """A folder of this name that nothing else has written into.

    Two remixes of the same song would otherwise share one parts folder: the
    second would overwrite the first's files, and deleting either job from the
    library with "delete files" would take the other's parts with it.
    """
    base = safe_filename(name, fallback="remix")
    folder = parent / base
    n = 2
    while folder.exists():
        folder = parent / f"{base} {n}"
        n += 1
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _verb(prompt: str) -> str:
    text = (prompt or "").strip()
    return (text[:40] + "...") if len(text) > 40 else (text or "remix")


def _read(path: str) -> np.ndarray:
    data, _rate = sf.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(data)


def _result_audio(result: GenerationResult, generator: Generator,
                  ctx: GeneratorContext) -> tuple[np.ndarray, int]:
    """The backing as samples, rendering a symbolic backend's score if needed."""
    if result.audio is not None:
        return np.asarray(result.audio.samples, dtype=np.float32), result.audio.sample_rate
    if result.song is None:
        raise BackendUnavailable(
            f"{getattr(generator, 'name', 'The backing generator')} produced nothing to mix."
        )
    if ctx.soundfont is not None:
        try:
            from ..audio.render import render_song

            rendered = render_song(result.song, ctx.soundfont,
                                   should_cancel=ctx.cancelled)
            return np.asarray(rendered.samples, dtype=np.float32), rendered.sample_rate
        except Exception:
            log.exception("SoundFont render failed; using the built-in synth")
    from ..audio.synth_fallback import render_song_fallback

    rendered = render_song_fallback(result.song, should_cancel=ctx.cancelled)
    return np.asarray(rendered.samples, dtype=np.float32), rendered.sample_rate
