"""Application service: the layer between the UI and the generators.

Owns the queue, decides how a result becomes files on disk, and maintains the
library.  Keeping this Qt-free means the whole generation path can be tested
without a display.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..audio.export import ExportOptions, TrackMetadata, export_audio, safe_filename
from ..audio.midi_io import write_midi
from ..config.paths import get_paths
from ..config.settings import Settings, load_settings
from .catalog import Catalog, ModelEntry, load_catalog
from .generator import Generator, GeneratorContext
from .hardware import SystemInfo, detect_system, recommended_backend
from .jobs import Job, JobQueue
from .models import AudioBuffer, GenerationRequest, OutputFormat, Song
from .registry import create_generator

__all__ = ["AppService", "LibraryItem"]

log = logging.getLogger(__name__)


@dataclass
class LibraryItem:
    """One finished generation, as recorded in the library."""

    id: str = ""
    title: str = "Untitled"
    prompt: str = ""
    backend: str = ""
    model: str = ""
    style: str = ""
    key: str = ""
    tempo: float = 0.0
    duration: float = 0.0
    seed: int | None = None
    created_at: float = field(default_factory=time.time)
    paths: list[str] = field(default_factory=list)
    favourite: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def audio_path(self) -> Path | None:
        for p in self.paths:
            if p.lower().endswith((".flac", ".wav")):
                return Path(p)
        return None

    @property
    def midi_path(self) -> Path | None:
        for p in self.paths:
            if p.lower().endswith(".mid"):
                return Path(p)
        return None

    def exists(self) -> bool:
        return any(Path(p).exists() for p in self.paths)


class AppService:
    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None):
        self.paths = get_paths()
        self.settings = settings or load_settings()
        self.catalog = catalog or load_catalog()
        self.system: SystemInfo = detect_system()
        self.queue = JobQueue(post_process=self._write_outputs)
        self.library: list[LibraryItem] = []
        self._load_library()

    # -- capability ---------------------------------------------------------

    def recommended_compute(self) -> str:
        if self.settings.compute_backend != "auto":
            return self.settings.compute_backend
        return recommended_backend(self.system)

    def vram_gb(self) -> float:
        gpu = self.system.primary_gpu
        return gpu.vram_gb if gpu else 0.0

    def usable_models(self, output_format: str | None = None) -> list[ModelEntry]:
        device = self.recommended_compute()
        return self.catalog.available(device, self.vram_gb(), output_format)

    def soundfont_path(self) -> Path | None:
        if self.settings.soundfont:
            p = Path(self.settings.soundfont)
            if p.exists():
                return p
        for candidate in sorted(self.paths.soundfonts.glob("*.sf*")):
            return candidate
        return None

    # -- submission ---------------------------------------------------------

    def make_context(self) -> GeneratorContext:
        return GeneratorContext(
            device=self.settings.device or "auto",
            models_dir=self.settings.resolved_models_dir(),
            soundfont=self.soundfont_path(),
            hf_token=self.settings.hf_token,
            offline=self.settings.offline,
        )

    def generator_for(self, model_id: str) -> tuple[ModelEntry | None, Generator | None]:
        entry = self.catalog.get(model_id)
        if entry is None:
            return None, None
        return entry, create_generator(entry)

    def submit(self, request: GenerationRequest, model_id: str | None = None) -> list[Job]:
        """Queue a request, expanding ``variations`` into separate jobs."""
        model_id = model_id or request.model_id or self.settings.default_backend
        entry, generator = self.generator_for(model_id)
        if generator is None:
            entry, generator = self.generator_for("builtin-composer")
        if generator is None:
            raise RuntimeError("No usable generator is available.")

        jobs: list[Job] = []
        count = max(1, int(request.variations or 1))
        for i in range(count):
            variant = GenerationRequest.from_dict(request.to_dict())
            variant.model_id = model_id
            variant.variations = 1
            # Give each variation its own seed so they differ but stay
            # reproducible from the seed the user seeded them with.
            if request.seed is not None:
                variant.seed = int(request.seed) + i
            label = request.prompt or (entry.name if entry else model_id)
            if count > 1:
                label = f"{label} ({i + 1}/{count})"
            jobs.append(self.queue.submit(variant, generator, self.make_context(), label))
        return jobs

    # -- output -------------------------------------------------------------

    def _write_outputs(self, job: Job) -> list[Path]:
        """Turn a finished result into files, converting between formats.

        A symbolic backend asked for FLAC is rendered through a SoundFont; an
        audio backend asked for MIDI is transcribed. That conversion lives here
        so no generator has to care what format was requested.
        """
        result = job.result
        if result is None:
            return []

        out_dir = self.settings.resolved_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self._unique_stem(out_dir, result.title or job.display_name())
        fmt = job.request.output_format
        written: list[Path] = []

        metadata = TrackMetadata(
            title=result.title or stem,
            genre=str(result.meta.get("style", "")),
            comment=job.request.prompt,
            date=time.strftime("%Y-%m-%d"),
            extra={"seed": str(result.seed), "backend": result.backend},
        )
        options = ExportOptions(
            sample_rate=self.settings.sample_rate,
            bit_depth=self.settings.bit_depth,
            target_lufs=self.settings.target_lufs,
        )

        song: Song | None = result.song
        audio: AudioBuffer | None = result.audio

        if fmt is OutputFormat.MIDI:
            if song is None and audio is not None:
                song = self._transcribe(audio, job)
            if song is not None:
                written.append(write_midi(song, out_dir / f"{stem}.mid"))
        else:
            if audio is None and song is not None:
                audio = self._render(song, job)
            if audio is not None:
                written.append(
                    export_audio(audio, out_dir / f"{stem}{fmt.extension}", fmt,
                                 options, metadata)
                )
            # A symbolic backend can hand over the score for free, so do.
            if song is not None and self.settings.also_write_midi:
                written.append(write_midi(song, out_dir / f"{stem}.mid"))

        if written:
            self._record(job, result, written)
        return written

    def _render(self, song: Song, job: Job) -> AudioBuffer:
        soundfont = self.soundfont_path()
        job.context.report(0.7, "Rendering audio", "render")
        if soundfont is not None:
            try:
                from ..audio.render import render_song

                return render_song(
                    song, soundfont,
                    progress=lambda f: job.context.report(0.7 + 0.25 * f, "Rendering", "render"),
                    should_cancel=job.context.cancelled,
                )
            except Exception:
                log.exception("SoundFont render failed; falling back to built-in synth")
        from ..audio.synth_fallback import render_song_fallback

        return render_song_fallback(
            song,
            sample_rate=self.settings.sample_rate,
            progress=lambda f: job.context.report(0.7 + 0.25 * f, "Rendering", "render"),
            should_cancel=job.context.cancelled,
        )

    def _transcribe(self, audio: AudioBuffer, job: Job) -> Song | None:
        job.context.report(0.7, "Transcribing to MIDI", "transcribe")
        try:
            from ..audio.transcribe import transcribe_audio

            return transcribe_audio(audio, progress=lambda f: job.context.report(
                0.7 + 0.25 * f, "Transcribing", "transcribe"))
        except Exception:
            log.exception("transcription failed")
            return None

    def _unique_stem(self, out_dir: Path, title: str) -> str:
        base = safe_filename(title, fallback="untitled")
        stem = base
        n = 2
        while any((out_dir / f"{stem}{ext}").exists() for ext in (".flac", ".wav", ".mid")):
            stem = f"{base} {n}"
            n += 1
        return stem

    # -- library ------------------------------------------------------------

    def _record(self, job: Job, result, paths: list[Path]) -> None:
        item = LibraryItem(
            id=job.id,
            title=result.title or job.display_name(),
            prompt=job.request.prompt,
            backend=result.backend,
            model=job.request.model_id or "",
            style=str(result.meta.get("style", "")),
            key=str(result.meta.get("key", "")),
            tempo=float(result.meta.get("tempo", 0.0) or 0.0),
            duration=result.duration_seconds,
            seed=result.seed,
            paths=[str(p) for p in paths],
            meta=dict(result.meta),
        )
        self.library.insert(0, item)
        self._save_library()

    def _load_library(self) -> None:
        try:
            raw = json.loads(self.paths.library_db.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.library = []
            return
        items = []
        known = set(LibraryItem.__dataclass_fields__)
        for entry in raw.get("items", []):
            items.append(LibraryItem(**{k: v for k, v in entry.items() if k in known}))
        self.library = items

    def _save_library(self) -> None:
        try:
            payload = {"items": [asdict(i) for i in self.library[:2000]]}
            tmp = self.paths.library_db.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            tmp.replace(self.paths.library_db)
        except OSError:
            log.exception("could not save library")

    def remove_from_library(self, item_id: str, delete_files: bool = False) -> None:
        for item in list(self.library):
            if item.id != item_id:
                continue
            if delete_files:
                for p in item.paths:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass
            self.library.remove(item)
        self._save_library()

    def toggle_favourite(self, item_id: str) -> None:
        for item in self.library:
            if item.id == item_id:
                item.favourite = not item.favourite
        self._save_library()

    def shutdown(self) -> None:
        self.queue.stop()
        self._save_library()
