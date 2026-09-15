"""Catalog, generators, the job queue and the service that ties them together."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from midimusic.core.catalog import ModelEntry, load_catalog, save_user_model
from midimusic.core.generator import GeneratorContext
from midimusic.core.hardware import GPU, Vendor, _gfx_for, apply_runtime_probe, detect_system
from midimusic.core.jobs import JobQueue
from midimusic.core.models import (
    GenerationRequest,
    JobStatus,
    OutputFormat,
    resolve_durations,
)
from midimusic.core.registry import ADAPTERS, available_generators, create_generator
from midimusic.core.runtime import RUNTIME_OPTIONS, options_for, resolve_packages
from midimusic.prompt.parser import parse_prompt


def _builtin():
    entry = load_catalog().get("builtin-composer")
    return entry, create_generator(entry)


class TestCatalog:
    def test_every_entry_has_a_registered_adapter(self):
        for entry in load_catalog().models:
            assert entry.adapter in ADAPTERS, f"{entry.id} -> {entry.adapter}"

    def test_every_adapter_class_resolves(self):
        catalog = load_catalog()
        assert len(available_generators(catalog)) == len(catalog.models)

    def test_non_commercial_weights_are_flagged(self):
        entry = load_catalog().get("musicgen-small")
        assert entry.commercial_use is False
        assert "commercial" in entry.license_warning().lower()

    def test_gated_models_are_marked(self):
        assert load_catalog().get("stable-audio-open").gated

    def test_availability_respects_vram(self):
        catalog = load_catalog()
        small = {m.id for m in catalog.available("cuda", 4.0)}
        large = {m.id for m in catalog.available("cuda", 24.0)}
        assert small < large
        assert "builtin-composer" in small  # the floor is always available

    def test_output_filter(self):
        catalog = load_catalog()
        assert all(m.supports("midi") for m in catalog.available("cpu", 0, "midi"))

    def test_user_entries_override_and_persist(self, isolated_paths):
        save_user_model(ModelEntry(id="mine", name="Mine", adapter="hf-musicgen",
                                   repo="me/mine", kind="audio"))
        reloaded = load_catalog()
        assert reloaded.get("mine") is not None
        assert reloaded.get("mine").source == "user"


class TestPromptParsing:
    @pytest.mark.parametrize("text,field,expected", [
        ("dark aggressive metal at 170bpm", "style", "metal"),
        ("dark aggressive metal at 170bpm", "tempo", 170.0),
        ("chill lofi study beats", "style", "lofi"),
        ("drum and bass, 174 bpm", "style", "dnb"),
        ("epic orchestral trailer", "style", "cinematic"),
        ("a song in F# minor", "key", "F# minor"),
        ("2 minutes of ambient", "duration_seconds", 120.0),
    ])
    def test_extraction(self, text, field, expected):
        assert getattr(parse_prompt(text), field) == expected

    def test_longest_match_wins(self):
        # "drum and bass" must beat the bare word "bass".
        assert parse_prompt("drum and bass").style == "dnb"

    def test_vocals_are_detected(self):
        assert parse_prompt("pop song with vocals").instrumental is False
        assert parse_prompt("instrumental pop").instrumental is True

    def test_unknown_text_still_yields_a_usable_style(self):
        assert parse_prompt("asdfgh qwerty").style in ("pop",)


class TestBuiltinGenerator:
    def test_produces_a_multitrack_song(self):
        _entry, gen = _builtin()
        result = gen.generate(
            GenerationRequest(prompt="dark cinematic", duration_seconds=30, seed=1),
            GeneratorContext(),
        )
        assert result.ok
        assert len(result.song.tracks) >= 4
        assert result.song.note_count > 50

    def test_same_seed_is_reproducible(self):
        _entry, gen = _builtin()
        req = GenerationRequest(prompt="lofi", duration_seconds=20, seed=99)
        a = gen.generate(req, GeneratorContext()).song
        b = gen.generate(req, GeneratorContext()).song
        assert [(n.pitch, n.start) for n in a.tracks[0].notes] == \
               [(n.pitch, n.start) for n in b.tracks[0].notes]

    def test_different_seeds_differ(self):
        _entry, gen = _builtin()
        a = gen.generate(GenerationRequest(prompt="lofi", duration_seconds=20, seed=1),
                         GeneratorContext()).song
        b = gen.generate(GenerationRequest(prompt="lofi", duration_seconds=20, seed=2),
                         GeneratorContext()).song
        assert a.note_count != b.note_count or a.tempo != b.tempo

    def test_explicit_controls_beat_the_prompt(self):
        _entry, gen = _builtin()
        result = gen.generate(
            GenerationRequest(prompt="fast metal at 200bpm in E minor",
                              style="ambient", key="C major", tempo=70.0,
                              duration_seconds=20, seed=1),
            GeneratorContext(),
        )
        assert result.meta["style"] == "ambient"
        assert result.meta["key"] == "C major"
        assert result.meta["tempo"] == 70.0

    def test_complexity_adds_instruments(self):
        _entry, gen = _builtin()
        def tracks(cx):
            return len(gen.generate(
                GenerationRequest(prompt="pop", duration_seconds=60, seed=7,
                                  extra={"complexity": cx}),
                GeneratorContext()).song.tracks)
        assert tracks(0.2) < tracks(1.0)

    def test_is_always_ready(self):
        _entry, gen = _builtin()
        assert gen.is_ready() and gen.missing_packages() == []


class TestJobQueue:
    def test_jobs_run_and_report_progress(self):
        _entry, gen = _builtin()
        seen: list[str] = []
        queue = JobQueue()
        queue.subscribe(lambda e: seen.append(e.kind))
        job = queue.submit(GenerationRequest(prompt="test", duration_seconds=20),
                           gen, GeneratorContext())
        for _ in range(100):
            if job.is_terminal:
                break
            time.sleep(0.05)
        queue.stop()
        assert job.status is JobStatus.DONE
        assert "progress" in seen and "finished" in seen

    def test_queued_job_can_be_cancelled(self):
        _entry, gen = _builtin()
        queue = JobQueue()
        job = queue.submit(GenerationRequest(prompt="x", duration_seconds=600),
                           gen, GeneratorContext())
        queue.cancel(job.id)
        time.sleep(0.4)
        queue.stop()
        assert job.status is JobStatus.CANCELLED

    def test_a_failing_generator_does_not_take_down_the_queue(self):
        _entry, gen = _builtin()
        queue = JobQueue(post_process=lambda job: (_ for _ in ()).throw(RuntimeError("boom")))
        job = queue.submit(GenerationRequest(prompt="x", duration_seconds=10),
                           gen, GeneratorContext())
        for _ in range(100):
            if job.is_terminal:
                break
            time.sleep(0.05)
        assert job.status is JobStatus.FAILED
        # The worker must still accept new work afterwards.
        queue._post_process = None
        second = queue.submit(GenerationRequest(prompt="y", duration_seconds=10),
                              gen, GeneratorContext())
        for _ in range(100):
            if second.is_terminal:
                break
            time.sleep(0.05)
        queue.stop()
        assert second.status is JobStatus.DONE


class TestService:
    def test_midi_generation_writes_a_file_and_records_it(self, service):
        jobs = service.submit(
            GenerationRequest(prompt="ambient drone", output_format=OutputFormat.MIDI,
                              duration_seconds=20, seed=1),
            "builtin-composer",
        )
        _wait(jobs)
        assert jobs[0].status is JobStatus.DONE
        assert jobs[0].output_paths[0].suffix == ".mid"
        assert len(service.library) == 1

    def test_audio_generation_also_writes_the_score(self, service):
        service.settings.also_write_midi = True
        jobs = service.submit(
            GenerationRequest(prompt="lofi", output_format=OutputFormat.FLAC,
                              duration_seconds=10, seed=1),
            "builtin-composer",
        )
        _wait(jobs, limit=400)
        suffixes = {p.suffix for p in jobs[0].output_paths}
        assert ".flac" in suffixes and ".mid" in suffixes

    def test_variations_get_distinct_seeds_and_filenames(self, service):
        jobs = service.submit(
            GenerationRequest(prompt="pop", output_format=OutputFormat.MIDI,
                              duration_seconds=15, seed=5, variations=3),
            "builtin-composer",
        )
        _wait(jobs)
        assert len({j.request.seed for j in jobs}) == 3
        names = {j.output_paths[0].name for j in jobs}
        assert len(names) == 3

    def test_a_length_range_is_resolved_before_the_job_runs(self, service):
        jobs = service.submit(
            GenerationRequest(prompt="pop", output_format=OutputFormat.MIDI,
                              min_duration_seconds=20, max_duration_seconds=40,
                              seed=3, variations=3),
            "builtin-composer",
        )
        lengths = [j.request.duration_seconds for j in jobs]
        assert all(20 <= n <= 40 for n in lengths)
        assert len(set(lengths)) == 3
        _wait(jobs, limit=400)
        assert all(j.status is JobStatus.DONE for j in jobs)
        # The music is as long as it was asked to be, not merely requested so.
        for job in jobs:
            assert job.result.song.duration_seconds == pytest.approx(
                job.request.duration_seconds, rel=0.35
            )

    def test_unknown_model_falls_back_rather_than_failing(self, service):
        jobs = service.submit(
            GenerationRequest(prompt="x", output_format=OutputFormat.MIDI,
                              duration_seconds=10),
            "no-such-model",
        )
        _wait(jobs)
        assert jobs[0].status is JobStatus.DONE


class TestDurationRange:
    def test_no_range_leaves_the_duration_alone(self):
        request = GenerationRequest(duration_seconds=60)
        assert resolve_durations(request, 3) == [60, 60, 60]

    def test_none_stays_none(self):
        assert resolve_durations(GenerationRequest(duration_seconds=None), 2) == [None, None]

    def test_lengths_are_spread_across_the_range(self):
        request = GenerationRequest(min_duration_seconds=60,
                                    max_duration_seconds=180, seed=11)
        lengths = resolve_durations(request, 4)
        assert all(60 <= n <= 180 for n in lengths)
        assert len(set(lengths)) == 4
        # Spread, not clustered: four slices of a two-minute band cannot all
        # land within a few seconds of each other.
        assert max(lengths) - min(lengths) > 40

    def test_the_same_seed_gives_the_same_lengths(self):
        request = GenerationRequest(min_duration_seconds=30,
                                    max_duration_seconds=90, seed=4)
        assert resolve_durations(request, 3) == resolve_durations(request, 3)

    def test_a_reversed_range_is_still_honoured(self):
        request = GenerationRequest(min_duration_seconds=180,
                                    max_duration_seconds=60, seed=2)
        assert all(60 <= n <= 180 for n in resolve_durations(request, 3))

    def test_a_backend_ceiling_clamps_the_range(self):
        request = GenerationRequest(min_duration_seconds=60,
                                    max_duration_seconds=300, seed=1)
        assert resolve_durations(request, 3, cap=30.0) == [30.0, 30.0, 30.0]

    def test_a_degenerate_range_is_a_fixed_length(self):
        request = GenerationRequest(min_duration_seconds=45, max_duration_seconds=45)
        assert resolve_durations(request, 4) == [45.0] * 4


class TestHardware:
    def test_detection_does_not_import_torch(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "torch", None)  # would break any import
        info = detect_system()
        assert info.cpu_cores > 0

    def test_gfx_resolves_from_device_id_first(self):
        assert _gfx_for("Vendor Branded Thing", 0x744C) == "gfx1100"
        assert _gfx_for("AMD Radeon RX 7800 XT", 0x7470) == "gfx1101"

    def test_each_card_gets_its_own_build(self):
        assert GPU("XTX", Vendor.AMD, 24576, gfx_arch="gfx1100").torch_extra() == \
            "torch[device-gfx1100]"
        assert GPU("7800", Vendor.AMD, 16384, gfx_arch="gfx1101").torch_extra() == \
            "torch[device-gfx1101]"

    def test_probe_fills_in_torch_facts(self):
        info = apply_runtime_probe(detect_system(), {"torch": "2.9.0", "cuda": True})
        assert info.torch_installed and info.torch_device == "cuda"


class TestRuntime:
    def test_amd_is_offered_a_gpu_option_before_cpu(self):
        ids = [o.id for o in options_for("amd")]
        assert ids[-1] == "cpu" and len(ids) > 1

    def test_the_broken_pinned_channel_is_gone(self):
        # repo.radeon.com/.../rocm-rel-7.2.1/ is a directory listing, not a
        # PEP 503 index, so --index-url could never resolve against it.
        assert not any(o.id == "rocm-windows-pinned" for o in RUNTIME_OPTIONS)
        assert not any("repo.radeon.com" in o.index_url for o in RUNTIME_OPTIONS)

    def test_gfx_extra_is_derived_not_hardcoded(self):
        rocm = next(o for o in RUNTIME_OPTIONS if o.id == "rocm-windows")
        assert "torch[device-gfx1101]" in resolve_packages(rocm, "gfx1101")
        assert "torch[device-gfx1100]" not in resolve_packages(rocm, "gfx1101")

    def test_unknown_architecture_falls_back_to_plain_torch(self):
        rocm = next(o for o in RUNTIME_OPTIONS if o.id == "rocm-windows")
        assert "torch" in resolve_packages(rocm, "")


def _wait(jobs, limit: int = 200) -> None:
    for _ in range(limit):
        if all(j.is_terminal for j in jobs):
            return
        time.sleep(0.05)


class TestOrchestralScore:
    """Turning a recording into per-section MIDI, rather than audio stems."""

    def _tracks(self):
        # (program, is_drum, note count) covering one instrument per section.
        return [
            {"name": "Strings", "program": 48, "is_drum": False,
             "notes": [[60, 0.0, 1.0, 90], [64, 1.0, 1.0, 88]]},
            {"name": "Brass", "program": 60, "is_drum": False,
             "notes": [[55, 0.5, 0.5, 100]]},
            {"name": "Reed", "program": 71, "is_drum": False,
             "notes": [[72, 0.0, 0.25, 70]]},
            {"name": "Drums", "program": 0, "is_drum": True,
             "notes": [[36, 0.0, 0.1, 110], [38, 0.5, 0.1, 100]]},
            {"name": "Piano", "program": 0, "is_drum": False,
             "notes": [[48, 0.0, 2.0, 80]]},
        ]

    def test_layers_are_grouped_by_section(self):
        from midimusic.core.score import build_sections

        layers, _full = build_sections(self._tracks(), tempo=120.0, title="Cue")
        names = [layer.name for layer in layers]
        assert names == ["Woodwinds", "Brass", "Percussion", "Keyboards", "Strings"]

    def test_the_full_score_is_in_conductors_order(self):
        from midimusic.core.score import build_sections

        _layers, full = build_sections(self._tracks(), tempo=120.0)
        # Woodwinds at the top of the page, strings at the bottom, as printed.
        order = [t.name for t in full.tracks]
        assert order[0] == "Reeds"
        assert order[-1] == "Ensemble"
        assert full.note_count == 7

    def test_percussion_keeps_the_drum_channel(self):
        from midimusic.core.score import build_sections

        _layers, full = build_sections(self._tracks(), tempo=120.0)
        drums = [t for t in full.tracks if t.is_drum]
        assert drums and all(t.channel == 9 for t in drums)
        assert all(t.channel != 9 for t in full.tracks if not t.is_drum)

    def test_seconds_become_beats_at_the_detected_tempo(self):
        from midimusic.core.score import build_sections

        # At 120 bpm a note one second in starts on beat two.
        _layers, full = build_sections(self._tracks(), tempo=120.0)
        strings = next(t for t in full.tracks if t.name == "Ensemble")
        assert strings.notes[1].start == pytest.approx(2.0)

        # The same notes at 60 bpm land half as far along.
        _layers, slower = build_sections(self._tracks(), tempo=60.0)
        strings = next(t for t in slower.tracks if t.name == "Ensemble")
        assert strings.notes[1].start == pytest.approx(1.0)

    def test_silent_tracks_are_dropped(self):
        from midimusic.core.score import build_sections

        layers, full = build_sections(
            [{"name": "Empty", "program": 40, "is_drum": False, "notes": []}], tempo=120.0
        )
        assert layers == [] and full.tracks == []

    def test_the_catalog_entry_fetches_only_its_own_checkpoint(self):
        entry = load_catalog().get("yourmt3-orchestral")
        assert entry is not None
        # The repo holds five checkpoints; pulling all of them would be several
        # gigabytes for a model that needs one.
        assert len(entry.files) == 1
        assert entry.files[0].endswith(".ckpt")
        assert entry.outputs == ("midi",)

    def test_it_asks_for_a_transformers_it_can_actually_use(self):
        # The vendored transcriber uses internals that transformers 5 removed,
        # so the pin is part of what the model needs, not a nicety.
        generator = create_generator(load_catalog().get("yourmt3-orchestral"))
        assert "transformers<5" in generator.required_packages()


class TestFrozenBuild:
    """What the packaged build has to be told, because imports cannot show it.

    Every backend is resolved from a string in the registry, so a bundler that
    follows imports finds none of them. The symptom is not a crash but a
    backend that is quietly absent from the packaged app, which is exactly the
    kind of thing a source checkout never reveals.
    """

    def test_the_module_list_covers_every_adapter(self):
        from midimusic.core.registry import (
            ADAPTERS,
            IN_PROCESS_ADAPTERS,
            adapter_modules,
        )

        listed = set(adapter_modules())
        for target in list(ADAPTERS.values()) + list(IN_PROCESS_ADAPTERS.values()):
            assert target.partition(":")[0] in listed, target

    def test_every_listed_module_imports(self):
        import importlib

        from midimusic.core.registry import adapter_modules

        for name in adapter_modules():
            assert importlib.import_module(name) is not None

    def test_no_backend_is_reachable_by_import_alone(self):
        # If this ever fails it is good news, but until then the list above is
        # the only thing keeping these modules in the bundle.
        from midimusic.core.registry import adapter_modules

        assert adapter_modules(), "the registry named no adapter modules"

    def test_the_spec_asks_the_registry_rather_than_repeating_it(self):
        spec = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "midimusic.spec"
        text = spec.read_text(encoding="utf-8")
        assert "adapter_modules()" in text
        # A second, hand-written copy of the list is what went stale before.
        body = text.split("hiddenimports = [", 1)[1].split("]", 1)[0]
        assert "midimusic." not in body, "the spec is hard-coding backend modules again"


class TestRemix:
    """Keeping one layer of a recording and generating the rest."""

    @staticmethod
    def _fake_separation(monkeypatch, rate=22050, seconds=6.0):
        """Stand in for Demucs, so the mix path can be tested without it."""
        import os
        from types import SimpleNamespace

        import numpy as np
        import soundfile as sf

        from midimusic.core import remix as remix_module

        levels = {"vocals": 0.30, "drums": 0.25, "bass": 0.20, "other": 0.15}
        calls: dict = {}

        def fake_run_worker(payload, on_progress=None, should_cancel=None):
            calls["payload"] = payload
            folder = payload["output_dir"]
            os.makedirs(folder, exist_ok=True)
            t = np.arange(int(rate * seconds)) / rate
            written = {}
            for index, (name, gain) in enumerate(levels.items()):
                wave = (np.sin(2 * np.pi * (110 * (index + 1)) * t) * gain).astype("float32")
                path = os.path.join(folder, f"{name}.wav")
                sf.write(path, np.stack([wave, wave], axis=1), rate, subtype="FLOAT")
                written[name] = path
            if on_progress:
                on_progress(1.0, "Separating", "separate")
            return SimpleNamespace(meta={"stems": written, "sample_rate": rate},
                                   output_path=None)

        monkeypatch.setattr(remix_module, "run_worker", fake_run_worker)
        return calls, rate, seconds

    def _request(self, source, out_dir, **extra):
        base = {
            "input_path": str(source), "output_dir": str(out_dir),
            "separator": "demucs-htdemucs", "bed_model": "builtin-composer",
            "keep_stems": ["vocals"], "save_stems": False, "max_seconds": 0.0,
        }
        base.update(extra)
        return GenerationRequest(prompt="hard techno", model_id="stem-remix",
                                 output_format=OutputFormat.FLAC,
                                 duration_seconds=None, seed=4, extra=base)

    def _source(self, tmp_path, rate=22050, seconds=6.0):
        import numpy as np
        import soundfile as sf

        t = np.arange(int(rate * seconds)) / rate
        wave = (np.sin(2 * np.pi * 220 * t) * 0.4).astype("float32")
        path = tmp_path / "song.wav"
        sf.write(str(path), np.stack([wave, wave], axis=1), rate)
        return path

    def test_it_keeps_one_layer_and_replaces_the_others(self, monkeypatch, tmp_path):
        self._fake_separation(monkeypatch)
        remix = create_generator(load_catalog().get("stem-remix"))
        source = self._source(tmp_path)
        result = remix.generate(self._request(source, tmp_path), GeneratorContext())

        assert result.meta["kept"] == ["vocals"]
        assert result.meta["replaced"] == ["bass", "drums", "other"]
        assert result.audio is not None
        assert result.audio.duration_seconds == pytest.approx(6.0, abs=0.05)

    def test_the_backing_is_written_at_the_recordings_own_tempo(self, monkeypatch, tmp_path):
        self._fake_separation(monkeypatch)
        remix = create_generator(load_catalog().get("stem-remix"))
        result = remix.generate(
            self._request(self._source(tmp_path), tmp_path), GeneratorContext()
        )
        # The built-in composer writes to the tempo it is handed, so the mix
        # can say the backing is locked to the recording rather than drifting.
        assert result.meta["beat_locked"] is True
        assert result.meta["tempo"] > 0

    def test_keeping_everything_is_refused_with_a_reason(self, monkeypatch, tmp_path):
        from midimusic.core.generator import BackendUnavailable

        self._fake_separation(monkeypatch)
        remix = create_generator(load_catalog().get("stem-remix"))
        request = self._request(self._source(tmp_path), tmp_path,
                                keep_stems=["vocals", "drums", "bass", "other"])
        with pytest.raises(BackendUnavailable, match="nothing to regenerate"):
            remix.generate(request, GeneratorContext())

    def test_the_parts_can_be_saved_alongside_the_mix(self, monkeypatch, tmp_path):
        self._fake_separation(monkeypatch)
        remix = create_generator(load_catalog().get("stem-remix"))
        result = remix.generate(
            self._request(self._source(tmp_path), tmp_path, save_stems=True),
            GeneratorContext(),
        )
        names = {p.stem for p in result.paths}
        assert "vocals" in names and "new backing" in names

    def test_the_kept_layer_is_still_audible_in_the_mix(self, monkeypatch, tmp_path):
        import numpy as np

        _calls, rate, _seconds = self._fake_separation(monkeypatch)
        remix = create_generator(load_catalog().get("stem-remix"))
        result = remix.generate(
            self._request(self._source(tmp_path), tmp_path), GeneratorContext()
        )
        # The kept stem is a 110 Hz tone. If the backing had swamped it, or the
        # mix had simply dropped it, that bin would be gone.
        mono = np.asarray(result.audio.samples).mean(axis=1)
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
        freqs = np.fft.rfftfreq(mono.size, 1.0 / rate)
        band = spectrum[(freqs > 105) & (freqs < 115)].max()
        assert band > spectrum.mean() * 20

    def test_the_prompt_carries_the_tempo_and_key(self):
        from midimusic.audio.analyze import AudioAnalysis
        from midimusic.core.remix import condition_prompt

        analysis = AudioAnalysis(tempo=128.0, key="A minor")
        assert condition_prompt("edm", analysis) == "edm, 128 bpm, in A minor"
        # Nothing measured, nothing added.
        assert condition_prompt("edm", AudioAnalysis()) == "edm"
        assert condition_prompt("", None) == "instrumental backing"


class TestModelPackages:
    def test_torch_is_never_reinstalled_from_pypi(self):
        from midimusic.core.runtime import model_packages

        # Doing so would replace a vendor ROCm or CUDA build with a CPU one.
        assert model_packages(["torch", "demucs"]) == ("demucs",)
        assert model_packages(["torch[device-gfx1100]", "diffusers"]) == ("diffusers",)

    def test_version_pins_survive(self):
        from midimusic.core.runtime import model_packages

        assert model_packages(["torch", "transformers<5"]) == ("transformers<5",)

    def test_every_catalog_model_names_installable_libraries(self):
        from midimusic.core.runtime import model_packages

        for entry in load_catalog().models:
            if entry.is_builtin:
                continue
            assert entry.extras, f"{entry.id} lists no libraries"
            assert model_packages(entry.extras), f"{entry.id} needs only torch?"


class TestBootstrap:
    """The packaged app must be able to build its own Python environment.

    Without this it would silently depend on the user already having Python
    installed, which defeats the point of shipping an installer.
    """

    def test_a_bootstrapper_is_available(self):
        from midimusic.core.bootstrap import find_uv

        assert find_uv() is not None, "no environment builder found"

    def test_it_creates_a_working_interpreter(self, tmp_path):
        import subprocess

        from midimusic.core.bootstrap import create_runtime, runtime_exists

        target = tmp_path / "runtime"
        assert not runtime_exists(target)
        python = create_runtime(target)
        assert runtime_exists(target)
        version = subprocess.run([str(python), "--version"],
                                 capture_output=True, text=True).stdout
        assert "3.12" in version

    def test_creating_twice_is_a_no_op(self, tmp_path):
        from midimusic.core.bootstrap import create_runtime

        target = tmp_path / "runtime"
        assert create_runtime(target) == create_runtime(target)

    def test_the_installer_builds_an_environment_rather_than_using_the_app(self):
        # In a frozen build sys.executable is MIDIMusic.exe, so an installer
        # that shelled out to it would relaunch the GUI instead of installing
        # anything. It must provision a real interpreter first.
        from midimusic.core import bootstrap as bootstrap_module
        from midimusic.core.runtime import RUNTIME_OPTIONS, install_runtime

        called: dict = {}

        def fake_create_runtime(*_args, **kwargs):
            called["created"] = True
            on_line = kwargs.get("on_line")
            if on_line:
                on_line("stub")
            raise RuntimeError("stop here; provisioning was reached")

        original = bootstrap_module.create_runtime
        bootstrap_module.create_runtime = fake_create_runtime
        try:
            option = next(o for o in RUNTIME_OPTIONS if o.id == "cpu")
            handle = install_runtime(option, extra_packages=())
            handle.thread.join(30)
        finally:
            bootstrap_module.create_runtime = original

        assert called.get("created"), "install did not provision an interpreter"


class TestDeconstruct:
    """Pulling a recording apart into layers."""

    def test_the_separator_is_registered_and_described(self):
        catalog = load_catalog()
        separators = catalog.by_kind("separator")
        assert separators, "no separator in the catalog"
        for entry in separators:
            assert entry.stems, f"{entry.id} declares no stems"
            assert "vocals" in entry.stems
            assert create_generator(entry) is not None

    def test_it_reports_what_it_needs_rather_than_failing_late(self):
        entry = load_catalog().get("demucs-htdemucs")
        generator = create_generator(entry)
        # Either ready, or it names what is missing. Never a silent failure.
        assert generator.is_ready() or generator.missing_packages()

    def test_a_missing_input_file_is_reported_clearly(self):
        from midimusic.core.generator import BackendUnavailable

        entry = load_catalog().get("demucs-htdemucs")
        generator = create_generator(entry)
        request = GenerationRequest(
            prompt="x", model_id=entry.id,
            extra={"input_path": "/definitely/not/here.wav"},
        )
        with pytest.raises((BackendUnavailable, Exception)) as caught:
            generator.generate(request, GeneratorContext())
        assert "not" in str(caught.value).lower() or "no such" in str(caught.value).lower()

    def test_drums_are_not_pitch_transcribed(self):
        # Running a pitch tracker over percussion yields noise, not a drum
        # part, so it must be skipped deliberately rather than attempted.
        from midimusic.core.deconstruct import DeconstructGenerator

        entry = load_catalog().get("demucs-htdemucs")
        generator = DeconstructGenerator(entry.id, entry)
        path, count, reason = generator._transcribe_stem(
            "drums", "/nonexistent.wav", Path("/tmp"), GeneratorContext()
        )
        assert path is None and count == 0
        assert "percussion" in reason or "drum" in reason

    def test_vocals_get_a_voice_program(self):
        from midimusic.core.deconstruct import GM_FOR_STEM

        # GM 52-54 are the choir/voice patches; a transcribed vocal line
        # should not play back as a piano.
        assert GM_FOR_STEM["vocals"] in (52, 53, 54)
        assert GM_FOR_STEM["bass"] != GM_FOR_STEM["vocals"]

    def test_stems_are_not_loudness_normalised(self):
        # Normalising each stem independently would destroy their relative
        # levels and make the set impossible to recombine into the mix.
        import inspect

        from midimusic.core import deconstruct

        source = inspect.getsource(deconstruct.DeconstructGenerator.generate)
        assert "target_lufs=None" in source
