"""Catalog, generators, the job queue and the service that ties them together."""

from __future__ import annotations

import time

import pytest

from midimusic.core.catalog import ModelEntry, load_catalog, save_user_model
from midimusic.core.generator import GeneratorContext
from midimusic.core.hardware import GPU, Vendor, _gfx_for, apply_runtime_probe, detect_system
from midimusic.core.jobs import JobQueue
from midimusic.core.models import GenerationRequest, JobStatus, OutputFormat
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

    def test_unknown_model_falls_back_rather_than_failing(self, service):
        jobs = service.submit(
            GenerationRequest(prompt="x", output_format=OutputFormat.MIDI,
                              duration_seconds=10),
            "no-such-model",
        )
        _wait(jobs)
        assert jobs[0].status is JobStatus.DONE


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
