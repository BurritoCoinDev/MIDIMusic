"""GUI smoke tests.

These run headless against Qt's offscreen platform. They are not trying to
verify how the app looks -- they check that every panel constructs, that the
wiring between panels holds, and that a full generation can be driven through
the UI without touching the widgets' internals.
"""

from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def window(qapp, service):
    from midimusic.ui.main_window import MainWindow

    win = MainWindow(service)
    yield win
    win.library.stop()


class TestMainWindow:
    def test_every_tab_constructs(self, window):
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert titles == [
            "Compose", "Deconstruct", "Remix", "Queue", "Library", "Models",
            "Settings",
        ]

    def test_switching_tabs_does_not_raise(self, window, qapp):
        for i in range(window.tabs.count()):
            window.tabs.setCurrentIndex(i)
            qapp.processEvents()

    def test_status_bar_reports_hardware(self, window):
        assert window.hardware_label.text()


class TestComposePanel:
    def test_models_are_listed_for_the_chosen_format(self, window):
        panel = window.compose
        assert panel.model.count() > 0

    def test_switching_output_format_refilters_models(self, window, qapp):
        panel = window.compose
        panel.output_format.setCurrentIndex(
            panel.output_format.findData("midi")
        )
        qapp.processEvents()
        ids = {panel.model.itemData(i) for i in range(panel.model.count())}
        assert "builtin-composer" in ids
        # An audio-only model must not be offered when MIDI is selected.
        assert "musicgen-small" not in ids

    def test_prompt_interpretation_is_shown(self, window, qapp):
        panel = window.compose
        panel.prompt.setPlainText("dark cinematic orchestral, 90 bpm, C minor")
        qapp.processEvents()
        assert "cinematic" in panel.hint.text()

    def test_request_reflects_the_controls(self, window):
        panel = window.compose
        panel.prompt.setPlainText("test prompt")
        panel.duration.set_values(45, 45)
        panel.variations.setValue(2)
        panel.complexity.setValue(80)
        request = panel.build_request()
        assert request.prompt == "test prompt"
        assert request.duration_seconds == 45
        assert request.min_duration_seconds == 45
        assert request.max_duration_seconds == 45
        assert request.variations == 2
        assert request.extra["complexity"] == pytest.approx(0.8)

    def test_a_length_range_gives_the_variations_different_lengths(self, window, service):
        panel = window.compose
        panel.prompt.setPlainText("range test")
        panel.duration.set_values(30, 90)
        request = panel.build_request()
        assert request.min_duration_seconds == 30
        assert request.max_duration_seconds == 90

        from midimusic.core.models import resolve_durations

        lengths = resolve_durations(request, 4, cap=900.0)
        assert all(30 <= n <= 90 for n in lengths)
        assert len(set(lengths)) == 4

    def test_the_ends_of_the_range_cannot_cross(self, window):
        panel = window.compose
        panel.duration.set_values(60, 120)
        panel.duration.minimum.setValue(200)
        assert panel.duration.maximum.value() == 200
        panel.duration.maximum.setValue(30)
        assert panel.duration.minimum.value() == 30

    def test_an_estimate_is_offered_before_committing(self, window, qapp):
        window.compose.prompt.setPlainText("ambient")
        qapp.processEvents()
        assert window.compose.estimate.text()


class TestGenerationThroughTheUi:
    def test_generate_produces_a_file_and_a_library_entry(self, window, qapp, service):
        panel = window.compose
        panel.prompt.setPlainText("gentle ambient piece")
        panel.output_format.setCurrentIndex(panel.output_format.findData("midi"))
        panel.duration.set_values(20, 20)
        panel.variations.setValue(1)
        qapp.processEvents()

        panel.submit()

        deadline = time.time() + 30
        while time.time() < deadline:
            qapp.processEvents()
            jobs = service.queue.jobs()
            if jobs and all(j.is_terminal for j in jobs):
                break
            time.sleep(0.05)

        jobs = service.queue.jobs()
        assert jobs and jobs[0].status.value == "done"
        assert jobs[0].output_paths
        assert service.library

        window.library.refresh()
        qapp.processEvents()
        assert window.library.list.count() >= 1

    def test_the_queue_shows_the_job(self, window, qapp, service):
        panel = window.compose
        panel.prompt.setPlainText("short test")
        panel.output_format.setCurrentIndex(panel.output_format.findData("midi"))
        panel.duration.set_values(15, 15)
        panel.submit()
        for _ in range(60):
            qapp.processEvents()
            if window.queue._rows:
                break
            time.sleep(0.05)
        assert window.queue._rows


class TestModelsPanel:
    def test_a_card_exists_for_every_model(self, window, service):
        # The layout holds one card per model plus a trailing stretch.
        assert window.models._list.count() - 1 == len(service.catalog.models)

    def test_a_runtime_option_is_offered(self, window):
        assert window.models.runtime.count() >= 1

    def test_builtin_offers_no_download(self, window, qapp, service):
        from midimusic.ui.widgets.models_panel import ModelCard

        card = ModelCard(service.catalog.get("builtin-composer"), service)
        card.show()
        qapp.processEvents()
        assert not card.action.isVisible()
        assert "Built in" in card.status.text()


class TestSettingsPanel:
    def test_saving_round_trips(self, window, service):
        panel = window.settings_panel
        panel.default_duration.set_values(150, 210)
        panel.lufs.setValue(-16.0)
        panel.save()

        from midimusic.config.settings import load_settings

        reloaded = load_settings(force=True)
        assert reloaded.duration_range() == (150, 210)
        assert reloaded.target_lufs == -16.0

    def test_soundfont_state_is_reported(self, window):
        assert window.settings_panel.sf_status.text()


class TestWaveform:
    def test_renders_and_seeks(self, qapp):
        import numpy as np

        from midimusic.ui.widgets.waveform import WaveformWidget

        widget = WaveformWidget()
        widget.resize(400, 80)
        samples = np.random.RandomState(0).randn(44100, 2).astype("float32") * 0.3
        widget.set_audio(samples, 1.0)
        widget.show()
        qapp.processEvents()
        assert not widget.grab().isNull()

        seen: list[float] = []
        widget.seeked.connect(seen.append)
        widget._seek_to(200)
        assert seen and 0 <= seen[0] <= 1.0

    def test_empty_state_does_not_crash(self, qapp):
        from midimusic.ui.widgets.waveform import WaveformWidget

        widget = WaveformWidget()
        widget.resize(200, 60)
        widget.clear()
        widget.show()
        qapp.processEvents()
        assert not widget.grab().isNull()


class TestEntryPoint:
    """The packaged app's startup path.

    A frozen build runs its entry script as top-level ``__main__`` with no
    package context. That is not how ``python -m midimusic`` runs, so the
    normal test suite cannot see a failure that only appears once packaged --
    which is exactly how an ImportError shipped in a release build. These tests
    invoke the entry points the way PyInstaller does.
    """

    @staticmethod
    def _run(script_args: list[str], timeout: int = 180):
        import os
        import subprocess
        import sys
        from pathlib import Path

        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            [sys.executable, *script_args],
            capture_output=True, text=True, timeout=timeout, cwd=str(root), env=env,
        )

    def test_launcher_runs_as_a_bare_script(self):
        # This is precisely how PyInstaller invokes the frozen entry point.
        from pathlib import Path

        launcher = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "launcher.py"
        assert launcher.exists(), "the frozen build's entry script is missing"
        result = self._run([str(launcher), "--selftest"])
        assert result.returncode == 0, (
            f"launcher failed as a bare script:\n{result.stdout}\n{result.stderr}"
        )

    def test_module_entry_runs_as_a_bare_script(self):
        # Belt and braces: __main__.py must not depend on relative imports
        # either, so running it directly cannot break.
        from pathlib import Path

        entry = Path(__file__).resolve().parents[1] / "src" / "midimusic" / "__main__.py"
        result = self._run([str(entry), "--selftest"])
        assert result.returncode == 0, (
            f"__main__.py failed as a bare script:\n{result.stdout}\n{result.stderr}"
        )

    def test_entry_has_no_relative_imports(self):
        # The failure mode is silent in normal test runs, so assert the shape
        # directly as well as the behaviour.
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "src" / "midimusic" / "__main__.py"
        ).read_text(encoding="utf-8")
        offenders = re.findall(r"^\s*from\s+\.", source, re.MULTILINE)
        assert not offenders, (
            "__main__.py uses relative imports; a frozen build runs it without "
            "package context and they will fail at startup"
        )

    def test_selftest_exits_cleanly_via_module(self):
        result = self._run(["-m", "midimusic", "--selftest"])
        assert result.returncode == 0
        assert "selftest ok" in result.stdout

    def test_version_flag(self):
        result = self._run(["-m", "midimusic", "--version"])
        assert result.returncode == 0
        assert "MIDIMusic" in result.stdout


class TestDeferredCallbacksSurviveTeardown:
    """Timers must not outlive the widgets they touch.

    A ``QTimer.singleShot`` without a context object keeps firing after its
    widget is destroyed, and the callback then touches a deleted C++ object.
    That raises inside the Qt event loop, where it does not propagate to the
    caller -- so it surfaces as a mysterious error rather than a traceback at
    the call site.
    """

    def test_save_confirmation_timer_dies_with_the_panel(self, qapp, service, monkeypatch):
        from PySide6 import QtCore

        from midimusic.ui.widgets import settings_panel as module

        # Shorten the delay so the test does not wait the full display time.
        monkeypatch.setattr(module, "SAVED_MESSAGE_MS", 50)

        panel = module.SettingsPanel(service)
        panel.save()
        panel.deleteLater()
        panel.setParent(None)
        del panel
        qapp.processEvents()

        errors: list[BaseException] = []

        def record(exc_type, exc, tb):
            errors.append(exc)

        monkeypatch.setattr(sys, "excepthook", record)

        # Spin well past the timer so a surviving callback would fire here.
        deadline = QtCore.QDeadlineTimer(400)
        while not deadline.hasExpired():
            qapp.processEvents()
        qapp.processEvents()

        assert not errors, f"a deferred callback outlived its widget: {errors}"


class TestDeconstructPanel:
    def test_it_offers_the_separators(self, window):
        panel = window.deconstruct
        assert panel.model.count() >= 1
        assert "vocals" in panel.stem_summary.text()

    def test_generate_is_disabled_until_a_song_is_chosen(self, window):
        assert not window.deconstruct.go.isEnabled()

    def test_choosing_a_file_enables_it_and_builds_a_request(self, window, qapp, tmp_path):
        import numpy as np
        import soundfile as sf

        source = tmp_path / "song.flac"
        sf.write(str(source), np.zeros((44100, 2), dtype="float32"), 44100)

        panel = window.deconstruct
        panel.set_source(str(source))
        qapp.processEvents()

        assert panel.go.isEnabled()
        request = panel.build_request()
        assert request.extra["input_path"] == str(source)
        assert request.extra["transcribe"] is True

    def test_it_offers_the_orchestral_transcriber_too(self, window):
        panel = window.deconstruct
        ids = [panel.model.itemData(i) for i in range(panel.model.count())]
        assert "yourmt3-orchestral" in ids

    def test_choosing_the_transcriber_switches_the_controls(self, window, qapp, tmp_path):
        import numpy as np
        import soundfile as sf

        source = tmp_path / "cue.flac"
        sf.write(str(source), np.zeros((44100, 2), dtype="float32"), 44100)

        panel = window.deconstruct
        panel.set_source(str(source))
        panel.model.setCurrentIndex(panel.model.findData("yourmt3-orchestral"))
        qapp.processEvents()

        # Stem format and per-stem transcription are meaningless for a model
        # that writes a score, so they go away rather than sitting there inert.
        assert panel.format.isHidden()
        assert panel.transcribe.isHidden()
        assert not panel.score_note.isHidden()
        assert "Strings" in panel.stem_summary.text()

        request = panel.build_request()
        assert request.model_id == "yourmt3-orchestral"
        assert request.output_format.value == "midi"

    def test_a_length_limit_reaches_the_request(self, window, qapp, tmp_path):
        import numpy as np
        import soundfile as sf

        source = tmp_path / "long.flac"
        sf.write(str(source), np.zeros((44100, 2), dtype="float32"), 44100)

        panel = window.deconstruct
        panel.set_source(str(source))
        panel.limit.setValue(90)
        qapp.processEvents()
        assert panel.build_request().extra["max_seconds"] == 90.0

    def test_the_vocal_note_appears_only_with_transcription_on(self, window, qapp):
        # isHidden, not isVisible: a widget inside a window that was never
        # shown reports isVisible() False whatever its own flag says, so the
        # question is whether it was explicitly hidden.
        panel = window.deconstruct
        panel.transcribe.setChecked(True)
        qapp.processEvents()
        assert not panel.vocal_note.isHidden()
        panel.transcribe.setChecked(False)
        qapp.processEvents()
        assert panel.vocal_note.isHidden()


class TestRemixPanel:
    def _song(self, tmp_path):
        import numpy as np
        import soundfile as sf

        path = tmp_path / "cue.flac"
        sf.write(str(path), np.zeros((44100 * 2, 2), dtype="float32"), 44100)
        return path

    def test_vocals_are_kept_by_default(self, window):
        panel = window.remix
        assert panel.keep_stems() == ["vocals"]
        assert "bass" in panel.keep_note.text()

    def test_remix_is_disabled_until_a_song_is_chosen(self, window):
        assert not window.remix.go.isEnabled()

    def test_it_builds_a_request_from_the_controls(self, window, qapp, tmp_path):
        panel = window.remix
        panel.set_source(str(self._song(tmp_path)))
        panel.prompt.setPlainText("hard techno")
        panel.save_stems.setChecked(True)
        panel.limit.setValue(60)
        qapp.processEvents()

        assert panel.go.isEnabled()
        request = panel.build_request()
        assert request.prompt == "hard techno"
        assert request.model_id == "stem-remix"
        assert request.extra["keep_stems"] == ["vocals"]
        assert request.extra["save_stems"] is True
        assert request.extra["max_seconds"] == 60.0

    def test_keeping_every_layer_blocks_the_button(self, window, qapp, tmp_path):
        panel = window.remix
        panel.set_source(str(self._song(tmp_path)))
        for box in panel._keep_boxes.values():
            box.setChecked(True)
        qapp.processEvents()
        assert not panel.go.isEnabled()
        assert "nothing is left" in panel.keep_note.text().lower()

    def test_it_says_whether_the_backing_will_stay_in_time(self, window, qapp):
        panel = window.remix
        panel.bed.setCurrentIndex(panel.bed.findData("builtin-composer"))
        qapp.processEvents()
        # The built-in composer is held to a tempo; a waveform model is not,
        # and the panel has to be straight about which one is in use.
        assert "stays under" in panel.bed_note.text()

        index = panel.bed.findData("musicgen-small")
        if index >= 0:
            panel.bed.setCurrentIndex(index)
            qapp.processEvents()
            assert "drift" in panel.bed_note.text()

    def test_changing_the_separator_rebuilds_the_layer_choices(self, window, qapp):
        panel = window.remix
        index = panel.separator.findData("demucs-6s")
        assert index >= 0
        panel.separator.setCurrentIndex(index)
        qapp.processEvents()
        assert "guitar" in panel._keep_boxes
        assert "piano" in panel._keep_boxes
