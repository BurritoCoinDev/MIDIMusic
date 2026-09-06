"""GUI smoke tests.

These run headless against Qt's offscreen platform. They are not trying to
verify how the app looks -- they check that every panel constructs, that the
wiring between panels holds, and that a full generation can be driven through
the UI without touching the widgets' internals.
"""

from __future__ import annotations

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
        assert titles == ["Compose", "Queue", "Library", "Models", "Settings"]

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
        panel.duration.setValue(45)
        panel.variations.setValue(2)
        panel.complexity.setValue(80)
        request = panel.build_request()
        assert request.prompt == "test prompt"
        assert request.duration_seconds == 45
        assert request.variations == 2
        assert request.extra["complexity"] == pytest.approx(0.8)

    def test_an_estimate_is_offered_before_committing(self, window, qapp):
        window.compose.prompt.setPlainText("ambient")
        qapp.processEvents()
        assert window.compose.estimate.text()


class TestGenerationThroughTheUi:
    def test_generate_produces_a_file_and_a_library_entry(self, window, qapp, service):
        panel = window.compose
        panel.prompt.setPlainText("gentle ambient piece")
        panel.output_format.setCurrentIndex(panel.output_format.findData("midi"))
        panel.duration.setValue(20)
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
        panel.duration.setValue(15)
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
        panel.default_duration.setValue(210)
        panel.lufs.setValue(-16.0)
        panel.save()

        from midimusic.config.settings import load_settings

        reloaded = load_settings(force=True)
        assert reloaded.default_duration == 210
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
