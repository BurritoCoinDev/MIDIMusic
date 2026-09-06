"""The Settings tab."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ...config.paths import disk_free, get_paths, human_bytes
from ...config.settings import save_settings
from ...core.service import AppService

__all__ = ["SettingsPanel"]


class _SoundFontBridge(QtCore.QObject):
    """Marshals the download thread's callbacks onto the UI thread."""

    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(str, str)


class PathRow(QtWidgets.QWidget):
    """A read/write directory field with a browse button."""

    changed = QtCore.Signal(str)

    def __init__(self, value: str, placeholder: str, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit(value)
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        row.addWidget(self.edit, 1)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)

    def _browse(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a folder", self.edit.text() or str(Path.home())
        )
        if directory:
            self.edit.setText(directory)

    def value(self) -> str:
        return self.edit.text().strip()


class SettingsPanel(QtWidgets.QWidget):
    settings_changed = QtCore.Signal()

    def __init__(self, service: AppService, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.service = service
        settings = service.settings

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Settings")
        title.setProperty("role", "title")
        layout.addWidget(title)

        # -- folders --------------------------------------------------------
        folders = QtWidgets.QGroupBox("Folders")
        f_form = QtWidgets.QFormLayout(folders)
        paths = get_paths()
        self.output_dir = PathRow(settings.output_dir, str(paths.output))
        f_form.addRow("Output", self.output_dir)
        self.models_dir = PathRow(settings.models_dir, str(paths.models))
        f_form.addRow("Models", self.models_dir)
        self.disk_label = QtWidgets.QLabel("")
        self.disk_label.setProperty("role", "dim")
        f_form.addRow("", self.disk_label)
        self.models_dir.changed.connect(self._update_disk)
        self._update_disk()
        layout.addWidget(folders)

        # -- audio ----------------------------------------------------------
        audio = QtWidgets.QGroupBox("Audio export")
        a_form = QtWidgets.QFormLayout(audio)
        self.sample_rate = QtWidgets.QComboBox()
        for rate in (32000, 44100, 48000, 96000):
            self.sample_rate.addItem(f"{rate} Hz", rate)
        idx = self.sample_rate.findData(settings.sample_rate)
        self.sample_rate.setCurrentIndex(max(0, idx))
        a_form.addRow("Sample rate", self.sample_rate)

        self.bit_depth = QtWidgets.QComboBox()
        for depth in (16, 24):
            self.bit_depth.addItem(f"{depth}-bit", depth)
        idx = self.bit_depth.findData(settings.bit_depth)
        self.bit_depth.setCurrentIndex(max(0, idx))
        a_form.addRow("Bit depth", self.bit_depth)

        self.lufs = QtWidgets.QDoubleSpinBox()
        self.lufs.setRange(-30.0, -6.0)
        self.lufs.setSingleStep(0.5)
        self.lufs.setSuffix(" LUFS")
        self.lufs.setValue(settings.target_lufs)
        a_form.addRow("Loudness target", self.lufs)

        self.also_midi = QtWidgets.QCheckBox(
            "Also save a MIDI score alongside audio exports"
        )
        self.also_midi.setChecked(settings.also_write_midi)
        a_form.addRow("", self.also_midi)

        self.soundfont = PathRow(settings.soundfont, "Default SoundFont (optional)")
        a_form.addRow("SoundFont", self.soundfont)
        sf_note = QtWidgets.QLabel(
            "Used to render MIDI to audio. Without one the built-in synth is used, "
            "which always works but sounds plainer."
        )
        sf_note.setProperty("role", "dim")
        sf_note.setWordWrap(True)
        a_form.addRow("", sf_note)

        sf_row = QtWidgets.QHBoxLayout()
        self.sf_button = QtWidgets.QPushButton("Download MuseScore General (38 MB, MIT)")
        self.sf_button.clicked.connect(self._download_soundfont)
        sf_row.addWidget(self.sf_button)
        self.sf_progress = QtWidgets.QProgressBar()
        self.sf_progress.setRange(0, 100)
        self.sf_progress.setVisible(False)
        sf_row.addWidget(self.sf_progress, 1)
        a_form.addRow("", sf_row)

        self.sf_status = QtWidgets.QLabel("")
        self.sf_status.setProperty("role", "dim")
        self.sf_status.setWordWrap(True)
        a_form.addRow("", self.sf_status)

        self._sf_bridge = _SoundFontBridge()
        self._sf_bridge.progress.connect(self._on_sf_progress)
        self._sf_bridge.finished.connect(self._on_sf_finished)
        self._sf_cancel = None
        self._refresh_soundfont_state()
        layout.addWidget(audio)

        # -- compute --------------------------------------------------------
        compute = QtWidgets.QGroupBox("Compute and downloads")
        c_form = QtWidgets.QFormLayout(compute)
        self.device = QtWidgets.QComboBox()
        for label, value in (("Automatic", "auto"), ("GPU", "cuda"), ("CPU", "cpu")):
            self.device.addItem(label, value)
        idx = self.device.findData(settings.device)
        self.device.setCurrentIndex(max(0, idx))
        c_form.addRow("Device", self.device)

        self.hf_token = QtWidgets.QLineEdit(settings.hf_token)
        self.hf_token.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("For gated models (stored on this machine)")
        c_form.addRow("Hugging Face token", self.hf_token)

        self.offline = QtWidgets.QCheckBox("Offline mode (never contact the network)")
        self.offline.setChecked(settings.offline)
        c_form.addRow("", self.offline)
        layout.addWidget(compute)

        # -- defaults -------------------------------------------------------
        defaults = QtWidgets.QGroupBox("Generation defaults")
        d_form = QtWidgets.QFormLayout(defaults)
        self.default_duration = QtWidgets.QSpinBox()
        self.default_duration.setRange(5, 900)
        self.default_duration.setSuffix(" s")
        self.default_duration.setValue(settings.default_duration)
        d_form.addRow("Length", self.default_duration)

        self.default_variations = QtWidgets.QSpinBox()
        self.default_variations.setRange(1, 8)
        self.default_variations.setValue(settings.default_variations)
        d_form.addRow("Variations", self.default_variations)

        self.default_complexity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.default_complexity.setRange(0, 100)
        self.default_complexity.setValue(int(settings.default_complexity * 100))
        d_form.addRow("Complexity", self.default_complexity)
        layout.addWidget(defaults)

        layout.addStretch(1)
        scroll.setWidget(inner)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll, 1)

        bar = QtWidgets.QFrame()
        bar.setProperty("role", "card")
        bar_row = QtWidgets.QHBoxLayout(bar)
        bar_row.setContentsMargins(18, 12, 18, 12)
        self.saved_label = QtWidgets.QLabel("")
        self.saved_label.setProperty("role", "ok")
        bar_row.addWidget(self.saved_label, 1)
        save = QtWidgets.QPushButton("Save settings")
        save.setProperty("role", "primary")
        save.clicked.connect(self.save)
        bar_row.addWidget(save)
        outer.addWidget(bar)

    def _refresh_soundfont_state(self) -> None:
        from ...core.assets import installed_soundfonts

        found = installed_soundfonts(get_paths().soundfonts)
        if found:
            self.sf_status.setText(f"Installed: {', '.join(p.name for p in found)}")
            self.sf_button.setText("Re-download MuseScore General")
        else:
            self.sf_status.setText("No SoundFont installed yet.")

    def _download_soundfont(self) -> None:
        from ...core.assets import SOUNDFONTS, download_soundfont

        if self._sf_cancel is not None:
            self._sf_cancel.set()
            return
        option = SOUNDFONTS[0]
        self.sf_progress.setVisible(True)
        self.sf_progress.setValue(0)
        self.sf_button.setText("Cancel")
        self._sf_cancel = download_soundfont(
            option,
            get_paths().soundfonts,
            on_progress=lambda f, m: self._sf_bridge.progress.emit(f, m),
            on_finished=lambda path, err: self._sf_bridge.finished.emit(
                str(path or ""), err
            ),
        )

    @QtCore.Slot(float, str)
    def _on_sf_progress(self, fraction: float, message: str) -> None:
        self.sf_progress.setValue(int(fraction * 100))
        self.sf_status.setText(message)

    @QtCore.Slot(str, str)
    def _on_sf_finished(self, path: str, error: str) -> None:
        self.sf_progress.setVisible(False)
        self._sf_cancel = None
        self.sf_button.setText("Download MuseScore General (38 MB, MIT)")
        if error:
            self.sf_status.setText(error)
            return
        if path:
            # Select it straight away; downloading it and then not using it
            # would be a confusing outcome.
            self.soundfont.edit.setText(path)
            self.service.settings.soundfont = path
            save_settings(self.service.settings)
        self._refresh_soundfont_state()

    def _update_disk(self) -> None:
        path = self.models_dir.value() or str(get_paths().models)
        self.disk_label.setText(f"{human_bytes(disk_free(path))} free on that drive")

    def save(self) -> None:
        s = self.service.settings
        s.output_dir = self.output_dir.value()
        s.models_dir = self.models_dir.value()
        s.sample_rate = int(self.sample_rate.currentData())
        s.bit_depth = int(self.bit_depth.currentData())
        s.target_lufs = float(self.lufs.value())
        s.also_write_midi = self.also_midi.isChecked()
        s.soundfont = self.soundfont.value()
        s.device = str(self.device.currentData())
        s.hf_token = self.hf_token.text().strip()
        s.offline = self.offline.isChecked()
        s.default_duration = int(self.default_duration.value())
        s.default_variations = int(self.default_variations.value())
        s.default_complexity = self.default_complexity.value() / 100.0
        save_settings(s)
        self.saved_label.setText("Saved")
        QtCore.QTimer.singleShot(2500, lambda: self.saved_label.setText(""))
        self.settings_changed.emit()
