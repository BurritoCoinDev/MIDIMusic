"""The Deconstruct tab: take a finished recording apart into its layers."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ...core.catalog import ModelEntry
from ...core.models import GenerationRequest, OutputFormat
from ...core.service import AppService

__all__ = ["DeconstructPanel"]

_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".wma", ".opus"}


class DropZone(QtWidgets.QFrame):
    """A drop target that also opens a file dialog when clicked."""

    file_chosen = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label = QtWidgets.QLabel("Drop a song here, or click to choose one")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.detail = QtWidgets.QLabel("WAV, FLAC, MP3, M4A, OGG")
        self.detail.setProperty("role", "dim")
        self.detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detail)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self._first_audio(event.mimeData()):
            event.acceptProposedAction()
            self.label.setText("Release to load")

    def dragLeaveEvent(self, event: QtCore.QEvent) -> None:
        self.label.setText("Drop a song here, or click to choose one")

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        path = self._first_audio(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.file_chosen.emit(path)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a song", "",
            "Audio (*.wav *.flac *.mp3 *.m4a *.ogg *.aiff *.aif *.opus);;All files (*)",
        )
        if path:
            self.file_chosen.emit(path)

    @staticmethod
    def _first_audio(mime: QtCore.QMimeData) -> str:
        for url in mime.urls():
            local = url.toLocalFile()
            if local and Path(local).suffix.lower() in _AUDIO_SUFFIXES:
                return local
        return ""


class DeconstructPanel(QtWidgets.QWidget):
    submitted = QtCore.Signal(list)

    def __init__(self, service: AppService, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self._source: Path | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Deconstruct")
        title.setProperty("role", "title")
        layout.addWidget(title)

        blurb = QtWidgets.QLabel(
            "Take a recording apart. A band splits into audio stems; an orchestral "
            "or film score splits into MIDI layers by section, because separation "
            "has little to grip on when there is no drum kit and no bass guitar."
        )
        blurb.setProperty("role", "dim")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self.drop = DropZone()
        self.drop.file_chosen.connect(self.set_source)
        layout.addWidget(self.drop)

        layout.addWidget(self._options_card())
        layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        outer.addWidget(self._action_bar())
        self._refresh_models()
        self._update_state()

    def _options_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setProperty("role", "card")
        box = QtWidgets.QVBoxLayout(frame)
        box.setContentsMargins(16, 14, 16, 16)
        box.setSpacing(10)

        heading = QtWidgets.QLabel("Layers")
        heading.setProperty("role", "subtitle")
        box.addWidget(heading)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)

        self.model = QtWidgets.QComboBox()
        self.model.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Split into", self.model)

        self.stem_summary = QtWidgets.QLabel("")
        self.stem_summary.setProperty("role", "dim")
        self.stem_summary.setWordWrap(True)
        form.addRow("", self.stem_summary)

        self.transcribe = QtWidgets.QCheckBox("Also transcribe each pitched layer to MIDI")
        self.transcribe.setChecked(True)
        self.transcribe.toggled.connect(self._update_state)
        form.addRow("", self.transcribe)

        self.vocal_note = QtWidgets.QLabel(
            "MIDI has no notion of a voice, so the vocal layer gives you the isolated "
            "audio plus its melody as notes. Drums are skipped: pitch tracking on "
            "percussion produces noise, not a drum part."
        )
        self.vocal_note.setProperty("role", "dim")
        self.vocal_note.setWordWrap(True)
        form.addRow("", self.vocal_note)

        self.score_note = QtWidgets.QLabel(
            "Transcribes the whole recording into one MIDI layer per section. It "
            "resolves instrument families, so you get the string body rather than "
            "first and second violins, and the result is an estimate of the score "
            "rather than the score itself."
        )
        self.score_note.setProperty("role", "dim")
        self.score_note.setWordWrap(True)
        form.addRow("", self.score_note)

        self.analyse = QtWidgets.QCheckBox("Estimate tempo and key")
        self.analyse.setChecked(True)
        self.analyse.setToolTip(
            "The tempo is also used to time the MIDI, so bars line up in a DAW."
        )
        form.addRow("", self.analyse)

        self.format = QtWidgets.QComboBox()
        self.format.addItem("FLAC (lossless)", OutputFormat.FLAC.value)
        self.format.addItem("WAV", OutputFormat.WAV.value)
        self.format_label = QtWidgets.QLabel("Stem format")
        form.addRow(self.format_label, self.format)

        self.limit = QtWidgets.QSpinBox()
        self.limit.setRange(0, 3600)
        self.limit.setSingleStep(30)
        self.limit.setValue(0)
        self.limit.setSuffix(" s")
        self.limit.setSpecialValueText("Whole recording")
        self.limit.valueChanged.connect(self._update_state)
        self.limit.setToolTip(
            "Stop after this much of the recording. Useful on a CPU, where a "
            "six-minute cue takes a while."
        )
        form.addRow("Process", self.limit)

        box.addLayout(form)

        self.warning = QtWidgets.QLabel("")
        self.warning.setProperty("role", "warn")
        self.warning.setWordWrap(True)
        self.warning.setVisible(False)
        box.addWidget(self.warning)
        return frame

    def _action_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setProperty("role", "card")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(18, 12, 18, 12)
        self.estimate = QtWidgets.QLabel("")
        self.estimate.setProperty("role", "dim")
        row.addWidget(self.estimate, 1)
        self.go = QtWidgets.QPushButton("Deconstruct")
        self.go.setToolTip("Queue this recording for deconstruction")
        self.go.setProperty("role", "primary")
        self.go.setMinimumWidth(150)
        self.go.clicked.connect(self.submit)
        row.addWidget(self.go)
        return bar

    # -- state --------------------------------------------------------------

    def _refresh_models(self) -> None:
        self.model.blockSignals(True)
        self.model.clear()
        for kind in ("separator", "transcriber"):
            for entry in self.service.catalog.by_kind(kind):
                self.model.addItem(entry.name, entry.id)
        self.model.blockSignals(False)
        self._on_model_changed()

    def current_kind(self) -> str:
        entry = self.current_model()
        return entry.kind if entry else "separator"

    def current_model(self) -> ModelEntry | None:
        model_id = self.model.currentData()
        return self.service.catalog.get(model_id) if model_id else None

    def _on_model_changed(self) -> None:
        entry = self.current_model()
        if entry is None:
            self.stem_summary.setText("")
            return
        separating = entry.kind == "separator"
        self.stem_summary.setText(
            ("Audio stems: " if separating else "MIDI layers: ") + ", ".join(entry.stems)
        )
        # The two paths produce different things, so the controls that only
        # apply to one of them go away rather than sitting there inert.
        for widget in (self.transcribe, self.format, self.format_label):
            widget.setVisible(separating)
        self.score_note.setVisible(not separating)
        self._update_state()

    def set_source(self, path: str) -> None:
        self._source = Path(path)
        self.drop.label.setText(self._source.name)
        try:
            import soundfile as sf

            info = sf.info(str(self._source))
            length = info.frames / max(1, info.samplerate)
            self.drop.detail.setText(
                f"{int(length // 60)}:{int(length % 60):02d}  |  "
                f"{info.samplerate} Hz  |  {info.channels} ch"
            )
        except Exception:
            self.drop.detail.setText("Could not read this file's details")
        self._update_state()

    def _update_state(self) -> None:
        entry = self.current_model()
        separating = entry.kind == "separator" if entry else True
        has_source = self._source is not None and self._source.exists()
        self.go.setEnabled(has_source and entry is not None)
        self.vocal_note.setVisible(separating and self.transcribe.isChecked())

        warnings: list[str] = []
        if entry is not None:
            generator = self.service.generator_for(entry.id)[1]
            missing = generator.missing_packages() if generator else ["adapter"]
            if missing:
                warnings.append(
                    f"Needs: {', '.join(missing)}. Install a compute runtime from the Models tab."
                )
        if separating and self.transcribe.isChecked():
            from ..._version_probe import transcription_available

            if not transcription_available():
                warnings.append(
                    "Transcription needs basic-pitch, which is not installed; "
                    "stems will still be written as audio."
                )
        self.warning.setText("  ".join(warnings))
        self.warning.setVisible(bool(warnings))

        if not has_source or entry is None:
            self.estimate.setText("Choose a song to get started")
            return
        generator = self.service.generator_for(entry.id)[1]
        if generator is None:
            self.estimate.setText("")
            return
        seconds = generator.estimated_seconds(self.build_request(), self.service.make_context())
        unit = f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.0f} min"
        self.estimate.setText(
            f"About {unit} on {self.service.recommended_compute()}"
        )

    # -- submit -------------------------------------------------------------

    def build_request(self) -> GenerationRequest:
        separating = self.current_kind() == "separator"
        fmt = OutputFormat(self.format.currentData() or OutputFormat.FLAC.value) \
            if separating else OutputFormat.MIDI
        verb = "deconstruct" if separating else "transcribe"
        return GenerationRequest(
            prompt=f"{verb} {self._source.name}" if self._source else verb,
            model_id=self.model.currentData() or "demucs-htdemucs",
            output_format=fmt,
            duration_seconds=None,
            extra={
                "input_path": str(self._source) if self._source else "",
                "output_dir": str(self.service.settings.resolved_output_dir()),
                "transcribe": self.transcribe.isChecked(),
                "analyse": self.analyse.isChecked(),
                "max_seconds": float(self.limit.value()),
                "sample_rate": self.service.settings.sample_rate,
                "bit_depth": self.service.settings.bit_depth,
            },
        )

    def submit(self) -> None:
        if self._source is None:
            return
        request = self.build_request()
        try:
            jobs = self.service.submit(request, request.model_id)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not start", str(exc))
            return
        self.submitted.emit(jobs)
