"""The Remix tab: keep part of a recording, and write the rest anew."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ...core.catalog import ModelEntry
from ...core.models import GenerationRequest, OutputFormat
from ...core.remix import DEFAULT_BED_MODEL, DEFAULT_KEEP
from ...core.service import AppService
from .deconstruct_panel import DropZone

__all__ = ["RemixPanel"]


class RemixPanel(QtWidgets.QWidget):
    submitted = QtCore.Signal(list)

    def __init__(self, service: AppService, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self._source: Path | None = None
        self._keep_boxes: dict[str, QtWidgets.QCheckBox] = {}

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

        title = QtWidgets.QLabel("Remix")
        title.setProperty("role", "title")
        layout.addWidget(title)

        blurb = QtWidgets.QLabel(
            "Keep the parts of a recording worth keeping — the vocal on a song, "
            "or the synths and guitars on an instrumental — and describe the "
            "backing you want written underneath them. Move the tempo as well "
            "and you have changed what genre it is."
        )
        blurb.setProperty("role", "dim")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self.drop = DropZone()
        self.drop.file_chosen.connect(self.set_source)
        layout.addWidget(self.drop)

        layout.addWidget(self._prompt_card())
        layout.addWidget(self._layers_card())
        layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        outer.addWidget(self._action_bar())

        self.refresh_models()
        self._update_state()

    # -- construction -------------------------------------------------------

    def _prompt_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setProperty("role", "card")
        box = QtWidgets.QVBoxLayout(frame)
        box.setContentsMargins(16, 14, 16, 16)
        box.setSpacing(8)

        heading = QtWidgets.QLabel("What should the new backing be?")
        heading.setProperty("role", "subtitle")
        box.addWidget(heading)

        self.prompt = QtWidgets.QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "e.g. driving four-to-the-floor EDM with a big synth bass and side-chained pads"
        )
        self.prompt.setFixedHeight(70)
        self.prompt.textChanged.connect(self._update_state)
        box.addWidget(self.prompt)

        note = QtWidgets.QLabel(
            "The recording's own tempo and key are measured and handed to the "
            "generator, so you do not need to name them."
        )
        note.setProperty("role", "dim")
        note.setWordWrap(True)
        box.addWidget(note)
        return frame

    def _layers_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setProperty("role", "card")
        box = QtWidgets.QVBoxLayout(frame)
        box.setContentsMargins(16, 14, 16, 16)
        box.setSpacing(10)

        heading = QtWidgets.QLabel("Layers and backends")
        heading.setProperty("role", "subtitle")
        box.addWidget(heading)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)

        self.separator = QtWidgets.QComboBox()
        self.separator.currentIndexChanged.connect(lambda _i: self._rebuild_keep_boxes())
        form.addRow("Separate with", self.separator)

        self.keep_row = QtWidgets.QWidget()
        self.keep_layout = QtWidgets.QHBoxLayout(self.keep_row)
        self.keep_layout.setContentsMargins(0, 0, 0, 0)
        self.keep_layout.setSpacing(12)
        form.addRow("Keep", self.keep_row)

        self.keep_note = QtWidgets.QLabel("")
        self.keep_note.setProperty("role", "dim")
        self.keep_note.setWordWrap(True)
        form.addRow("", self.keep_note)

        self.bed = QtWidgets.QComboBox()
        self.bed.currentIndexChanged.connect(self._update_state)
        form.addRow("Write the backing with", self.bed)

        self.bed_note = QtWidgets.QLabel("")
        self.bed_note.setProperty("role", "dim")
        self.bed_note.setWordWrap(True)
        form.addRow("", self.bed_note)

        self.tempo = QtWidgets.QSpinBox()
        self.tempo.setRange(0, 200)
        self.tempo.setSingleStep(2)
        self.tempo.setSuffix(" bpm")
        self.tempo.setSpecialValueText("Keep the original")
        self.tempo.valueChanged.connect(self._update_state)
        form.addRow("Tempo", self.tempo)

        self.tempo_note = QtWidgets.QLabel("")
        self.tempo_note.setProperty("role", "dim")
        self.tempo_note.setWordWrap(True)
        form.addRow("", self.tempo_note)

        self.limit = QtWidgets.QSpinBox()
        self.limit.setRange(0, 3600)
        self.limit.setSingleStep(30)
        self.limit.setSuffix(" s")
        self.limit.setSpecialValueText("Whole recording")
        self.limit.valueChanged.connect(self._update_state)
        form.addRow("Process", self.limit)

        self.save_stems = QtWidgets.QCheckBox(
            "Also save the kept layers and the new backing separately"
        )
        self.save_stems.setToolTip("So you can re-balance the mix yourself afterwards")
        form.addRow("", self.save_stems)

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
        self.go = QtWidgets.QPushButton("Remix")
        self.go.setProperty("role", "primary")
        self.go.setMinimumWidth(150)
        self.go.clicked.connect(self.submit)
        row.addWidget(self.go)
        return bar

    # -- state --------------------------------------------------------------

    def refresh_models(self) -> None:
        # The catalog changes whenever anything finishes downloading, and this
        # runs on every one of those. Put the user's choices back afterwards:
        # silently resetting the separator, the backing model and the kept
        # layers while they are on another tab is how someone ends up
        # remixing something they did not ask for.
        keeping = self.keep_stems() or None
        chosen_separator = self.separator.currentData()
        chosen_bed = self.bed.currentData()

        self.separator.blockSignals(True)
        self.separator.clear()
        for entry in self.service.catalog.by_kind("separator"):
            self.separator.addItem(entry.name, entry.id)
        self._restore(self.separator, chosen_separator)
        self.separator.blockSignals(False)

        self.bed.blockSignals(True)
        self.bed.clear()
        for entry in self.service.usable_models("flac"):
            if entry.kind in ("symbolic", "audio"):
                self.bed.addItem(entry.name, entry.id)
        self._restore(self.bed, chosen_bed or DEFAULT_BED_MODEL)
        self.bed.blockSignals(False)

        self._rebuild_keep_boxes(keeping)

    @staticmethod
    def _restore(combo: QtWidgets.QComboBox, value) -> None:
        index = combo.findData(value) if value else -1
        if index >= 0:
            combo.setCurrentIndex(index)

    def _rebuild_keep_boxes(self, keeping: list[str] | None = None) -> None:
        while self.keep_layout.count():
            item = self.keep_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Cut the connection now rather than trusting deleteLater: a
                # box that is still wired up can be toggled between here and
                # its deletion, and _update_state would then read a dict that
                # no longer holds it.
                widget.toggled.disconnect()
                widget.setParent(None)
                widget.deleteLater()
        self._keep_boxes = {}

        entry = self.current_separator()
        stems = entry.stems if entry else DEFAULT_KEEP
        for stem in stems:
            box = QtWidgets.QCheckBox(stem)
            box.setChecked(stem in keeping if keeping is not None else stem in DEFAULT_KEEP)
            box.toggled.connect(self._update_state)
            self.keep_layout.addWidget(box)
            self._keep_boxes[stem] = box
        self.keep_layout.addStretch(1)
        self._update_state()

    def current_separator(self) -> ModelEntry | None:
        model_id = self.separator.currentData()
        return self.service.catalog.get(model_id) if model_id else None

    def current_bed(self) -> ModelEntry | None:
        model_id = self.bed.currentData()
        return self.service.catalog.get(model_id) if model_id else None

    def keep_stems(self) -> list[str]:
        return [name for name, box in self._keep_boxes.items() if box.isChecked()]

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
        keep = self.keep_stems()
        entry = self.current_separator()
        bed_entry = self.current_bed()
        has_source = self._source is not None and self._source.exists()
        replaced = [s for s in (entry.stems if entry else ()) if s not in keep]
        self.go.setEnabled(has_source and bool(replaced) and bed_entry is not None)

        self.keep_note.setText(
            "Everything else is replaced: " + ", ".join(replaced) if replaced
            else "Nothing is left to replace — clear at least one layer."
        )

        warnings: list[str] = []
        generator = self.service.generator_for("stem-remix")[1]
        missing = generator.missing_packages() if generator else ["adapter"]
        if missing:
            warnings.append(
                f"Needs: {', '.join(missing)}. Install it from the Models tab."
            )

        if bed_entry is not None:
            caps_source = self.service.generator_for(bed_entry.id)[1]
            caps = caps_source.capabilities() if caps_source else None
            if caps is not None:
                if caps.honours_tempo:
                    self.bed_note.setText(
                        "Writes at exactly the tempo it is given, so the new backing "
                        "stays under the kept layer."
                    )
                else:
                    self.bed_note.setText(
                        "A waveform model cannot be held to a tempo. It is told the "
                        "tempo and key in the prompt, but the backing will drift "
                        "against the kept layer rather than locking to it."
                    )
                length = self._length_seconds()
                if length and caps.max_duration and length > caps.max_duration:
                    repeats = int(length // caps.max_duration) + 1
                    warnings.append(
                        f"{bed_entry.name} writes at most {caps.max_duration:.0f}s, so "
                        f"the backing is looped about {repeats} times to cover the track."
                    )
            missing_bed = caps_source.missing_packages() if caps_source else []
            if missing_bed:
                warnings.append(f"{bed_entry.name} needs: {', '.join(missing_bed)}.")

        if self.tempo.value():
            self.tempo_note.setText(
                f"The layers you keep are re-timed to {self.tempo.value()} bpm without "
                "changing their pitch, and the new backing is written at that tempo. "
                "Genre is partly tempo: most house and techno sits around 120-130, "
                "drum and bass nearer 174."
            )
            # A phase vocoder holds up over a moderate change and smears past
            # one, so an impossible ask is met with the nearest tempo that
            # still sounds like the same performance.
            warnings.append(
                "A large tempo change smears transients. Anything beyond half or "
                "double the original is pulled back to what re-timing can do "
                "convincingly."
            )
        else:
            self.tempo_note.setText(
                "The backing is written at whatever tempo the recording turns out to "
                "be. Set a tempo to move the whole thing, which is usually what "
                "changing genre needs."
            )

        self.warning.setText("  ".join(warnings))
        self.warning.setVisible(bool(warnings))

        if not has_source:
            self.estimate.setText("Choose a song to get started")
            return
        if generator is None:
            self.estimate.setText("")
            return
        seconds = generator.estimated_seconds(self.build_request(), self.service.make_context())
        unit = f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.0f} min"
        self.estimate.setText(f"About {unit} on {self.service.recommended_compute()}")

    def _length_seconds(self) -> float:
        limit = float(self.limit.value())
        if not self._source:
            return limit
        try:
            import soundfile as sf

            info = sf.info(str(self._source))
            length = info.frames / max(1, info.samplerate)
        except Exception:
            return limit
        return min(length, limit) if limit > 0 else length

    # -- submit -------------------------------------------------------------

    def build_request(self) -> GenerationRequest:
        return GenerationRequest(
            prompt=self.prompt.toPlainText().strip(),
            model_id="stem-remix",
            output_format=OutputFormat.FLAC,
            duration_seconds=None,
            extra={
                "input_path": str(self._source) if self._source else "",
                "output_dir": str(self.service.settings.resolved_output_dir()),
                "separator": self.separator.currentData() or "demucs-htdemucs",
                "bed_model": self.bed.currentData() or DEFAULT_BED_MODEL,
                "keep_stems": self.keep_stems(),
                "target_tempo": float(self.tempo.value()),
                "save_stems": self.save_stems.isChecked(),
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
