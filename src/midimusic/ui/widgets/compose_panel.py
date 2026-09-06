"""The compose panel: where a prompt becomes a queued generation."""

from __future__ import annotations

import random

from PySide6 import QtCore, QtWidgets

from ...core.catalog import ModelEntry
from ...core.models import GenerationRequest, OutputFormat
from ...core.service import AppService
from ...prompt.parser import parse_prompt
from ...theory.pitch import NOTE_NAMES
from ...theory.structure import FORMS
from ...theory.style import list_styles

__all__ = ["ComposePanel"]

_KEYS = ["Auto"] + [
    f"{n} {m}" for n in NOTE_NAMES for m in ("major", "minor")
]


def _card(title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QFormLayout]:
    frame = QtWidgets.QFrame()
    frame.setProperty("role", "card")
    outer = QtWidgets.QVBoxLayout(frame)
    outer.setContentsMargins(16, 14, 16, 16)
    outer.setSpacing(10)
    if title:
        label = QtWidgets.QLabel(title)
        label.setProperty("role", "subtitle")
        outer.addWidget(label)
    form = QtWidgets.QFormLayout()
    form.setSpacing(10)
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    outer.addLayout(form)
    return frame, form


class ComposePanel(QtWidgets.QWidget):
    """Prompt, musical controls and the generate button."""

    submitted = QtCore.Signal(list)  # list[Job]

    def __init__(self, service: AppService, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self._models: list[ModelEntry] = []
        self._build()
        self.refresh_models()
        self._on_prompt_changed()

    # -- construction -------------------------------------------------------

    def _build(self) -> None:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        layout.addWidget(self._prompt_card())
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(self._model_card(), 1)
        row.addWidget(self._music_card(), 1)
        layout.addLayout(row)
        layout.addWidget(self._advanced_card())
        layout.addStretch(1)

        scroll.setWidget(inner)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll, 1)
        outer.addWidget(self._action_bar())

    def _prompt_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setProperty("role", "card")
        box = QtWidgets.QVBoxLayout(frame)
        box.setContentsMargins(16, 14, 16, 16)
        box.setSpacing(8)

        title = QtWidgets.QLabel("Describe the music")
        title.setProperty("role", "subtitle")
        box.addWidget(title)

        self.prompt = QtWidgets.QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "e.g. dark cinematic orchestral with heavy percussion, 120 bpm, C minor"
        )
        self.prompt.setFixedHeight(76)
        self.prompt.textChanged.connect(self._on_prompt_changed)
        box.addWidget(self.prompt)

        self.hint = QtWidgets.QLabel("")
        self.hint.setProperty("role", "dim")
        self.hint.setWordWrap(True)
        box.addWidget(self.hint)

        self.instrumental = QtWidgets.QCheckBox("Instrumental (no vocals)")
        self.instrumental.setChecked(True)
        self.instrumental.toggled.connect(self._on_instrumental)
        box.addWidget(self.instrumental)

        self.lyrics_label = QtWidgets.QLabel("Lyrics")
        self.lyrics_label.setProperty("role", "dim")
        box.addWidget(self.lyrics_label)
        self.lyrics = QtWidgets.QPlainTextEdit()
        self.lyrics.setPlaceholderText(
            "[Verse]\nWrite your lyrics here\n\n[Chorus]\nStructure tags help the model"
        )
        self.lyrics.setFixedHeight(110)
        box.addWidget(self.lyrics)
        self._on_instrumental(True)
        return frame

    def _model_card(self) -> QtWidgets.QFrame:
        frame, form = _card("Model and output")

        self.output_format = QtWidgets.QComboBox()
        # Store the plain value: Qt unwraps a str-Enum to its string, so the
        # enum would not survive the round trip through currentData().
        self.output_format.addItem("FLAC (audio)", OutputFormat.FLAC.value)
        self.output_format.addItem("MIDI (score)", OutputFormat.MIDI.value)
        self.output_format.addItem("WAV (audio)", OutputFormat.WAV.value)
        self.output_format.currentIndexChanged.connect(self.refresh_models)
        form.addRow("Output", self.output_format)

        self.model = QtWidgets.QComboBox()
        self.model.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model", self.model)

        self.model_info = QtWidgets.QLabel("")
        self.model_info.setProperty("role", "dim")
        self.model_info.setWordWrap(True)
        form.addRow("", self.model_info)

        self.model_warning = QtWidgets.QLabel("")
        self.model_warning.setProperty("role", "warn")
        self.model_warning.setWordWrap(True)
        self.model_warning.setVisible(False)
        form.addRow("", self.model_warning)
        frame.layout().addStretch(1)
        return frame

    def _music_card(self) -> QtWidgets.QFrame:
        frame, form = _card("Musical direction")

        self.style = QtWidgets.QComboBox()
        self.style.addItem("Auto (from prompt)", "")
        for name in list_styles():
            self.style.addItem(name.replace("_", " ").title(), name)
        form.addRow("Style", self.style)

        self.key = QtWidgets.QComboBox()
        self.key.addItems(_KEYS)
        form.addRow("Key", self.key)

        tempo_row = QtWidgets.QHBoxLayout()
        self.tempo = QtWidgets.QSpinBox()
        self.tempo.setRange(0, 300)
        self.tempo.setSpecialValueText("Auto")
        self.tempo.setSuffix(" bpm")
        tempo_row.addWidget(self.tempo, 1)
        self.structure = QtWidgets.QComboBox()
        self.structure.addItem("Auto", "")
        for name in FORMS:
            self.structure.addItem(name.replace("_", " ").title(), name)
        tempo_row.addWidget(self.structure, 1)
        form.addRow("Tempo / form", tempo_row)

        self.duration = QtWidgets.QSpinBox()
        self.duration.setRange(5, 900)
        self.duration.setValue(int(self.service.settings.default_duration))
        self.duration.setSuffix(" s")
        self.duration.valueChanged.connect(self._update_estimate)
        form.addRow("Length", self.duration)

        self.complexity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.complexity.setRange(0, 100)
        self.complexity.setValue(int(self.service.settings.default_complexity * 100))
        self.complexity_label = QtWidgets.QLabel("")
        self.complexity_label.setProperty("role", "dim")
        self.complexity.valueChanged.connect(self._on_complexity)
        cx_row = QtWidgets.QVBoxLayout()
        cx_row.setSpacing(2)
        cx_row.addWidget(self.complexity)
        cx_row.addWidget(self.complexity_label)
        form.addRow("Complexity", cx_row)
        self._on_complexity(self.complexity.value())
        return frame

    def _advanced_card(self) -> QtWidgets.QFrame:
        frame, form = _card("Advanced")

        seed_row = QtWidgets.QHBoxLayout()
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(-1, 2_000_000_000)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText("Random")
        seed_row.addWidget(self.seed, 1)
        dice = QtWidgets.QPushButton("New seed")
        dice.setProperty("role", "ghost")
        dice.clicked.connect(lambda: self.seed.setValue(random.randrange(1, 2_000_000_000)))
        seed_row.addWidget(dice)
        form.addRow("Seed", seed_row)

        self.variations = QtWidgets.QSpinBox()
        self.variations.setRange(1, 8)
        self.variations.setValue(int(self.service.settings.default_variations))
        self.variations.valueChanged.connect(self._update_estimate)
        form.addRow("Variations", self.variations)

        self.energy = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.energy.setRange(0, 100)
        self.energy.setValue(50)
        form.addRow("Energy", self.energy)

        self.brightness = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brightness.setRange(0, 100)
        self.brightness.setValue(50)
        form.addRow("Brightness", self.brightness)

        self.negative = QtWidgets.QLineEdit()
        self.negative.setPlaceholderText("Things to avoid (supported models only)")
        form.addRow("Avoid", self.negative)
        return frame

    def _action_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setProperty("role", "card")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(18, 12, 18, 12)

        self.estimate = QtWidgets.QLabel("")
        self.estimate.setProperty("role", "dim")
        row.addWidget(self.estimate, 1)

        self.generate_button = QtWidgets.QPushButton("Generate")
        self.generate_button.setProperty("role", "primary")
        self.generate_button.setMinimumWidth(150)
        self.generate_button.clicked.connect(self.submit)
        row.addWidget(self.generate_button)
        return bar

    # -- behaviour ----------------------------------------------------------

    def _on_instrumental(self, checked: bool) -> None:
        self.lyrics.setVisible(not checked)
        self.lyrics_label.setVisible(not checked)

    def _on_complexity(self, value: int) -> None:
        words = [
            (25, "Sparse - a few core parts"),
            (50, "Moderate - a small band"),
            (80, "Full - layered arrangement"),
            (100, "Dense - everything, fully produced"),
        ]
        for limit, text in words:
            if value <= limit:
                self.complexity_label.setText(f"{value}%  {text}")
                break

    def _on_prompt_changed(self) -> None:
        text = self.prompt.toPlainText().strip()
        if not text:
            self.hint.setText("Tip: mention a genre, mood, tempo and key.")
            return
        parsed = parse_prompt(text)
        self.hint.setText(f"Read as: {parsed.describe()}")
        self._update_estimate()

    def _on_model_changed(self) -> None:
        entry = self.current_model()
        if entry is None:
            self.model_info.setText("")
            self.model_warning.setVisible(False)
            return

        bits = [entry.description or entry.name]
        facts = []
        if entry.size_gb:
            facts.append(f"{entry.size_gb:g} GB download")
        if entry.vram_gb:
            facts.append(f"{entry.vram_gb:g} GB VRAM")
        if entry.vocals:
            facts.append("vocals")
        facts.append(entry.license)
        bits.append(" | ".join(facts))
        self.model_info.setText("\n".join(bits))

        warnings = []
        if entry.license_warning():
            warnings.append(entry.license_warning())
        generator = self.service.generator_for(entry.id)[1]
        if generator is not None:
            missing = generator.missing_packages()
            if missing:
                warnings.append(f"Needs: {', '.join(missing)}. Install from the Models tab.")
        if entry.gated:
            warnings.append("Gated: accept the licence on Hugging Face and add a token.")
        self.model_warning.setText("  ".join(warnings))
        self.model_warning.setVisible(bool(warnings))

        caps = generator.capabilities() if generator else None
        if caps:
            self.duration.setMaximum(max(5, int(caps.max_duration)))
            self.instrumental.setEnabled(caps.supports_vocals)
            if not caps.supports_vocals:
                self.instrumental.setChecked(True)
            self.negative.setEnabled(caps.supports_negative_prompt)
            for widget in (self.key, self.tempo, self.structure, self.style, self.complexity):
                widget.setEnabled(caps.honours_key or caps.honours_tempo or entry.is_builtin)
        self._update_estimate()

    def _update_estimate(self) -> None:
        entry = self.current_model()
        if entry is None:
            self.estimate.setText("")
            return
        generator = self.service.generator_for(entry.id)[1]
        if generator is None:
            self.estimate.setText("")
            return
        request = self.build_request()
        seconds = generator.estimated_seconds(request, self.service.make_context())
        total = seconds * max(1, self.variations.value())
        unit = f"{total:.0f}s" if total < 90 else f"{total / 60:.0f} min"
        count = self.variations.value()
        text = (
            f"About {unit} for {count} variation{'s' if count > 1 else ''} on "
            f"{self.service.recommended_compute()}"
        )
        # A generation measured in tens of minutes is a decision, not a detail.
        # Say so plainly rather than letting the user find out by waiting.
        if total > 900:
            text += "  -  this model is slow without a GPU"
            self.estimate.setProperty("role", "warn")
        else:
            self.estimate.setProperty("role", "dim")
        self.estimate.style().unpolish(self.estimate)
        self.estimate.style().polish(self.estimate)
        self.estimate.setText(text)

    # -- data ---------------------------------------------------------------

    def refresh_models(self) -> None:
        self._models = self.service.usable_models(self.selected_format().value)
        current = self.model.currentData()
        self.model.blockSignals(True)
        self.model.clear()
        for entry in self._models:
            label = entry.name
            if entry.flagship:
                label += "  *"
            if entry.tier == 0:
                label += "  (no download)"
            self.model.addItem(label, entry.id)
        if current:
            idx = self.model.findData(current)
            if idx >= 0:
                self.model.setCurrentIndex(idx)
        self.model.blockSignals(False)
        self._on_model_changed()

    def selected_format(self) -> OutputFormat:
        return OutputFormat(self.output_format.currentData() or OutputFormat.FLAC.value)

    def current_model(self) -> ModelEntry | None:
        model_id = self.model.currentData()
        return self.service.catalog.get(model_id) if model_id else None

    def build_request(self) -> GenerationRequest:
        seed = self.seed.value()
        key = self.key.currentText()
        return GenerationRequest(
            prompt=self.prompt.toPlainText().strip(),
            model_id=self.model.currentData() or "builtin-composer",
            output_format=self.selected_format(),
            style=self.style.currentData() or None,
            key=None if key == "Auto" else key,
            tempo=float(self.tempo.value()) if self.tempo.value() > 0 else None,
            duration_seconds=float(self.duration.value()),
            structure=self.structure.currentData() or None,
            instrumental=self.instrumental.isChecked(),
            lyrics=self.lyrics.toPlainText().strip(),
            seed=None if seed < 0 else seed,
            variations=self.variations.value(),
            negative_prompt=self.negative.text().strip(),
            extra={
                "complexity": self.complexity.value() / 100.0,
                "energy": self.energy.value() / 100.0,
                "brightness": self.brightness.value() / 100.0,
            },
        )

    def submit(self) -> None:
        request = self.build_request()
        try:
            jobs = self.service.submit(request, request.model_id)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not start", str(exc))
            return
        self.submitted.emit(jobs)
