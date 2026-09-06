"""The Models tab: what is installed, what can be downloaded, and the runtime."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...config.paths import human_bytes
from ...core.catalog import ModelEntry
from ...core.downloader import download_model, local_size, model_is_present, remove_model
from ...core.runtime import install_runtime, options_for
from ...core.service import AppService

__all__ = ["ModelsPanel"]


class ModelCard(QtWidgets.QFrame):
    """One catalog entry, with its licence, size and install state."""

    changed = QtCore.Signal()

    def __init__(self, entry: ModelEntry, service: AppService,
                 parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.entry = entry
        self.service = service
        self._handle = None
        self.setProperty("role", "card")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 14)
        layout.setSpacing(7)

        top = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel(entry.name + ("  *" if entry.flagship else ""))
        name.setProperty("role", "subtitle")
        top.addWidget(name)
        top.addStretch(1)

        badge = QtWidgets.QLabel(entry.kind.upper())
        badge.setProperty("role", "dim")
        top.addWidget(badge)
        layout.addLayout(top)

        desc = QtWidgets.QLabel(entry.description or "")
        desc.setProperty("role", "dim")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        facts = []
        if entry.size_gb:
            facts.append(f"{entry.size_gb:g} GB")
        if entry.vram_gb:
            facts.append(f"{entry.vram_gb:g} GB VRAM")
        facts.append(entry.license)
        facts.append("outputs: " + ", ".join(entry.outputs))
        if entry.vocals:
            facts.append("vocals")
        meta = QtWidgets.QLabel("  |  ".join(facts))
        meta.setProperty("role", "dim")
        meta.setWordWrap(True)
        layout.addWidget(meta)

        warning = entry.license_warning()
        if warning:
            label = QtWidgets.QLabel(warning)
            label.setProperty("role", "warn")
            label.setWordWrap(True)
            layout.addWidget(label)

        if entry.notes:
            notes = QtWidgets.QLabel(entry.notes)
            notes.setProperty("role", "dim")
            notes.setWordWrap(True)
            layout.addWidget(notes)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QtWidgets.QLabel("")
        self.status.setProperty("role", "dim")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        self.action = QtWidgets.QPushButton("Download")
        self.action.setProperty("role", "primary")
        self.action.clicked.connect(self._on_action)
        buttons.addWidget(self.action)
        self.remove = QtWidgets.QPushButton("Remove")
        self.remove.setProperty("role", "danger")
        self.remove.clicked.connect(self._on_remove)
        buttons.addWidget(self.remove)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._bridge = _Bridge()
        self._bridge.progress.connect(self._on_progress)
        self._bridge.finished.connect(self._on_finished)
        self.refresh()

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        entry = self.entry
        models_dir = self.service.settings.resolved_models_dir()
        generator = self.service.generator_for(entry.id)[1]
        missing = generator.missing_packages() if generator else ["adapter"]

        if not entry.needs_download:
            # Nothing to fetch or delete for a built-in backend, so offer neither.
            self.status.setText("Built in. Always ready.")
            self.action.setVisible(False)
            self.remove.setVisible(False)
            self.progress.setVisible(False)
            return

        present = model_is_present(entry, models_dir)
        size = local_size(entry, models_dir)
        size_text = f"Installed ({human_bytes(size)})" if present else "Not downloaded"

        self.action.setVisible(True)
        if missing:
            self.status.setText(f"{size_text}. Needs: {', '.join(missing)}")
            self.action.setText("Download anyway")
        else:
            self.status.setText(f"{size_text}. Ready.")
            self.action.setText("Re-download" if present else "Download")
        self.action.setEnabled(True)
        self.remove.setVisible(present)

    def _on_action(self) -> None:
        if self._handle is not None and not self._handle.done:
            self._handle.stop()
            self.action.setText("Cancelling")
            return
        if self.entry.gated and not self.service.settings.hf_token:
            QtWidgets.QMessageBox.information(
                self, "Access token needed",
                "This model is gated. Accept its licence on Hugging Face, then add "
                "an access token in Settings.",
            )
            return
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.action.setText("Cancel")
        self._handle = download_model(
            self.entry,
            self.service.settings.resolved_models_dir(),
            token=self.service.settings.hf_token,
            on_progress=lambda f, m: self._bridge.progress.emit(f, m),
            on_finished=lambda h: self._bridge.finished.emit(h.error or ""),
        )

    def _on_remove(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "Remove model", f"Delete the downloaded files for {self.entry.name}?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer is QtWidgets.QMessageBox.StandardButton.Yes:
            remove_model(self.entry, self.service.settings.resolved_models_dir())
            self.refresh()
            self.changed.emit()

    @QtCore.Slot(float, str)
    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(fraction * 100))
        self.status.setText(message)

    @QtCore.Slot(str)
    def _on_finished(self, error: str) -> None:
        self.progress.setVisible(False)
        self._handle = None
        if error:
            self.status.setText(error)
            self.status.setProperty("role", "error")
        self.refresh()
        self.changed.emit()


class _Bridge(QtCore.QObject):
    """Marshals download callbacks from their worker thread onto the UI thread."""

    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(str)


class _InstallBridge(QtCore.QObject):
    """Same, for the runtime installer's streamed pip output."""

    line = QtCore.Signal(str)
    finished = QtCore.Signal(str)


class ModelsPanel(QtWidgets.QWidget):
    catalog_changed = QtCore.Signal()

    def __init__(self, service: AppService, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self._install = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Models")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        add = QtWidgets.QPushButton("Add custom model")
        add.clicked.connect(self._add_custom)
        header.addWidget(add)
        layout.addLayout(header)

        layout.addWidget(self._runtime_card())

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        container = QtWidgets.QWidget()
        self._list = QtWidgets.QVBoxLayout(container)
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(10)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        self.refresh()

    def _runtime_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setProperty("role", "card")
        box = QtWidgets.QVBoxLayout(frame)
        box.setContentsMargins(16, 13, 16, 14)
        box.setSpacing(8)

        title = QtWidgets.QLabel("Compute runtime")
        title.setProperty("role", "subtitle")
        box.addWidget(title)

        gpu = self.service.system.primary_gpu
        detected = f"{gpu.name} ({gpu.vram_gb} GB)" if gpu else "No discrete GPU detected"
        info = QtWidgets.QLabel(
            f"{detected}\n{self.service.system.cpu} - "
            f"{self.service.system.cpu_cores} cores, {self.service.system.ram_gb} GB RAM\n"
            f"PyTorch: {self.service.system.torch_version or 'not installed'}"
        )
        info.setProperty("role", "dim")
        box.addWidget(info)

        row = QtWidgets.QHBoxLayout()
        self.runtime = QtWidgets.QComboBox()
        vendor = gpu.vendor.value if gpu else "unknown"
        for option in options_for(vendor):
            self.runtime.addItem(f"{option.name}  (~{option.approx_gb:g} GB)", option)
        row.addWidget(self.runtime, 1)
        self.install_button = QtWidgets.QPushButton("Install")
        self.install_button.setProperty("role", "primary")
        self.install_button.clicked.connect(self._install_runtime)
        row.addWidget(self.install_button)
        box.addLayout(row)

        self.runtime_notes = QtWidgets.QLabel("")
        self.runtime_notes.setProperty("role", "dim")
        self.runtime_notes.setWordWrap(True)
        box.addWidget(self.runtime_notes)
        self.runtime.currentIndexChanged.connect(self._on_runtime_changed)
        self._on_runtime_changed()

        self.install_log = QtWidgets.QPlainTextEdit()
        self.install_log.setReadOnly(True)
        self.install_log.setMaximumHeight(120)
        self.install_log.setVisible(False)
        box.addWidget(self.install_log)

        self._install_bridge = _InstallBridge()
        self._install_bridge.line.connect(self._append_log)
        self._install_bridge.finished.connect(self._on_install_finished)
        return frame

    def _on_runtime_changed(self) -> None:
        option = self.runtime.currentData()
        self.runtime_notes.setText(
            f"{option.description}\n{option.notes}".strip() if option else ""
        )

    def _install_runtime(self) -> None:
        option = self.runtime.currentData()
        if option is None:
            return
        if self._install is not None and not self._install.done:
            self._install.stop()
            return
        self.install_log.setVisible(True)
        self.install_log.clear()
        self.install_button.setText("Cancel")
        gpu = self.service.system.primary_gpu
        self._install = install_runtime(
            option,
            on_line=lambda line: self._install_bridge.line.emit(line),
            on_finished=lambda h: self._install_bridge.finished.emit(h.error or ""),
            gfx_arch=gpu.gfx_arch if gpu else "",
        )

    @QtCore.Slot(str)
    def _append_log(self, line: str) -> None:
        self.install_log.appendPlainText(line)

    @QtCore.Slot(str)
    def _on_install_finished(self, error: str) -> None:
        self.install_button.setText("Install")
        self._install = None
        self._append_log(f"\nFailed: {error}" if error else "\nDone. Restart to use the new runtime.")

    # -- catalog ------------------------------------------------------------

    def refresh(self) -> None:
        while self._list.count():
            item = self._list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for entry in sorted(self.service.catalog.models, key=lambda m: (m.tier, m.name)):
            card = ModelCard(entry, self.service)
            card.changed.connect(self.catalog_changed)
            self._list.addWidget(card)
        self._list.addStretch(1)

    def _add_custom(self) -> None:
        dialog = AddModelDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        from ...core.catalog import save_user_model

        entry = dialog.entry()
        save_user_model(entry)
        self.service.catalog.upsert(entry)
        self.refresh()
        self.catalog_changed.emit()


class AddModelDialog(QtWidgets.QDialog):
    """Add any Hugging Face repo that one of the existing adapters can load."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add a model")
        self.setMinimumWidth(460)
        form = QtWidgets.QFormLayout(self)

        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("Display name")
        form.addRow("Name", self.name)

        self.repo = QtWidgets.QLineEdit()
        self.repo.setPlaceholderText("owner/repository")
        form.addRow("HF repo", self.repo)

        self.adapter = QtWidgets.QComboBox()
        from ...core.registry import ADAPTERS

        for key in ADAPTERS:
            if key != "builtin":
                self.adapter.addItem(key)
        form.addRow("Adapter", self.adapter)

        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["audio", "symbolic"])
        form.addRow("Kind", self.kind)

        self.size = QtWidgets.QDoubleSpinBox()
        self.size.setRange(0, 500)
        self.size.setSuffix(" GB")
        form.addRow("Download size", self.size)

        self.vram = QtWidgets.QDoubleSpinBox()
        self.vram.setRange(0, 200)
        self.vram.setSuffix(" GB")
        form.addRow("VRAM needed", self.vram)

        self.licence = QtWidgets.QLineEdit()
        self.licence.setPlaceholderText("e.g. Apache-2.0")
        form.addRow("Licence", self.licence)

        note = QtWidgets.QLabel(
            "The adapter decides how the model is loaded. Pick the one matching its "
            "architecture; a MusicGen-style repo needs the MusicGen adapter."
        )
        note.setProperty("role", "dim")
        note.setWordWrap(True)
        form.addRow(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def entry(self) -> ModelEntry:
        repo = self.repo.text().strip()
        kind = self.kind.currentText()
        return ModelEntry(
            id=repo.replace("/", "-").lower() or "custom-model",
            name=self.name.text().strip() or repo,
            kind=kind,
            adapter=self.adapter.currentText(),
            repo=repo,
            license=self.licence.text().strip() or "unknown",
            size_gb=self.size.value(),
            vram_gb=self.vram.value(),
            devices=("cuda", "rocm", "cpu"),
            outputs=("flac", "wav") if kind == "audio" else ("midi", "flac", "wav"),
            tier=4,
            description="Added by you.",
            source="user",
        )
