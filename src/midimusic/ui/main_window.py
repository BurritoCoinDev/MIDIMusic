"""The application window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..config.paths import get_paths
from ..config.settings import save_settings
from ..core.jobs import Job
from ..core.service import AppService
from .theme import DARK, LIGHT, build_stylesheet
from .widgets.compose_panel import ComposePanel
from .widgets.library_panel import LibraryPanel, reveal_in_explorer
from .widgets.models_panel import ModelsPanel
from .widgets.queue_panel import QueuePanel
from .widgets.settings_panel import SettingsPanel

__all__ = ["MainWindow"]

log = logging.getLogger(__name__)

APP_TITLE = "MIDIMusic"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, service: AppService):
        super().__init__()
        self.service = service
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 820)
        self.setMinimumSize(940, 640)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)

        self.compose = ComposePanel(service)
        self.queue = QueuePanel(service)
        self.library = LibraryPanel(service)
        self.models = ModelsPanel(service)
        self.settings_panel = SettingsPanel(service)

        self.tabs.addTab(self.compose, "Compose")
        self.tabs.addTab(self.queue, "Queue")
        self.tabs.addTab(self.library, "Library")
        self.tabs.addTab(self.models, "Models")
        self.tabs.addTab(self.settings_panel, "Settings")
        self.setCentralWidget(self.tabs)

        self.compose.submitted.connect(self._on_submitted)
        self.queue.job_finished.connect(self._on_job_finished)
        self.queue.show_output.connect(self._reveal_job_output)
        self.models.catalog_changed.connect(self.compose.refresh_models)
        self.settings_panel.settings_changed.connect(self._on_settings_changed)

        self._build_menu()
        self._build_status_bar()
        self._restore_geometry()

    # -- chrome -------------------------------------------------------------

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        act = QtGui.QAction("Open output folder", self)
        act.triggered.connect(self._open_output)
        file_menu.addAction(act)
        act = QtGui.QAction("Open settings folder", self)
        act.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(get_paths().config))
            )
        )
        file_menu.addAction(act)
        file_menu.addSeparator()
        act = QtGui.QAction("E&xit", self)
        act.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        gen_menu = menu.addMenu("&Generate")
        act = QtGui.QAction("Generate now", self)
        act.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        act.triggered.connect(self.compose.submit)
        gen_menu.addAction(act)
        act = QtGui.QAction("Cancel all", self)
        act.triggered.connect(self.service.queue.cancel_all)
        gen_menu.addAction(act)

        view_menu = menu.addMenu("&View")
        self.theme_action = QtGui.QAction("Light theme", self, checkable=True)
        self.theme_action.setChecked(self.service.settings.theme == "light")
        self.theme_action.toggled.connect(self._on_theme_toggled)
        view_menu.addAction(self.theme_action)

        help_menu = menu.addMenu("&Help")
        act = QtGui.QAction("About", self)
        act.triggered.connect(self._about)
        help_menu.addAction(act)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        gpu = self.service.system.primary_gpu
        hardware = f"{gpu.name} ({gpu.vram_gb} GB)" if gpu else "CPU only"
        self.hardware_label = QtWidgets.QLabel(
            f"{hardware}  |  {self.service.recommended_compute()}"
        )
        bar.addPermanentWidget(self.hardware_label)
        self.queue_label = QtWidgets.QLabel("Idle")
        bar.addWidget(self.queue_label)

        self._status_timer = QtCore.QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(700)

    def _update_status(self) -> None:
        pending = self.service.queue.pending
        current = self.service.queue.current
        if current is not None:
            self.queue_label.setText(
                f"Generating: {current.display_name()} - {current.progress.percent}%"
            )
        elif pending:
            self.queue_label.setText(f"{pending} queued")
        else:
            self.queue_label.setText("Idle")

    # -- events -------------------------------------------------------------

    def _on_submitted(self, jobs: list[Job]) -> None:
        self.tabs.setCurrentWidget(self.queue)
        count = len(jobs)
        self.statusBar().showMessage(
            f"Queued {count} generation{'s' if count != 1 else ''}", 4000
        )

    def _on_job_finished(self, _job: Job) -> None:
        self.library.refresh()

    def _reveal_job_output(self, job_id: str) -> None:
        job = self.service.queue.get(job_id)
        if job and job.output_paths:
            reveal_in_explorer(Path(job.output_paths[0]))

    def _on_settings_changed(self) -> None:
        self.compose.refresh_models()
        self.library.refresh()
        self.hardware_label.setText(
            f"{self.hardware_label.text().split('|')[0].strip()}  |  "
            f"{self.service.recommended_compute()}"
        )

    def _on_theme_toggled(self, light: bool) -> None:
        self.service.settings.theme = "light" if light else "dark"
        save_settings(self.service.settings)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(LIGHT if light else DARK))

    def _open_output(self) -> None:
        path = self.service.settings.resolved_output_dir()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _about(self) -> None:
        from .. import __version__

        gpu = self.service.system.primary_gpu
        QtWidgets.QMessageBox.about(
            self, f"About {APP_TITLE}",
            f"<b>{APP_TITLE}</b> {__version__}<br><br>"
            "Local music generation with open-weight models.<br>"
            "Exports MIDI and FLAC. Nothing leaves your machine.<br><br>"
            f"<b>Hardware</b><br>{self.service.system.cpu}<br>"
            f"{gpu.name if gpu else 'No discrete GPU'}<br>"
            f"PyTorch: {self.service.system.torch_version or 'not installed'}",
        )

    # -- geometry -----------------------------------------------------------

    def _restore_geometry(self) -> None:
        saved = self.service.settings.window_geometry
        if saved:
            try:
                self.restoreGeometry(QtCore.QByteArray.fromBase64(saved.encode("ascii")))
            except Exception:
                log.debug("could not restore window geometry")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.service.queue.pending:
            answer = QtWidgets.QMessageBox.question(
                self, "Still generating",
                f"{self.service.queue.pending} generation(s) are still running. "
                "Quit anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer is not QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        try:
            self.service.settings.window_geometry = bytes(
                self.saveGeometry().toBase64()
            ).decode("ascii")
            save_settings(self.service.settings)
        except Exception:
            log.debug("could not save window geometry")
        self.library.stop()
        self.service.shutdown()
        event.accept()
