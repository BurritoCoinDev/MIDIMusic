"""The library: everything generated, with playback and export."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ...core.service import AppService, LibraryItem
from ..player import AudioPlayer, playback_available
from .waveform import WaveformWidget

__all__ = ["LibraryPanel", "reveal_in_explorer"]


def reveal_in_explorer(path: Path) -> None:
    """Open the file manager with ``path`` selected."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError:
        pass


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class LibraryPanel(QtWidgets.QWidget):
    def __init__(self, service: AppService, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self.player = AudioPlayer(on_finished=self._on_playback_finished)
        self._current: LibraryItem | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Library")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.setMaximumWidth(240)
        self.search.textChanged.connect(self.refresh)
        header.addWidget(self.search)
        open_folder = QtWidgets.QPushButton("Open folder")
        open_folder.clicked.connect(self._open_output_folder)
        header.addWidget(open_folder)
        layout.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.list = QtWidgets.QListWidget()
        self.list.currentItemChanged.connect(self._on_selected)
        self.list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        splitter.addWidget(self.list)
        splitter.addWidget(self._detail_pane())
        splitter.setSizes([380, 620])
        layout.addWidget(splitter, 1)

        self._position_timer = QtCore.QTimer(self)
        self._position_timer.timeout.connect(self._update_position)
        self._position_timer.start(120)
        self.refresh()

    def _detail_pane(self) -> QtWidgets.QWidget:
        pane = QtWidgets.QFrame()
        pane.setProperty("role", "card")
        box = QtWidgets.QVBoxLayout(pane)
        box.setContentsMargins(16, 16, 16, 16)
        box.setSpacing(12)

        self.detail_title = QtWidgets.QLabel("Nothing selected")
        self.detail_title.setProperty("role", "subtitle")
        self.detail_title.setWordWrap(True)
        box.addWidget(self.detail_title)

        self.detail_meta = QtWidgets.QLabel("")
        self.detail_meta.setProperty("role", "dim")
        self.detail_meta.setWordWrap(True)
        box.addWidget(self.detail_meta)

        self.waveform = WaveformWidget()
        self.waveform.seeked.connect(self.player.seek)
        box.addWidget(self.waveform)

        transport = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setProperty("role", "primary")
        self.play_button.clicked.connect(self._toggle_play)
        transport.addWidget(self.play_button)

        self.time_label = QtWidgets.QLabel("0:00 / 0:00")
        self.time_label.setProperty("role", "mono")
        transport.addWidget(self.time_label)
        transport.addStretch(1)

        self.volume = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(90)
        self.volume.setMaximumWidth(120)
        self.volume.valueChanged.connect(lambda v: self.player.set_volume(v / 100.0))
        transport.addWidget(QtWidgets.QLabel("Volume"))
        transport.addWidget(self.volume)
        box.addLayout(transport)

        if not playback_available():
            note = QtWidgets.QLabel("Playback is unavailable on this system; files still export.")
            note.setProperty("role", "warn")
            note.setWordWrap(True)
            box.addWidget(note)

        self.prompt_view = QtWidgets.QPlainTextEdit()
        self.prompt_view.setReadOnly(True)
        self.prompt_view.setMaximumHeight(80)
        box.addWidget(self.prompt_view)

        actions = QtWidgets.QHBoxLayout()
        for label, slot in (
            ("Show in folder", self._reveal),
            ("Copy path", self._copy_path),
            ("Delete", self._delete),
        ):
            button = QtWidgets.QPushButton(label)
            if label == "Delete":
                button.setProperty("role", "danger")
            button.clicked.connect(slot)
            actions.addWidget(button)
        actions.addStretch(1)
        box.addLayout(actions)
        box.addStretch(1)
        return pane

    # -- data ---------------------------------------------------------------

    def refresh(self) -> None:
        query = self.search.text().strip().lower()
        self.list.clear()
        for item in self.service.library:
            haystack = f"{item.title} {item.prompt} {item.style} {item.backend}".lower()
            if query and query not in haystack:
                continue
            entry = QtWidgets.QListWidgetItem()
            bits = [b for b in (item.style, item.key,
                                f"{item.tempo:.0f} bpm" if item.tempo else "",
                                _fmt_time(item.duration)) if b]
            entry.setText(f"{item.title}\n{'  '.join(bits)}")
            entry.setData(QtCore.Qt.ItemDataRole.UserRole, item.id)
            if not item.exists():
                entry.setForeground(QtGui.QColor("#9295A8"))
                entry.setToolTip("Files are missing from disk")
            self.list.addItem(entry)

    def _selected_item(self) -> LibraryItem | None:
        current = self.list.currentItem()
        if current is None:
            return None
        item_id = current.data(QtCore.Qt.ItemDataRole.UserRole)
        for item in self.service.library:
            if item.id == item_id:
                return item
        return None

    def _on_selected(self) -> None:
        item = self._selected_item()
        self._current = item
        if item is None:
            self.detail_title.setText("Nothing selected")
            self.detail_meta.setText("")
            self.prompt_view.setPlainText("")
            self.waveform.clear()
            return

        self.detail_title.setText(item.title)
        facts = [f for f in (item.backend, item.style, item.key,
                             f"{item.tempo:.0f} bpm" if item.tempo else "",
                             _fmt_time(item.duration),
                             f"seed {item.seed}" if item.seed is not None else "") if f]
        self.detail_meta.setText("  |  ".join(facts))
        self.prompt_view.setPlainText(item.prompt or "(no prompt)")

        self.player.stop()
        audio = item.audio_path
        if audio and audio.exists() and self.player.load_file(audio):
            self.waveform.set_audio(self.player._samples, self.player.duration)
            self.play_button.setEnabled(True)
        else:
            self.waveform.clear()
            # A MIDI-only result has nothing to preview without rendering it.
            self.play_button.setEnabled(False)
        self._update_position()

    # -- transport ----------------------------------------------------------

    def _toggle_play(self) -> None:
        playing = self.player.toggle()
        self.play_button.setText("Pause" if playing else "Play")

    def _on_playback_finished(self) -> None:
        self.play_button.setText("Play")

    def _update_position(self) -> None:
        if not self.player.loaded:
            self.time_label.setText("0:00 / 0:00")
            return
        self.waveform.set_position(self.player.position, self.player.duration)
        self.time_label.setText(
            f"{_fmt_time(self.player.position)} / {_fmt_time(self.player.duration)}"
        )

    # -- actions ------------------------------------------------------------

    def _open_output_folder(self) -> None:
        path = self.service.settings.resolved_output_dir()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _reveal(self) -> None:
        item = self._current
        if item and item.paths:
            reveal_in_explorer(Path(item.paths[0]))

    def _copy_path(self) -> None:
        item = self._current
        if item and item.paths:
            QtWidgets.QApplication.clipboard().setText(item.paths[0])

    def _delete(self) -> None:
        item = self._current
        if item is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete", f"Delete '{item.title}' and its files from disk?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer is QtWidgets.QMessageBox.StandardButton.Yes:
            self.player.stop()
            self.service.remove_from_library(item.id, delete_files=True)
            self.refresh()

    def _context_menu(self, point: QtCore.QPoint) -> None:
        item = self._selected_item()
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Show in folder", self._reveal)
        menu.addAction("Copy path", self._copy_path)
        menu.addSeparator()
        menu.addAction("Delete", self._delete)
        menu.exec(self.list.mapToGlobal(point))

    def stop(self) -> None:
        self.player.stop()
