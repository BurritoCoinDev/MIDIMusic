"""Waveform display with a seek head."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ...audio.dsp import peak_envelope

__all__ = ["WaveformWidget"]


class WaveformWidget(QtWidgets.QWidget):
    """Draws a min/max envelope and lets the user click to seek."""

    seeked = QtCore.Signal(float)  # position in seconds

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._envelope: np.ndarray | None = None
        self._position = 0.0
        self._duration = 0.0
        self._accent = QtGui.QColor("#7C5CFF")
        self._wave = QtGui.QColor("#4A4D63")
        self._bg = QtGui.QColor("#1A1C25")
        self.setMinimumHeight(72)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def set_colors(self, accent: str, wave: str, background: str) -> None:
        self._accent = QtGui.QColor(accent)
        self._wave = QtGui.QColor(wave)
        self._bg = QtGui.QColor(background)
        self.update()

    def set_audio(self, samples: np.ndarray | None, duration: float = 0.0) -> None:
        if samples is None or getattr(samples, "size", 0) == 0:
            self._envelope = None
            self._duration = 0.0
        else:
            self._envelope = peak_envelope(np.asarray(samples), buckets=1400)
            self._duration = duration or 0.0
        self._position = 0.0
        self.update()

    def clear(self) -> None:
        self.set_audio(None)

    def set_position(self, seconds: float, duration: float | None = None) -> None:
        if duration:
            self._duration = duration
        self._position = seconds
        self.update()

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()

        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), 8, 8)
        painter.fillPath(path, self._bg)
        painter.setClipPath(path)

        if self._envelope is None or self._envelope.size == 0:
            painter.setPen(QtGui.QColor("#5F6478"))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No audio loaded")
            return

        n = self._envelope.shape[0]
        mid = rect.height() / 2.0
        scale = (rect.height() / 2.0) * 0.92
        width = rect.width()
        played_x = (self._position / self._duration * width) if self._duration else 0.0

        # One vertical line per pixel column, coloured by play position.
        for x in range(width):
            idx = int(x / max(1, width) * n)
            lo, hi = self._envelope[min(idx, n - 1)]
            y1 = mid - hi * scale
            y2 = mid - lo * scale
            if y2 - y1 < 1.0:
                y1, y2 = mid - 0.5, mid + 0.5
            painter.setPen(self._accent if x <= played_x else self._wave)
            painter.drawLine(QtCore.QPointF(x, y1), QtCore.QPointF(x, y2))

        if self._duration:
            painter.setPen(QtGui.QPen(QtGui.QColor("#E8E9F0"), 1.5))
            painter.drawLine(QtCore.QPointF(played_x, 0), QtCore.QPointF(played_x, rect.height()))

    # -- interaction --------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._seek_to(event.position().x())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def _seek_to(self, x: float) -> None:
        if not self._duration or self.width() <= 0:
            return
        fraction = max(0.0, min(1.0, x / self.width()))
        self._position = fraction * self._duration
        self.update()
        self.seeked.emit(self._position)
