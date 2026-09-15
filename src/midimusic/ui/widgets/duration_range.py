"""A length band: the shortest and longest a generated track may be.

Two spinboxes rather than one, because asking for "three to five minutes" is
how people actually think about track length, and because it gives a batch of
variations a reason to differ from each other beyond the seed.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["DurationRange", "format_seconds"]


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}:{seconds % 60:02d}"


class DurationRange(QtWidgets.QWidget):
    """Minimum and maximum length, kept in order at all times."""

    changed = QtCore.Signal()

    def __init__(self, low: int, high: int, floor: int = 5, ceiling: int = 900,
                 parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._floor = floor

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.minimum = QtWidgets.QSpinBox()
        self.maximum = QtWidgets.QSpinBox()
        for box in (self.minimum, self.maximum):
            box.setRange(floor, ceiling)
            box.setSuffix(" s")
            box.setSingleStep(15)
            row.addWidget(box, 1)
            if box is self.minimum:
                separator = QtWidgets.QLabel("to")
                separator.setProperty("role", "dim")
                row.addWidget(separator)

        self.minimum.setValue(max(floor, min(ceiling, int(low))))
        self.maximum.setValue(max(self.minimum.value(), min(ceiling, int(high))))
        self.minimum.valueChanged.connect(self._on_minimum)
        self.maximum.valueChanged.connect(self._on_maximum)

    # -- ordering -----------------------------------------------------------

    def _on_minimum(self, value: int) -> None:
        # Push the other end rather than refusing the edit: silently clamping
        # what someone just typed reads as the control being broken.
        if value > self.maximum.value():
            self.maximum.blockSignals(True)
            self.maximum.setValue(value)
            self.maximum.blockSignals(False)
        self.changed.emit()

    def _on_maximum(self, value: int) -> None:
        if value < self.minimum.value():
            self.minimum.blockSignals(True)
            self.minimum.setValue(value)
            self.minimum.blockSignals(False)
        self.changed.emit()

    # -- values -------------------------------------------------------------

    def values(self) -> tuple[int, int]:
        low, high = self.minimum.value(), self.maximum.value()
        return (low, high) if low <= high else (high, low)

    def set_values(self, low: int, high: int) -> None:
        for box, value in ((self.minimum, low), (self.maximum, high)):
            box.blockSignals(True)
            box.setValue(int(value))
            box.blockSignals(False)
        self.changed.emit()

    def set_ceiling(self, seconds: int) -> None:
        """Cap both ends, for a backend that cannot go beyond some length."""
        seconds = max(self._floor, int(seconds))
        for box in (self.minimum, self.maximum):
            box.blockSignals(True)
            box.setMaximum(seconds)
            box.blockSignals(False)

    def is_fixed(self) -> bool:
        low, high = self.values()
        return high - low < 1

    def describe(self) -> str:
        low, high = self.values()
        if self.is_fixed():
            return f"Every track {format_seconds(low)} long."
        return (
            f"Each track lands somewhere between {format_seconds(low)} and "
            f"{format_seconds(high)}; variations are spread across the range."
        )
