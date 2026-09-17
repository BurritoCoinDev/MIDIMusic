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
        # What the user asked for, as opposed to what the current backend
        # permits. Kept separately so that picking a short-form model and then
        # changing your mind gives the range back instead of leaving it
        # collapsed to that model's maximum.
        self._wanted = (self.minimum.value(), self.maximum.value())
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
        self._wanted = self.values()
        self.changed.emit()

    def _on_maximum(self, value: int) -> None:
        if value < self.minimum.value():
            self.minimum.blockSignals(True)
            self.minimum.setValue(value)
            self.minimum.blockSignals(False)
        self._wanted = self.values()
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
        self._wanted = self.values()
        self.changed.emit()

    def set_ceiling(self, seconds: int) -> None:
        """Cap both ends, for a backend that cannot go beyond some length.

        Qt clamps a spinbox's value silently when its maximum drops, and with
        signals blocked nothing downstream hears about it -- so the hint under
        the control went on describing a range the widget no longer held. The
        clamp is applied deliberately here, against the range the user asked
        for rather than against whatever the last backend left behind, and
        anything that actually moved is announced.
        """
        seconds = max(self._floor, int(seconds))
        before = self.values()
        wanted = self._wanted
        for box, value in ((self.minimum, wanted[0]), (self.maximum, wanted[1])):
            box.blockSignals(True)
            box.setMaximum(seconds)
            box.setValue(min(int(value), seconds))
            box.blockSignals(False)
        if self.values() != before:
            self.changed.emit()

    def is_fixed(self) -> bool:
        low, high = self.values()
        return high - low < 1

    def is_capped(self) -> bool:
        """Whether a backend's ceiling is currently holding the range down."""
        return self.values() != self._wanted

    def describe(self) -> str:
        low, high = self.values()
        if self.is_fixed():
            text = f"Every track {format_seconds(low)} long."
        else:
            text = (
                f"Each track lands somewhere between {format_seconds(low)} and "
                f"{format_seconds(high)}; variations are spread across the range."
            )
        if self.is_capped():
            # Say why, rather than letting the number quietly disagree with
            # what was typed.
            text += (
                f" Capped at {format_seconds(high)} by this model; "
                f"your {format_seconds(self._wanted[1])} comes back with a "
                "model that can manage it."
            )
        return text
