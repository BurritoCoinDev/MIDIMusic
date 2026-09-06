"""The queue: what is generating now and what is waiting."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...core.jobs import Job, JobEvent
from ...core.models import JobStatus
from ...core.service import AppService

__all__ = ["QueuePanel"]

_STATUS_TEXT = {
    JobStatus.QUEUED: "Waiting",
    JobStatus.RUNNING: "Generating",
    JobStatus.DONE: "Done",
    JobStatus.FAILED: "Failed",
    JobStatus.CANCELLED: "Cancelled",
}


class JobRow(QtWidgets.QFrame):
    """One job, with progress and a cancel button."""

    cancelled = QtCore.Signal(str)
    opened = QtCore.Signal(str)

    def __init__(self, job: Job, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.job = job
        self.setProperty("role", "card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 12)
        layout.setSpacing(7)

        top = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel(job.display_name())
        self.title.setWordWrap(True)
        top.addWidget(self.title, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setProperty("role", "dim")
        top.addWidget(self.status)

        self.action = QtWidgets.QPushButton("Cancel")
        self.action.setProperty("role", "ghost")
        self.action.clicked.connect(self._on_action)
        top.addWidget(self.action)
        layout.addLayout(top)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)

        self.detail = QtWidgets.QLabel("")
        self.detail.setProperty("role", "dim")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.refresh()

    def _on_action(self) -> None:
        if self.job.status is JobStatus.DONE:
            self.opened.emit(self.job.id)
        else:
            self.cancelled.emit(self.job.id)

    def refresh(self) -> None:
        job = self.job
        self.status.setText(_STATUS_TEXT.get(job.status, ""))
        self.bar.setValue(job.progress.percent)
        self.bar.setVisible(job.status is JobStatus.RUNNING)

        if job.status is JobStatus.RUNNING:
            msg = job.progress.message or "Working"
            self.detail.setText(f"{msg} - {job.elapsed:.0f}s elapsed")
            self.detail.setProperty("role", "dim")
            self.action.setText("Cancel")
            self.action.setVisible(True)
        elif job.status is JobStatus.DONE:
            names = ", ".join(p.name for p in job.output_paths) or "no files written"
            self.detail.setText(f"{names}  ({job.elapsed:.1f}s)")
            self.detail.setProperty("role", "ok")
            self.action.setText("Show")
            self.action.setVisible(bool(job.output_paths))
        elif job.status is JobStatus.FAILED:
            self.detail.setText(job.error or "Generation failed")
            self.detail.setProperty("role", "error")
            self.action.setVisible(False)
        elif job.status is JobStatus.CANCELLED:
            self.detail.setText("Cancelled")
            self.detail.setProperty("role", "dim")
            self.action.setVisible(False)
        else:
            self.detail.setText("Queued")
            self.detail.setProperty("role", "dim")
            self.action.setText("Cancel")
            self.action.setVisible(True)
        # Re-polish so the role-based colour actually changes.
        self.detail.style().unpolish(self.detail)
        self.detail.style().polish(self.detail)


class QueuePanel(QtWidgets.QWidget):
    job_finished = QtCore.Signal(object)
    show_output = QtCore.Signal(str)

    def __init__(self, service: AppService, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self._rows: dict[str, JobRow] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Queue")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        clear = QtWidgets.QPushButton("Clear finished")
        clear.clicked.connect(self._clear_finished)
        header.addWidget(clear)
        cancel_all = QtWidgets.QPushButton("Cancel all")
        cancel_all.setProperty("role", "danger")
        cancel_all.clicked.connect(self.service.queue.cancel_all)
        header.addWidget(cancel_all)
        layout.addLayout(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._container = QtWidgets.QWidget()
        self._list = QtWidgets.QVBoxLayout(self._container)
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(10)
        self._list.addStretch(1)
        scroll.setWidget(self._container)
        layout.addWidget(scroll, 1)

        self.empty = QtWidgets.QLabel("Nothing queued. Generate something from Compose.")
        self.empty.setProperty("role", "dim")
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty)

        # Job events arrive on the worker thread, so hop to the UI thread.
        self._bridge = _EventBridge()
        self._bridge.event.connect(self._on_event)
        self.service.queue.subscribe(self._bridge.emit_event)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(400)
        self._sync()

    def _clear_finished(self) -> None:
        self.service.queue.clear_finished()
        self._sync()

    @QtCore.Slot(object)
    def _on_event(self, event: JobEvent) -> None:
        self._sync()
        if event.kind == "finished":
            self.job_finished.emit(event.job)

    def _tick(self) -> None:
        for row in self._rows.values():
            if row.job.status is JobStatus.RUNNING:
                row.refresh()

    def _sync(self) -> None:
        jobs = self.service.queue.jobs()
        seen = set()
        for job in jobs:
            seen.add(job.id)
            row = self._rows.get(job.id)
            if row is None:
                row = JobRow(job)
                row.cancelled.connect(self.service.queue.cancel)
                row.opened.connect(self.show_output)
                self._rows[job.id] = row
                self._list.insertWidget(0, row)
            row.refresh()
        for job_id in list(self._rows):
            if job_id not in seen:
                row = self._rows.pop(job_id)
                row.setParent(None)
                row.deleteLater()
        self.empty.setVisible(not jobs)


class _EventBridge(QtCore.QObject):
    """Marshals worker-thread job events onto the UI thread."""

    event = QtCore.Signal(object)

    def emit_event(self, evt: JobEvent) -> None:
        self.event.emit(evt)
