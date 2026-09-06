"""The generation queue.

Generation is slow (seconds for the built-in composer, minutes for a neural
backend), so it never runs on the UI thread.  Jobs run one at a time on a
worker thread: models are large and two concurrent generations would fight
over VRAM, so serialising is a feature rather than a limitation.

The queue is deliberately Qt-free so it can be tested headlessly; the UI layer
subscribes with plain callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

from .generator import GenerationCancelled, Generator, GeneratorContext
from .models import GenerationRequest, GenerationResult, JobStatus, Progress

__all__ = ["Job", "JobQueue", "JobEvent"]

log = logging.getLogger(__name__)


@dataclass
class Job:
    request: GenerationRequest
    generator: Generator
    context: GeneratorContext
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.QUEUED
    progress: Progress = field(default_factory=Progress)
    result: GenerationResult | None = None
    error: str = ""
    label: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    output_paths: list[Path] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

    def display_name(self) -> str:
        if self.label:
            return self.label
        text = (self.request.prompt or "").strip()
        return (text[:48] + "...") if len(text) > 48 else (text or "Untitled")


@dataclass
class JobEvent:
    kind: str  # queued | started | progress | finished | failed | cancelled
    job: Job


class JobQueue:
    """A single-worker queue with progress reporting and cancellation."""

    def __init__(self, post_process: Callable[[Job], list[Path]] | None = None):
        self._queue: Queue[Job | None] = Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._listeners: list[Callable[[JobEvent], None]] = []
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._current: Job | None = None
        self._post_process = post_process

    # -- subscription -------------------------------------------------------

    def subscribe(self, listener: Callable[[JobEvent], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[JobEvent], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _emit(self, kind: str, job: Job) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(JobEvent(kind, job))
            except Exception:
                # A broken listener must never take down the worker thread.
                log.exception("job listener failed")

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="midimusic-jobs", daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        self._running.clear()
        if self._current is not None:
            self._current.cancel()
        self._queue.put(None)
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)

    # -- submission ---------------------------------------------------------

    def submit(self, request: GenerationRequest, generator: Generator,
               context: GeneratorContext, label: str = "") -> Job:
        job = Job(request=request, generator=generator, context=context, label=label)
        # Wire cancellation and progress through to the generator.
        context.cancelled = job._cancel.is_set
        context.progress = lambda p, j=job: self._on_progress(j, p)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._queue.put(job)
        self._emit("queued", job)
        self.start()
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.is_terminal:
            return False
        job.cancel()
        if job.status is JobStatus.QUEUED:
            # Not started yet, so finish it here; the worker will skip it.
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
            self._emit("cancelled", job)
        return True

    def cancel_all(self) -> None:
        for job in self.jobs():
            if not job.is_terminal:
                self.cancel(job.id)

    def clear_finished(self) -> None:
        with self._lock:
            keep = [jid for jid in self._order if not self._jobs[jid].is_terminal]
            self._jobs = {jid: self._jobs[jid] for jid in keep}
            self._order = keep

    # -- inspection ---------------------------------------------------------

    def jobs(self) -> list[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in self._order]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    @property
    def pending(self) -> int:
        return sum(1 for j in self.jobs() if not j.is_terminal)

    @property
    def current(self) -> Job | None:
        return self._current

    # -- worker -------------------------------------------------------------

    def _on_progress(self, job: Job, progress: Progress) -> None:
        job.progress = progress
        self._emit("progress", job)

    def _run(self) -> None:
        while self._running.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except Empty:
                continue
            if job is None:
                break
            if job.cancelled:
                if job.status is not JobStatus.CANCELLED:
                    job.status = JobStatus.CANCELLED
                    job.finished_at = time.time()
                    self._emit("cancelled", job)
                continue
            self._execute(job)
        self._running.clear()

    def _execute(self, job: Job) -> None:
        self._current = job
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._emit("started", job)
        try:
            result = job.generator.generate(job.request, job.context)
            if job.cancelled:
                raise GenerationCancelled()
            job.result = result
            if self._post_process is not None:
                job.output_paths = self._post_process(job) or []
                result.paths = list(job.output_paths)
            job.status = JobStatus.DONE
            job.finished_at = time.time()
            self._emit("finished", job)
        except GenerationCancelled:
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
            self._emit("cancelled", job)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = JobStatus.FAILED
            job.finished_at = time.time()
            log.error("job %s failed: %s\n%s", job.id, job.error, traceback.format_exc())
            self._emit("failed", job)
        finally:
            self._current = None
