"""Audio backends that run in the provisioned runtime, not in this process.

The application bundle deliberately contains no torch, so these adapters do not
import it either.  They describe the job, hand it to the worker, and read the
audio back from the file it writes.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..core.generator import (
    BackendUnavailable,
    Capabilities,
    GenerationCancelled,
    Generator,
    GeneratorContext,
)
from ..core.models import AudioBuffer, GenerationRequest, GenerationResult
from ..core.worker_client import (
    WorkerCancelled,
    WorkerError,
    probe_runtime,
    run_worker,
    runtime_python,
)

__all__ = ["RemoteAudioGenerator"]

# Which worker command serves which catalog adapter.
_COMMANDS = {
    "hf-musicgen": "hf-musicgen",
    "diffusers-audio": "diffusers-audio",
    "ace-step": "ace-step",
    "diffrhythm": "diffusers-audio",
}

# What each backend needs present in the runtime before it can work.
_REQUIREMENTS = {
    "hf-musicgen": ("torch", "transformers"),
    "diffusers-audio": ("torch", "diffusers"),
    "ace-step": ("torch", "acestep"),
}


class RemoteAudioGenerator(Generator):
    """Runs an audio model out of process."""

    id = "remote-audio"
    kind = "audio"

    def __init__(self, model_id: str = "", entry=None):
        super().__init__(model_id, entry)
        self.name = getattr(entry, "name", "Remote model")
        self._probe: dict | None = None

    # -- description --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        entry = self.entry
        return Capabilities(
            outputs=("flac", "wav"),
            max_duration=float(getattr(entry, "max_duration", 60.0)),
            supports_lyrics=bool(getattr(entry, "lyrics", False)),
            supports_vocals=bool(getattr(entry, "vocals", False)),
            supports_seed=True,
            supports_negative_prompt=True,
            needs_gpu=False,
            sample_rate=int(getattr(entry, "sample_rate", 44100)),
        )

    def _command(self) -> str:
        adapter = getattr(self.entry, "adapter", "")
        command = _COMMANDS.get(adapter)
        if command is None:
            raise BackendUnavailable(f"No worker command for adapter {adapter!r}.")
        return command

    def probe(self, refresh: bool = False) -> dict:
        if self._probe is None or refresh:
            self._probe = probe_runtime()
        return self._probe

    def missing_packages(self) -> list[str]:
        if runtime_python() is None:
            return ["compute runtime"]
        info = self.probe()
        if not info:
            return ["compute runtime"]
        try:
            needed = _REQUIREMENTS.get(self._command(), ("torch",))
        except BackendUnavailable:
            return ["adapter"]
        return [pkg for pkg in needed if not info.get(pkg, pkg == "torch" and "torch" in info)]

    def is_ready(self, ctx: GeneratorContext | None = None) -> bool:
        return not self.missing_packages()

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        seconds = min(request.duration_seconds or 30.0, self.capabilities().max_duration)
        info = self.probe()
        on_gpu = bool(info.get("cuda"))
        adapter = getattr(self.entry, "adapter", "")
        if adapter == "ace-step":
            return max(20.0, seconds * (0.35 if on_gpu else 5.0))
        if adapter == "hf-musicgen":
            return max(5.0, seconds * (1.5 if on_gpu else 25.0))
        steps = int(request.extra.get("steps", 100))
        return max(10.0, steps * (0.2 if on_gpu else 3.0))

    # -- work ---------------------------------------------------------------

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        entry = self.entry
        caps = self.capabilities()

        with tempfile.TemporaryDirectory(prefix="midimusic-") as tmp:
            output = Path(tmp) / "generated.wav"
            payload = {
                "cmd": self._command(),
                "repo": getattr(entry, "repo", ""),
                "output_path": str(output),
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "lyrics": request.lyrics,
                "instrumental": request.instrumental,
                "duration": min(request.duration_seconds or 30.0, caps.max_duration),
                "max_duration": caps.max_duration,
                "seed": request.seed,
                "temperature": request.temperature,
                "guidance": request.guidance,
                "steps": request.extra.get("steps"),
                "sample_rate": caps.sample_rate,
                "device": ctx.device,
                "token": ctx.hf_token,
                "offline": ctx.offline,
                "models_dir": str(ctx.models_dir) if ctx.models_dir else "",
                "env": dict(getattr(entry, "env", None) or {}),
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            try:
                result = run_worker(
                    payload,
                    on_progress=lambda f, m, s: ctx.report(f, m, s),
                    should_cancel=ctx.cancelled,
                )
            except WorkerCancelled as exc:
                raise GenerationCancelled() from exc
            except WorkerError as exc:
                if exc.remote_traceback:
                    import logging

                    logging.getLogger(__name__).error(
                        "worker traceback:\n%s", exc.remote_traceback
                    )
                raise

            if result.output_path is None or not result.output_path.exists():
                raise WorkerError("The worker reported success but wrote no audio.")

            samples, rate = sf.read(str(result.output_path), dtype="float32",
                                    always_2d=True)
            buffer = AudioBuffer(np.ascontiguousarray(samples), int(rate))

        ctx.report(1.0, "Done", "generate")
        return self._result(
            request, started,
            audio=buffer,
            title=(request.prompt or self.name)[:60] or self.name,
            duration_seconds=buffer.duration_seconds,
            meta={"model": getattr(entry, "repo", ""), **result.meta, "isolated": True},
        )
