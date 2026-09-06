"""The generator interface every backend implements.

One interface covers both symbolic and audio backends.  A generator declares
what it can produce; the app converts between MIDI and audio around it, so a
MIDI-only model can still yield FLAC (rendered through a SoundFont) and an
audio-only model can still yield MIDI (through transcription).
"""

from __future__ import annotations

import abc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import GenerationRequest, GenerationResult, OutputFormat, Progress

__all__ = [
    "BackendUnavailable",
    "Capabilities",
    "GenerationCancelled",
    "Generator",
    "GeneratorContext",
]


class GenerationCancelled(Exception):
    """Raised inside a generator when the user cancels."""


class BackendUnavailable(Exception):
    """Raised when a backend's dependencies or weights are not present.

    Carries what is missing so the UI can offer to install or download it
    rather than just reporting a failure.
    """

    def __init__(self, message: str, missing_packages: list[str] | None = None,
                 missing_model: str | None = None):
        super().__init__(message)
        self.missing_packages = missing_packages or []
        self.missing_model = missing_model


@dataclass
class Capabilities:
    """What a backend can do, so the UI can enable the right controls."""

    outputs: tuple[str, ...] = ("midi",)
    max_duration: float = 300.0
    supports_lyrics: bool = False
    supports_vocals: bool = False
    supports_seed: bool = True
    supports_continuation: bool = False
    supports_negative_prompt: bool = False
    needs_gpu: bool = False
    sample_rate: int = 44100
    honours_key: bool = False      # can the request's key/tempo be enforced?
    honours_tempo: bool = False
    honours_structure: bool = False


@dataclass
class GeneratorContext:
    """Everything a generator needs from its environment."""

    device: str = "cpu"
    models_dir: Path | None = None
    soundfont: Path | None = None
    hf_token: str = ""
    offline: bool = False
    progress: Callable[[Progress], None] | None = None
    cancelled: Callable[[], bool] | None = None
    extra: dict = field(default_factory=dict)

    def report(self, fraction: float, message: str = "", stage: str = "") -> None:
        if self.progress is not None:
            self.progress(Progress(fraction, message, stage))

    def check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise GenerationCancelled()

    def is_cancelled(self) -> bool:
        return bool(self.cancelled and self.cancelled())


class Generator(abc.ABC):
    """Base class for every generation backend."""

    id: str = "base"
    name: str = "Generator"
    kind: str = "symbolic"  # symbolic | audio

    def __init__(self, model_id: str = "", entry=None):
        self.model_id = model_id
        self.entry = entry

    # -- description --------------------------------------------------------

    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """What this backend supports."""

    def required_packages(self) -> list[str]:
        """Pip requirements that must be importable before generating."""
        return []

    def missing_packages(self) -> list[str]:
        import importlib.util

        aliases = {"torch": "torch", "transformers": "transformers",
                   "diffusers": "diffusers", "onnxruntime": "onnxruntime"}
        missing = []
        for pkg in self.required_packages():
            mod = aliases.get(pkg, pkg).split("[")[0].replace("-", "_")
            if importlib.util.find_spec(mod) is None:
                missing.append(pkg)
        return missing

    def is_ready(self, ctx: GeneratorContext | None = None) -> bool:
        return not self.missing_packages()

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        """Rough wall-clock estimate, used to set expectations in the UI."""
        return max(2.0, (request.duration_seconds or 60.0) * 0.1)

    # -- lifecycle ----------------------------------------------------------

    def load(self, ctx: GeneratorContext) -> None:
        """Load weights. Called before the first generate; may be slow."""

    def unload(self) -> None:
        """Release weights and VRAM."""

    # -- work ---------------------------------------------------------------

    @abc.abstractmethod
    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        """Produce one result. Must honour ctx.check_cancelled()."""

    # -- helpers ------------------------------------------------------------

    def _result(self, request: GenerationRequest, started: float, **kw) -> GenerationResult:
        result = GenerationResult(
            request_id=request.id, backend=self.id, seed=request.seed, **kw
        )
        result.elapsed_seconds = time.time() - started
        if result.song is not None and not result.duration_seconds:
            result.duration_seconds = result.song.duration_seconds
        elif result.audio is not None and not result.duration_seconds:
            result.duration_seconds = result.audio.duration_seconds
        return result

    def supports_output(self, fmt: OutputFormat) -> bool:
        return fmt.value in self.capabilities().outputs
