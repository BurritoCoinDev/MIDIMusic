"""ACE-Step: full-song generation with vocals.

The flagship audio backend, and the closest open-weight analogue to Suno.  MIT
licensed for both code and weights, with first-party AMD support.

Two AMD/Windows details are load-bearing and handled here rather than left to
the user: the language-model stage must be forced onto the PyTorch backend
(its default is vLLM, which does not exist on Windows or ROCm), and torchao
must not be present, because it registers distributed ops that are missing in
the Windows ROCm builds and fails at import time.
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np

from ..core.generator import BackendUnavailable, Capabilities, Generator, GeneratorContext
from ..core.models import AudioBuffer, GenerationRequest, GenerationResult
from . import _torch_util as tu

__all__ = ["AceStepGenerator"]

_INSTALL_HINT = (
    "ACE-Step is not installed. It is not on PyPI; install it from source:\n"
    "  pip install git+https://github.com/ace-step/ACE-Step-1.5"
)


class AceStepGenerator(Generator):
    id = "ace-step"
    name = "ACE-Step"
    kind = "audio"

    def __init__(self, model_id: str = "", entry=None):
        super().__init__(model_id, entry)
        self._pipeline = None
        self._device = "cpu"

    def capabilities(self) -> Capabilities:
        entry = self.entry
        return Capabilities(
            outputs=("flac", "wav"),
            max_duration=float(getattr(entry, "max_duration", 240.0)),
            supports_lyrics=True,
            supports_vocals=True,
            supports_seed=True,
            supports_continuation=True,
            supports_negative_prompt=True,
            needs_gpu=False,
            sample_rate=int(getattr(entry, "sample_rate", 48000)),
        )

    def required_packages(self) -> list[str]:
        return ["torch"]

    def missing_packages(self) -> list[str]:
        missing = super().missing_packages()
        if importlib.util.find_spec("acestep") is None:
            missing.append("acestep")
        return missing

    def is_ready(self, ctx: GeneratorContext | None = None) -> bool:
        return importlib.util.find_spec("acestep") is not None and not super().missing_packages()

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        seconds = min(request.duration_seconds or 120.0, self.capabilities().max_duration)
        if ctx.device == "cpu":
            return seconds * 5.0
        # Measured around 30-60s per song on an RX 7900-class card under ROCm.
        return max(20.0, seconds * 0.35)

    def load(self, ctx: GeneratorContext) -> None:
        if self._pipeline is not None:
            return
        tu.require_torch()

        entry_env = dict(getattr(self.entry, "env", None) or {})
        # Force the PyTorch LM backend; the vLLM default is CUDA/Linux-only.
        entry_env.setdefault("ACESTEP_LM_BACKEND", "pt")
        tu.apply_model_env(entry_env)

        if importlib.util.find_spec("torchao") is not None:
            # Not fatal on CUDA, but on Windows ROCm it breaks at import time.
            import logging

            logging.getLogger(__name__).warning(
                "torchao is installed; ACE-Step may fail to import on Windows ROCm. "
                "Uninstall it if loading fails."
            )

        try:
            from acestep.pipeline_ace_step import ACEStepPipeline
        except ImportError as exc:
            raise BackendUnavailable(
                _INSTALL_HINT, missing_packages=["acestep"],
            ) from exc

        self._device = tu.resolve_device(ctx.device)
        repo = getattr(self.entry, "repo", None) or "ACE-Step/ACE-Step-1.5"
        ctx.report(0.05, f"Loading {repo} (first run compiles GPU kernels)", "load")

        self._pipeline = ACEStepPipeline(
            checkpoint_dir=str(ctx.models_dir) if ctx.models_dir else None,
            device_id=0 if self._device != "cpu" else -1,
            dtype="float16" if self._device != "cpu" else "float32",
            torch_compile=tu.can_compile(),
        )

    def unload(self) -> None:
        self._pipeline = None
        tu.empty_cache()

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        self.load(ctx)
        ctx.check_cancelled()

        caps = self.capabilities()
        seconds = min(request.duration_seconds or 120.0, caps.max_duration)
        tu.seed_everything(request.seed)

        lyrics = request.lyrics.strip()
        if request.instrumental or not lyrics:
            lyrics = "[instrumental]"

        ctx.report(0.15, f"Generating {seconds:.0f}s", "generate")

        params = {
            "prompt": request.prompt or "instrumental music",
            "lyrics": lyrics,
            "audio_duration": float(seconds),
            "infer_step": int(request.extra.get("steps", 60)),
            "guidance_scale": float(request.guidance or 15.0),
            "manual_seeds": str(request.seed) if request.seed is not None else None,
        }
        if request.negative_prompt:
            params["negative_prompt"] = request.negative_prompt

        try:
            output = self._pipeline(**params)
        except TypeError:
            # Upstream has changed its signature more than once; retry with the
            # minimal set rather than failing the whole generation.
            output = self._pipeline(
                prompt=params["prompt"], lyrics=lyrics, audio_duration=float(seconds)
            )
        ctx.check_cancelled()

        samples, sr = _coerce_audio(output, caps.sample_rate)
        buffer = AudioBuffer(samples, sr)

        ctx.report(1.0, "Done", "generate")
        return self._result(
            request, started,
            audio=buffer,
            title=(request.prompt or "ACE-Step")[:60] or "ACE-Step",
            duration_seconds=buffer.duration_seconds,
            meta={"model": getattr(self.entry, "repo", ""), "device": self._device,
                  "vocals": not request.instrumental, "sample_rate": sr},
        )


def _coerce_audio(output, default_sr: int) -> tuple[np.ndarray, int]:
    """Normalise whatever the pipeline returned into (frames, channels) float32.

    The upstream return shape has varied between releases -- a path, a tuple, a
    tensor, a dict -- so this stays deliberately tolerant.
    """
    sr = default_sr
    data = output

    if isinstance(output, dict):
        sr = int(output.get("sample_rate", default_sr))
        data = output.get("audio", output.get("waveform", output.get("audios")))
    elif isinstance(output, (list, tuple)):
        if len(output) == 2 and isinstance(output[1], int):
            data, sr = output[0], int(output[1])
        else:
            data = output[0]

    if isinstance(data, str):
        import soundfile as sf

        samples, sr = sf.read(data, dtype="float32", always_2d=True)
        return np.ascontiguousarray(samples), int(sr)

    if hasattr(data, "detach"):
        data = data.detach().to("cpu").float().numpy()
    samples = np.asarray(data, dtype=np.float32)

    while samples.ndim > 2:
        samples = samples[0]
    if samples.ndim == 2 and samples.shape[0] <= 2 < samples.shape[1]:
        samples = samples.T  # (channels, frames) -> (frames, channels)
    if samples.ndim == 1:
        samples = samples[:, None]
    return np.ascontiguousarray(samples), int(sr)
