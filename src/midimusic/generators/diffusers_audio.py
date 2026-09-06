"""Stable Audio Open through diffusers.

Good for loops, textures and sound effects rather than full songs.  The weights
are gated on Hugging Face and carry the Stability Community License, so the
catalog warns before a download and the adapter fails with a clear message when
no token is configured.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.generator import BackendUnavailable, Capabilities, Generator, GeneratorContext
from ..core.models import AudioBuffer, GenerationRequest, GenerationResult
from . import _torch_util as tu

__all__ = ["DiffusersAudioGenerator"]


class DiffusersAudioGenerator(Generator):
    id = "stable-audio"
    name = "Stable Audio Open"
    kind = "audio"

    def __init__(self, model_id: str = "", entry=None):
        super().__init__(model_id, entry)
        self._pipe = None
        self._device = "cpu"

    def capabilities(self) -> Capabilities:
        entry = self.entry
        return Capabilities(
            outputs=("flac", "wav"),
            max_duration=float(getattr(entry, "max_duration", 47.0)),
            supports_seed=True,
            supports_negative_prompt=True,
            needs_gpu=False,
            sample_rate=int(getattr(entry, "sample_rate", 44100)),
        )

    def required_packages(self) -> list[str]:
        return ["torch", "diffusers", "transformers"]

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        steps = int(request.extra.get("steps", 100))
        return max(10.0, steps * (0.2 if ctx.device != "cpu" else 3.0))

    def load(self, ctx: GeneratorContext) -> None:
        if self._pipe is not None:
            return
        torch = tu.require_torch()
        try:
            from diffusers import StableAudioPipeline
        except ImportError as exc:
            raise BackendUnavailable(
                "diffusers is required for Stable Audio.", missing_packages=["diffusers"]
            ) from exc

        if getattr(self.entry, "gated", False) and not ctx.hf_token:
            raise BackendUnavailable(
                "Stable Audio Open is gated. Accept its licence on Hugging Face and "
                "add an access token in Settings.",
                missing_model=getattr(self.entry, "repo", ""),
            )

        repo = getattr(self.entry, "repo", None) or "stabilityai/stable-audio-open-1.0"
        self._device = tu.resolve_device(ctx.device)
        ctx.report(0.05, f"Loading {repo}", "load")
        self._pipe = StableAudioPipeline.from_pretrained(
            repo, torch_dtype=tu.torch_dtype_for(self._device),
            token=ctx.hf_token or None, local_files_only=ctx.offline,
        ).to(self._device)
        _ = torch

    def unload(self) -> None:
        self._pipe = None
        tu.empty_cache()

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        torch = tu.require_torch()
        self.load(ctx)
        ctx.check_cancelled()

        caps = self.capabilities()
        seconds = min(request.duration_seconds or 30.0, caps.max_duration)
        steps = int(request.extra.get("steps", 100))

        generator = torch.Generator(self._device)
        if request.seed is not None:
            generator.manual_seed(int(request.seed))

        def on_step(_pipe, step: int, _timestep, kwargs):
            ctx.report(0.15 + 0.8 * (step / max(1, steps)), "Denoising", "generate")
            if ctx.is_cancelled():
                raise KeyboardInterrupt
            return kwargs

        ctx.report(0.15, f"Generating {seconds:.0f}s", "generate")
        output = self._pipe(
            prompt=request.prompt or "ambient texture",
            negative_prompt=request.negative_prompt or None,
            num_inference_steps=steps,
            audio_end_in_s=float(seconds),
            num_waveforms_per_prompt=1,
            generator=generator,
            callback_on_step_end=on_step,
        )
        ctx.check_cancelled()

        samples = output.audios[0].to(torch.float32).cpu().numpy()
        if samples.ndim == 2 and samples.shape[0] <= 2 < samples.shape[1]:
            samples = samples.T
        buffer = AudioBuffer(
            np.ascontiguousarray(samples, dtype=np.float32),
            int(self._pipe.vae.sampling_rate),
        )

        ctx.report(1.0, "Done", "generate")
        return self._result(
            request, started, audio=buffer,
            title=(request.prompt or "Stable Audio")[:60] or "Stable Audio",
            duration_seconds=buffer.duration_seconds,
            meta={"model": getattr(self.entry, "repo", ""), "device": self._device,
                  "steps": steps},
        )
