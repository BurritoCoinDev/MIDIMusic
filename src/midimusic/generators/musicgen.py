"""MusicGen through Hugging Face transformers.

The most portable audio backend here: a plain autoregressive transformer over
an EnCodec tokenizer, with no custom kernels, so it runs wherever torch runs --
CUDA, ROCm, or CPU.  Its weights are CC-BY-NC, which the catalog surfaces
before a download starts.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.generator import BackendUnavailable, Capabilities, Generator, GeneratorContext
from ..core.models import AudioBuffer, GenerationRequest, GenerationResult
from . import _torch_util as tu

__all__ = ["MusicGenGenerator"]


class MusicGenGenerator(Generator):
    id = "musicgen"
    name = "MusicGen"
    kind = "audio"

    def __init__(self, model_id: str = "", entry=None):
        super().__init__(model_id, entry)
        self._model = None
        self._processor = None
        self._device = "cpu"

    def capabilities(self) -> Capabilities:
        entry = self.entry
        return Capabilities(
            outputs=("flac", "wav"),
            max_duration=float(getattr(entry, "max_duration", 30.0)),
            supports_lyrics=False,
            supports_vocals=False,
            supports_seed=True,
            supports_negative_prompt=False,
            needs_gpu=False,
            sample_rate=int(getattr(entry, "sample_rate", 32000)),
        )

    def required_packages(self) -> list[str]:
        return ["torch", "transformers"]

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        seconds = min(request.duration_seconds or 15.0, self.capabilities().max_duration)
        # MusicGen decodes 50 tokens per second of audio; CPU is far slower.
        rate = 1.5 if ctx.device != "cpu" else 25.0
        return max(5.0, seconds * rate)

    def load(self, ctx: GeneratorContext) -> None:
        if self._model is not None:
            return
        torch = tu.require_torch()
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except ImportError as exc:
            raise BackendUnavailable(
                "transformers is required for MusicGen.",
                missing_packages=["transformers"],
            ) from exc

        repo = getattr(self.entry, "repo", None) or "facebook/musicgen-small"
        tu.apply_model_env(getattr(self.entry, "env", None))
        self._device = tu.resolve_device(ctx.device)
        dtype = tu.torch_dtype_for(self._device)

        ctx.report(0.05, f"Loading {repo}", "load")
        kwargs = {"token": ctx.hf_token or None, "local_files_only": ctx.offline}
        self._processor = AutoProcessor.from_pretrained(repo, **kwargs)
        self._model = MusicgenForConditionalGeneration.from_pretrained(
            repo, dtype=dtype, **kwargs
        )
        self._model.to(self._device)
        self._model.eval()
        # Never compile: there is no Triton on Windows, and eager is correct
        # everywhere.
        _ = torch

    def unload(self) -> None:
        self._model = None
        self._processor = None
        tu.empty_cache()

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        torch = tu.require_torch()
        self.load(ctx)
        ctx.check_cancelled()

        caps = self.capabilities()
        seconds = min(request.duration_seconds or 15.0, caps.max_duration)
        # MusicGen's frame rate is 50 Hz; tokens map directly to duration.
        max_new_tokens = int(seconds * 50)

        tu.seed_everything(request.seed)
        prompt = request.prompt or "instrumental music"

        inputs = self._processor(text=[prompt], padding=True, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        ctx.report(0.15, f"Generating {seconds:.0f}s of audio", "generate")

        streamer = _ProgressStopper(ctx, max_new_tokens)
        with torch.inference_mode():
            audio = self._model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=float(request.guidance or 3.0),
                temperature=max(0.1, float(request.temperature or 1.0)),
                max_new_tokens=max_new_tokens,
                stopping_criteria=streamer.criteria(),
            )
        ctx.check_cancelled()

        sr = int(self._model.config.audio_encoder.sampling_rate)
        samples = audio[0].to(torch.float32).cpu().numpy()
        if samples.ndim == 2:  # (channels, frames) -> (frames, channels)
            samples = samples.T
        buffer = AudioBuffer(np.ascontiguousarray(samples, dtype=np.float32), sr)

        ctx.report(1.0, "Done", "generate")
        return self._result(
            request, started,
            audio=buffer,
            title=(request.prompt or "MusicGen")[:60] or "MusicGen",
            duration_seconds=buffer.duration_seconds,
            meta={"model": getattr(self.entry, "repo", ""), "device": self._device,
                  "sample_rate": sr},
        )


class _ProgressStopper:
    """Reports progress and honours cancellation during generation.

    transformers has no progress callback for generate(), but stopping criteria
    are consulted every step, which serves both purposes.
    """

    def __init__(self, ctx: GeneratorContext, total_tokens: int):
        self.ctx = ctx
        self.total = max(1, total_tokens)
        self.step = 0

    def criteria(self):
        from transformers import StoppingCriteria, StoppingCriteriaList

        outer = self

        class _Criteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                outer.step += 1
                if outer.step % 10 == 0:
                    outer.ctx.report(
                        0.15 + 0.8 * min(1.0, outer.step / outer.total),
                        "Generating audio", "generate",
                    )
                return outer.ctx.is_cancelled()

        return StoppingCriteriaList([_Criteria()])
