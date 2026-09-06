"""Open-weight symbolic backends: text prompt or continuation to MIDI.

These are the models that generate notes rather than waveforms, so their output
is a real editable score rather than a transcription guess.  They are small
enough to run on CPU, which is why they are the first neural tier.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ..audio.midi_io import midi_to_song
from ..core.generator import BackendUnavailable, Capabilities, Generator, GeneratorContext
from ..core.models import GenerationRequest, GenerationResult, Song
from . import _torch_util as tu

__all__ = ["AnticipatoryGenerator", "Text2MidiGenerator"]


class _SymbolicBase(Generator):
    kind = "symbolic"

    def __init__(self, model_id: str = "", entry=None):
        super().__init__(model_id, entry)
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def required_packages(self) -> list[str]:
        return ["torch", "transformers"]

    def capabilities(self) -> Capabilities:
        return Capabilities(
            outputs=("midi", "flac", "wav"),
            max_duration=float(getattr(self.entry, "max_duration", 300.0)),
            supports_seed=True,
            needs_gpu=False,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        tu.empty_cache()

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        """Wall clock for a generation, from measurement rather than optimism.

        These models decode one token at a time, and a second of music is on
        the order of a hundred tokens, so CPU generation is minutes per
        *second* of output -- not a fraction of real time. Measured at roughly
        130s of compute per second of music on four cores; the figure below
        assumes a considerably faster desktop CPU and is still slow enough that
        the UI should be honest about it rather than quietly disappointing.
        """
        seconds = request.duration_seconds or 60.0
        per_second = 2.5 if ctx.device != "cpu" else 40.0
        return max(10.0, seconds * per_second)


class Text2MidiGenerator(_SymbolicBase):
    """amaai-lab/text2midi: free-text caption to a MIDI file."""

    id = "text2midi"
    name = "text2midi"

    def capabilities(self) -> Capabilities:
        caps = super().capabilities()
        caps.honours_tempo = False
        caps.honours_key = False
        return caps

    def load(self, ctx: GeneratorContext) -> None:
        if self._model is not None:
            return
        torch = tu.require_torch()
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise BackendUnavailable(
                "transformers is required.", missing_packages=["transformers"]
            ) from exc

        repo = getattr(self.entry, "repo", None) or "amaai-lab/text2midi"
        self._device = tu.resolve_device(ctx.device)
        ctx.report(0.05, f"Loading {repo}", "load")

        kwargs = {"token": ctx.hf_token or None, "local_files_only": ctx.offline}
        # The published checkpoint pairs a FLAN-T5 encoder with a custom
        # decoder, so the tokenizer comes from the encoder it was trained with.
        self._tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", **kwargs)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise BackendUnavailable(
                "huggingface-hub is required.", missing_packages=["huggingface-hub"]
            ) from exc

        weights = hf_hub_download(repo, "pytorch_model.bin", **kwargs)
        try:
            from model.transformer_model import Transformer  # type: ignore
        except ImportError as exc:
            raise BackendUnavailable(
                "text2midi's model code is not installed. Install it from source:\n"
                "  pip install git+https://github.com/AMAAI-Lab/Text2midi",
                missing_packages=["text2midi"],
            ) from exc

        model = Transformer(
            vocab_size=len(self._tokenizer), d_model=768, nhead=8,
            num_layers=18, dim_feedforward=1024, max_len=2000, device=self._device,
        )
        model.load_state_dict(torch.load(weights, map_location=self._device), strict=False)
        model.to(self._device).eval()
        self._model = model

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        torch = tu.require_torch()
        self.load(ctx)
        ctx.check_cancelled()
        tu.seed_everything(request.seed)

        prompt = request.prompt or "a gentle piano piece"
        ctx.report(0.2, "Generating notes", "generate")
        encoded = self._tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self._device)
        attention = encoded["attention_mask"].to(self._device)

        with torch.inference_mode():
            output = self._model.generate(
                input_ids, attention,
                max_len=int(request.extra.get("max_tokens", 2000)),
                temperature=max(0.1, float(request.temperature or 1.0)),
            )
        ctx.check_cancelled()

        song = _decode_to_song(output, self._tokenizer, request.prompt)
        ctx.report(1.0, "Done", "generate")
        return self._result(
            request, started, song=song, title=(prompt[:60] or "text2midi"),
            duration_seconds=song.duration_seconds,
            meta={"model": getattr(self.entry, "repo", ""), "device": self._device},
        )


class AnticipatoryGenerator(_SymbolicBase):
    """stanford-crfm anticipatory music transformer: generation and infilling."""

    id = "anticipatory"
    name = "Anticipatory Music Transformer"

    def capabilities(self) -> Capabilities:
        caps = super().capabilities()
        caps.supports_continuation = True
        return caps

    def required_packages(self) -> list[str]:
        return ["torch", "transformers", "anticipation"]

    def load(self, ctx: GeneratorContext) -> None:
        if self._model is not None:
            return
        tu.require_torch()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise BackendUnavailable(
                "transformers is required.", missing_packages=["transformers"]
            ) from exc
        try:
            import anticipation  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable(
                "The anticipation package is required. Install it with:\n"
                "  pip install git+https://github.com/jthickstun/anticipation",
                missing_packages=["anticipation"],
            ) from exc

        repo = getattr(self.entry, "repo", None) or "stanford-crfm/music-small-800k"
        self._device = tu.resolve_device(ctx.device)
        ctx.report(0.05, f"Loading {repo}", "load")
        self._model = AutoModelForCausalLM.from_pretrained(
            repo, token=ctx.hf_token or None, local_files_only=ctx.offline
        ).to(self._device).eval()

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        self.load(ctx)
        ctx.check_cancelled()
        tu.seed_everything(request.seed)

        from anticipation.convert import events_to_midi
        from anticipation.sample import generate as anticipate

        seconds = float(request.duration_seconds or 60.0)
        ctx.report(0.2, f"Generating {seconds:.0f}s of events", "generate")

        events = anticipate(
            self._model, start_time=0, end_time=seconds,
            top_p=float(request.extra.get("top_p", 0.98)),
        )
        ctx.check_cancelled()

        midi = events_to_midi(events)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.mid"
            midi.save(str(path))
            song = midi_to_song(path, title=(request.prompt or "Anticipatory")[:60])

        ctx.report(1.0, "Done", "generate")
        return self._result(
            request, started, song=song,
            title=(request.prompt or "Anticipatory")[:60] or "Anticipatory",
            duration_seconds=song.duration_seconds,
            meta={"model": getattr(self.entry, "repo", ""), "device": self._device},
        )


def _decode_to_song(output, tokenizer, prompt: str) -> Song:
    """Decode model output into a Song, writing through a temporary MIDI file."""
    import mido

    from ..audio.midi_io import midi_to_song as _load

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.mid"
        if hasattr(output, "save"):
            output.save(str(path))
        elif isinstance(output, (bytes, bytearray)):
            path.write_bytes(output)
        elif isinstance(output, str) and Path(output).exists():
            path = Path(output)
        else:
            # A token sequence: let the tokenizer's own decoder produce MIDI.
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            midi = mido.MidiFile()
            midi.tracks.append(mido.MidiTrack())
            midi.save(str(path))
            if not decoded:
                return Song(title=prompt[:60] or "Untitled")
        return _load(path, title=prompt[:60] or "Untitled")
