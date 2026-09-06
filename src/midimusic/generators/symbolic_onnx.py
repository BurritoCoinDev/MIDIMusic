"""ONNX symbolic backend.

Notable because it needs no torch at all: onnxruntime is a small wheel and the
model runs on CPU, so this backend works on a machine with no GPU and no
multi-gigabyte compute runtime installed.
"""

from __future__ import annotations

import time

from ..core.generator import BackendUnavailable, Capabilities, Generator, GeneratorContext
from ..core.models import GenerationRequest, GenerationResult, Note, Song, Track

__all__ = ["OnnxMidiGenerator"]


class OnnxMidiGenerator(Generator):
    id = "onnx-midi"
    name = "MIDI Composer (ONNX)"
    kind = "symbolic"

    def __init__(self, model_id: str = "", entry=None):
        super().__init__(model_id, entry)
        self._session = None
        self._tokenizer = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            outputs=("midi", "flac", "wav"),
            max_duration=float(getattr(self.entry, "max_duration", 300.0)),
            supports_seed=True,
            supports_continuation=True,
            needs_gpu=False,
        )

    def required_packages(self) -> list[str]:
        return ["onnxruntime"]

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        return max(5.0, (request.duration_seconds or 60.0) * 0.5)

    def load(self, ctx: GeneratorContext) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailable(
                "onnxruntime is required for the ONNX MIDI model.",
                missing_packages=["onnxruntime"],
            ) from exc
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise BackendUnavailable(
                "huggingface-hub is required to fetch the model.",
                missing_packages=["huggingface-hub"],
            ) from exc

        repo = getattr(self.entry, "repo", None) or "skytnt/midi-model-tv2o-medium"
        revision = getattr(self.entry, "revision", "main")
        ctx.report(0.05, f"Loading {repo}", "load")

        # Pin the revision: older ONNX exports are not compatible with the
        # current decoding code, and "latest" silently breaks.
        local = snapshot_download(
            repo, revision=revision, token=ctx.hf_token or None,
            local_files_only=ctx.offline,
            cache_dir=str(ctx.models_dir) if ctx.models_dir else None,
            allow_patterns=["*.onnx", "*.json", "*.txt", "*.data"],
        )

        from pathlib import Path

        onnx_files = sorted(Path(local).rglob("*.onnx"))
        if not onnx_files:
            raise BackendUnavailable(
                f"No ONNX weights found in {repo}.", missing_model=repo
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = {
            f.stem: ort.InferenceSession(str(f), options, providers=["CPUExecutionProvider"])
            for f in onnx_files
        }

    def unload(self) -> None:
        self._session = None

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        self.load(ctx)
        ctx.check_cancelled()
        ctx.report(0.2, "Generating notes", "generate")

        # The upstream decoding loop is model-specific and changes between
        # exports; rather than silently emit nonsense when it does not match,
        # report clearly that this export is unsupported.
        raise BackendUnavailable(
            "This ONNX export needs the upstream decoding loop, which is not bundled. "
            "Use text2midi or the Anticipatory model for symbolic generation, or the "
            "built-in composer.",
            missing_model=getattr(self.entry, "repo", ""),
        )


def _events_to_song(events, title: str, tempo: float = 120.0) -> Song:
    """Build a Song from (start_beats, duration_beats, pitch, velocity, channel)."""
    song = Song(title=title, tempo=tempo)
    by_channel: dict[int, Track] = {}
    for start, duration, pitch, velocity, channel in events:
        track = by_channel.get(channel)
        if track is None:
            track = Track(name=f"Channel {channel}", channel=channel,
                          is_drum=(channel == 9))
            by_channel[channel] = track
        track.notes.append(Note(int(pitch), float(start), float(duration),
                                int(velocity), int(channel)))
    for track in by_channel.values():
        track.sort()
    song.tracks = [t for t in by_channel.values() if t.notes]
    return song
