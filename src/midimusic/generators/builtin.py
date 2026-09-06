"""The built-in composer, wrapped as a generator.

Always available, always fast, no download.  It is also the fallback the app
uses when a neural backend is missing its weights, so the user is never left
with nothing.
"""

from __future__ import annotations

import random
import time

from ..audio.export import safe_filename
from ..core.generator import Capabilities, Generator, GeneratorContext
from ..core.models import GenerationRequest, GenerationResult, Song
from ..prompt.parser import parse_prompt
from ..theory.composer import Composer, CompositionSpec
from ..theory.pitch import Scale, parse_key
from ..theory.structure import ROLES
from ..theory.style import get_style

__all__ = ["BuiltinComposerGenerator", "build_spec"]


def build_spec(request: GenerationRequest, rng: random.Random | None = None) -> CompositionSpec:
    """Resolve a request plus its prompt into a fully-specified composition.

    Explicit request fields always win over anything inferred from the prompt,
    so the UI controls are authoritative and the prompt only fills the gaps.
    """
    parsed = parse_prompt(request.prompt, request.style or "pop")
    seed = request.seed if request.seed is not None else random.randrange(1 << 30)
    rng = rng or random.Random(seed)

    style = get_style(request.style or parsed.style or "pop")

    key_text = request.key or parsed.key
    if key_text:
        scale = Scale(*parse_key(key_text))
    else:
        mode = rng.choice(style.scales)
        scale = Scale(rng.randrange(12), mode)

    tempo = request.tempo or parsed.tempo
    if not tempo:
        tempo = rng.uniform(*style.tempo)

    duration = request.duration_seconds or parsed.duration_seconds or 120.0

    extra = request.extra or {}
    complexity = extra.get("complexity")
    if complexity is None:
        complexity = parsed.complexity if parsed.complexity is not None else 0.7

    roles = None
    if extra.get("roles"):
        roles = tuple(r for r in extra["roles"] if r in ROLES)

    return CompositionSpec(
        style=style,
        scale=scale,
        tempo=float(tempo),
        duration_seconds=float(duration),
        seed=seed,
        form_name=request.structure or extra.get("form"),
        title=extra.get("title") or _title_from(request, style.name),
        density=extra.get("density"),
        complexity=float(complexity),
        brightness=float(extra.get("brightness", parsed.brightness
                                   if parsed.brightness is not None else 0.5)),
        energy=float(extra.get("energy", parsed.energy
                               if parsed.energy is not None else 0.5)),
        modulate=bool(extra.get("modulate", True)),
        roles=roles,
        prompt=request.prompt,
    )


def _title_from(request: GenerationRequest, style_name: str) -> str:
    words = [w for w in (request.prompt or "").split() if w.isalnum()][:5]
    base = " ".join(words).title() if words else f"{style_name.title()} Sketch"
    return safe_filename(base, fallback=f"{style_name.title()} Sketch")


class BuiltinComposerGenerator(Generator):
    id = "builtin"
    name = "Built-in Composer"
    kind = "symbolic"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            outputs=("midi", "flac", "wav"),
            max_duration=900.0,
            supports_lyrics=False,
            supports_vocals=False,
            supports_seed=True,
            supports_continuation=False,
            needs_gpu=False,
            honours_key=True,
            honours_tempo=True,
            honours_structure=True,
        )

    def required_packages(self) -> list[str]:
        return []

    def is_ready(self, ctx: GeneratorContext | None = None) -> bool:
        return True

    def estimated_seconds(self, request: GenerationRequest, ctx: GeneratorContext) -> float:
        # Composition is near-instant; rendering dominates for audio output.
        seconds = (request.duration_seconds or 120.0)
        return 1.0 + (seconds * 0.12 if request.output_format.value != "midi" else 0.0)

    def generate(self, request: GenerationRequest, ctx: GeneratorContext) -> GenerationResult:
        started = time.time()
        ctx.report(0.05, "Planning the arrangement", "compose")
        spec = build_spec(request)
        ctx.check_cancelled()

        song: Song = Composer(spec).compose()
        ctx.report(0.6, f"Arranged {len(song.tracks)} parts", "compose")
        ctx.check_cancelled()

        result = self._result(
            request, started,
            song=song,
            title=song.title,
            duration_seconds=song.duration_seconds,
            meta={
                "style": spec.style.name,
                "key": str(spec.scale),
                "tempo": round(spec.tempo, 1),
                "bars": song.meta.get("bars", 0),
                "tracks": [t.name for t in song.tracks],
                "complexity": spec.complexity,
            },
        )
        result.seed = spec.seed
        ctx.report(1.0, "Done", "compose")
        return result
