"""Maps catalog adapter names to generator classes.

Imports are lazy and by name, so a missing optional dependency disables one
backend instead of preventing the app from starting.
"""

from __future__ import annotations

import importlib
import logging

from .catalog import Catalog, ModelEntry, load_catalog
from .generator import Generator

__all__ = ["ADAPTERS", "create_generator", "available_generators", "register_adapter"]

log = logging.getLogger(__name__)

# adapter name -> "module:ClassName"
ADAPTERS: dict[str, str] = {
    "builtin": "midimusic.generators.builtin:BuiltinComposerGenerator",
    "hf-musicgen": "midimusic.generators.musicgen:MusicGenGenerator",
    "ace-step": "midimusic.generators.ace_step:AceStepGenerator",
    "diffusers-audio": "midimusic.generators.diffusers_audio:DiffusersAudioGenerator",
    "hf-text2midi": "midimusic.generators.symbolic_hf:Text2MidiGenerator",
    "hf-anticipatory": "midimusic.generators.symbolic_hf:AnticipatoryGenerator",
    "onnx-midi": "midimusic.generators.symbolic_onnx:OnnxMidiGenerator",
    "diffrhythm": "midimusic.generators.diffusers_audio:DiffusersAudioGenerator",
}


def register_adapter(name: str, target: str) -> None:
    """Register a new adapter kind at runtime (used by plugins)."""
    ADAPTERS[name] = target


def _resolve(target: str) -> type[Generator] | None:
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        log.debug("adapter %s unavailable: %s", target, exc)
        return None


def create_generator(entry: ModelEntry) -> Generator | None:
    """Instantiate the generator for a catalog entry."""
    target = ADAPTERS.get(entry.adapter)
    if not target:
        log.warning("no adapter registered for %r (model %s)", entry.adapter, entry.id)
        return None
    cls = _resolve(target)
    if cls is None:
        return None
    try:
        return cls(model_id=entry.id, entry=entry)
    except Exception:
        log.exception("failed to construct generator for %s", entry.id)
        return None


def available_generators(catalog: Catalog | None = None) -> list[tuple[ModelEntry, Generator]]:
    """Every catalog entry that has a working adapter class."""
    catalog = catalog or load_catalog()
    out: list[tuple[ModelEntry, Generator]] = []
    for entry in catalog.models:
        gen = create_generator(entry)
        if gen is not None:
            out.append((entry, gen))
    return out
