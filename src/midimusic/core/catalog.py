"""The model catalog: what the app knows how to run.

Deliberately data-driven.  A model is a JSON entry naming an adapter kind, so
adding one is a config change, not a code change.  Three sources are merged, in
increasing priority: the bundled catalog, an optional remote catalog, and the
user's own file -- which means a new model can be added without waiting for an
app release.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.paths import get_paths

__all__ = ["ModelEntry", "Catalog", "load_catalog", "BUNDLED_CATALOG"]

BUNDLED_CATALOG = Path(__file__).resolve().parent.parent / "data" / "models.json"


@dataclass
class ModelEntry:
    id: str
    name: str = ""
    kind: str = "audio"            # audio | symbolic
    adapter: str = "builtin"       # which adapter class loads it
    repo: str = ""                 # Hugging Face repo id
    revision: str = "main"
    license: str = "unknown"
    license_note: str = ""
    commercial_use: Any = "unknown"  # True | False | "conditional"
    gated: bool = False
    size_gb: float = 0.0
    vram_gb: float = 0.0
    devices: tuple[str, ...] = ("cpu",)
    tier: int = 3
    flagship: bool = False
    experimental: bool = False
    outputs: tuple[str, ...] = ("flac",)
    vocals: bool = False
    lyrics: bool = False
    sample_rate: int = 44100
    max_duration: float = 60.0
    extras: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    incompatible: tuple[str, ...] = ()
    requires_binary: tuple[str, ...] = ()
    description: str = ""
    notes: str = ""
    source: str = "bundled"        # bundled | remote | user

    @property
    def is_builtin(self) -> bool:
        return self.adapter == "builtin"

    @property
    def needs_download(self) -> bool:
        return bool(self.repo) and not self.is_builtin

    def supports(self, output_format: str) -> bool:
        return output_format in self.outputs

    def runs_on(self, device: str) -> bool:
        if device in ("auto", ""):
            return True
        # ROCm and CUDA both present as "cuda" to torch, so treat them alike
        # for capability purposes.
        alias = {"rocm-windows": "rocm", "cuda": "cuda", "rocm": "rocm"}
        return alias.get(device, device) in self.devices or device in self.devices

    def fits(self, vram_gb: float) -> bool:
        return self.vram_gb <= 0 or vram_gb <= 0 or vram_gb >= self.vram_gb

    def license_warning(self) -> str:
        """A short warning to show before the user commits to a download."""
        if self.commercial_use is False:
            return f"{self.license}: personal use only, not licensed for commercial release."
        if self.commercial_use == "conditional":
            return f"{self.license}: conditional commercial use. {self.license_note}".strip()
        return ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str = "bundled") -> "ModelEntry":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        clean: dict[str, Any] = {}
        for key, value in d.items():
            if key not in known:
                continue
            if key in ("devices", "outputs", "extras", "incompatible", "requires_binary"):
                value = tuple(value or ())
            clean[key] = value
        clean.setdefault("id", d.get("id", "unknown"))
        clean.setdefault("name", clean["id"])
        entry = cls(**clean)
        entry.source = source
        return entry


@dataclass
class Catalog:
    models: list[ModelEntry] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)

    def get(self, model_id: str) -> ModelEntry | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def by_kind(self, kind: str) -> list[ModelEntry]:
        return [m for m in self.models if m.kind == kind]

    def for_output(self, output_format: str) -> list[ModelEntry]:
        return [m for m in self.models if m.supports(output_format)]

    def available(self, device: str = "auto", vram_gb: float = 0.0,
                  output_format: str | None = None) -> list[ModelEntry]:
        """Models this machine can actually run, best first."""
        out = [
            m for m in self.models
            if m.runs_on(device)
            and m.fits(vram_gb)
            and (output_format is None or m.supports(output_format))
        ]
        return sorted(out, key=lambda m: (m.tier, not m.flagship, m.name))

    def upsert(self, entry: ModelEntry) -> None:
        for i, m in enumerate(self.models):
            if m.id == entry.id:
                self.models[i] = entry
                return
        self.models.append(entry)

    def remove(self, model_id: str) -> bool:
        before = len(self.models)
        self.models = [m for m in self.models if m.id != model_id]
        return len(self.models) < before


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_catalog(user_file: Path | None = None, remote_cache: Path | None = None) -> Catalog:
    """Merge bundled, remote and user catalogs. Later sources win by model id."""
    catalog = Catalog()
    bundled = _read(BUNDLED_CATALOG)
    for raw in bundled.get("models", []):
        catalog.upsert(ModelEntry.from_dict(raw, "bundled"))
    catalog.dropped = list(bundled.get("dropped", []))

    paths = get_paths()
    remote = remote_cache if remote_cache is not None else paths.cache / "models.remote.json"
    for raw in _read(remote).get("models", []):
        catalog.upsert(ModelEntry.from_dict(raw, "remote"))

    user = user_file if user_file is not None else paths.user_catalog
    data = _read(user)
    for raw in data.get("models", []):
        catalog.upsert(ModelEntry.from_dict(raw, "user"))
    for removed in data.get("disabled", []):
        catalog.remove(str(removed))
    return catalog


def save_user_model(entry: ModelEntry, user_file: Path | None = None) -> None:
    """Add or update a user-defined model so it survives restarts."""
    path = user_file or get_paths().user_catalog
    data = _read(path)
    models = [m for m in data.get("models", []) if m.get("id") != entry.id]
    payload = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in entry.__dict__.items()
        if k != "source"
    }
    models.append(payload)
    data["models"] = models
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
