"""Persisted user settings."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .paths import get_paths

__all__ = ["Settings", "load_settings", "save_settings"]


@dataclass
class Settings:
    # Output
    output_dir: str = ""
    models_dir: str = ""
    output_format: str = "flac"
    sample_rate: int = 44100
    bit_depth: int = 24
    target_lufs: float = -14.0
    also_write_midi: bool = True   # audio backends also save a transcribed score

    # Generation defaults
    default_backend: str = "builtin"
    default_style: str = "pop"
    default_duration: int = 120
    default_complexity: float = 0.7
    default_variations: int = 2
    soundfont: str = ""

    # Runtime
    compute_backend: str = "auto"   # auto | cuda | rocm-windows | cpu | ...
    device: str = "auto"
    allow_downloads: bool = True
    hf_token: str = ""
    offline: bool = False

    # UI
    theme: str = "dark"
    accent: str = "#7C5CFF"
    window_geometry: str = ""
    autoplay: bool = True
    catalog_url: str = ""  # optional remote catalog for new models

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}
        obj = cls(**clean)
        obj.extra = {k: v for k, v in (data or {}).items() if k not in known}
        return obj

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir) if self.output_dir else get_paths().output

    def resolved_models_dir(self) -> Path:
        return Path(self.models_dir) if self.models_dir else get_paths().models


_LOCK = threading.Lock()
_CACHE: Settings | None = None


def load_settings(force: bool = False) -> Settings:
    global _CACHE
    with _LOCK:
        if _CACHE is not None and not force:
            return _CACHE
        path = get_paths().settings_file
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _CACHE = Settings.from_dict(data)
        except (OSError, json.JSONDecodeError):
            _CACHE = Settings()
        return _CACHE


def save_settings(settings: Settings) -> None:
    global _CACHE
    with _LOCK:
        _CACHE = settings
        path = get_paths().settings_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        payload = {**settings.to_dict(), **settings.extra}
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)  # atomic, so a crash cannot truncate settings
        except OSError:
            pass
