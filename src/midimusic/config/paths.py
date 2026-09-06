"""Where the app keeps things on disk.

Windows conventions matter here: models are large and belong in LOCALAPPDATA
(which does not roam), settings are small and belong in APPDATA, and the user
must be able to move the model directory to another drive without reinstalling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["AppPaths", "get_paths", "set_models_dir", "human_bytes", "disk_free"]

APP_NAME = "MIDIMusic"


def _base_dirs() -> tuple[Path, Path, Path]:
    """Return (config_dir, data_dir, cache_dir) for this platform."""
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        return appdata / APP_NAME, local / APP_NAME, local / APP_NAME / "cache"
    if sys.platform == "darwin":
        base = Path.home() / "Library/Application Support" / APP_NAME
        return base, base, Path.home() / "Library/Caches" / APP_NAME
    xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    xdg_data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return xdg_config / APP_NAME, xdg_data / APP_NAME, xdg_cache / APP_NAME


class AppPaths:
    """Resolved application directories."""

    def __init__(self, config: Path, data: Path, cache: Path):
        self.config = config
        self.data = data
        self.cache = cache

    @property
    def settings_file(self) -> Path:
        return self.config / "settings.json"

    @property
    def library_db(self) -> Path:
        return self.data / "library.json"

    @property
    def logs(self) -> Path:
        return self.data / "logs"

    @property
    def user_styles(self) -> Path:
        return self.config / "styles.json"

    @property
    def user_catalog(self) -> Path:
        return self.config / "models.user.json"

    @property
    def soundfonts(self) -> Path:
        return self.data / "soundfonts"

    @property
    def runtime(self) -> Path:
        """Where the provisioned torch environment lives."""
        return self.data / "runtime"

    @property
    def models(self) -> Path:
        override = _models_override(self)
        return override if override else self.data / "models"

    @property
    def output(self) -> Path:
        override = _output_override(self)
        if override:
            return override
        music = Path.home() / "Music"
        return (music if music.exists() else Path.home()) / APP_NAME

    def ensure(self) -> "AppPaths":
        for d in (self.config, self.data, self.cache, self.logs,
                  self.soundfonts, self.models, self.output):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        return self


def _read_override(paths: AppPaths, key: str) -> Path | None:
    """Read a directory override from settings without importing settings.

    Kept deliberately dependency-free: settings imports paths, so paths must
    not import settings.
    """
    try:
        import json

        with open(paths.settings_file, encoding="utf-8") as fh:
            value = json.load(fh).get(key)
        return Path(value) if value else None
    except (OSError, ValueError, TypeError):
        return None


def _models_override(paths: AppPaths) -> Path | None:
    env = os.environ.get("MIDIMUSIC_MODELS_DIR")
    if env:
        return Path(env)
    return _read_override(paths, "models_dir")


def _output_override(paths: AppPaths) -> Path | None:
    env = os.environ.get("MIDIMUSIC_OUTPUT_DIR")
    if env:
        return Path(env)
    return _read_override(paths, "output_dir")


_PATHS: AppPaths | None = None


def get_paths() -> AppPaths:
    global _PATHS
    if _PATHS is None:
        _PATHS = AppPaths(*_base_dirs()).ensure()
    return _PATHS


def set_models_dir(path: str | Path) -> None:
    """Point Hugging Face caches at the app's model directory."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(p)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(p / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def disk_free(path: str | Path) -> int:
    """Free bytes on the volume holding ``path`` (0 if it cannot be read)."""
    import shutil

    try:
        p = Path(path)
        while not p.exists() and p.parent != p:
            p = p.parent
        return shutil.disk_usage(p).free
    except OSError:
        return 0
