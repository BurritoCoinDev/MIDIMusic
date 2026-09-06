"""Shared fixtures.

Tests run headless: Qt uses the offscreen platform and the app's directories
are redirected into a temporary tree so a test run never touches real settings,
the real library, or the user's music folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Point every app directory at a throwaway tree."""
    from midimusic.config import paths as paths_module

    app_paths = paths_module.AppPaths(
        tmp_path / "config", tmp_path / "data", tmp_path / "cache"
    ).ensure()
    monkeypatch.setattr(paths_module, "_PATHS", app_paths)
    monkeypatch.setattr(paths_module, "get_paths", lambda: app_paths)

    from midimusic.config import settings as settings_module

    monkeypatch.setattr(settings_module, "_CACHE", None)
    monkeypatch.setattr(settings_module, "get_paths", lambda: app_paths)
    return app_paths


@pytest.fixture
def qapp():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def service():
    from midimusic.core.service import AppService

    svc = AppService()
    yield svc
    svc.shutdown()
