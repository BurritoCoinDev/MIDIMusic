"""Application entry point.

Imports here are absolute, not relative. A frozen build runs its entry script
as top-level ``__main__`` with no package context, so ``from .config import x``
raises "attempted relative import with no known parent package" the moment the
packaged app starts. Absolute imports work identically in both cases.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["main"]


def _build_app(argv: list[str], offscreen: bool = False):
    """Construct the application, service and window without running the loop.

    Shared by normal startup and by ``--selftest`` so the self-test exercises
    the real path rather than an approximation of it.
    """
    if offscreen:
        # Lets the self-test run on a machine or CI runner with no desktop.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtGui, QtWidgets

    from midimusic.config.logging_setup import setup_logging
    from midimusic.config.paths import get_paths, set_models_dir
    from midimusic.config.settings import load_settings
    from midimusic.core.service import AppService
    from midimusic.theory.style import load_user_styles
    from midimusic.ui.main_window import MainWindow
    from midimusic.ui.theme import DARK, LIGHT, build_stylesheet

    setup_logging()
    log = logging.getLogger("midimusic")

    paths = get_paths()
    settings = load_settings()
    # Point the Hugging Face caches at our models directory before anything
    # imports a library that reads those variables at import time.
    set_models_dir(settings.resolved_models_dir())
    loaded = load_user_styles(paths.user_styles)
    if loaded:
        log.info("loaded %d user styles", loaded)

    app = QtWidgets.QApplication(argv)
    app.setApplicationName("MIDIMusic")
    app.setOrganizationName("MIDIMusic")
    app.setStyleSheet(build_stylesheet(LIGHT if settings.theme == "light" else DARK))

    icon = paths.data / "icon.png"
    if icon.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon)))

    service = AppService(settings=settings)
    window = MainWindow(service)
    return app, service, window, log


def _selftest(argv: list[str]) -> int:
    """Build the whole application, then exit. Returns 0 when it all worked.

    This exists because "the process is still running" is not evidence that a
    packaged app started: a frozen build that raises during startup shows an
    error dialog and *keeps running*, so a liveness check reports success while
    the user sees a traceback. Exercising construction and exiting with a
    status code is something CI can actually verify.
    """
    try:
        app, service, window, log = _build_app(argv, offscreen=True)
    except Exception:
        logging.getLogger("midimusic").exception("selftest failed during startup")
        import traceback

        traceback.print_exc()
        return 1

    try:
        window.show()
        app.processEvents()
        missing = _missing_backends()
        if missing:
            # A frozen build drops these silently, because they are only ever
            # imported by name. The app still starts; the backends are simply
            # gone from the model list, which is not something a startup check
            # would otherwise notice.
            message = "backends missing from this build: " + ", ".join(missing)
            log.error("selftest failed: %s", message)
            print(f"selftest failed: {message}", file=sys.stderr)
            return 1
        summary = service.system.summary()
        log.info("selftest ok on %s (%d backends)", summary, _backend_count())
        print(f"selftest ok: {summary}")
    except Exception:
        logging.getLogger("midimusic").exception("selftest failed after startup")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        try:
            window.close()
            service.shutdown()
        except Exception:
            pass
    return 0


def _backend_modules() -> tuple[str, ...]:
    from midimusic.core.registry import adapter_modules

    return adapter_modules()


def _backend_count() -> int:
    return len(_backend_modules())


def _missing_backends() -> list[str]:
    """Adapter modules this build cannot import."""
    import importlib

    missing = []
    for name in _backend_modules():
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)

    if "--version" in args:
        from midimusic import __version__

        print(f"MIDIMusic {__version__}")
        return 0

    if "--selftest" in args:
        return _selftest([a for a in args if a != "--selftest"])

    app, _service, window, log = _build_app(args)
    window.show()
    log.info("started on %s", _service.system.summary())
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
