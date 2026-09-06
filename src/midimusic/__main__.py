"""Application entry point."""

from __future__ import annotations

import logging
import sys


def main(argv: list[str] | None = None) -> int:
    from PySide6 import QtGui, QtWidgets

    from .config.logging_setup import setup_logging
    from .config.paths import get_paths, set_models_dir
    from .config.settings import load_settings
    from .core.service import AppService
    from .theory.style import load_user_styles
    from .ui.main_window import MainWindow
    from .ui.theme import DARK, LIGHT, build_stylesheet

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

    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("MIDIMusic")
    app.setOrganizationName("MIDIMusic")
    app.setStyleSheet(build_stylesheet(LIGHT if settings.theme == "light" else DARK))

    icon = paths.data / "icon.png"
    if icon.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon)))

    service = AppService(settings=settings)
    window = MainWindow(service)
    window.show()
    log.info("started on %s", service.system.summary())
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
