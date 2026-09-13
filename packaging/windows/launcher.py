"""Frozen-build entry point.

PyInstaller executes its entry script as top-level ``__main__``, outside any
package, so pointing it at ``midimusic/__main__.py`` breaks that module's
imports. Importing the package from a separate launcher gives the real entry
point normal package context.
"""

from midimusic.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
