# PyInstaller spec for MIDIMusic.
#
# onedir, deliberately: a onefile build of a Qt + audio application unpacks
# itself to a temp directory on every launch, which is slow, trips antivirus
# heuristics, and breaks the assumption that the app can find its own assets.
# onedir starts fast and updates by replacing files.
#
# Torch is intentionally NOT bundled. The correct build depends on the user's
# GPU (AMD's Windows ROCm wheels, CUDA, or CPU), so the app installs it after
# setup from the Models tab. That keeps the installer around 140 MB instead of
# several gigabytes and means an AMD user gets a working GPU path.
#
# What IS bundled is uv, the tool that builds that environment. A frozen app
# cannot install packages with its own interpreter -- sys.executable is
# MIDIMusic.exe -- so without uv there would be no way to provision anything,
# and the app would depend on the user already having Python installed.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None
ROOT = Path(SPECPATH).resolve().parents[1]

datas = [
    (str(ROOT / "src" / "midimusic" / "data"), "midimusic/data"),
    # The worker is run by the provisioned runtime's interpreter, so it must
    # survive as a real .py file rather than only as bytecode in the archive.
    (str(ROOT / "src" / "midimusic" / "worker" / "runner.py"), "midimusic/worker"),
]
assets = ROOT / "src" / "midimusic" / "assets"
if assets.exists():
    datas.append((str(assets), "midimusic/assets"))

# uv builds the compute runtime. Bundling it is what keeps the app
# self-contained: everything needed to bootstrap ships in the installer, and
# only the multi-gigabyte GPU wheels are fetched on demand.
try:
    import uv as _uv_package

    _uv_binary = Path(_uv_package.find_uv_bin())
    if _uv_binary.exists():
        binaries_extra = [(str(_uv_binary), "uv")]
    else:
        raise FileNotFoundError(_uv_binary)
except Exception as exc:  # pragma: no cover - build-time only
    raise SystemExit(
        "uv is required to build a self-contained bundle: pip install uv"
    ) from exc

# soundfile and sounddevice carry the native libraries we depend on.
binaries = collect_dynamic_libs("soundfile") + collect_dynamic_libs("sounddevice")
binaries += binaries_extra
datas += collect_data_files("soundfile") + collect_data_files("tinysoundfont")

hiddenimports = [
    "midimusic.generators.builtin",
    "midimusic.generators.musicgen",
    "midimusic.generators.ace_step",
    "midimusic.generators.diffusers_audio",
    "midimusic.generators.symbolic_hf",
    "midimusic.generators.symbolic_onnx",
    "soundfile",
    "sounddevice",
    "tinysoundfont",
    "mido.backends.rtmidi",
]

# Everything heavy and optional is provisioned at runtime, so excluding it here
# keeps the build both smaller and faster.
excludes = [
    "torch", "torchaudio", "torchvision", "transformers", "diffusers",
    "onnxruntime", "basic_pitch", "tensorflow", "matplotlib", "scipy",
    "IPython", "notebook", "pandas", "PySide6.QtWebEngineCore",
    "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia",
    "tkinter", "test", "unittest",
]

a = Analysis(
    [str(ROOT / "src" / "midimusic" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MIDIMusic",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX compression is a common antivirus false-positive trigger
    console=False,  # a GUI app must not flash a console window
    icon=str(ROOT / "packaging" / "windows" / "icon.ico")
    if (ROOT / "packaging" / "windows" / "icon.ico").exists()
    else None,
    version=str(ROOT / "packaging" / "windows" / "version_info.txt")
    if (ROOT / "packaging" / "windows" / "version_info.txt").exists()
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="MIDIMusic",
)
