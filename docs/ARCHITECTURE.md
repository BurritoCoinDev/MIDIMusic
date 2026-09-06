# Architecture

Decisions and the evidence behind them. The point of this file is that the
next person to touch the code — including a future me — does not re-litigate
choices that were made for a reason, and does not "fix" something that is
deliberate.

---

## The shape of it

```
Compose panel ─┐
               ├─► AppService ─► JobQueue (one worker thread)
Settings ──────┘                     │
                                     ├─► Generator (in-process)
                                     │     └─ builtin composer, symbolic models
                                     │
                                     └─► RemoteAudioGenerator
                                           └─ subprocess ─► worker/runner.py
                                                              └─ torch, models
                                     │
                                     ▼
                               result → files
                          Song ─► MIDI          (mido)
                          Song ─► FLAC          (SoundFont or numpy synth)
                          Audio ─► FLAC         (libsndfile)
                          Audio ─► MIDI         (basic-pitch)
```

`AppService` owns the queue and decides how a result becomes files. Format
conversion lives there, not in the generators — which is why a MIDI-only model
can still produce FLAC and an audio-only model can still produce a score.

---

## Decisions

### Two pipelines, not one

Waveform models contain no notes. Transcribing their output is lossy and
degrades on dense mixes. If the goal is an editable score, a symbolic model is
the right tool and no amount of post-processing makes a diffusion model into
one. So both exist, and the UI is honest about which is which.

### The built-in composer is not a placeholder

It is the reason the app is usable on first launch, offline, on any hardware.
It is also the fallback when a neural backend's weights are missing, so the
user is never left with nothing. It is a real arranger: voice-leading,
motif development, song form, 14 instrument roles.

### Neural backends run out of process

The frozen bundle excludes torch. This is not an optimisation — the installer
would otherwise be multiple gigabytes and still be wrong for most users,
because the correct torch build depends on the GPU. So:

- the application bundle contains no torch and cannot import it;
- `worker/runner.py` runs in a provisioned runtime and imports nothing from
  `midimusic`, because it runs under an interpreter that has never heard of
  this package;
- they speak newline-delimited JSON over pipes, and audio comes back as a file
  rather than a base64 payload down the pipe.

A native crash in a GPU stack therefore fails one job. Cancellation kills the
process, which is the only reliable way to interrupt a native library.

### AMD gets real GPU acceleration on Windows

Verified against live indexes rather than assumed:

- `https://repo.amd.com/rocm/whl-multi-arch/` is a real PEP 503 index with
  win_amd64 torch wheels for cp310–cp314, and `amd-torch-device-gfxNNNN`
  packages behind the `torch[device-gfxNNNN]` extras.
- `https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/simple/` returns **404**.
  It is a directory listing, not an index. An earlier version of this code
  pointed `--index-url` at it, which could never have worked.
- `torch-directml`'s last release is from September 2024 and pins torch 2.4.1.
  It is a dead end, not a fallback.
- ZLUDA's own documentation still describes PyTorch support as unfinished.

GPU architecture is resolved from the PCI **DeviceId**, not from substrings of
the adapter description, which vary by OEM and locale. The install target is
derived from that architecture, so a 7800 XT gets gfx1101 rather than the
gfx1100 build every AMD user would otherwise have received.

pip gets `--index-url <vendor>` plus `--extra-index-url pypi.org`. Using two
`--index-url` flags would silently install a CUDA build over the requested one.

### Detection never imports torch

`detect_system()` has to run *before* torch exists — choosing which torch to
install is its whole purpose — and importing a GPU stack into the UI process
risks a native crash taking the window with it. Torch facts come from the
out-of-process probe, which runs a matmul and an attention call rather than
trusting that `cuda.is_available()` means the kernels work.

### The probe is cached

It spawns an interpreter and imports torch, taking seconds. The UI asks
"is this backend ready?" once per model card. Without caching the Models tab
would spawn a dozen interpreters and take the better part of a minute to open.
`clear_probe_cache()` after installing a runtime.

### The catalog is data

Models live in `data/models.json` with an adapter name, licence, size, VRAM,
device support and per-model environment quirks. Adding one is a config change.
Users can add their own entries, and a remote catalog can add models without
an app release.

### Licence hygiene

- **mutagen** (GPL-2.0-or-later) was removed. libsndfile writes Vorbis comments
  natively, so it was unnecessary as well as incompatible with an MIT app.
- **soxr** is optional because its wheel statically links LGPL-2.1 libsoxr.
  The core resampler is a windowed-sinc polyphase filter in NumPy, measured at
  −116 dB sidelobes converting 32 kHz to 44.1 kHz.
- The build is `onedir` so Qt and libsndfile remain replaceable, which is what
  makes LGPL dynamic-linking compliance straightforward.
- Model licences are surfaced in the UI before a download. "Open weights" and
  "you may sell what it makes" are different claims.

### Windows specifics

- Filenames are sanitised for reserved device names (`CON`, `COM1`), forbidden
  characters, and trailing dots.
- Models live under `%LOCALAPPDATA%` and the directory is movable, because a
  model collection outgrows a system drive.
- `wmic` is not used: it is removed from current Windows 11. GPU enumeration is
  DXGI through ctypes, with a PowerShell CIM query as fallback.
  `Win32_VideoController.AdapterRAM` is a 32-bit field that saturates at 4 GB,
  so it is a last resort only.

---

## Things deliberately not done

**Training and fine-tuning.** Out of scope by request. The app runs existing
open-weight models.

**QtMultimedia for playback.** It would add a large dependency and force
writing a temporary file to audition audio already decoded in memory.
`sounddevice` plays a NumPy buffer directly.

**`torch.compile` on the GPU path.** There is no Triton for Windows, so
compiling fails outright rather than falling back. `can_compile()` returns
False on Windows and is checked before any use.

**onefile packaging.** It re-extracts on every launch, is a common antivirus
false-positive trigger, and puts Qt's DLLs somewhere a user cannot replace
them.

**YuE and MiniMax Music 3.** Recorded in the catalog's `dropped` list with
reasons: FlashAttention-2 is load-bearing for YuE's memory use and has no
practical Windows build; MiniMax's licence requires displaying its branding in
your product's UI.

---

## CI

Two test legs (Linux and Windows, Python 3.12) and a Windows build job.

The build job is the valuable one, because packaging is the part that cannot
be verified anywhere else: it asserts that the worker script survived as a real
file rather than only as bytecode, that `uv` is bundled, and that the bundle
has not quietly swallowed torch. It then **starts the built executable** and
checks it is still running twenty-five seconds later, which is how a missing
hidden import or an absent Qt platform plugin gets caught before a user finds
it. The bundle and installer are uploaded as artifacts.

`packaging/windows/build.ps1` performs the same checks locally, so a developer
build is held to the same standard as a CI one.

## Testing

Tests assert musical facts, not just that functions return. Voice leading is
checked to actually minimise movement. `bVII` in A minor is checked to be G,
not F#. The resampler is measured for aliasing. Loudness normalisation is
measured against its target. The MIDI round trip is checked to be lossless,
including the case of overlapping same-pitch notes that MIDI cannot represent.

GUI tests run headless on Qt's offscreen platform and drive a full generation
through the UI rather than poking at widget internals.
