# MIDIMusic

[![CI](https://github.com/BurritoCoinDev/MIDIMusic/actions/workflows/ci.yml/badge.svg)](https://github.com/BurritoCoinDev/MIDIMusic/actions/workflows/ci.yml)

A Windows desktop application that generates music locally with open-weight
models, and exports **MIDI** and **FLAC**. Nothing is uploaded, nothing is
metered, and it keeps working offline.

It is what Suno does, on your own machine, with weights you control.

---

## Why two pipelines

This matters more than any other design decision here, so it is worth being
blunt about it: **audio models and MIDI models are different models.**

Waveform models such as ACE-Step and MusicGen generate sound. There are no
notes inside them to export — asking one for a MIDI file is asking a
photograph for its brush strokes. Symbolic models generate notes directly, and
their output is a real score you can open in a DAW and edit.

So MIDIMusic runs both:

| You want | Path |
|---|---|
| Finished audio, optionally with vocals | Audio models → FLAC |
| An editable score | The built-in composer or a symbolic model → MIDI |
| Audio *and* a rough score | Audio model → FLAC, then transcription → MIDI |

The transcription bridge (basic-pitch) is honest about what it is: good on
sparse or solo material, rough on dense mixes. It is a starting point, not a
faithful decomposition.

---

## It works before you download anything

Most local-AI apps are useless until a multi-gigabyte download finishes. This
one ships a **music-theory composer** that runs instantly on any machine with
no model weights, no GPU and no network.

It is not a random note generator. It knows scales and modes, builds chords by
roman numeral, voice-leads between them so block chords move by a semitone or
two instead of jumping around, writes melodies as a motif that gets developed
across the song rather than sampled bar by bar, and arranges up to fourteen
parts across a real song form — with the harmonic rhythm subdividing, later
choruses varying, and a key lift into the final repeat.

Neural backends are an upgrade on top of that floor, never a prerequisite.

---

## Models

The catalog is a JSON file. Adding a model is a config change, not a code
change — and there is an **Add custom model** dialog for pointing the app at
any Hugging Face repo one of the existing adapters can load.

| Model | Kind | Licence | Notes |
|---|---|---|---|
| Built-in Composer | MIDI | MIT | No download, no GPU, instant |
| **ACE-Step 1.5** | Audio | MIT | Full songs with vocals. The flagship |
| text2midi | MIDI | Apache-2.0 | Free text to score |
| Anticipatory Music Transformer | MIDI | Apache-2.0 | Generation and infilling |
| MusicGen (small/medium) | Audio | **CC-BY-NC** | 30s cap, no vocals, very portable |
| Stable Audio Open | Audio | Stability Community | Gated; loops and textures |
| DiffRhythm | Audio | Mixed — see notes | Fast full songs; VAE licence is unclear |

Licences are shown in the app **before** a download starts, because "open
weights" and "you may ship what it makes" are not the same claim. MusicGen's
weights are non-commercial. Stable Audio's licence has a revenue ceiling.
DiffRhythm badges Apache-2.0 on the generator while requiring a VAE under a
different licence entirely.

Two models are deliberately **not** included, with reasons recorded in the
catalog: YuE (FlashAttention-2 is load-bearing for memory and has no practical
Windows build) and MiniMax Music 3 (its licence requires displaying MiniMax
branding in your product's UI).

---

## AMD, NVIDIA, Intel, or nothing

**AMD users get real GPU acceleration on Windows.** This is recent and widely
misreported, so to be specific: AMD now publishes native Windows ROCm PyTorch
wheels, with gfx1100 (RX 7900 XT/XTX) as a named target. You do not need WSL2.
You do not need DirectML — `torch-directml` has been unmaintained since 2024
and pins torch 2.4. You do not need ZLUDA, whose own documentation still lists
PyTorch support as unfinished.

The app detects your card's PCI device ID, resolves the LLVM target from it,
and installs the matching build — a 7800 XT gets the gfx1101 wheels, not the
gfx1100 ones.

| Hardware | Backend |
|---|---|
| AMD RDNA3/RDNA4 on Windows | ROCm for Windows |
| NVIDIA | CUDA |
| Intel Arc | XPU |
| Anything else, or no GPU | CPU |

After installing a runtime the app **measures** it — running a matmul and an
attention call in the new environment — rather than trusting that
`cuda.is_available()` means the kernels work.

### Models run in a separate process

Neural backends do not run inside the application. They run in a provisioned
runtime, driven over a small JSON protocol. Three reasons:

1. The installer stays around 120 MB instead of several gigabytes.
2. One build serves CPU, CUDA and ROCm users — the GPU-specific wheels live in
   the runtime, not the bundle.
3. A native crash in a GPU stack fails one generation instead of taking your
   session and your unsaved queue with it.

---

## Installing

### From the installer

Download the latest `MIDIMusic-x.y.z-Setup.exe` from Releases and run it. It
installs per-user, so there is no administrator prompt.

**You do not need Python installed.** The installer carries its own Python
runtime, Qt, the audio libraries, and `uv` — everything needed to run the app
and to build the separate environment that neural models use. There is no
prerequisite, no PATH surgery, and no "install Python 3.12 first".

What the installer does *not* carry is PyTorch (about 4 GB, and which build is
correct depends on your GPU) or model weights (2–9 GB each). Those are fetched
on demand from the **Models** tab, which builds the environment for you. The
built-in composer needs none of it and works the moment the installer finishes,
offline.

### From source

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m midimusic
```

Python 3.12 is the target. Everything in the core dependency set has a
prebuilt Windows wheel: no compiler, no ffmpeg, no external DLLs.

---

## Using it

**Compose** — describe what you want. The prompt is parsed for genre, key,
tempo, mood and length, and it shows you its reading so you can correct it.
Anything you set explicitly overrides the prompt. Complexity is a dial from a
sparse trio to a fully-produced arrangement.

**Queue** — generation runs one job at a time in the background, with live
progress and a cancel button. The window stays responsive.

**Library** — everything you have made, with a waveform you can click to seek,
playback, and reveal-in-folder.

**Models** — what is installed, what it costs to download, what licence it
carries, and your compute runtime.

**Settings** — output folder, model folder (movable to another drive), sample
rate, bit depth, loudness target, and a Hugging Face token for gated models.

---

## What is bundled versus fetched

| | Where it lives |
|---|---|
| The app, Qt, Python runtime, audio libraries | In the installer (~140 MB) |
| `uv`, which builds the model environment | In the installer |
| Built-in composer | In the installer — works offline, immediately |
| PyTorch for your GPU (~4 GB) | Fetched on request, into a separate environment |
| Model weights (2–9 GB each) | Fetched on request, licence shown first |
| SoundFont for nicer MIDI playback (38 MB) | Optional, offered in Settings |

The split is deliberate: bundling every GPU variant of PyTorch would make a
multi-gigabyte installer that is wrong for most people who download it.

---

## Output

FLAC is written at 16 or 24-bit through libsndfile, loudness-normalised to a
target you choose (−14 LUFS by default, the streaming convention), with
Vorbis comments carrying the title, genre, prompt and seed.

MIDI is written as a type-1 file with tempo, time signature, key signature,
track names, program changes and section markers — so it lands in a DAW as a
usable arrangement rather than one anonymous blob of notes.

Every generation records its seed. The same seed and settings reproduce the
same piece.

---

## Building the Windows installer

CI builds this on every push to the default branch and uploads it, so the
quickest way to get a copy is the **Actions** tab → the latest run →
**MIDIMusic-windows**. To build it yourself:

```powershell
cd packaging\windows
.\build.ps1
```

Produces a PyInstaller `onedir` bundle and, if Inno Setup is present, an
installer in `packaging\windows\Output`. The script verifies the result
before it hands it to you: that the worker script survived as a real file,
that `uv` is bundled, that the package imports, and that the bundle has not
quietly swallowed torch.

Pass `-SkipInstaller` to build just the application folder.

Unsigned builds trip SmartScreen until they accumulate reputation. Signing
helps, but note that since March 2024 an EV certificate no longer grants
instant reputation — OV and EV now accrue it identically.

---

## Development

```bash
pytest              # the full suite, headless
ruff check src/     # lint
```

CI runs the suite on Linux and Windows, then builds the Windows bundle,
verifies it (worker script present as a real file, `uv` bundled, size sane),
launches the built `.exe` to confirm it does not die on startup, and uploads
the result as an artifact.

Tests cover the theory engine as music (voice leading really does minimise
movement; `bVII` in A minor really is G), the audio pipeline (the resampler
is measured for aliasing, loudness normalisation for accuracy), the MIDI round
trip, the job queue, and the GUI headlessly via Qt's offscreen platform.

---

## Licence

MIT — see [LICENSE](LICENSE).

The app links Qt via PySide6 (LGPLv3) and libsndfile (LGPL-2.1+) as
replaceable dynamic libraries, which is why the build is `onedir` rather than
`onefile`. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

**Model weights carry their own licences, which are not MIT and are not all
commercial-friendly.** The app shows each licence before you download. What
you may do with the music a model produces is governed by that model's terms,
not by this one.
