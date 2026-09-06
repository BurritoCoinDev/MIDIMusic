# Third-party licences

MIDIMusic is MIT licensed. It links and ships other people's work, and this
file records what and under which terms.

## Libraries in the application bundle

| Component | Licence | How it is used |
|---|---|---|
| Qt 6 via PySide6-Essentials | **LGPLv3** | GUI toolkit. Dynamically linked, replaceable |
| libsndfile (inside `soundfile`) | **LGPL-2.1+** | FLAC/WAV encode and decode, and tagging |
| PortAudio (inside `sounddevice`) | MIT | Audio playback |
| TinySoundFont (inside `tinysoundfont`) | MIT | Offline SoundFont rendering |
| NumPy | BSD-3-Clause | Signal processing |
| mido | MIT | MIDI file reading and writing |
| pyloudnorm | MIT | ITU-R BS.1770 loudness measurement |
| platformdirs | MIT | Per-platform application directories |

### LGPL compliance

Two dependencies are LGPL: **Qt/PySide6** (LGPLv3) and **libsndfile**
(LGPL-2.1+). Both are dynamically linked and shipped as ordinary replaceable
files. This drives several concrete build decisions:

1. **The build is `onedir`, never `onefile`.** A onefile bundle unpacks itself
   into a temporary directory at each launch, which leaves the user with no
   practical way to substitute their own build of Qt or libsndfile. Onedir
   keeps `PySide6/`, the `Qt6*.dll` files and `_soundfile_data/` as loose
   files that can be replaced.
2. **Nothing is statically linked, obfuscated, or integrity-checked** in a way
   that would prevent substituting those libraries.
3. **Source for the exact versions shipped** is offered alongside each release.
4. The application's Help menu names these libraries and their licences.

### Deliberate exclusions

Two libraries were removed rather than shipped, and this is worth recording so
they do not quietly return:

- **mutagen** (GPL-2.0-or-later) was used for FLAC tagging. GPL is not
  compatible with shipping an MIT application, and libsndfile writes Vorbis
  comments natively, so mutagen was unnecessary as well as unwanted.
- **soxr** is optional rather than required. Its wheel statically links
  libsoxr (LGPL-2.1), and static linking brings relinking obligations that
  dynamic linking does not. The core resampler is a windowed-sinc polyphase
  implementation in NumPy; install `midimusic[resample]` to use soxr instead.

## Optional runtime components

These are installed by the user, into a separate environment, and are not part
of the application bundle.

| Component | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| Hugging Face Transformers | Apache-2.0 |
| Diffusers | Apache-2.0 |
| basic-pitch | Apache-2.0 |
| ONNX Runtime | MIT |

## Model weights

**Model weights are licensed separately from this application, and not all of
them permit commercial use.** The app displays each licence before a download
begins.

| Model | Weights licence | Commercial use |
|---|---|---|
| ACE-Step 1.5 | MIT | Yes |
| text2midi | Apache-2.0 | Yes |
| Anticipatory Music Transformer | Apache-2.0 | Yes (trained on Lakh MIDI, CC-BY 4.0) |
| MIDI Composer (skytnt) | Apache-2.0 | Yes |
| MusicGen | **CC-BY-NC-4.0** | **No** — non-commercial only |
| Stable Audio Open | Stability Community Licence | Conditional, below a revenue threshold |
| DiffRhythm | Mixed | Unclear — generators badge Apache-2.0, but the required VAE is under the Stability Community Licence and one release declares no licence at all |

What you may do with generated music is governed by the licence of the model
that produced it. That is a question about the weights, not about MIDIMusic.

## SoundFonts

MuseScore General (MIT) is offered as an optional download. It is fetched from
its upstream host at your request and is not redistributed in the installer.
