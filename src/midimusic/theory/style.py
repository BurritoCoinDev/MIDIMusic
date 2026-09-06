"""Genre presets: the knowledge base that turns a word like "funk" into music.

A :class:`Style` binds together tempo, mode, harmony, groove, arrangement and
instrumentation.  Presets live in :data:`STYLES` but can be overridden or
extended from JSON on disk, so adding a genre never requires touching code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["Style", "STYLES", "get_style", "load_user_styles", "list_styles"]

# General MIDI program numbers, by role, for readability below.
GM = {
    "piano": 0, "bright_piano": 1, "e_piano": 4, "e_piano2": 5, "harpsichord": 6,
    "celesta": 8, "music_box": 10, "vibraphone": 11, "marimba": 12, "organ": 16,
    "rock_organ": 18, "church_organ": 19, "accordion": 21, "harmonica": 22,
    "nylon_guitar": 24, "steel_guitar": 25, "jazz_guitar": 26, "clean_guitar": 27,
    "muted_guitar": 28, "overdrive_guitar": 29, "distortion_guitar": 30,
    "acoustic_bass": 32, "finger_bass": 33, "pick_bass": 34, "fretless_bass": 35,
    "slap_bass": 36, "synth_bass": 38, "synth_bass2": 39,
    "violin": 40, "viola": 41, "cello": 42, "contrabass": 43, "harp": 46, "timpani": 47, "strings": 48, "slow_strings": 51,
    "synth_strings": 50, "choir": 52, "voice_oohs": 53, "orchestra_hit": 55,
    "trumpet": 56, "trombone": 57, "tuba": 58, "muted_trumpet": 59,
    "french_horn": 60, "brass": 61, "synth_brass": 62,
    "soprano_sax": 64, "alto_sax": 65, "tenor_sax": 66, "oboe": 68,
    "clarinet": 71, "flute": 73, "pan_flute": 75,
    "square_lead": 80, "saw_lead": 81, "calliope_lead": 82, "chiff_lead": 83,
    "voice_lead": 85, "fifths_lead": 86, "bass_lead": 87,
    "new_age_pad": 88, "warm_pad": 89, "poly_pad": 90, "choir_pad": 91,
    "bowed_pad": 92, "metallic_pad": 93, "halo_pad": 94, "sweep_pad": 95,
    "rain": 96, "soundtrack": 97, "crystal": 98, "atmosphere": 99,
    "brightness": 100, "goblins": 101, "echoes": 102, "sci_fi": 103,
    "sitar": 104, "banjo": 105, "shamisen": 106, "koto": 107, "kalimba": 108,
    "steel_drums": 114, "woodblock": 115, "taiko": 116, "melodic_tom": 117,
    "synth_drum": 118,
}


@dataclass
class Style:
    """Everything needed to arrange a song in one genre."""

    name: str
    tempo: tuple[int, int] = (100, 120)
    scales: tuple[str, ...] = ("major", "minor")
    progressions: tuple[tuple[str, ...], ...] = (("I", "V", "vi", "IV"),)
    drums: str = "pop"
    bass: str = "roots"
    comp: str = "quarters"
    form: str = "verse_chorus"
    sevenths: bool = False
    swing: float = 0.0
    humanize: float = 0.5  # 0 = machine tight, 1 = loose
    # GM programs per role.
    programs: dict[str, int] = field(default_factory=lambda: {
        "bass": GM["finger_bass"], "chords": GM["piano"], "lead": GM["saw_lead"],
        "pad": GM["warm_pad"], "arp": GM["square_lead"], "counter": GM["strings"],
    })
    # Melody character.
    melody_density: float = 0.55   # fraction of available steps used
    melody_range: tuple[int, int] = (60, 84)
    melody_leapiness: float = 0.3  # 0 = stepwise, 1 = arpeggiated leaps
    chord_octave: int = 3
    bass_octave: int = 2
    beats_per_bar: int = 4
    description: str = ""

    def merged(self, **over) -> "Style":
        d = asdict(self)
        d.update({k: v for k, v in over.items() if v is not None})
        return Style(**d)


def _st(name, **kw) -> Style:
    return Style(name=name, **kw)


STYLES: dict[str, Style] = {
    "pop": _st("pop", tempo=(100, 124), scales=("major", "minor"),
        progressions=(("I", "V", "vi", "IV"), ("vi", "IV", "I", "V"),
                      ("I", "vi", "IV", "V"), ("IV", "I", "V", "vi")),
        drums="pop", bass="eighths", comp="quarters", form="pop",
        programs={"bass": GM["finger_bass"], "chords": GM["bright_piano"],
                  "lead": GM["voice_lead"], "pad": GM["warm_pad"],
                  "arp": GM["square_lead"], "counter": GM["strings"]},
        description="Bright, hook-driven, four-chord."),

    "rock": _st("rock", tempo=(110, 145), scales=("major", "mixolydian", "minor"),
        progressions=(("I", "bVII", "IV", "I"), ("I", "IV", "V", "IV"),
                      ("vi", "IV", "I", "V"), ("I", "V", "IV", "IV")),
        drums="rock", bass="eighths", comp="eighths", form="verse_chorus",
        humanize=0.6,
        programs={"bass": GM["pick_bass"], "chords": GM["overdrive_guitar"],
                  "lead": GM["distortion_guitar"], "pad": GM["rock_organ"],
                  "arp": GM["clean_guitar"], "counter": GM["organ"]},
        melody_range=(55, 79), description="Guitar-driven, mixolydian bVII."),

    "metal": _st("metal", tempo=(140, 190), scales=("minor", "phrygian", "locrian"),
        progressions=(("i", "bVI", "bVII", "i"), ("i", "bII", "i", "bVII"),
                      ("i", "iv", "bVI", "V")),
        drums="metal", bass="driving", comp="eighths", form="verse_chorus",
        humanize=0.25,
        programs={"bass": GM["pick_bass"], "chords": GM["distortion_guitar"],
                  "lead": GM["distortion_guitar"], "pad": GM["synth_strings"],
                  "arp": GM["overdrive_guitar"], "counter": GM["strings"]},
        melody_range=(52, 76), chord_octave=2, bass_octave=1,
        description="Dark, fast, palm-muted."),

    "punk": _st("punk", tempo=(160, 200), scales=("major", "mixolydian"),
        progressions=(("I", "IV", "V", "V"), ("I", "V", "vi", "IV"), ("I", "bVII", "IV", "I")),
        drums="punk", bass="driving", comp="eighths", form="verse_chorus",
        humanize=0.7,
        programs={"bass": GM["pick_bass"], "chords": GM["distortion_guitar"],
                  "lead": GM["overdrive_guitar"], "pad": GM["rock_organ"],
                  "arp": GM["clean_guitar"], "counter": GM["organ"]},
        description="Fast, loud, three chords."),

    "funk": _st("funk", tempo=(96, 116), scales=("dorian", "minor_pentatonic", "mixolydian"),
        progressions=(("i7", "i7", "IV7", "i7"), ("i7", "bVII", "i7", "i7"),
                      ("i7", "iv7", "i7", "V7")),
        drums="funk", bass="funk", comp="stabs", form="verse_chorus",
        sevenths=True, swing=0.12, humanize=0.65,
        programs={"bass": GM["slap_bass"], "chords": GM["e_piano"],
                  "lead": GM["alto_sax"], "pad": GM["organ"],
                  "arp": GM["muted_guitar"], "counter": GM["brass"]},
        melody_leapiness=0.45, description="Syncopated, one-chord vamps."),

    "jazz": _st("jazz", tempo=(110, 160), scales=("major", "dorian", "melodic_minor"),
        progressions=(("ii7", "V7", "Imaj7", "Imaj7"), ("Imaj7", "vi7", "ii7", "V7"),
                      ("iii7", "vi7", "ii7", "V7"), ("Imaj7", "bIII7", "ii7", "V7")),
        drums="jazz", bass="walking", comp="charleston", form="aaba",
        sevenths=True, swing=0.62, humanize=0.75,
        programs={"bass": GM["acoustic_bass"], "chords": GM["piano"],
                  "lead": GM["tenor_sax"], "pad": GM["vibraphone"],
                  "arp": GM["jazz_guitar"], "counter": GM["muted_trumpet"]},
        melody_leapiness=0.5, description="Swing feel, ii-V-I."),

    "blues": _st("blues", tempo=(80, 120), scales=("blues", "minor_pentatonic", "mixolydian"),
        progressions=(("I7", "I7", "I7", "I7", "IV7", "IV7", "I7", "I7",
                       "V7", "IV7", "I7", "V7"),),
        drums="blues", bass="walking", comp="quarters", form="twelve_bar",
        sevenths=True, swing=0.6, humanize=0.8,
        programs={"bass": GM["acoustic_bass"], "chords": GM["rock_organ"],
                  "lead": GM["overdrive_guitar"], "pad": GM["organ"],
                  "arp": GM["clean_guitar"], "counter": GM["harmonica"]},
        melody_range=(55, 79), description="Twelve-bar, shuffle."),

    "hiphop": _st("hiphop", tempo=(80, 96), scales=("minor", "dorian", "minor_pentatonic"),
        progressions=(("i", "bVI", "bIII", "bVII"), ("i7", "iv7", "i7", "i7"),
                      ("i", "bVII", "bVI", "V")),
        drums="hiphop", bass="roots", comp="sustained", form="verse_chorus",
        sevenths=True, swing=0.18, humanize=0.5,
        programs={"bass": GM["synth_bass"], "chords": GM["e_piano"],
                  "lead": GM["vibraphone"], "pad": GM["warm_pad"],
                  "arp": GM["kalimba"], "counter": GM["strings"]},
        melody_density=0.4, description="Boom-bap, sampled feel."),

    "trap": _st("trap", tempo=(130, 150), scales=("minor", "phrygian", "harmonic_minor"),
        progressions=(("i", "bVI", "bVII", "i"), ("i", "iv", "i", "bVII")),
        drums="trap", bass="roots", comp="sustained", form="verse_chorus",
        humanize=0.2,
        programs={"bass": GM["synth_bass2"], "chords": GM["celesta"],
                  "lead": GM["square_lead"], "pad": GM["halo_pad"],
                  "arp": GM["music_box"], "counter": GM["choir"]},
        melody_density=0.35, bass_octave=1, description="808s, hi-hat rolls."),

    "lofi": _st("lofi", tempo=(70, 88), scales=("dorian", "major", "minor"),
        progressions=(("ii7", "V7", "Imaj7", "vi7"), ("Imaj7", "vi7", "ii7", "V7"),
                      ("i7", "iv7", "bVII", "bIII")),
        drums="lofi", bass="roots", comp="half", form="loop",
        sevenths=True, swing=0.22, humanize=0.85,
        programs={"bass": GM["acoustic_bass"], "chords": GM["e_piano"],
                  "lead": GM["vibraphone"], "pad": GM["warm_pad"],
                  "arp": GM["music_box"], "counter": GM["nylon_guitar"]},
        melody_density=0.35, description="Warm, hazy, jazzy loops."),

    "house": _st("house", tempo=(120, 128), scales=("minor", "dorian", "major"),
        progressions=(("i7", "iv7", "bVII", "bIII"), ("i", "bVI", "bIII", "bVII"),
                      ("ii7", "V7", "Imaj7", "Imaj7")),
        drums="house", bass="house", comp="offbeat", form="electronic",
        sevenths=True, humanize=0.15,
        programs={"bass": GM["synth_bass"], "chords": GM["e_piano"],
                  "lead": GM["saw_lead"], "pad": GM["poly_pad"],
                  "arp": GM["square_lead"], "counter": GM["synth_strings"]},
        description="Four-on-the-floor, offbeat stabs."),

    "techno": _st("techno", tempo=(128, 140), scales=("minor", "phrygian", "locrian"),
        progressions=(("i", "i", "bVI", "bVII"), ("i", "i", "i", "i")),
        drums="techno", bass="driving", comp="stabs", form="electronic",
        humanize=0.08,
        programs={"bass": GM["synth_bass2"], "chords": GM["saw_lead"],
                  "lead": GM["sci_fi"], "pad": GM["sweep_pad"],
                  "arp": GM["square_lead"], "counter": GM["metallic_pad"]},
        melody_density=0.3, description="Hypnotic, minimal, driving."),

    "dnb": _st("dnb", tempo=(170, 178), scales=("minor", "dorian", "harmonic_minor"),
        progressions=(("i", "bVI", "bVII", "i"), ("i7", "iv7", "bVII", "bIII")),
        drums="dnb", bass="roots", comp="stabs", form="electronic",
        sevenths=True, humanize=0.2,
        programs={"bass": GM["synth_bass2"], "chords": GM["poly_pad"],
                  "lead": GM["saw_lead"], "pad": GM["halo_pad"],
                  "arp": GM["square_lead"], "counter": GM["atmosphere"]},
        bass_octave=1, description="Breakbeats at 174, sub bass."),

    "synthwave": _st("synthwave", tempo=(100, 118), scales=("minor", "dorian"),
        progressions=(("i", "bVI", "bIII", "bVII"), ("i", "bVII", "bVI", "V"),
                      ("vi", "IV", "I", "V")),
        drums="synthwave", bass="octaves", comp="arpeggio", form="electronic",
        humanize=0.2,
        programs={"bass": GM["synth_bass"], "chords": GM["poly_pad"],
                  "lead": GM["saw_lead"], "pad": GM["sweep_pad"],
                  "arp": GM["square_lead"], "counter": GM["synth_brass"]},
        description="Neon, 80s, arpeggiated."),

    "disco": _st("disco", tempo=(112, 126), scales=("major", "dorian", "minor"),
        progressions=(("i7", "iv7", "bVII", "bIII"), ("Imaj7", "vi7", "ii7", "V7")),
        drums="disco", bass="octaves", comp="offbeat", form="verse_chorus",
        sevenths=True, humanize=0.35,
        programs={"bass": GM["finger_bass"], "chords": GM["clean_guitar"],
                  "lead": GM["brass"], "pad": GM["strings"],
                  "arp": GM["e_piano"], "counter": GM["strings"]},
        description="Four-on-the-floor, octave bass, strings."),

    "reggae": _st("reggae", tempo=(70, 92), scales=("major", "minor", "dorian"),
        progressions=(("I", "V", "vi", "IV"), ("i", "bVII", "i", "bVII")),
        drums="reggae", bass="reggae", comp="offbeat", form="verse_chorus",
        humanize=0.6,
        programs={"bass": GM["finger_bass"], "chords": GM["clean_guitar"],
                  "lead": GM["organ"], "pad": GM["organ"],
                  "arp": GM["muted_guitar"], "counter": GM["trombone"]},
        description="Offbeat skank, deep bass."),

    "country": _st("country", tempo=(96, 132), scales=("major", "mixolydian"),
        progressions=(("I", "IV", "I", "V"), ("I", "V", "vi", "IV"), ("I", "IV", "V", "I")),
        drums="country", bass="root_fifth", comp="strum", form="verse_chorus",
        humanize=0.6,
        programs={"bass": GM["acoustic_bass"], "chords": GM["steel_guitar"],
                  "lead": GM["steel_guitar"], "pad": GM["strings"],
                  "arp": GM["banjo"], "counter": GM["violin"]},
        description="Acoustic, I-IV-V, twang."),

    "folk": _st("folk", tempo=(88, 120), scales=("major", "dorian", "minor"),
        progressions=(("I", "IV", "I", "V"), ("i", "bVII", "bVI", "bVII"), ("I", "V", "vi", "IV")),
        drums="country", bass="root_fifth", comp="strum", form="verse_chorus",
        humanize=0.7,
        programs={"bass": GM["acoustic_bass"], "chords": GM["nylon_guitar"],
                  "lead": GM["flute"], "pad": GM["strings"],
                  "arp": GM["nylon_guitar"], "counter": GM["violin"]},
        description="Fingerpicked, modal, acoustic."),

    "bossa": _st("bossa", tempo=(120, 140), scales=("major", "dorian", "melodic_minor"),
        progressions=(("Imaj7", "ii7", "V7", "Imaj7"), ("ii7", "V7", "Imaj7", "vi7")),
        drums="bossa", bass="root_fifth", comp="charleston", form="aaba",
        sevenths=True, humanize=0.55,
        programs={"bass": GM["acoustic_bass"], "chords": GM["nylon_guitar"],
                  "lead": GM["flute"], "pad": GM["vibraphone"],
                  "arp": GM["nylon_guitar"], "counter": GM["soprano_sax"]},
        description="Brazilian, gentle, jazzy sevenths."),

    "latin": _st("latin", tempo=(96, 130), scales=("minor", "phrygian_dominant", "harmonic_minor"),
        progressions=(("i", "bVII", "bVI", "V"), ("i", "iv", "V", "i")),
        drums="latin", bass="syncopated", comp="offbeat", form="verse_chorus",
        humanize=0.55,
        programs={"bass": GM["acoustic_bass"], "chords": GM["nylon_guitar"],
                  "lead": GM["trumpet"], "pad": GM["strings"],
                  "arp": GM["nylon_guitar"], "counter": GM["brass"]},
        description="Clave-driven, Spanish cadence."),

    "ambient": _st("ambient", tempo=(60, 80), scales=("lydian", "major", "dorian"),
        progressions=(("Imaj7", "IVmaj7", "Imaj7", "vi7"), ("Imaj7", "iii7", "IVmaj7", "Imaj7")),
        drums="ambient", bass="pedal", comp="pad", form="ambient",
        sevenths=True, humanize=0.9,
        programs={"bass": GM["warm_pad"], "chords": GM["new_age_pad"],
                  "lead": GM["crystal"], "pad": GM["atmosphere"],
                  "arp": GM["celesta"], "counter": GM["halo_pad"]},
        melody_density=0.2, melody_leapiness=0.15, description="Slow, drifting, textural."),

    "cinematic": _st("cinematic", tempo=(80, 110), scales=("minor", "dorian", "harmonic_minor"),
        progressions=(("i", "bVI", "bIII", "bVII"), ("i", "iv", "bVI", "V"),
                      ("i", "bVII", "bVI", "bVII")),
        drums="cinematic", bass="pedal", comp="sustained", form="cinematic",
        humanize=0.5,
        programs={"bass": GM["cello"], "chords": GM["slow_strings"],
                  "lead": GM["french_horn"], "pad": GM["choir"],
                  "arp": GM["harp"],
                  "counter": GM["violin"]},
        melody_density=0.35, description="Orchestral, building, epic."),

    "classical": _st("classical", tempo=(72, 132), scales=("major", "minor", "harmonic_minor"),
        progressions=(("I", "IV", "V", "I"), ("I", "vi", "ii", "V"),
                      ("i", "iv", "V", "i"), ("I", "V/V", "V", "I")),
        drums="ambient", bass="root_fifth", comp="arpeggio", form="aaba",
        humanize=0.7,
        programs={"bass": GM["cello"], "chords": GM["piano"],
                  "lead": GM["violin"], "pad": GM["strings"],
                  "arp": GM["piano"], "counter": GM["flute"]},
        melody_leapiness=0.35, description="Tonal, functional harmony."),

    "chiptune": _st("chiptune", tempo=(130, 170), scales=("major", "minor", "lydian"),
        progressions=(("I", "V", "vi", "IV"), ("i", "bVI", "bIII", "bVII")),
        drums="punk", bass="octaves", comp="arp_16", form="loop",
        humanize=0.05,
        programs={"bass": GM["synth_bass"], "chords": GM["square_lead"],
                  "lead": GM["square_lead"], "pad": GM["saw_lead"],
                  "arp": GM["chiff_lead"], "counter": GM["calliope_lead"]},
        melody_density=0.7, description="8-bit, fast arpeggios."),
}

_USER_STYLES: dict[str, Style] = {}


def list_styles() -> list[str]:
    return sorted(set(STYLES) | set(_USER_STYLES))


def get_style(name: str) -> Style:
    """Look up a style by name, falling back to pop for unknown genres."""
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _USER_STYLES.get(key) or STYLES.get(key) or STYLES["pop"]


def load_user_styles(path: Path) -> int:
    """Load or override styles from a JSON file. Returns how many were loaded.

    The file maps style name to a partial :class:`Style`; missing fields are
    inherited from the built-in style of the same name, or from pop.
    """
    if not path.exists():
        return 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    count = 0
    for name, over in (raw or {}).items():
        if not isinstance(over, dict):
            continue
        key = name.strip().lower().replace(" ", "_")
        base = STYLES.get(key, STYLES["pop"])
        try:
            for tup in ("tempo", "scales", "melody_range"):
                if tup in over and isinstance(over[tup], list):
                    over[tup] = tuple(over[tup])
            if "progressions" in over:
                over["progressions"] = tuple(tuple(p) for p in over["progressions"])
            merged = base.merged(**over)
            merged.name = key
            _USER_STYLES[key] = merged
            count += 1
        except (TypeError, ValueError):
            continue
    return count
