"""Audio export, rendering and the MIDI round trip."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import soundfile as sf

from midimusic.audio import dsp
from midimusic.audio.export import ExportOptions, TrackMetadata, export_audio, safe_filename
from midimusic.audio.midi_io import midi_to_song, write_midi
from midimusic.audio.synth_fallback import render_song_fallback
from midimusic.core.models import AudioBuffer, Note, OutputFormat, Song, Track
from midimusic.theory.composer import compose


@pytest.fixture
def tone():
    rate = 44100
    t = np.linspace(0, 2.0, rate * 2, endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioBuffer(np.stack([mono, mono], axis=1), rate)


class TestDsp:
    def test_to_stereo_handles_every_shape(self):
        assert dsp.to_stereo(np.zeros(100, np.float32)).shape == (100, 2)
        assert dsp.to_stereo(np.zeros((100, 1), np.float32)).shape == (100, 2)
        assert dsp.to_stereo(np.zeros((100, 4), np.float32)).shape == (100, 2)

    def test_resampler_produces_the_right_length(self):
        x = np.zeros((32000, 2), np.float32)
        assert dsp.resample(x, 32000, 44100).shape[0] == 44100

    def test_resampler_preserves_pitch_without_aliasing(self):
        # A 440 Hz tone converted 32k -> 44.1k must still be 440 Hz, and the
        # next-largest spectral peak must be far below it.
        t = np.linspace(0, 1, 32000, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = dsp.resample(tone, 32000, 44100)
        spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        peak_hz = int(np.argmax(spectrum)) * 44100 / len(out)
        assert abs(peak_hz - 440) < 2
        sidelobe = np.sort(spectrum)[:-8].max() / spectrum.max()
        assert 20 * np.log10(sidelobe) < -60

    def test_loudness_normalisation_hits_the_target(self, tone):
        out = dsp.loudness_normalize(tone.samples, tone.sample_rate, -14.0)
        assert abs(dsp.measure(out, tone.sample_rate)["lufs"] - (-14.0)) < 0.6

    def test_normalisation_never_clips(self, tone):
        loud = (tone.samples * 8).astype(np.float32)
        out = dsp.loudness_normalize(loud, tone.sample_rate, -6.0)
        assert float(np.abs(out).max()) <= 1.0

    def test_fades_silence_the_edges(self, tone):
        out = dsp.apply_fades(tone.samples, tone.sample_rate, 0.05, 0.05)
        assert abs(float(out[0].max())) < 1e-3
        assert abs(float(out[-1].max())) < 1e-3

    def test_dither_stays_inside_the_bit_depth(self, tone):
        out = dsp.dither_to_int(tone.samples, 16)
        assert out.min() >= -32768 and out.max() <= 32767

    def test_peak_envelope_shape(self, tone):
        env = dsp.peak_envelope(tone.samples, 100)
        assert env.shape == (100, 2)
        assert (env[:, 0] <= env[:, 1]).all()

    def test_empty_input_is_survivable(self):
        empty = np.zeros((0, 2), np.float32)
        assert dsp.resample(empty, 44100, 48000).shape[0] == 0
        assert dsp.measure(empty, 44100)["peak_db"] == -np.inf


class TestExport:
    def test_flac_is_written_and_readable(self, tone, tmp_path):
        path = export_audio(tone, tmp_path / "a.flac", OutputFormat.FLAC)
        assert path.exists()
        data, rate = sf.read(str(path))
        assert rate == 44100 and data.shape[1] == 2

    def test_tags_round_trip_without_a_gpl_library(self, tone, tmp_path):
        meta = TrackMetadata(title="Night Drive", artist="MIDIMusic",
                             album="Gen", genre="synthwave", comment="dark")
        path = export_audio(tone, tmp_path / "t.flac", OutputFormat.FLAC,
                            ExportOptions(), meta)
        with sf.SoundFile(str(path)) as handle:
            assert handle.title == "Night Drive"
            assert handle.artist == "MIDIMusic"
            assert handle.genre == "synthwave"

    @pytest.mark.parametrize("depth", [16, 24])
    def test_bit_depths(self, tone, tmp_path, depth):
        path = export_audio(tone, tmp_path / f"d{depth}.flac", OutputFormat.FLAC,
                            ExportOptions(bit_depth=depth))
        assert sf.info(str(path)).subtype == f"PCM_{depth}"

    def test_resamples_to_the_requested_rate(self, tone, tmp_path):
        path = export_audio(tone, tmp_path / "r.flac", OutputFormat.FLAC,
                            ExportOptions(sample_rate=48000))
        assert sf.info(str(path)).samplerate == 48000

    @pytest.mark.parametrize("name,expected", [
        ("normal name", "normal name"),
        ('bad<>:"/\\|?*chars', "bad_________chars"),
        ("CON", "untitled"),          # a reserved Windows device name
        ("trailing dots...", "trailing dots"),
        ("", "untitled"),
    ])
    def test_filenames_are_safe_on_windows(self, name, expected):
        assert safe_filename(name) == expected


class TestMidiIo:
    def test_round_trip_loses_nothing(self, tmp_path):
        song = compose("jazz", "Bb major", duration_seconds=30, seed=3, complexity=0.9)
        path = write_midi(song, tmp_path / "s.mid")
        assert midi_to_song(path).note_count == song.note_count

    def test_overlapping_same_pitch_notes_are_clipped_not_dropped(self, tmp_path):
        # Two of the same pitch overlapping on one channel cannot both sound in
        # MIDI, so the writer shortens the first rather than losing one.
        track = Track(name="T", channel=0)
        track.notes = [Note(60, 0.0, 2.0, 100, 0), Note(60, 1.0, 2.0, 100, 0)]
        song = Song(tracks=[track], tempo=120)
        back = midi_to_song(write_midi(song, tmp_path / "o.mid"))
        assert back.note_count == 2
        first, second = sorted(back.tracks[0].notes, key=lambda n: n.start)
        assert first.end <= second.start + 1e-3

    def test_tempo_and_metadata_survive(self, tmp_path):
        song = compose("techno", "F# minor", tempo=137.0, duration_seconds=20, seed=1)
        back = midi_to_song(write_midi(song, tmp_path / "m.mid"))
        assert abs(back.tempo - 137.0) < 0.5
        assert len(back.tracks) == len(song.tracks)


class TestFallbackSynth:
    def test_always_produces_audible_output(self):
        song = compose("pop", "C major", duration_seconds=10, seed=2)
        buffer = render_song_fallback(song)
        assert buffer.frames > 0
        assert float(np.abs(buffer.samples).max()) > 0.01

    def test_never_clips(self):
        song = compose("metal", "E minor", duration_seconds=10, seed=2, complexity=1.0)
        buffer = render_song_fallback(song)
        assert float(np.abs(buffer.samples).max()) <= 1.0

    def test_empty_song_does_not_crash(self):
        assert render_song_fallback(Song(tempo=120)).frames >= 0


class TestDependencyHygiene:
    """Guards against dependencies that break a plain install.

    A core dependency that needs a C compiler turns `pip install` into a build,
    and the failure lands before any of this code runs. tinysoundfont drags in
    pyaudio, which has no wheels outside Windows, so it is declared with a
    platform marker rather than unconditionally.
    """

    def test_compiler_requiring_deps_are_windows_only(self):
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        core = data["project"]["dependencies"]

        needs_a_compiler_elsewhere = ("tinysoundfont",)
        for name in needs_a_compiler_elsewhere:
            matches = [d for d in core if d.startswith(name)]
            assert matches, f"{name} disappeared from the core dependencies"
            for dep in matches:
                assert "sys_platform" in dep, (
                    f"{name} must carry a platform marker; without one a plain "
                    "pip install fails on Linux and macOS trying to build pyaudio"
                )

    def test_the_app_produces_audio_without_a_soundfont_renderer(self, tmp_path, monkeypatch):
        # The renderer is optional. Losing it must cost quality, not function.
        from midimusic.audio import render

        monkeypatch.setattr(render, "is_available", lambda: False)

        from midimusic.audio.synth_fallback import render_song_fallback
        from midimusic.theory.composer import compose

        song = compose("lofi", "D dorian", duration_seconds=10, seed=1)
        buffer = render_song_fallback(song)
        path = export_audio(buffer, tmp_path / "fallback.flac", OutputFormat.FLAC)
        assert path.exists() and path.stat().st_size > 1000


class TestAnalysis:
    """Tempo and key estimation.

    These are estimates, so the assertions are about being usefully close and
    honestly calibrated rather than exactly right. Key detection in particular
    cannot fully separate a key from its relative -- they contain identical
    notes -- which is why the estimator reports alternatives.
    """

    @staticmethod
    def _render(style: str, key: str, tempo: float):
        from midimusic.audio.synth_fallback import render_song_fallback
        from midimusic.theory.composer import compose

        # modulate=False: the final-chorus key lift would put the piece in two
        # keys, which is not what a key estimate is being asked about.
        song = compose(style, key, tempo=tempo, duration_seconds=22, seed=3,
                       complexity=0.6, modulate=False)
        return render_song_fallback(song)

    @pytest.mark.parametrize("style,tempo", [("pop", 120), ("lofi", 85), ("jazz", 140)])
    def test_tempo_is_close(self, style, tempo):
        from midimusic.audio.analyze import estimate_tempo

        buffer = self._render(style, "C major", tempo)
        found, confidence = estimate_tempo(buffer.samples, buffer.sample_rate)
        assert abs(found - tempo) < 4, f"expected ~{tempo}, got {found}"
        assert 0.0 <= confidence <= 1.0

    def test_key_lands_in_the_right_neighbourhood(self):
        from midimusic.audio.analyze import estimate_key

        buffer = self._render("pop", "C major", 120)
        key, confidence, chroma, alternatives = estimate_key(
            buffer.samples, buffer.sample_rate
        )
        assert key, "no key was estimated"
        assert len(chroma) == 12
        assert 0.0 <= confidence <= 1.0
        # C major, its relative A minor and its parallel C minor are all
        # defensible readings of this material; anything else is a real miss.
        assert key in {"C major", "A minor", "C minor"}, key
        assert alternatives and alternatives[0][0] == key

    def test_analysis_degrades_rather_than_raising(self):
        from midimusic.audio.analyze import analyze_audio

        tiny = np.zeros((64, 2), dtype=np.float32)
        result = analyze_audio(tiny, 44100)
        assert result.tempo == 0.0 and result.key == ""

    def test_describe_flags_an_ambiguous_key(self):
        from midimusic.audio.analyze import AudioAnalysis

        confident = AudioAnalysis(tempo=120, key="C major", key_confidence=0.9,
                                  key_alternatives=[("C major", 1.0), ("A minor", 0.5)])
        # Match the parenthetical form, not a bare "or": "major" contains one.
        assert "(or " not in confident.describe()
        unsure = AudioAnalysis(tempo=120, key="C major", key_confidence=0.1,
                               key_alternatives=[("C major", 1.0), ("A minor", 0.95)])
        assert "(or A minor)" in unsure.describe()


class TestMixingHelpers:
    """The pieces that put a generated backing under a kept performance."""

    def _tone(self, seconds=2.0, rate=44100, freq=220.0, gain=0.5):
        import numpy as np

        t = np.arange(int(rate * seconds)) / rate
        wave = (np.sin(2 * np.pi * freq * t) * gain).astype("float32")
        return np.stack([wave, wave], axis=1)

    def test_a_short_bed_is_repeated_to_cover_the_track(self):
        from midimusic.audio import dsp

        rate = 44100
        short = self._tone(2.0, rate)
        out = dsp.fit_length(short, rate * 7, rate, crossfade=0.25)
        assert out.shape == (rate * 7, 2)

    def test_a_long_bed_is_trimmed(self):
        from midimusic.audio import dsp

        rate = 44100
        out = dsp.fit_length(self._tone(5.0, rate), rate * 2, rate)
        assert out.shape[0] == rate * 2

    def test_tiling_keeps_musical_time(self):
        import numpy as np

        from midimusic.audio import dsp

        # A marker at a fixed offset inside the clip. Each repetition must land
        # one clip length later; if the join eats its crossfade out of the
        # timeline instead, the copies creep earlier and a looped backing walks
        # off the beat it was aligned to.
        rate = 1000
        clip = np.zeros((2000, 1), dtype="float32")
        clip[800:805, 0] = 1.0
        out = dsp.fit_length(clip, 8000, rate)
        hits = np.flatnonzero(out[:, 0] > 0.5)
        starts = [int(p) for i, p in enumerate(hits) if i == 0 or p - hits[i - 1] > 5]
        strides = [b - a for a, b in pairwise(starts)]
        assert strides, "the clip was never repeated"
        slip = 2000 - min(strides)
        assert slip <= dsp.crossfade_frames(2000, rate) + 1, strides
        assert slip / rate < 0.05, f"{slip / rate:.3f}s lost per join"

    def test_the_reported_repeat_count_matches_the_tiling(self):
        import numpy as np

        from midimusic.audio import dsp

        rate = 1000
        clip = np.zeros((2000, 1), dtype="float32")
        clip[800:805, 0] = 1.0
        for target in (2000, 3000, 8000, 20000):
            out = dsp.fit_length(clip, target, rate)
            hits = np.flatnonzero(out[:, 0] > 0.5)
            seen = len([p for i, p in enumerate(hits)
                        if i == 0 or p - hits[i - 1] > 5])
            # Counting copies as ceil(target / len) ignores what each join
            # costs and under-reports, which is what the UI then tells the user.
            assert dsp.tiles_needed(2000, target, rate) >= seen, target

    def test_a_join_cannot_swell_past_the_material_it_splices(self):
        import numpy as np

        from midimusic.audio import dsp

        # A clip whose head and tail are identical: the crossfade then adds a
        # signal to itself, and an equal-power law sums two halves of 0.707 to
        # 1.414 -- a 3 dB swell straight past full scale. Sustained material
        # spliced into itself does exactly this whenever the phase lines up.
        rate, n = 44100, 44100
        overlap = dsp.crossfade_frames(n, rate)
        clip = (np.random.default_rng(0).normal(0, 0.2, (n, 1))).astype("float32")
        clip = np.clip(clip, -0.8, 0.8)
        clip[:overlap] = 0.8
        clip[-overlap:] = 0.8

        out = dsp.fit_length(clip, n * 3, rate)
        assert float(np.abs(out).max()) <= 0.8 + 1e-3, (
            f"the join reached {float(np.abs(out).max()):.3f} from sources of 0.8"
        )

    def test_the_joins_are_crossfaded_rather_than_cut(self):
        import numpy as np

        from midimusic.audio import dsp

        rate = 8000
        # A tone that starts and ends at very different levels: a butt splice
        # would leave a step at the join, which is what clicks.
        ramp = np.linspace(0.0, 1.0, rate, dtype="float32")[:, None].repeat(2, axis=1)
        joined = dsp.fit_length(ramp, rate * 2, rate, crossfade=0.2)
        step = float(np.abs(np.diff(joined[:, 0])).max())
        hard = float(np.abs(np.diff(np.concatenate([ramp, ramp])[:, 0])).max())
        assert step < hard

    def test_an_empty_bed_yields_silence_of_the_right_length(self):
        import numpy as np

        from midimusic.audio import dsp

        out = dsp.fit_length(np.zeros((0, 2), dtype="float32"), 500, 44100)
        assert out.shape == (500, 2) and not out.any()

    def test_loudness_matching_brings_a_quiet_bed_up(self):
        from midimusic.audio import dsp

        rate = 44100
        reference = self._tone(2.0, rate, gain=0.5)
        quiet = self._tone(2.0, rate, gain=0.12)  # about 12 dB down
        matched = dsp.match_loudness(quiet, reference, rate)
        before = dsp.measure(quiet, rate)["rms_db"]
        after = dsp.measure(matched, rate)["rms_db"]
        target = dsp.measure(reference, rate)["rms_db"]
        assert after > before
        assert abs(after - target) < 1.5

    def test_loudness_matching_will_not_shout(self):
        import numpy as np

        from midimusic.audio import dsp

        rate = 44100
        # Near silence against a loud reference would otherwise ask for an
        # enormous gain and turn the noise floor into the mix.
        silence = (np.random.default_rng(0).normal(0, 1e-7, (rate, 2))).astype("float32")
        loud = self._tone(1.0, rate, gain=0.9)
        matched = dsp.match_loudness(silence, loud, rate, max_gain_db=24.0)
        assert float(np.abs(matched).max()) < 0.01

    def test_mixing_holds_the_ceiling(self):
        import numpy as np

        from midimusic.audio import dsp

        rate = 44100
        layers = [self._tone(1.0, rate, freq=f, gain=0.6) for f in (110, 220, 330)]
        mixed = dsp.mix(layers, ceiling_db=-1.0)
        assert float(np.abs(mixed).max()) <= 10 ** (-1.0 / 20.0) + 1e-6

    def test_mixing_pads_shorter_layers(self):
        from midimusic.audio import dsp

        rate = 8000
        mixed = dsp.mix([self._tone(1.0, rate), self._tone(0.25, rate)])
        assert mixed.shape[0] == rate


class TestBeatPhase:
    """Finding where a recording's pulse falls, not just how fast it is."""

    def _clicks(self, rate, tempo, offset, seconds=12.0):
        import numpy as np

        n = int(rate * seconds)
        x = np.zeros(n, dtype="float32")
        period = 60.0 / tempo
        k = 0
        while True:
            i = int((offset + k * period) * rate)
            if i + 400 >= n:
                break
            x[i:i + 400] = np.sin(2 * np.pi * 900 * np.arange(400) / rate) * \
                np.linspace(1, 0, 400)
            k += 1
        return x

    def test_it_finds_the_offset_of_the_pulse(self):
        from midimusic.audio.analyze import beat_phase

        rate, tempo, offset = 22050, 120.0, 0.125
        found = beat_phase(self._clicks(rate, tempo, offset), rate, tempo)
        assert abs(found - offset) < 0.02

    def test_the_answer_is_always_inside_one_beat(self):
        from midimusic.audio.analyze import beat_phase

        rate, tempo = 22050, 100.0
        period = 60.0 / tempo
        for offset in (0.0, 0.1, 0.3, 0.5):
            found = beat_phase(self._clicks(rate, tempo, offset % period), rate, tempo)
            assert 0.0 <= found < period

    def test_it_declines_to_guess_without_a_tempo(self):
        from midimusic.audio.analyze import beat_phase

        assert beat_phase(self._clicks(22050, 120.0, 0.1), 22050, 0.0) == 0.0

    def test_it_survives_a_clip_too_short_to_measure(self):
        import numpy as np

        from midimusic.audio.analyze import beat_phase

        assert beat_phase(np.zeros(64, dtype="float32"), 22050, 120.0) == 0.0
