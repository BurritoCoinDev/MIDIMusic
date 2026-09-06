"""Audio export, rendering and the MIDI round trip."""

from __future__ import annotations

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
