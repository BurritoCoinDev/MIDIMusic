"""The music theory engine: these assertions are about music, not code."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from midimusic.theory.chords import parse_roman, realize_progression, voice_lead
from midimusic.theory.melody import Melodist
from midimusic.theory.pitch import Scale, note_name, parse_key, parse_note, prefers_flats
from midimusic.theory.rhythm import (
    DRUM_PATTERNS,
    build_bass_bar,
    build_comp_bar,
    build_drum_bar,
    parse_steps,
)
from midimusic.theory.structure import FORMS, apply_final_lift, build_form
from midimusic.theory.style import get_style, list_styles


class TestScales:
    def test_major_scale_intervals(self):
        assert [note_name(Scale(0, "major").degree_to_pitch(d)) for d in range(7)] == [
            "C4", "D4", "E4", "F4", "G4", "A4", "B4"
        ]

    def test_dorian_has_major_sixth_and_minor_third(self):
        scale = Scale(*parse_key("D dorian"))
        pcs = scale.pitch_classes
        assert 5 in pcs  # F natural: the minor third
        assert 11 in pcs  # B natural: the major sixth that defines dorian
        assert 6 not in pcs  # no F#

    def test_negative_degree_is_the_leading_tone_below(self):
        scale = Scale(0, "major")
        assert note_name(scale.degree_to_pitch(-1)) == "B3"

    def test_degree_beyond_the_scale_wraps_an_octave(self):
        scale = Scale(0, "major")
        assert scale.degree_to_pitch(7) == scale.degree_to_pitch(0) + 12

    def test_quantize_snaps_outside_notes_into_the_scale(self):
        scale = Scale(0, "major")
        assert scale.quantize(61) in (60, 62)  # C# is not in C major
        assert scale.quantize(60) == 60

    @pytest.mark.parametrize("text,expected", [
        ("C4", 60), ("Bb3", 58), ("F#5", 78), ("a0", 21),
    ])
    def test_note_parsing(self, text, expected):
        assert parse_note(text) == expected

    def test_key_signature_chooses_accidentals(self):
        assert prefers_flats(Scale(*parse_key("Bb major")))
        assert not prefers_flats(Scale(*parse_key("D major")))
        assert str(Scale(*parse_key("Eb minor"))) == "Eb minor"


class TestChords:
    def test_diatonic_sevenths_in_c_major(self):
        scale = Scale(0, "major")
        got = [parse_roman(r, scale, True).name() for r in
               ("I", "ii", "iii", "IV", "V", "vi", "vii")]
        assert got == ["Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5"]

    def test_flat_seven_in_a_minor_is_g_not_f_sharp(self):
        # bVII is read against the parallel major, which is what makes the
        # common i-bVII-bVI progression come out right in a minor key.
        scale = Scale(*parse_key("A minor"))
        assert [c.name() for c in realize_progression(["i", "bVII", "bVI"], scale)] == [
            "Am", "G", "F"
        ]

    def test_altered_root_takes_quality_from_case_not_the_unaltered_degree(self):
        # bVII in C major is Bb major. The vii degree is diminished, but that
        # is a fact about B, not about Bb.
        scale = Scale(0, "major")
        assert parse_roman("bVII", scale).name() == "Bb"
        assert parse_roman("bIII", scale).name() == "Eb"

    def test_secondary_dominant(self):
        assert parse_roman("V/V", Scale(0, "major")).name() == "D"
        assert parse_roman("V/V", Scale(7, "major")).name() == "A"

    def test_unparseable_symbol_degrades_to_the_tonic(self):
        assert parse_roman("!!!", Scale(0, "major")).root == 0

    def test_voice_leading_minimises_movement(self):
        scale = Scale(0, "major")
        chords = realize_progression(["I", "vi", "IV", "V"], scale)
        voicings = voice_lead(chords, low=52, high=76)
        assert len(voicings) == 4
        for before, after in pairwise(voicings):
            movement = sum(abs(a - b) for a, b in zip(before, after, strict=False))
            # A root-position reset would move far more than this.
            assert movement <= 12, f"{before} -> {after} moves {movement} semitones"

    def test_voicings_stay_inside_the_register(self):
        chords = realize_progression(["I", "IV", "V", "vi"], Scale(0, "major"))
        for voicing in voice_lead(chords, low=52, high=76):
            assert all(52 <= p <= 76 for p in voicing)


class TestRhythm:
    def test_step_string_expands_to_velocities(self):
        assert parse_steps("x...x...", 8) == [1.0, 0, 0, 0, 1.0, 0, 0, 0]

    def test_short_pattern_loops_to_fill_the_bar(self):
        assert len(parse_steps("x.", 16)) == 16

    def test_lower_intensity_thins_the_arrangement(self):
        kit = DRUM_PATTERNS["funk"]
        assert len(build_drum_bar(kit, intensity=0.3)) < len(build_drum_bar(kit, intensity=1.0))

    def test_fill_lands_at_the_end_of_the_bar(self):
        events = build_drum_bar(DRUM_PATTERNS["rock"], fill=True)
        assert any(step >= 12 for step, _v, _n in events)

    def test_walking_bass_approaches_the_next_chord(self):
        events = build_bass_bar("walking", [45, 49, 52], next_root=50)
        assert abs(events[-1][2] - 50) == 1  # a semitone approach

    def test_arpeggio_does_not_repeat_one_note(self):
        events = build_comp_bar("arpeggio", [60, 64, 67])
        assert len({pitch for _s, _v, pitch, _l in events}) > 1

    def test_every_style_names_a_real_drum_kit(self):
        for name in list_styles():
            assert get_style(name).drums in DRUM_PATTERNS


class TestStructure:
    @pytest.mark.parametrize("template", list(FORMS))
    def test_forms_hit_their_target_duration(self, template):
        form = build_form(template, 150.0, 120.0)
        actual = form.duration_seconds(120.0)
        assert abs(actual - 150.0) / 150.0 < 0.15, f"{template} landed at {actual}s"

    def test_sections_stay_musically_sized(self):
        for section in build_form("pop", 200.0, 120.0).sections:
            assert section.bars >= 2
            assert section.bars % 2 == 0

    def test_final_repeat_gets_a_key_lift(self):
        sections = build_form("pop", 240.0, 120.0).sections
        apply_final_lift(sections, 2)
        assert sum(1 for s in sections if s.transpose) == 1

    def test_no_target_duration_leaves_the_template_alone(self):
        assert build_form("pop", None, 120.0).total_bars == sum(
            s.bars for s in FORMS["pop"]
        )


class TestMelody:
    def test_strong_beats_land_on_chord_tones(self):
        scale = Scale(*parse_key("A minor"))
        rng = random.Random(4)
        mel = Melodist(scale, rng, 60, 84)
        chord = realize_progression(["i"], scale)[0]
        notes = mel.realize_bar(mel.make_motif(), chord, 0.0)
        downbeat = [p for start, _d, p in notes if abs(start) < 1e-6]
        for pitch in downbeat:
            assert pitch % 12 in chord.pitch_classes

    def test_melody_stays_in_range(self):
        scale = Scale(0, "major")
        mel = Melodist(scale, random.Random(1), low=60, high=72)
        for bar in range(8):
            chord = realize_progression(["I"], scale)[0]
            for _s, _d, pitch in mel.realize_bar(mel.make_motif(), chord, bar * 4.0):
                assert 60 <= pitch <= 72

    def test_development_keeps_the_rhythm_recognisable(self):
        mel = Melodist(Scale(0, "major"), random.Random(2), 60, 84)
        motif = mel.make_motif()
        variants = [mel.develop(motif, i) for i in range(6)]
        assert any(v.rhythm == motif.rhythm for v in variants)
