"""MIDI import and export.

Written with mido directly rather than through a higher-level wrapper so the
file carries tempo, time signature, key signature, track names, program
changes and section markers -- the things that make a MIDI export actually
useful when it lands in a DAW.
"""

from __future__ import annotations

from pathlib import Path

import mido

from ..core.models import Note, Song, Track

__all__ = ["song_to_midi", "write_midi", "read_midi", "midi_to_song"]

TICKS_PER_BEAT = 480

_KEY_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _key_signature(key: str) -> str:
    """Map our key string to a MIDI key-signature name mido accepts."""
    parts = (key or "C major").split()
    root = parts[0] if parts else "C"
    minor = len(parts) > 1 and "min" in parts[1].lower()
    root = root.replace("♭", "b").replace("♯", "#")
    # mido accepts names like 'C', 'F#m', 'Abm'.
    valid = {
        "C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F", "Cb", "C#", "Gb",
    }
    if root not in valid:
        root = "C"
    return f"{root}m" if minor else root


def song_to_midi(song: Song, ticks_per_beat: int = TICKS_PER_BEAT) -> mido.MidiFile:
    """Build a type-1 MIDI file from a :class:`Song`."""
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    # Track 0 carries tempo, meter, key and markers.
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=song.title[:120] or "Song", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(max(1.0, song.tempo)), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature",
            numerator=song.beats_per_bar,
            denominator=song.beat_unit,
            time=0,
        )
    )
    try:
        meta.append(mido.MetaMessage("key_signature", key=_key_signature(song.key), time=0))
    except (ValueError, KeyError):
        pass

    marker_events: list[tuple[int, mido.MetaMessage]] = []
    for name, start, _end in song.sections:
        tick = int(round(start * ticks_per_beat))
        marker_events.append((tick, mido.MetaMessage("marker", text=name, time=0)))
    for symbol, start, _end in song.chords:
        tick = int(round(start * ticks_per_beat))
        marker_events.append((tick, mido.MetaMessage("text", text=symbol, time=0)))

    last = 0
    for tick, msg in sorted(marker_events, key=lambda x: x[0]):
        msg.time = max(0, tick - last)
        meta.append(msg)
        last = tick
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    for track in song.tracks:
        mid.tracks.append(_track_to_midi(track, ticks_per_beat))
    return mid


def _track_to_midi(track: Track, ticks_per_beat: int) -> mido.MidiTrack:
    mt = mido.MidiTrack()
    mt.append(mido.MetaMessage("track_name", name=(track.name or "Track")[:120], time=0))
    channel = 9 if track.is_drum else max(0, min(15, track.channel))
    if not track.is_drum:
        mt.append(
            mido.Message("program_change", channel=channel,
                         program=max(0, min(127, track.program)), time=0)
        )

    events: list[tuple[int, int, mido.Message]] = []
    for n, start, end in _resolve_overlaps(track.notes, ticks_per_beat):
        vel = max(1, min(127, int(n.velocity)))
        pitch = max(0, min(127, int(n.pitch)))
        # Sort key puts note-offs before note-ons at the same tick so repeated
        # pitches retrigger cleanly instead of cutting each other short.
        events.append((start, 1, mido.Message("note_on", channel=channel, note=pitch,
                                              velocity=vel, time=0)))
        events.append((end, 0, mido.Message("note_off", channel=channel, note=pitch,
                                            velocity=0, time=0)))

    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _order, msg in events:
        msg.time = max(0, tick - last)
        mt.append(msg)
        last = tick
    mt.append(mido.MetaMessage("end_of_track", time=0))
    return mt


def _resolve_overlaps(
    notes: list[Note], ticks_per_beat: int
) -> list[tuple[Note, int, int]]:
    """Clip notes so no two of the same pitch overlap on the same channel.

    A MIDI channel has one voice per pitch: if a note-on for a pitch arrives
    while that pitch is already sounding, the following note-off silences both.
    Rather than let a DAW resolve that arbitrarily (and lose notes on reload),
    shorten the earlier note to end just before the later one begins.
    """
    prepared: list[tuple[Note, int, int]] = []
    for n in notes:
        start = max(0, int(round(n.start * ticks_per_beat)))
        end = max(start + 1, int(round(n.end * ticks_per_beat)))
        prepared.append((n, start, end))

    by_pitch: dict[int, list[int]] = {}
    for idx, (n, start, _end) in enumerate(prepared):
        by_pitch.setdefault(int(n.pitch), []).append(idx)

    for indices in by_pitch.values():
        indices.sort(key=lambda i: prepared[i][1])
        for a, b in zip(indices, indices[1:], strict=False):
            n_a, start_a, end_a = prepared[a]
            start_b = prepared[b][1]
            if end_a > start_b:
                prepared[a] = (n_a, start_a, max(start_a + 1, start_b - 1))

    # Drop zero-length leftovers created by exactly coincident notes.
    return [(n, s, e) for (n, s, e) in prepared if e > s]


def write_midi(song: Song, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    song_to_midi(song).save(str(path))
    return path


def read_midi(path: str | Path) -> mido.MidiFile:
    return mido.MidiFile(str(path))


def midi_to_song(path: str | Path, title: str = "") -> Song:
    """Load a MIDI file back into a :class:`Song` (for import and re-export)."""
    mf = mido.MidiFile(str(path))
    tpb = mf.ticks_per_beat or TICKS_PER_BEAT
    song = Song(title=title or Path(path).stem)

    tempo = 500000
    for msg in mf.tracks[0] if mf.tracks else []:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break
    song.tempo = mido.tempo2bpm(tempo)

    for i, mt in enumerate(mf.tracks):
        track = Track(name=f"Track {i}", channel=0)
        open_notes: dict[tuple[int, int], tuple[float, int]] = {}
        t = 0
        for msg in mt:
            t += msg.time
            if msg.type == "track_name":
                track.name = msg.name
            elif msg.type == "program_change":
                track.program = msg.program
                track.channel = msg.channel
                track.is_drum = msg.channel == 9
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)] = (t / tpb, msg.velocity)
                if msg.channel == 9:
                    track.is_drum = True
            elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in open_notes:
                    start, vel = open_notes.pop(key)
                    dur = max(0.01, t / tpb - start)
                    track.notes.append(Note(msg.note, start, dur, vel, msg.channel))
        if track.notes:
            track.sort()
            song.tracks.append(track)
    return song
