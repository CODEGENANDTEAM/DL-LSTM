"""Tests for the MIDI -> WAV renderer (src/decode.py, scripts/render_audio.py).

The renderer never changes which notes are played; these tests pin down how
they sound: loudness, envelope, band-limiting, mixing, and determinism.
"""

from __future__ import annotations

import importlib.util
import math
import wave
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from src.decode import (
    INSTRUMENTS,
    loudness_stats,
    partial_spec,
    render_wav,
    shape_velocities,
    synthesize_midi,
)

FS = 44100
ROOT = Path(__file__).resolve().parent.parent


def _midi(notes, tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """notes: iterable of (pitch, start_s, end_s, velocity)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


def _tune(seconds: float = 8.0) -> pretty_midi.PrettyMIDI:
    """A plain melody over half-note triads, all velocity 100 (like model output)."""
    notes = []
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    chords = [(48, 52, 55), (53, 57, 60), (55, 59, 62), (48, 52, 55)]
    t = 0.0
    i = 0
    while t < seconds:
        notes.append((scale[i % len(scale)] + 12, t, t + 0.25, 100))
        if i % 4 == 0:
            for p in chords[(i // 4) % len(chords)]:
                notes.append((p, t, t + 1.0, 100))
        t += 0.25
        i += 1
    return _midi(notes)


def _read_wav(path: Path):
    with wave.open(str(path), "rb") as fh:
        assert fh.getsampwidth() == 2
        ch, rate = fh.getnchannels(), fh.getframerate()
        data = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2")
    return data.reshape(-1, ch).T.astype(np.float64) / 32768.0, rate, data


def _rms_db(x: np.ndarray) -> float:
    return 10.0 * math.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# --------------------------------------------------------------- loudness


@pytest.mark.parametrize("target", [-14.0, -20.0])
def test_output_hits_target_loudness(tmp_path: Path, target: float) -> None:
    stats = render_wav(_tune(), tmp_path / "tune.wav", target_rms_db=target)
    audio, rate, pcm = _read_wav(stats.path)
    assert rate == FS and audio.shape[0] == 2
    measured = loudness_stats(audio, rate)["rms_db"]
    assert abs(measured - target) < 1.0, measured
    # plain whole-file RMS should also be close for a piece with no long gaps
    assert abs(_rms_db(audio) - target) < 1.5
    assert np.abs(pcm.astype(np.int64)).max() < 32767


def test_dense_chord_piece_never_clips(tmp_path: Path) -> None:
    """Pathological: 36-note fortissimo clusters hammered repeatedly."""
    notes = []
    for k in range(8):
        t = k * 0.5
        notes += [(p, t, t + 0.45, 127) for p in range(36, 108, 2)]
    notes += [(60, 4.0, 6.0, 20)]  # and one very quiet note after them
    stats = render_wav(_midi(notes), tmp_path / "dense.wav")
    audio, rate, pcm = _read_wav(stats.path)
    assert np.abs(pcm.astype(np.int64)).max() <= int(32767 * 10 ** (-1 / 20)) + 1  # -1 dBFS
    assert abs(loudness_stats(audio, rate)["rms_db"] - (-14.0)) < 1.0


# --------------------------------------------------------------- envelope


def _raw(midi, **kw):
    kw = {"reverb": 0.0, "target_rms_db": None, "velocity_shaping": "off", **kw}
    return synthesize_midi(midi, fs=FS, **kw)


def test_single_note_decays() -> None:
    audio = _raw(_midi([(60, 0.0, 3.0, 100)]))[0][: 3 * FS]   # the held part only
    q = len(audio) // 4
    first, last = np.mean(audio[:q] ** 2), np.mean(audio[-q:] ** 2)
    assert last < first * 0.1           # at least 10 dB quieter by the end


def test_higher_notes_decay_faster() -> None:
    def drop_db(pitch: int) -> float:
        a = _raw(_midi([(pitch, 0.0, 3.0, 100)]))[0]
        return _rms_db(a[: FS // 4]) - _rms_db(a[2 * FS: 2 * FS + FS // 2])
    assert drop_db(84) > drop_db(48) + 6.0


def test_attack_has_no_click() -> None:
    audio = _raw(_midi([(60, 0.0, 1.0, 127)]))[0]
    # the very first sample is silent and the first ms stays small
    assert abs(audio[0]) < 1e-6
    assert np.abs(audio[: FS // 1000]).max() < np.abs(audio).max()


def test_note_off_releases_quickly() -> None:
    audio = _raw(_midi([(60, 0.0, 0.5, 100)]))[0]
    held = np.abs(audio[int(0.4 * FS): int(0.5 * FS)]).max()
    after = np.abs(audio[int(0.8 * FS):]).max() if audio.size > 0.8 * FS else 0.0
    assert after < held * 0.05


def test_velocity_scales_loudness() -> None:
    soft = _raw(_midi([(60, 0.0, 1.0, 40)]))
    loud = _raw(_midi([(60, 0.0, 1.0, 120)]))
    assert _rms_db(loud) - _rms_db(soft) > 6.0


# --------------------------------------------------------------- spectrum


@pytest.mark.parametrize("instrument", sorted(INSTRUMENTS))
@pytest.mark.parametrize("fs", [8000, 22050, 44100])
def test_no_partial_at_or_above_nyquist(instrument: str, fs: int) -> None:
    for pitch in range(0, 128):
        for vel in (1, 64, 127):
            freqs, amps, rates = partial_spec(pitch, vel, instrument, fs)
            assert freqs.shape == amps.shape == rates.shape
            if freqs.size:
                assert freqs.max() < fs / 2, (instrument, pitch, vel, freqs.max())
                assert np.all(rates >= 0)


def test_rendered_top_note_has_no_aliases() -> None:
    """C8 at 22.05 kHz: only partials below Nyquist, so no energy far below f0."""
    fs = 22050
    audio = synthesize_midi(_midi([(108, 0.0, 0.5, 127)]), fs=fs, reverb=0.0,
                            target_rms_db=None, velocity_shaping="off")[0]
    spec = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / fs)
    f0 = 440.0 * 2 ** ((108 - 69) / 12)
    band = (freqs > 200) & (freqs < f0 * 0.8)   # aliases of the 2nd+ partial would land here
    assert spec[band].max() < spec.max() * 0.05


# --------------------------------------------------------------- mixing


@pytest.mark.parametrize("n", [3, 6])
def test_chord_is_not_n_times_louder(n: int) -> None:
    single = _raw(_midi([(60, 0.0, 1.0, 100)]))
    chord_pitches = [48, 52, 55, 60, 64, 67][:n]
    chord = _raw(_midi([(p, 0.0, 1.0, 100) for p in chord_pitches]))
    ratio_db = _rms_db(chord) - _rms_db(single)
    assert ratio_db < 0.5 * 20 * math.log10(n)          # well under N x (and under sqrt N x)
    assert np.abs(chord).max() < n * np.abs(single).max() * 0.75


# --------------------------------------------------------------- velocity shaping


def test_velocity_shaping_varies_flat_input_deterministically() -> None:
    pm = _tune(4.0)
    notes = sorted(pm.instruments[0].notes, key=lambda n: (n.start, n.pitch))
    starts = np.array([n.start for n in notes])
    pitches = np.array([n.pitch for n in notes])
    flat = np.full(len(notes), 100)
    a = shape_velocities(starts, pitches, flat, midi=pm, seed=3)
    b = shape_velocities(starts, pitches, flat, midi=pm, seed=3)
    assert np.array_equal(a, b)
    assert np.unique(a).size > 3
    assert a.min() >= 1 and a.max() <= 127
    assert abs(float(np.mean(a)) - 100) < 8          # shapes, does not re-level
    # the top of each chord is louder than its inner voice
    first = np.flatnonzero(starts == 0.0)
    order = first[np.argsort(pitches[first])]
    assert a[order[-1]] > a[order[1]]


def test_auto_shaping_keeps_expressive_velocities() -> None:
    pm = _midi([(60, 0.0, 0.5, 30), (64, 0.5, 1.0, 110)])
    auto = _raw(pm, velocity_shaping="auto")
    off = _raw(pm, velocity_shaping="off")
    assert np.array_equal(auto, off)


def test_rendering_does_not_modify_the_midi() -> None:
    pm = _tune(2.0)
    before = [(n.pitch, n.start, n.end, n.velocity) for n in pm.instruments[0].notes]
    synthesize_midi(pm, fs=FS)
    after = [(n.pitch, n.start, n.end, n.velocity) for n in pm.instruments[0].notes]
    assert before == after


# --------------------------------------------------------------- robustness


def test_deterministic_bytes(tmp_path: Path) -> None:
    a = render_wav(_tune(3.0), tmp_path / "a.wav").path.read_bytes()
    b = render_wav(_tune(3.0), tmp_path / "b.wav").path.read_bytes()
    assert a == b


@pytest.mark.parametrize("instrument", sorted(INSTRUMENTS))
def test_empty_and_tiny_midis_do_not_crash(tmp_path: Path, instrument: str) -> None:
    empty = pretty_midi.PrettyMIDI()
    stats = render_wav(empty, tmp_path / "empty.wav", instrument=instrument)
    audio, _, _ = _read_wav(stats.path)
    assert audio.shape[1] > 0 and np.all(audio == 0)

    tiny = _midi([(60, 0.0, 0.001, 64)])
    stats = render_wav(tiny, tmp_path / "tiny.wav", instrument=instrument)
    audio, _, pcm = _read_wav(stats.path)
    assert audio.shape[1] > 0 and np.all(np.isfinite(audio))
    assert np.abs(pcm.astype(np.int64)).max() < 32767

    # notes off the audible range for the sample rate (all partials above Nyquist)
    high = _midi([(127, 0.0, 0.5, 100)])
    render_wav(high, tmp_path / "high.wav", fs=8000, instrument=instrument)


def test_missing_fluidsynth_gives_clear_error(tmp_path: Path) -> None:
    if importlib.util.find_spec("fluidsynth") is not None:
        pytest.skip("pyfluidsynth is installed")
    sf2 = tmp_path / "fake.sf2"
    sf2.write_bytes(b"")
    with pytest.raises(RuntimeError, match="FluidSynth"):
        synthesize_midi(_tune(1.0), soundfont=sf2)


def test_cli_renders_directory(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("render_audio", ROOT / "scripts" / "render_audio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _tune(2.0).write(str(tmp_path / "x.mid"))
    out = tmp_path / "out"
    assert mod.main([str(tmp_path), "--out-dir", str(out), "--instrument", "epiano"]) == 0
    audio, rate, _ = _read_wav(out / "x.wav")
    assert rate == 44100 and audio.shape[0] == 2
    assert mod.main([str(tmp_path / "x.mid"), "--out-dir", str(out / "o"),
                     "--instrument", "organ", "--no-reverb", "--rate", "22050"]) == 0
    assert _read_wav(out / "o" / "x.wav")[1] == 22050
