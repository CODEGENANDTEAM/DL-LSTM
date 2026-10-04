"""Regression tests for the efficiency / latent-bug sweep.

Bug tests pin a defect that produced no error (each docstring says what it
did). Equivalence tests pin that a faster or leaner path produces exactly what
the old, obviously-correct path produced -- the old code is reproduced here
verbatim as the reference.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config_hash, load_config, parse_override_value  # noqa: E402
from src.data.types import RESERVED_SYMBOLS  # noqa: E402
from src.data.vocab import Vocab  # noqa: E402

pretty_midi = pytest.importorskip("pretty_midi")


def _load_script(name: str):
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_sweep_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# --------------------------------------------------------------------------
# vocab.py
# --------------------------------------------------------------------------


class _GenEncoder:
    """An encoder whose vocabulary() is a generator -- legal per the
    Encoder protocol, which types it Optional[Iterable[str]]."""

    def vocabulary(self):
        return (s for s in ["<REST:2>", "<REST:3>"])


class _ListEncoder:
    def vocabulary(self):
        return ["<REST:2>", "<REST:3>"]


def test_declared_symbols_survive_pruning_when_vocabulary_is_a_generator() -> None:
    """Vocab.build consumed a generator once to union it in, then built the
    exemption set from the exhausted generator: declared symbols were pruned
    by min_freq/max_size like any rare chord, silently."""
    seqs = [["60"] * 50 + ["62"] * 40 + ["<REST:2>"]]
    for enc in (_ListEncoder(), _GenEncoder()):
        v = Vocab.build(seqs, encoder=enc, min_freq=10)
        assert "<REST:2>" in v and "<REST:3>" in v, type(enc).__name__
        v = Vocab.build(seqs, encoder=enc, max_size=1)
        assert "<REST:2>" in v and "<REST:3>" in v, type(enc).__name__


def test_max_size_keeps_declared_and_orders_deterministically() -> None:
    """max_size keeps the N most frequent corpus symbols (ties broken by
    symbol), declared symbols on top of that, reserved ids 0-4 always, and
    assigns ids in sorted-symbol order -- not frequency or set order."""
    seqs = [["a"] * 5 + ["b"] * 5 + ["c"] * 3 + ["d"] * 1 + ["z"] * 9]
    v = Vocab.build(seqs, encoder=_ListEncoder(), max_size=3)
    assert v.itos[:5] == list(RESERVED_SYMBOLS)
    # top 3 by (-count, symbol): z(9), a(5), b(5); c and d pruned; declared kept
    assert v.itos[5:] == sorted(["z", "a", "b", "<REST:2>", "<REST:3>"])
    assert v.encode(["c", "d"]) == [v.unk_id, v.unk_id]
    # the same inputs in another order give the same ids
    v2 = Vocab.build([list(reversed(seqs[0]))], encoder=_ListEncoder(), max_size=3)
    assert v2.itos == v.itos


def test_style_tokens_are_never_pruned() -> None:
    """A style token occurs once per sequence, so a rare style (or a tight
    max_size) sent it to <UNK>: style conditioning for that style silently
    disappeared from training and from generation priming."""
    seqs = [["<STYLE:Blues>", "60", "60"]] + [["<STYLE:Classical>"] + ["62"] * 20] * 20
    v = Vocab.build(seqs, min_freq=5, keep=["<STYLE:Blues>", "<STYLE:Classical>"])
    assert "<STYLE:Blues>" in v and "<STYLE:Classical>" in v
    v = Vocab.build(seqs, max_size=1, keep=["<STYLE:Blues>", "<STYLE:Classical>"])
    assert "<STYLE:Blues>" in v
    # symbols that never occur are not added by keep
    v = Vocab.build(seqs, keep=["<STYLE:Jazz>"])
    assert "<STYLE:Jazz>" not in v


def test_prepare_data_vocab_keeps_style_tokens() -> None:
    prep = _load_script("prepare_data")
    cfg = load_config(None, **{"encoding.min_freq": 50, "encoding.style_tokens": True})
    seqs = [["<STYLE:Blues>", "60"]] + [["<STYLE:Classical>"] + ["60"] * 60]
    v = prep._build_vocab(seqs, _ListEncoder(), cfg)
    assert "<STYLE:Blues>" in v and "<STYLE:Classical>" in v


# --------------------------------------------------------------------------
# config / override parsing
# --------------------------------------------------------------------------


def test_prepare_data_parses_overrides_like_the_other_scripts() -> None:
    """prepare_data used bare yaml.safe_load, so ``--set data.grid=2.5e-1``
    stayed the STRING "2.5e-1" there but became the float 0.25 in run_train /
    run_generate. The processed cache is keyed on the config hash, so the
    trainer then looked for a cache directory prepare_data never wrote."""
    prep = _load_script("prepare_data")
    parsed = prep._parse_overrides(["data.grid=2.5e-1", "train.lr=1e-4"])
    assert parsed == {"data.grid": 0.25, "train.lr": 1e-4}
    a = load_config(None, **parsed)
    b = load_config(None, **{"data.grid": parse_override_value("2.5e-1"),
                             "train.lr": parse_override_value("1e-4")})
    assert config_hash(a, "data", "augment", "encoding") == config_hash(b, "data", "augment", "encoding")


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "Infinity", "NaN", ".inf_run"])
def test_override_strings_that_float_accepts_stay_strings(text: str) -> None:
    """The 1e-4 fix-up converted ANY string float() accepts, so
    ``--set name=inf`` or ``name=nan`` became a float (and a NaN config value
    never compares equal, so it hashes to a key that cannot be matched)."""
    assert parse_override_value(text) == text


@pytest.mark.parametrize("text,value", [("1e-4", 1e-4), ("2.5E+3", 2500.0), ("-3e2", -300.0),
                                        ("1.5e-3", 1.5e-3), ("256", 256), ("0.25", 0.25),
                                        ("false", False), ("[-3,3]", [-3, 3]), ("x", "x")])
def test_override_numbers_still_parse(text: str, value) -> None:
    assert parse_override_value(text) == value


# --------------------------------------------------------------------------
# GPU memory estimate
# --------------------------------------------------------------------------


def test_memory_estimate_matches_the_measured_runs() -> None:
    """Calibrated against the runs recorded in configs/adl_mixed.yaml
    (512h x 3L, embed 256, seq 256, fp32): vocab 37,847 measured 8.5 GB
    allocated at batch 64 and 6.5 GB at batch 48."""
    from src.config import estimate_train_memory

    for batch, measured in ((64, 8.5e9), (48, 6.5e9)):
        est = estimate_train_memory(batch, 256, 37847, hidden_dim=512, num_layers=3, embed_dim=256)
        assert abs(est["total"] - measured) / measured < 0.15, (batch, est)


def test_memory_warning_flags_the_spill_configs() -> None:
    from src.config import gpu_memory_warning

    card = int(12.2e9)
    base = {"model.hidden_dim": 512, "model.num_layers": 3, "model.embed_dim": 256,
            "model.seq_len": 256}
    spilled = load_config(None, **base, **{"train.batch_size": 128})
    assert gpu_memory_warning(spilled, 25028, card) is not None     # v2 batch 128
    spilled = load_config(None, **base, **{"train.batch_size": 64})
    assert gpu_memory_warning(spilled, 37847, card) is not None     # v3 batch 64
    ok = load_config(None, **base, **{"train.batch_size": 48})
    assert gpu_memory_warning(ok, 37020, card) is None              # v4 batch 48


# --------------------------------------------------------------------------
# dataset.py
# --------------------------------------------------------------------------


def test_batched_loader_matches_per_window_collate() -> None:
    from torch.utils.data import DataLoader

    from src.data.dataset import MusicDataset, make_dataloaders

    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 500, size=int(n)).tolist() for n in rng.integers(5, 90, size=40)]
    cfg = load_config(None, **{"model.window_stride": 3, "train.batch_size": 7})
    for shuffle_seed in (0, 1):
        torch.manual_seed(shuffle_seed)
        ref = list(DataLoader(MusicDataset(seqs, 16, stride=3, cover_tail=True),
                              batch_size=7, shuffle=True))
        torch.manual_seed(shuffle_seed)
        train, val, _ = make_dataloaders(cfg, seqs, seqs, seq_len=16)
        new = list(train)
        assert len(ref) == len(new)
        for (rx, ry), (nx, ny) in zip(ref, new):
            assert rx.dtype == nx.dtype == torch.int64
            assert torch.equal(rx, nx) and torch.equal(ry, ny)
            assert nx.is_contiguous() and ny.is_contiguous()
    seq_ref = list(DataLoader(MusicDataset(seqs, 16, stride=3, cover_tail=True), batch_size=7))
    for (rx, ry), (nx, ny) in zip(seq_ref, list(val)):
        assert torch.equal(rx, nx) and torch.equal(ry, ny)


def test_get_batch_bounds() -> None:
    from src.data.dataset import MusicDataset

    ds = MusicDataset([list(range(30))], 8, stride=4, cover_tail=True)
    x, y = ds.get_batch([-1, 0])
    assert torch.equal(x[0], ds[-1][0]) and torch.equal(y[1], ds[0][1])
    with pytest.raises(IndexError):
        ds.get_batch([len(ds)])


# --------------------------------------------------------------------------
# decode.py -- the chunked renderer must equal the whole-array one
# --------------------------------------------------------------------------


def _ref_loudness(x, fs):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    power = np.mean(x * x, axis=0)
    peak = float(np.abs(x).max()) if x.size else 0.0

    def db(p):
        return 10.0 * math.log10(p) if p > 0 else float("-inf")

    plain = float(power.mean()) if power.size else 0.0
    blk = int(0.4 * fs)
    nb = power.size // blk
    if nb >= 1:
        bp = power[: nb * blk].reshape(nb, blk).mean(axis=1)
        bp = bp[bp > 1e-7]
        if bp.size:
            bp = bp[bp > bp.mean() * 0.01]
        gated = float(bp.mean()) if bp.size else 0.0
    else:
        gated = plain
    return {"rms_db": db(gated), "plain_rms_db": db(plain),
            "peak_db": 20.0 * math.log10(peak) if peak > 0 else float("-inf")}


def _ref_limit(x, fs, ceiling_db=-1.0, lookahead=0.005, release_db_per_s=80.0):
    ceiling = 10.0 ** (ceiling_db / 20.0)
    n = x.shape[-1]
    if n == 0:
        return x
    B = max(1, int(lookahead * fs))
    nb = -(-n // B)
    padded = np.zeros(nb * B)
    padded[:n] = np.abs(x).max(axis=0)
    blockpk = padded.reshape(nb, B).max(axis=1)
    g = np.minimum(0.0, 20.0 * np.log10(ceiling / np.maximum(blockpk, 1e-12)))
    gn = g.copy()
    gn[1:] = np.minimum(gn[1:], g[:-1])
    gn[:-1] = np.minimum(gn[:-1], g[1:])
    r = release_db_per_s * B / fs
    k = np.arange(nb)
    G = k * r + np.minimum.accumulate(gn - k * r)
    gain = 10.0 ** (np.interp(np.arange(n), (k + 0.5) * B, G) / 20.0)
    return np.clip(x * gain, -ceiling, ceiling)


def _ref_compress(x, fs, threshold_db, ratio=2.0, attack=0.010, release=0.200, block=0.005):
    n = x.shape[-1]
    B = max(1, int(block * fs))
    nb = -(-n // B)
    padded = np.zeros(nb * B)
    padded[:n] = np.mean(x * x, axis=0)
    level = 10.0 * np.log10(np.maximum(padded.reshape(nb, B).mean(axis=1), 1e-12))
    over = level - threshold_db
    knee = 6.0
    target = np.where(over <= -knee / 2, 0.0,
                      np.where(over >= knee / 2, over, (over + knee / 2) ** 2 / (2 * knee)))
    target = -target * (1.0 - 1.0 / ratio)
    a_att, a_rel = math.exp(-block / attack), math.exp(-block / release)
    g = np.empty(nb)
    cur = 0.0
    for i, t in enumerate(target.tolist()):
        coef = a_att if t < cur else a_rel
        cur = coef * cur + (1.0 - coef) * t
        g[i] = cur
    gain = 10.0 ** (np.interp(np.arange(n), (np.arange(nb) + 0.5) * B, g) / 20.0)
    return x * gain


def _ref_normalise(x, fs, target_db):
    cur = _ref_loudness(x, fs)["rms_db"]
    if not math.isfinite(cur):
        return x
    x = _ref_compress(x * 10.0 ** ((target_db - cur) / 20.0), fs, target_db + 4.0)
    gain_db = target_db - _ref_loudness(x, fs)["rms_db"]
    y = x
    for _ in range(8):
        y = _ref_limit(x * 10.0 ** (gain_db / 20.0), fs)
        err = target_db - _ref_loudness(y, fs)["rms_db"]
        if abs(err) < 0.05:
            break
        gain_db += err
    return y


@pytest.fixture
def small_chunks(monkeypatch):
    """Force many chunks / OLA segments on a few seconds of audio."""
    from src import decode

    monkeypatch.setattr(decode, "_CHUNK_BLOCKS", 2)
    monkeypatch.setattr(decode, "_OLA_BLOCK", 5000)
    return decode


def _signal(seconds: float, fs: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    env = np.abs(np.sin(2 * np.pi * 0.7 * t)) ** 3
    return np.vstack([env * rng.standard_normal(n), env * rng.standard_normal(n)]) * 0.9


@pytest.mark.parametrize("seconds", [0.1, 0.4, 3.3, 7.95])
def test_chunked_loudness_and_normalise_are_bit_identical(small_chunks, seconds) -> None:
    decode, fs = small_chunks, 8000
    x = _signal(seconds, fs)
    ref, new = _ref_loudness(x, fs), decode.loudness_stats(x, fs)
    assert ref["rms_db"] == new["rms_db"] and ref["peak_db"] == new["peak_db"]
    assert new["plain_rms_db"] == pytest.approx(ref["plain_rms_db"], abs=1e-9)
    assert np.array_equal(_ref_limit(x * 3.0, fs), decode._limit(x * 3.0, fs))
    assert np.array_equal(_ref_compress(x, fs, -20.0), decode._compress(x, fs, -20.0))
    expected = _ref_normalise(x, fs, -14.0)
    assert np.array_equal(expected, decode._normalise(x, fs, -14.0))
    inplace = x.copy()
    assert np.array_equal(expected, decode._normalise(inplace, fs, -14.0, inplace=True))


def test_overlap_add_reverb_matches_single_fft(small_chunks) -> None:
    decode = small_chunks
    x = _signal(2.0, 8000)
    ir = decode._room_ir(8000)
    n = x.shape[-1] + ir.shape[-1] - 1
    nfft = decode._fast_len(n)
    ref = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[..., :n]
    out = decode._convolve(x, ir)
    assert out.shape == ref.shape
    assert np.max(np.abs(out - ref)) < 1e-12 * np.max(np.abs(ref))


def test_render_wav_bytes_unchanged_by_chunking(tmp_path: Path, monkeypatch) -> None:
    """The same MIDI renders to the same WAV bytes with tiny and default
    chunk sizes (covers synthesis accumulation, reverb mix, trim, normalise
    and int16 interleaving across chunk edges)."""
    from src import decode

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for i in range(24):
        for p in (48 + i % 5, 60 + i % 7, 67):
            inst.notes.append(pretty_midi.Note(velocity=60 + i, pitch=p, start=0.25 * i, end=0.25 * i + 0.6))
    midi.instruments.append(inst)
    a = decode.render_wav(midi, tmp_path / "a.wav", fs=8000)
    monkeypatch.setattr(decode, "_CHUNK_BLOCKS", 1)
    monkeypatch.setattr(decode, "_OLA_BLOCK", 3000)
    monkeypatch.setattr(decode, "_WAV_CACHE_BYTES", 1)   # force cache eviction
    b = decode.render_wav(midi, tmp_path / "b.wav", fs=8000)
    ra, rb = (tmp_path / "a.wav").read_bytes(), (tmp_path / "b.wav").read_bytes()
    assert len(ra) == len(rb)
    diff = np.abs(np.frombuffer(ra[44:], "<i2").astype(int) - np.frombuffer(rb[44:], "<i2").astype(int))
    assert diff.max() <= 1          # OLA segment size may move one rounding
    assert a.rms_db == pytest.approx(b.rms_db, abs=1e-3)


# --------------------------------------------------------------------------
# app/transcribe.py
# --------------------------------------------------------------------------


def _fake_basic_pitch(monkeypatch) -> None:
    def predict_and_save(audio_path_list, output_directory, **_kw):
        midi = pretty_midi.PrettyMIDI(initial_tempo=100)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.13, end=0.61))
        midi.instruments.append(inst)
        midi.write(str(Path(output_directory) / "x_basic_pitch.mid"))

    pkg = types.ModuleType("basic_pitch")
    inference = types.ModuleType("basic_pitch.inference")
    inference.predict_and_save = predict_and_save  # type: ignore[attr-defined]
    pkg.inference = inference  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "basic_pitch", pkg)
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", inference)


def test_transcribed_seed_cleans_up_its_temp_dir(tmp_path: Path, monkeypatch) -> None:
    """generate._transcribe_seed wrote the transcription (and its quantized
    copy) into tempfile.mkdtemp() and never removed it: one leaked directory
    per --seed-audio run. transcribed_seed() owns the directory's lifetime."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.transcribe import transcribed_seed

    _fake_basic_pitch(monkeypatch)
    audio = tmp_path / "hum.wav"
    audio.write_bytes(b"RIFF")
    with transcribed_seed(audio, grid=0.25) as midi_path:
        assert midi_path.is_file()
        scratch = midi_path.parent
        notes = pretty_midi.PrettyMIDI(str(midi_path)).instruments[0].notes
        beat = 60.0 / 100
        assert abs(notes[0].start / beat - 0.25) < 1e-6   # snapped onto the grid
    assert not scratch.exists()


# --------------------------------------------------------------------------
# scripts: file discovery and evaluation config
# --------------------------------------------------------------------------


def _tiny_midi(path: Path) -> Path:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for i in range(4):
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60 + i, start=0.5 * i, end=0.5 * i + 0.4))
    midi.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def test_render_audio_lists_midi_and_midi_suffixes_in_stable_order(tmp_path: Path) -> None:
    """glob("*.mid") never matched .midi, and matched .MID on Windows only."""
    render = _load_script("render_audio")
    for name in ("b.MID", "a.midi", "C.mid"):
        _tiny_midi(tmp_path / name)
    (tmp_path / "notes.txt").write_text("x")
    _tiny_midi(tmp_path / "sub" / "d.mid")               # top level only
    assert [p.name for p in render.midi_files_in(tmp_path)] == ["a.midi", "b.MID", "C.mid"]


def test_evaluate_keeps_generated_pieces_under_any_time_signature_filter(tmp_path: Path) -> None:
    """Generated MIDI carries no time signature (= 4/4); a config allowing only
    3/4 made run_evaluate drop every generated file."""
    import yaml

    ev = _load_script("run_evaluate")
    cfg_path = tmp_path / "waltz.yaml"
    cfg_path.write_text(yaml.safe_dump({"data": {"time_signatures": ["3/4"]}}))
    from src.decode import write_midi
    from src.data.types import NoteEvent, Piece

    piece = Piece(events=[NoteEvent(60 + i % 5, 0.25 * i, 0.25) for i in range(8)])
    write_midi(piece, tmp_path / "gen" / "g_00.midi")
    eval_cfg = ev._generated_config(str(cfg_path), [])
    assert len(ev._load_pieces(str(tmp_path / "gen"), eval_cfg)) == 1


def test_nested_config_attribute_assignment_persists() -> None:
    """``cfg.model`` returned a fresh Config copy on every access, so
    ``cfg.model.seq_len = 7`` wrote into a temporary and was silently lost."""
    cfg = load_config(None)
    before = config_hash(cfg)
    cfg.model.seq_len = 7
    assert cfg.model.seq_len == 7 and cfg["model"]["seq_len"] == 7
    cfg.model.seq_len = load_config(None)["model"]["seq_len"]
    assert config_hash(cfg) == before             # wrapping does not change the hash
    import json
    json.dumps(cfg)                               # still plain-JSON serialisable
