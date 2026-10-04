"""Regression tests for the data-layer audit.

Each test pins one bug that was found and fixed (see the docstrings), plus the
equivalence guarantees the performance work relies on: the fast paths must
produce exactly what the slow, obviously-correct paths produce.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.types import NoteEvent, Piece

pretty_midi = pytest.importorskip("pretty_midi")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _write_midi(path: Path, tracks, bpm: float = 120.0, tempo_changes=None) -> Path:
    """tracks: list of lists of (pitch, start_beats, end_beats)."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    sec = 60.0 / bpm
    for notes in tracks:
        inst = pretty_midi.Instrument(program=0)
        for pitch, start, end in notes:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch,
                                               start=start * sec, end=end * sec))
        midi.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    if tempo_changes:
        _rewrite_with_tempo_changes(path, bpm, tempo_changes)
    return path


def _rewrite_with_tempo_changes(path: Path, bpm: float, changes) -> None:
    """Insert set_tempo meta messages at the given beat positions (via mido)."""
    import mido

    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    track0 = mid.tracks[0]
    # absolute times
    events, t = [], 0
    for msg in track0:
        t += msg.time
        events.append((t, msg))
    for beat, new_bpm in changes:
        events.append((int(beat * tpb), mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(new_bpm))))
    events.sort(key=lambda e: (e[0], 0 if e[1].type == "set_tempo" else 1))
    events = [e for e in events if e[1].type != "end_of_track"]
    events.append((max(at for at, _ in events), mido.MetaMessage("end_of_track")))
    new, last = mido.MidiTrack(), 0
    for at, msg in events:
        new.append(msg.copy(time=at - last))
        last = at
    mid.tracks[0] = new
    mid.save(str(path))


def _melody(n: int, pitch0: int = 60):
    return [(pitch0 + (i % 12), i * 0.5, i * 0.5 + 0.5) for i in range(n)]


# --------------------------------------------------------------------------
# BUG: decode clock drifts on a grid that is not exactly representable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["note_chord", "interval", "pitch_duration"])
def test_decode_clock_does_not_drift_on_non_dyadic_grid(scheme: str) -> None:
    """encode.py decoders advanced ``cursor = round(cursor + grid, 6)``.

    With a triplet grid (1/3 beat) each step rounds away ~3e-7 beats; after
    three steps the cursor reads 0.999999 instead of 1.0 and the error grows
    linearly -- a few thousand steps in, onsets no longer match the quantized
    starts they were encoded from. The decoders now count integer steps.
    """
    from src.data.encode import get_encoder
    from src.data.parse import quantize

    grid = 1.0 / 3.0
    cfg = load_config(**{"encoding.scheme": scheme, "data.grid": grid,
                         "encoding.include_duration": False})
    encoder = get_encoder(cfg)
    starts = [quantize(i * grid * 2, grid) for i in range(3000)]
    piece = Piece(events=[NoteEvent(60 + (i % 5), s, round(grid, 6)) for i, s in enumerate(starts)])
    restored = encoder.decode(encoder.encode(piece))
    assert [e.start for e in restored.events] == starts


# --------------------------------------------------------------------------
# BUG: identical notes at one onset produce "60.60"-style chord symbols
# --------------------------------------------------------------------------


def test_parse_merges_duplicate_notes_at_same_onset(tmp_path: Path) -> None:
    """Two tracks (or a re-struck overlapping note) sounding the same pitch at
    the same quantized onset survived parse as two NoteEvents, so note_chord
    wrote "48.60.60". On ADL that minted 1,881 of 37,847 vocabulary symbols
    (5%) that are exact aliases of another chord and occur 69k times in train.
    """
    from src.data.parse import load_piece

    melody = _melody(60)
    doubled = [(p, s, e - 0.25) for p, s, e in melody[:20]]  # same onsets, shorter
    path = _write_midi(tmp_path / "dup.mid", [melody, doubled])
    cfg = load_config(**{"data.time_signatures": []})
    piece = load_piece(path, cfg)
    assert piece is not None
    keys = [(e.start, e.pitch) for e in piece.events]
    assert len(keys) == len(set(keys)), "duplicate (onset, pitch) survived parse"
    # the merged note keeps the longer duration
    first = [e for e in piece.events if e.start == 0.0][0]
    assert first.duration == 0.5


@pytest.mark.parametrize("scheme", ["note_chord", "interval"])
def test_encoders_never_emit_a_duplicated_pitch(scheme: str) -> None:
    from src.data.encode import get_encoder

    cfg = load_config(**{"encoding.scheme": scheme, "encoding.include_duration": False})
    encoder = get_encoder(cfg)
    piece = Piece(events=[NoteEvent(48, 0.0, 1.0), NoteEvent(60, 0.0, 0.5), NoteEvent(60, 0.0, 1.0)])
    symbols = encoder.encode(piece)
    if scheme == "note_chord":
        assert symbols == ["48.60"]
    else:
        assert symbols == ["-12", "C+12"]


# --------------------------------------------------------------------------
# BUG: cfg.encoding.style_tokens was ignored by prepare_data
# --------------------------------------------------------------------------


def _corpus(tmp_path: Path, per_style: int = 4, styles=("Jazz", "Blues")) -> Path:
    raw = tmp_path / "raw"
    for s_i, style in enumerate(styles):
        for i in range(per_style):
            _write_midi(raw / style / f"{style.lower()}_{i}.mid",
                        [_melody(80 + i + 10 * s_i, pitch0=50 + 3 * i + s_i)])
    return raw


def _prepare(tmp_path: Path, raw: Path, *extra: str) -> Path:
    from scripts.prepare_data import main as prepare_main

    processed = tmp_path / "processed"
    args = ["--set", f"data.raw_dir={raw.as_posix()}",
            "--set", f"data.processed_dir={processed.as_posix()}",
            "--set", "data.time_signatures=[]",
            "--set", "data.val_split=0.25", "--set", "data.test_split=0.25",
            "--set", "augment.transpose_range=[-1, 1]"]
    for item in extra:
        args += ["--set", item]
    assert prepare_main(args) == 0
    (cache,) = [d for d in processed.iterdir() if d.is_dir()]
    return cache


def test_prepare_data_honours_style_tokens(tmp_path: Path) -> None:
    """prepare_data encoded with ``encoder.encode`` directly instead of
    ``dataset.pieces_to_symbols``, so ``encoding.style_tokens: true`` never
    reached the cache: configs/adl_mixed.yaml (the "style-conditioned" run)
    has no <STYLE:...> symbol in its vocabulary at all.
    """
    from src.data.vocab import Vocab

    cache = _prepare(tmp_path, _corpus(tmp_path), "encoding.style_tokens=true")
    vocab = Vocab.load(cache / "vocab.json")
    assert "<STYLE:Jazz>" in vocab and "<STYLE:Blues>" in vocab
    style_ids = {vocab["<STYLE:Jazz>"], vocab["<STYLE:Blues>"]}
    for split in ("train", "val", "test"):
        tokens = np.load(cache / f"{split}_tokens.npy")
        lengths = np.load(cache / f"{split}_lengths.npy")
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        for s in starts:
            assert int(tokens[s]) in style_ids, f"{split} sequence lacks its style token"


def test_style_tokens_off_adds_nothing(tmp_path: Path) -> None:
    from src.data.vocab import Vocab

    cache = _prepare(tmp_path, _corpus(tmp_path), "encoding.style_tokens=false")
    assert not any(s.startswith("<STYLE:") for s in Vocab.load(cache / "vocab.json").itos)


# --------------------------------------------------------------------------
# BUG: duplicate pieces leak from train into val/test
# --------------------------------------------------------------------------


def test_held_out_pieces_never_duplicate_a_training_piece() -> None:
    """Splitting by piece does not help when the corpus holds the same piece
    twice under different file names: ADL has 2 val and 3 test pieces whose
    token sequences are identical to a training sequence. A copy in a
    different key is the same leak, since train is augmented across keys.
    """
    from src.data.augment import split_pieces

    cfg = load_config()
    base = [NoteEvent(60 + (i % 7), i * 0.25, 0.25) for i in range(60)]
    unique = [Piece(events=[NoteEvent(40 + k + (i % 5), i * 0.5, 0.25) for i in range(60)],
                    source=f"u{k}") for k in range(30)]
    dupes = [Piece(events=list(base), source=f"d{k}") for k in range(5)]
    dupes += [Piece(events=[e.transposed(3) for e in base], source="d_transposed")]
    train, val, test = split_pieces(unique + dupes, cfg)

    def key(piece):
        lo = min(e.pitch for e in piece.events)
        return tuple((e.start, e.pitch - lo) for e in piece.events)

    train_keys = {key(p) for p in train}
    for p in val + test:
        assert key(p) not in train_keys, f"{p.source} duplicates a training piece"
    val_keys = {key(p) for p in val}
    for p in test:
        assert key(p) not in val_keys, f"{p.source} duplicates a validation piece"


def test_split_of_unique_pieces_is_unchanged() -> None:
    """The dedupe only removes held-out duplicates; a corpus without any gets
    exactly the old partition (so existing splits stay reproducible)."""
    import random

    from src.data.augment import split_pieces

    cfg = load_config()
    pieces = [Piece(events=[NoteEvent(40, 0.0, 0.25), NoteEvent(41 + k, 0.25, 0.25)],
                    source=f"p{k}") for k in range(50)]
    train, val, test = split_pieces(pieces, cfg)
    order = list(range(50))
    random.Random(int(cfg.data.split_seed)).shuffle(order)
    n = int(round(50 * 0.1))
    assert [p.source for p in val] == [f"p{i}" for i in sorted(order[:n])]
    assert [p.source for p in test] == [f"p{i}" for i in sorted(order[n:2 * n])]
    assert len(train) == 40


# --------------------------------------------------------------------------
# BUG: with stride > 1 the end of every piece is never a training target
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seq_len", [1, 4, 8, 16])
def test_loader_windows_supervise_every_token(stride: int, seq_len: int) -> None:
    """Windows start at k*stride only, so the last (len - seq_len - 1) % stride
    tokens of each piece were never a target: 825k of ADL's 12.75M training
    tokens (6.5%), always the piece endings. make_dataloaders now adds one
    end-aligned window per piece where the strided ones fall short.
    """
    from src.data.dataset import MusicDataset

    if stride > seq_len:
        pytest.skip("stride > seq_len skips tokens mid-piece by construction")
    sequences = [list(range(1000 * i, 1000 * i + n)) for i, n in enumerate(range(0, 40))]
    dataset = MusicDataset(sequences, seq_len, stride=stride, cover_tail=True)
    covered = set()
    for index in range(len(dataset)):
        x, y = dataset[index]
        assert x.shape == y.shape == (seq_len,)
        assert y[:-1].tolist() == x[1:].tolist()
        assert len({int(t) // 1000 for t in x.tolist() + y.tolist()}) == 1
        covered.update(int(t) for t in y.tolist())
    expected = {t for s in sequences if len(s) > seq_len for t in s[1:]}
    assert covered == expected


def test_make_dataloaders_covers_the_tail_by_default() -> None:
    from src.data.dataset import make_dataloaders

    cfg = load_config(**{"model.window_stride": 4, "train.batch_size": 2})
    train, _, _ = make_dataloaders(cfg, [list(range(30))], seq_len=8)
    ends = {int(y[-1]) for x, y in train.dataset}
    assert 29 in ends


# --------------------------------------------------------------------------
# numpy-backed cache: API, equivalence and guards
# --------------------------------------------------------------------------


def _cfg_for(tmp_path: Path, raw: Path | None = None):
    over = {"data.processed_dir": (tmp_path / "processed").as_posix()}
    if raw is not None:
        over["data.raw_dir"] = raw.as_posix()
    return load_config(**over)


def test_load_processed_round_trips_as_numpy(tmp_path: Path) -> None:
    from src.data.dataset import RaggedSequences, load_processed, load_split, save_processed
    from src.data.vocab import Vocab

    cfg = _cfg_for(tmp_path)
    vocab = Vocab(["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<REST>"] + [str(i) for i in range(40)])
    splits = {"train": [[5, 6, 7], [], [8, 9]], "val": [[10, 11]], "test": []}
    save_processed(cfg, splits, vocab)
    loaded, v = load_processed(cfg)
    assert v.itos == vocab.itos
    for name, seqs in splits.items():
        assert isinstance(loaded[name], RaggedSequences)
        assert [s.tolist() for s in loaded[name]] == seqs
        assert [list(map(int, s)) for s in load_split(cfg, name)] == seqs
    assert load_split(cfg, "val")[0].tolist() == [10, 11]
    assert loaded["train"][-1].tolist() == [8, 9]


def test_dataset_from_ragged_matches_dataset_from_lists() -> None:
    from src.data.dataset import MusicDataset, RaggedSequences

    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 500, size=n).tolist() for n in rng.integers(0, 60, size=40)]
    a = MusicDataset(seqs, 7, stride=3, cover_tail=True)
    b = MusicDataset(RaggedSequences.from_sequences(seqs), 7, stride=3, cover_tail=True)
    assert len(a) == len(b) > 0
    for i in range(len(a)):
        (xa, ya), (xb, yb) = a[i], b[i]
        assert xa.dtype == torch_long() and xa.tolist() == xb.tolist() and ya.tolist() == yb.tolist()


def torch_long():
    import torch

    return torch.long


def test_load_processed_rejects_ids_beyond_the_vocab(tmp_path: Path) -> None:
    from src.data.dataset import load_processed, save_processed
    from src.data.vocab import Vocab

    cfg = _cfg_for(tmp_path)
    vocab = Vocab(["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<REST>", "60"])
    save_processed(cfg, {"train": [[5, 99]], "val": [], "test": []}, vocab)
    with pytest.raises(ValueError, match="different builds"):
        load_processed(cfg)


def test_stale_raw_corpus_is_reported(tmp_path: Path, caplog) -> None:
    """The cache is keyed on the config, so editing the raw MIDI under the same
    raw_dir silently reused the old cache. meta.json now records a stat()
    fingerprint of the raw files and load_processed warns when it changes."""
    from src.data.dataset import load_processed, save_processed

    raw = tmp_path / "raw"
    _write_midi(raw / "a.mid", [_melody(10)])
    cfg = _cfg_for(tmp_path, raw)
    save_processed(cfg, {"train": [[5]], "val": [], "test": []})
    with caplog.at_level(logging.WARNING):
        load_processed(cfg)
    assert "changed since" not in caplog.text
    _write_midi(raw / "b.mid", [_melody(12)])
    with caplog.at_level(logging.WARNING):
        load_processed(cfg)
    assert "changed since" in caplog.text


def test_old_pipeline_cache_is_reported(tmp_path: Path, caplog) -> None:
    from src.data.dataset import DATA_PIPELINE_VERSION, load_processed, save_processed

    cfg = _cfg_for(tmp_path)
    out = save_processed(cfg, {"train": [[5]], "val": [], "test": []})
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    meta.pop("pipeline_version")
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        load_processed(cfg)
    assert (f"v1; this code is v{DATA_PIPELINE_VERSION}" in caplog.text) == (DATA_PIPELINE_VERSION > 1)


def test_prepare_sets_aside_a_cache_with_a_different_vocabulary(tmp_path: Path) -> None:
    """A checkpoint is only valid with the vocabulary it was trained on, and the
    cache directory name does not change when the code does. prepare_data must
    not silently overwrite a cache whose vocabulary differs from the new one:
    the old directory is renamed aside (restorable), an identical rebuild is
    written in place without any backup."""
    raw = _corpus(tmp_path)
    processed = tmp_path / "processed"
    cache = _prepare(tmp_path, raw)
    _prepare(tmp_path, raw)  # same vocabulary: no backup
    assert not [d for d in processed.iterdir() if ".replaced-" in d.name]

    fake = {"version": 1, "symbols": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<REST>", "1"]}
    (cache / "vocab.json").write_text(json.dumps(fake), encoding="utf-8")
    from scripts.prepare_data import main as prepare_main

    assert prepare_main(["--set", f"data.raw_dir={raw.as_posix()}",
                         "--set", f"data.processed_dir={processed.as_posix()}",
                         "--set", "data.time_signatures=[]",
                         "--set", "data.val_split=0.25", "--set", "data.test_split=0.25",
                         "--set", "augment.transpose_range=[-1, 1]"]) == 0
    backups = [d for d in processed.iterdir() if ".replaced-" in d.name]
    assert len(backups) == 1, "the replaced cache should be kept, not destroyed"
    assert json.loads((backups[0] / "vocab.json").read_text(encoding="utf-8")) == fake
    assert json.loads((cache / "vocab.json").read_text(encoding="utf-8"))["symbols"] != fake["symbols"]


# --------------------------------------------------------------------------
# performance paths are exact
# --------------------------------------------------------------------------


@pytest.mark.parametrize("include_duration", [False, True])
@pytest.mark.parametrize("scheme", ["note_chord", "interval", "pitch_duration"])
def test_encode_transpositions_equals_encoding_each_transposition(scheme, include_duration) -> None:
    from src.data.encode import get_encoder

    cfg = load_config(**{"encoding.scheme": scheme, "encoding.include_duration": include_duration})
    encoder = get_encoder(cfg)
    rng = np.random.default_rng(1)
    events = [NoteEvent(int(p), float(s) * 0.25, float(d) * 0.25)
              for p, s, d in zip(rng.integers(30, 90, 400), rng.integers(0, 600, 400),
                                 rng.integers(1, 9, 400))]
    piece = Piece(events=sorted(set(events), key=lambda e: (e.start, e.pitch)))
    offsets = [0, -6, -1, 1, 5]
    assert encoder.encode_transpositions(piece, offsets) == [
        encoder.encode(piece.transposed(k)) for k in offsets
    ]


def test_parallel_parse_equals_serial(tmp_path: Path) -> None:
    from src.data.parse import load_corpus

    raw = _corpus(tmp_path, per_style=3)
    cfg = load_config(**{"data.time_signatures": [], "data.include_styles": []})
    serial = load_corpus(raw, cfg, workers=1)
    parallel = load_corpus(raw, cfg, workers=2)
    assert [(p.source, p.style, p.events) for p in serial] == \
           [(p.source, p.style, p.events) for p in parallel]


def test_note_event_pickles_round_trip() -> None:
    import pickle

    e = NoteEvent(61, 1.25, 0.5, 99)
    assert pickle.loads(pickle.dumps(e)) == e


def test_discovery_order_is_case_insensitive_by_component(tmp_path: Path) -> None:
    """The seeded split depends on file order. sorted(WindowsPath) compares
    case-insensitively per component; a PosixPath sort does not, so the same
    corpus used to split differently on Linux. discover_midi_files now spells
    out the Windows order everywhere (this fails only on a case-sensitive FS
    with the old code, so on Windows it guards the equivalence)."""
    from src.data.parse import discover_midi_files

    for name in ["b.mid", "A.mid", "a2.MID", "sub/Z.mid", "Sub2/c.midi", "x.txt"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    got = [p.relative_to(tmp_path).as_posix() for p in discover_midi_files(tmp_path)]
    assert got == sorted(got, key=lambda s: tuple(part.lower() for part in s.split("/")))
    assert "x.txt" not in got and len(got) == 5


def test_tempo_map_opt_in_places_notes_on_their_notated_beats(tmp_path: Path) -> None:
    """Only the first tempo is used by default (kept for cache compatibility);
    data.tempo_map=true integrates the whole tempo map."""
    from src.data.parse import load_piece

    notes = [(60 + (i % 3), float(i), float(i) + 1.0) for i in range(60)]
    path = _write_midi(tmp_path / "t.mid", [notes], bpm=120.0, tempo_changes=[(8, 60.0)])
    exact = load_piece(path, load_config(**{"data.time_signatures": [], "data.tempo_map": True}))
    naive = load_piece(path, load_config(**{"data.time_signatures": []}))
    assert [e.start for e in exact.events] == [float(i) for i in range(60)]
    assert [e.start for e in naive.events] != [float(i) for i in range(60)]
