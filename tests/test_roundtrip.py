"""Round-trip tests -- build-order steps 1 and 2.

These are the tests that matter. Nearly every "trained for six hours and the
output is noise" bug is actually a broken round-trip that was never checked:
the model learned fine, but the symbols it emits are being decoded back to the
wrong notes.

Run with:  D:/PYTH/python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.types import NoteEvent, Piece


def make_piece() -> Piece:
    """A short quantized fragment: a C major scale fragment plus one triad."""
    events = [
        NoteEvent(pitch=60, start=0.0, duration=1.0),
        NoteEvent(pitch=62, start=1.0, duration=0.5),
        NoteEvent(pitch=64, start=1.5, duration=0.5),
        NoteEvent(pitch=65, start=2.0, duration=1.0),
        # a triad -- three notes sharing an onset
        NoteEvent(pitch=60, start=3.0, duration=2.0),
        NoteEvent(pitch=64, start=3.0, duration=2.0),
        NoteEvent(pitch=67, start=3.0, duration=2.0),
    ]
    return Piece(events=events, source="synthetic", tempo=120.0)


@pytest.mark.parametrize("scheme", ["note_chord", "interval", "pitch_duration"])
def test_encode_decode_roundtrip(scheme: str) -> None:
    """decode(encode(piece)) must reproduce the pitch content.

    pitch_duration is monophonic by design, so it keeps only the top note of
    each chord -- assert against that reduction rather than the full piece.
    """
    from src.data.encode import get_encoder

    cfg = load_config(**{"encoding.scheme": scheme})
    encoder = get_encoder(cfg)
    piece = make_piece()

    symbols = encoder.encode(piece)
    assert symbols, f"{scheme}: encoder produced no symbols"

    restored = encoder.decode(symbols)
    assert len(restored) > 0, f"{scheme}: decode produced no events"

    if scheme == "pitch_duration":
        expected = _top_note_per_onset(piece)
    else:
        expected = piece.pitches

    assert restored.pitches == expected, (
        f"{scheme}: pitch sequence changed across the round-trip\n"
        f"  in:  {expected}\n  out: {restored.pitches}"
    )


@pytest.mark.parametrize("scheme", ["note_chord", "interval", "pitch_duration"])
def test_decode_survives_garbage(scheme: str) -> None:
    """A sampling model emits malformed symbols. Decode must skip them, not raise."""
    from src.data.encode import get_encoder
    from src.data.types import BOS, EOS, PAD, UNK

    cfg = load_config(**{"encoding.scheme": scheme})
    encoder = get_encoder(cfg)

    symbols = encoder.encode(make_piece())
    polluted = [BOS, "!!not-a-symbol!!", PAD] + symbols + [UNK, "999.999", EOS]

    restored = encoder.decode(polluted)  # must not raise
    assert isinstance(restored, Piece)


def test_transpose_is_reversible() -> None:
    piece = make_piece()
    assert piece.transposed(5).transposed(-5).pitches == piece.pitches


def test_split_happens_before_augment_has_no_leakage() -> None:
    """Transposed copies of a validation piece must never appear in training.

    If this fails, val loss looks great and means nothing.
    """
    from src.data.augment import split_pieces, transpose_augment

    cfg = load_config()
    pieces = [Piece(events=make_piece().events, source=f"p{i}") for i in range(20)]

    train, val, test = split_pieces(pieces, cfg)
    train_aug = transpose_augment(train, cfg)

    train_sources = {p.source for p in train_aug}
    held_out = {p.source for p in val} | {p.source for p in test}
    assert not (train_sources & held_out), "augmented training data leaks held-out pieces"


def test_vocab_ordering_is_deterministic() -> None:
    """Vocab built twice from the same symbols must map identically.

    Set-iteration order is not stable across runs; if the vocab were built that
    way, reloading a checkpoint would map every token to the wrong note.
    """
    from src.data.vocab import Vocab

    sequences = [["60", "64", "67"], ["62", "60", "71"], ["67", "60"]]
    a = Vocab.build(sequences)
    b = Vocab.build(list(reversed([list(s) for s in sequences])))

    shared = set(a.itos) & set(b.itos)
    assert shared, "no shared symbols to compare"
    for sym in shared:
        assert a.encode([sym]) == b.encode([sym]), f"unstable id for {sym!r}"


# --------------------------------------------------------------------------
# windowing / teacher forcing
#
# The dataset hands the model x and y = x shifted left by one. Get the shift
# wrong by a position and nothing crashes: the model is simply trained to copy
# its own input, val loss drops beautifully, and generation emits the seed back
# verbatim. These tests pin the shift, and pin the piece boundary that keeps
# windows from splicing the end of one piece onto the start of another.
# --------------------------------------------------------------------------


def _distinct_sequences() -> list[list[int]]:
    """Three pieces whose id ranges do not overlap.

    Disjoint ranges are the whole trick: any window that straddled a boundary
    would contain ids from two different decades and is trivially detectable.
    """
    return [
        list(range(100, 120)),   # 20 tokens
        list(range(200, 213)),   # 13 tokens
        list(range(300, 331)),   # 31 tokens
    ]


@pytest.mark.parametrize("seq_len", [1, 4, 8])
@pytest.mark.parametrize("stride", [1, 3, 5])
def test_dataset_targets_are_inputs_shifted_by_one(seq_len: int, stride: int) -> None:
    """x and y are both length seq_len, and y[i] == x[i + 1]."""
    from src.data.dataset import MusicDataset

    dataset = MusicDataset(_distinct_sequences(), seq_len, stride=stride)
    assert len(dataset) > 0, "fixture produced no windows"

    for index in range(len(dataset)):
        x, y = dataset[index]
        assert x.shape == (seq_len,), f"x has shape {tuple(x.shape)}, want ({seq_len},)"
        assert y.shape == (seq_len,), f"y has shape {tuple(y.shape)}, want ({seq_len},)"
        assert x.dtype == y.dtype, "x and y must share a dtype for the loss"
        for i in range(seq_len - 1):
            assert int(y[i]) == int(x[i + 1]), (
                f"window {index} is not shifted by one at position {i}: "
                f"y[{i}]={int(y[i])} but x[{i + 1}]={int(x[i + 1])}\n"
                f"  x={x.tolist()}\n  y={y.tolist()}"
            )


@pytest.mark.parametrize("stride", [1, 2, 7])
def test_windows_never_cross_a_piece_boundary(stride: int) -> None:
    """Every window -- inputs AND targets -- comes from a single piece.

    Concatenating pieces into one stream would train the model on transitions
    from the end of one piece into the start of the next, which occur nowhere
    in the data and which it will happily reproduce when generating.
    """
    from src.data.dataset import MusicDataset

    sequences = _distinct_sequences()
    # Which piece each id belongs to; disjoint ranges make this unambiguous.
    owner = {token: piece for piece, seq in enumerate(sequences) for token in seq}

    dataset = MusicDataset(sequences, 6, stride=stride)
    assert len(dataset) > 0

    for index in range(len(dataset)):
        x, y = dataset[index]
        pieces = {owner[int(t)] for t in x.tolist()} | {owner[int(t)] for t in y.tolist()}
        assert len(pieces) == 1, (
            f"window {index} (stride {stride}) spans pieces {sorted(pieces)}\n"
            f"  x={x.tolist()}\n  y={y.tolist()}"
        )


@pytest.mark.parametrize("stride", [1, 2, 5])
def test_last_target_is_the_token_after_the_window(stride: int) -> None:
    """y[-1] is the token immediately following x in the source sequence.

    This is the old scalar target, and it must still be reachable: it is the
    only position the sampler reads at generation time.
    """
    from src.data.dataset import MusicDataset

    sequences = _distinct_sequences()
    seq_len = 6
    dataset = MusicDataset(sequences, seq_len, stride=stride)

    # Rebuild the expected (piece, offset) enumeration independently of the
    # bisect index, so a bug in that index cannot hide behind itself.
    expected = [
        (piece, k * stride)
        for piece, seq in enumerate(sequences)
        for k in range(max(0, (len(seq) - seq_len + stride - 1) // stride))
    ]
    assert len(dataset) == len(expected), "window count disagrees with the enumeration"

    for index, (piece, offset) in enumerate(expected):
        source = sequences[piece]
        x, y = dataset[index]
        assert x.tolist() == source[offset : offset + seq_len]
        assert int(y[-1]) == source[offset + seq_len], (
            f"window {index}: y[-1]={int(y[-1])} but the token after the "
            f"window is {source[offset + seq_len]}"
        )


def test_every_window_stays_in_bounds_for_ragged_lengths() -> None:
    """The +1 the shifted target needs must never run off the end of a piece.

    Lengths straddling seq_len+1 are where an off-by-one in the window count
    surfaces as an IndexError -- or worse, as a silently truncated final
    target. Sequences shorter than seq_len+1 must contribute zero windows.
    """
    from src.data.dataset import MusicDataset

    seq_len = 5
    for length in range(0, 3 * seq_len):
        sequence = list(range(length))
        for stride in (1, 2, 3, seq_len):
            dataset = MusicDataset([sequence], seq_len, stride=stride)
            if length < seq_len + 1:
                assert len(dataset) == 0, (
                    f"length {length} < seq_len+1 ({seq_len + 1}) yielded "
                    f"{len(dataset)} windows at stride {stride}"
                )
                continue
            for index in range(len(dataset)):
                x, y = dataset[index]  # must not raise
                assert len(x) == len(y) == seq_len, (
                    f"length {length}, stride {stride}, window {index}: "
                    f"got {len(x)}/{len(y)} tokens, want {seq_len}"
                )


def _top_note_per_onset(piece: Piece) -> list[int]:
    by_onset: dict[float, int] = {}
    for event in piece.events:
        by_onset[event.start] = max(by_onset.get(event.start, 0), event.pitch)
    return [by_onset[k] for k in sorted(by_onset)]


# --------------------------------------------------------------------------
# run-length rests
#
# The old time model spent one token per grid step, so 56% of ADL training
# tokens and 75% of generated tokens were <REST>. Collapsing runs into <REST:k>
# halves the sequences -- but a rest bug is the nastiest kind in this pipeline:
# get a rest length wrong and nothing raises, no pitch changes, every note after
# it simply slides in time. So these tests check ONSETS, not just pitches.
# --------------------------------------------------------------------------

ALL_SCHEMES = ["note_chord", "interval", "pitch_duration"]
GRIDS = [0.25, 0.5, 0.125]


def _rest_cfg(scheme: str, grid: float, *, run_length: bool, max_run: int = 16):
    return load_config(
        **{
            "encoding.scheme": scheme,
            "encoding.run_length_rests": run_length,
            "encoding.max_rest_run": max_run,
            "data.grid": grid,
        }
    )


def _spaced_piece(grid: float, gaps: list[int], first_pitch: int = 60) -> Piece:
    """Notes whose onsets are separated by ``gaps`` grid steps each.

    Onsets are built as ``index * grid`` from integer step counts rather than by
    accumulating floats, so they are exactly on the grid the encoder will ask
    about -- the same reason parse.quantize rounds to 6 places.
    """
    events = []
    step = 0
    for i, gap in enumerate([0] + gaps):
        step += gap
        events.append(
            NoteEvent(
                pitch=first_pitch + (i % 5),
                start=round(step * grid, 6),
                duration=grid,
            )
        )
    return Piece(events=events, source="spaced", tempo=120.0)


def _expected_onsets(piece: Piece, scheme: str) -> list[float]:
    """Onsets a round-trip should reproduce, allowing for monophonic reduction."""
    if scheme == "pitch_duration":
        return sorted({e.start for e in piece.events})
    return [e.start for e in sorted(piece.events, key=lambda e: (e.start, e.pitch))]


def _expected_pitches(piece: Piece, scheme: str) -> list[int]:
    if scheme == "pitch_duration":
        return _top_note_per_onset(piece)
    return piece.pitches


def _rest_runs(symbols: list[str]) -> list[int]:
    """The k of every <REST:k> in order; bare <REST> counts as 1."""
    from src.data.encode import parse_rest_run

    from src.data.types import REST

    return [
        1 if s == REST else parse_rest_run(s)  # type: ignore[misc]
        for s in symbols
        if s == REST or parse_rest_run(s) is not None
    ]


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize("grid", GRIDS)
def test_run_length_rests_roundtrip_pitches_and_onsets(scheme: str, grid: float) -> None:
    """With runs ON, pitches AND onsets survive exactly, at every grid."""
    from src.data.encode import get_encoder

    encoder = get_encoder(_rest_cfg(scheme, grid, run_length=True))
    # A deliberately irregular rhythm: adjacent notes, short gaps, and a gap
    # exactly at the cap, so the single-token and multi-token paths both run.
    piece = _spaced_piece(grid, [1, 1, 4, 2, 16, 1, 9, 3])

    symbols = encoder.encode(piece)
    restored = encoder.decode(symbols)

    assert restored.pitches == _expected_pitches(piece, scheme)
    assert [e.start for e in restored.events] == _expected_onsets(piece, scheme), (
        f"{scheme} @ grid {grid}: onsets shifted across the round-trip\n"
        f"  symbols: {symbols}"
    )


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize("grid", GRIDS)
def test_flag_off_matches_one_token_per_grid_step(scheme: str, grid: float) -> None:
    """run_length_rests=false reproduces the original encoding byte for byte.

    Expanding every <REST:k> from the flag-on encoding must give back exactly
    the flag-off symbol list -- which is the definition of the old behaviour and
    what makes the two usable as ablation arms of the same experiment.
    """
    from src.data.encode import get_encoder, parse_rest_run

    from src.data.types import REST

    piece = _spaced_piece(grid, [1, 3, 1, 20, 5, 2])
    on = get_encoder(_rest_cfg(scheme, grid, run_length=True)).encode(piece)
    off = get_encoder(_rest_cfg(scheme, grid, run_length=False)).encode(piece)

    assert all(parse_rest_run(s) is None for s in off), (
        f"{scheme}: flag-off encoding still contains run-length rests: {off}"
    )

    expanded: list[str] = []
    for symbol in on:
        steps = parse_rest_run(symbol)
        expanded.extend([REST] * steps if steps is not None else [symbol])
    assert expanded == off, (
        f"{scheme} @ grid {grid}: flag-on encoding does not expand to the "
        f"flag-off one\n  expanded: {expanded}\n  off:      {off}"
    )


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_flag_off_roundtrip_is_unchanged(scheme: str) -> None:
    """The old arm still round-trips: pitches and onsets, on the shared fixture."""
    from src.data.encode import get_encoder

    encoder = get_encoder(_rest_cfg(scheme, 0.25, run_length=False))
    piece = make_piece()

    restored = encoder.decode(encoder.encode(piece))
    assert restored.pitches == _expected_pitches(piece, scheme)
    assert [e.start for e in restored.events] == _expected_onsets(piece, scheme)


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize("max_run", [2, 4, 16])
def test_gap_longer_than_cap_decomposes_and_roundtrips(scheme: str, max_run: int) -> None:
    """A gap past the cap becomes several tokens and still lands on the same beat.

    The cap is the vocabulary bound, so it MUST be enforced on emission -- and
    the decomposition must sum back to the original gap, or every note after a
    long silence slides.
    """
    from src.data.encode import get_encoder

    grid = 0.25
    gap = 4 * max_run + 3  # several full runs plus a remainder
    encoder = get_encoder(_rest_cfg(scheme, grid, run_length=True, max_run=max_run))
    piece = _spaced_piece(grid, [gap])

    symbols = encoder.encode(piece)
    runs = _rest_runs(symbols)
    assert runs, f"{scheme}: a {gap}-step gap produced no rest tokens"
    assert max(runs) <= max_run, (
        f"{scheme}: emitted a rest run of {max(runs)} above the cap {max_run}"
    )
    assert sum(runs) == gap - 1, (
        f"{scheme}: rest runs sum to {sum(runs)} steps, want {gap - 1} "
        f"(the onset symbol itself accounts for the last step)"
    )
    # Greedy largest-first: every token but the last is a full-length run.
    assert all(r == max_run for r in runs[:-1]), f"{scheme}: not greedy: {runs}"

    restored = encoder.decode(symbols)
    assert [e.start for e in restored.events] == _expected_onsets(piece, scheme), (
        f"{scheme}: onset moved across a {gap}-step gap at cap {max_run}"
    )


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize(
    "bad",
    [
        "<REST:>",
        "<REST:abc>",
        "<REST:0>",
        "<REST:-3>",
        "<REST:2.5>",
        "<REST: 4>",
        "<REST:99999999>",  # absurd: past any sane cap
        "<REST:+4>",
        "<REST>>",
    ],
)
def test_malformed_rest_runs_do_not_shift_later_onsets(scheme: str, bad: str) -> None:
    """A rest we cannot read must not move the clock.

    This is the subtle failure mode: guessing a length for an illegible rest
    keeps every note but silently slides all of them, which sounds broken while
    looking fine in every pitch-based assertion. Skipping the token loses
    nothing but the silence it claimed.
    """
    from src.data.encode import get_encoder

    encoder = get_encoder(_rest_cfg(scheme, 0.25, run_length=True))
    piece = _spaced_piece(0.25, [2, 5, 1, 3])
    clean = encoder.encode(piece)

    baseline = [e.start for e in encoder.decode(clean).events]
    assert baseline == _expected_onsets(piece, scheme), "fixture itself does not round-trip"

    # Inject the bad token at every position, including the ends.
    for at in range(len(clean) + 1):
        polluted = clean[:at] + [bad] + clean[at:]
        restored = encoder.decode(polluted)  # must not raise
        assert isinstance(restored, Piece)
        assert [e.start for e in restored.events] == baseline, (
            f"{scheme}: {bad!r} at position {at} shifted onsets\n"
            f"  want: {baseline}\n  got:  {[e.start for e in restored.events]}"
        )


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_run_length_rests_cut_token_count_on_a_sparse_piece(scheme: str) -> None:
    """The whole point: fewer tokens, and specifically fewer rest tokens.

    A sparse piece is the realistic case on real performance data at a 16th
    grid -- the measured 56% rest share is exactly this shape.
    """
    from src.data.encode import get_encoder

    grid = 0.25
    piece = _spaced_piece(grid, [8] * 40)  # a note every two beats
    on = get_encoder(_rest_cfg(scheme, grid, run_length=True)).encode(piece)
    off = get_encoder(_rest_cfg(scheme, grid, run_length=False)).encode(piece)

    assert len(on) < 0.5 * len(off), (
        f"{scheme}: run-length rests saved too little: {len(off)} -> {len(on)} tokens"
    )
    assert len(_rest_runs(on)) < 0.2 * len(_rest_runs(off)), (
        f"{scheme}: rest-token count barely moved: "
        f"{len(_rest_runs(off))} -> {len(_rest_runs(on))}"
    )


def test_interval_vocabulary_declares_every_rest_run() -> None:
    """The interval encoder's a-priori set is the ONLY source of its symbols.

    A k missing from it is a rest length the model can never emit, however much
    silence the corpus contains -- and with the flag off, a k present in it is a
    symbol the old arm never uses and must not be charged for.
    """
    from src.data.encode import format_rest_run, get_encoder

    from src.data.types import REST

    max_run = 12
    on = set(get_encoder(_rest_cfg("interval", 0.25, run_length=True, max_run=max_run)).vocabulary())
    off = set(get_encoder(_rest_cfg("interval", 0.25, run_length=False, max_run=max_run)).vocabulary())

    expected = {format_rest_run(k) for k in range(2, max_run + 1)}
    assert expected <= on, f"missing rest runs: {sorted(expected - on)}"
    assert format_rest_run(max_run + 1) not in on, "declared a run above the cap"
    assert REST in on and REST in off
    assert on - off == expected, f"flag-off set is not the old one: {sorted(on - off)}"


def test_max_rest_run_below_one_is_rejected() -> None:
    """A config typo here would silently emit zero-length rests. Fail loudly."""
    from src.data.encode import get_encoder

    with pytest.raises(ValueError):
        get_encoder(_rest_cfg("note_chord", 0.25, run_length=True, max_run=0))
