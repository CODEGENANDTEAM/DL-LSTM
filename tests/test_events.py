"""EventEncoder (``encoding.scheme: event``): T{k} / N{p} / D{k} [/ V{b}].

The contract is stronger than the other schemes': pitch, onset AND duration
round-trip exactly (up to the duration cap), the vocabulary is declared
complete, and no malformed symbol a sampler can emit may move a later onset.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.encode import EventEncoder, get_encoder
from src.data.types import BOS, EOS, PAD, REST, RESERVED_SYMBOLS, UNK, NoteEvent, Piece


def make_encoder(**overrides) -> EventEncoder:
    cfg = load_config(**{"encoding.scheme": "event", **overrides})
    encoder = get_encoder(cfg)
    assert isinstance(encoder, EventEncoder)
    return encoder


def key(piece: Piece):
    return [(e.pitch, e.start, e.duration) for e in piece.events]


def canonical(piece: Piece):
    return sorted(key(piece), key=lambda t: (t[1], t[0]))


def random_piece(seed: int, grid: float = 0.25, n: int = 300, max_dur_steps: int = 32) -> Piece:
    """Random quantised polyphony, including long gaps and chords."""
    rng = random.Random(seed)
    events = []
    step = rng.randrange(0, 20)          # leading silence too
    for _ in range(n):
        for pitch in rng.sample(range(21, 109), rng.choice([1, 1, 1, 2, 3, 4])):
            events.append(NoteEvent(pitch, round(step * grid, 6),
                                    round(rng.randint(1, max_dur_steps) * grid, 6),
                                    rng.randint(1, 127)))
        step += rng.choice([1, 1, 2, 3, 4, 8, 16, 17, 40, 100])
    return Piece(events=sorted(events, key=lambda e: (e.start, e.pitch)))


def make_piece() -> Piece:
    return Piece(events=[
        NoteEvent(60, 0.5, 1.0),          # leading half-beat of silence
        NoteEvent(64, 0.5, 0.25),         # chord, different durations
        NoteEvent(67, 0.5, 2.0),
        NoteEvent(62, 1.0, 0.5),          # overlaps the chord still sounding
        NoteEvent(65, 11.0, 8.0),         # 40-step gap, 32-step (cap) duration
    ])


# -- round trip -------------------------------------------------------------


def test_registered_and_carries_durations() -> None:
    encoder = make_encoder(**{"encoding.include_duration": False})
    # generate.py skips fill_durations exactly when this is True.
    assert encoder.include_duration is True
    assert encoder.name == "event"


def test_exact_symbols_for_a_known_piece() -> None:
    assert make_encoder().encode(make_piece()) == [
        "T2", "N60", "D4", "N64", "D1", "N67", "D8",
        "T2", "N62", "D2",
        "T16", "T16", "T8", "N65", "D32",
    ]


@pytest.mark.parametrize("grid", [0.25, 0.5, 1 / 3])
@pytest.mark.parametrize("seed", range(4))
def test_roundtrip_is_exact(grid: float, seed: int) -> None:
    encoder = make_encoder(**{"data.grid": grid})
    piece = random_piece(seed, grid=grid)
    restored = encoder.decode(encoder.encode(piece))
    assert canonical(restored) == canonical(piece)


def test_roundtrip_exact_chords_and_long_gaps() -> None:
    encoder = make_encoder()
    piece = make_piece()
    assert canonical(encoder.decode(encoder.encode(piece))) == canonical(piece)


def test_durations_clamp_to_the_cap_and_floor_at_one_step() -> None:
    encoder = make_encoder()
    piece = Piece(events=[NoteEvent(60, 0.0, 40 * 0.25), NoteEvent(62, 1.0, 0.1)])
    symbols = encoder.encode(piece)
    assert symbols == ["N60", "D32", "T4", "N62", "D1"]
    assert key(encoder.decode(symbols)) == [(60, 0.0, 8.0), (62, 1.0, 0.25)]


def test_duplicate_pitch_at_one_onset_keeps_the_longest() -> None:
    encoder = make_encoder()
    piece = Piece(events=[NoteEvent(60, 0.0, 0.25), NoteEvent(60, 0.0, 1.0)])
    assert encoder.encode(piece) == ["N60", "D4"]


# -- caps -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cap,gap,expected",
    [
        (16, 1, ["T1"]),
        (16, 16, ["T16"]),
        (16, 17, ["T16", "T1"]),
        (16, 40, ["T16", "T16", "T8"]),
        (16, 32, ["T16", "T16"]),
        (4, 10, ["T4", "T4", "T2"]),
        (1, 3, ["T1", "T1", "T1"]),
    ],
)
def test_time_shift_decomposes_greedily(cap: int, gap: int, expected) -> None:
    encoder = make_encoder(**{"encoding.max_time_shift": cap})
    piece = Piece(events=[NoteEvent(60, 0.0, 0.25), NoteEvent(62, gap * 0.25, 0.25)])
    symbols = encoder.encode(piece)
    assert symbols == ["N60", "D1", *expected, "N62", "D1"]
    assert canonical(encoder.decode(symbols)) == canonical(piece)


def test_decode_honours_shifts_and_durations_beyond_this_configs_caps() -> None:
    """Tokens from a config with bigger caps must still decode on time."""
    encoder = make_encoder()
    restored = encoder.decode(["T40", "N60", "D48", "T1", "N61", "D1"])
    assert key(restored) == [(60, 10.0, 12.0), (61, 10.25, 0.25)]


def test_bad_caps_are_rejected() -> None:
    for field in ("max_time_shift", "max_duration_steps"):
        with pytest.raises(ValueError):
            make_encoder(**{f"encoding.{field}": 0})
    with pytest.raises(ValueError):
        make_encoder(**{"encoding.velocity_bins": 128})


# -- malformed symbols ------------------------------------------------------

GARBAGE = [
    "", "xyz", "!!", "T0", "T-3", "T99999", "Tabc", "T1.5", "T 4", "N128", "N-1",
    "N6O", "N60.64", "Nx", "D0", "D-2", "Dx", "D1.000", "V3", "60.64.67",
    "<REST:abc>", "٣", "T٣", "N²",
]
IGNORED = [PAD, UNK, BOS, EOS, REST, "<REST:4>", "<STYLE:jazz>", "<STYLE:>"]
# Empty strings and anything in angle brackets are skipped like reserved tokens
# (the pending note survives them); everything else unreadable breaks the
# N -> D pairing.
UNREADABLE = [s for s in GARBAGE if s and not s.startswith("<")]


def onsets(piece: Piece):
    return [(e.pitch, e.start) for e in piece.events]


@pytest.mark.parametrize("bad", GARBAGE + IGNORED)
def test_garbage_between_notes_changes_nothing(bad: str) -> None:
    encoder = make_encoder()
    symbols = encoder.encode(make_piece())
    reference = key(encoder.decode(symbols))
    # After every complete N/D pair and before every T: the piece must be
    # bit-identical, pitches, onsets and durations.
    for position in range(len(symbols) + 1):
        if position < len(symbols) and symbols[position].startswith("D"):
            continue
        polluted = symbols[:position] + [bad] + symbols[position:]
        assert key(encoder.decode(polluted)) == reference, (position, bad)


@pytest.mark.parametrize("bad", UNREADABLE)
def test_garbage_between_n_and_d_only_costs_that_duration(bad: str) -> None:
    encoder = make_encoder()
    symbols = ["N60", bad, "D8", "T4", "N62", "D4"]
    restored = encoder.decode(symbols)
    assert onsets(restored) == [(60, 0.0), (62, 1.0)]
    # The D cannot be trusted to belong to N60 across unreadable text.
    assert restored.events[0].duration == 0.5
    assert restored.events[1].duration == 1.0


@pytest.mark.parametrize("ignored", IGNORED + ["", "<REST:abc>"])
def test_reserved_between_n_and_d_keeps_the_pairing(ignored: str) -> None:
    encoder = make_encoder()
    restored = encoder.decode(["N60", ignored, "D8", "T4", "N62", "D4"])
    assert key(restored) == [(60, 0.0, 2.0), (62, 1.0, 1.0)]


def test_orphan_durations_are_ignored() -> None:
    encoder = make_encoder()
    restored = encoder.decode([
        "D4",                      # at the start: nothing to attach to
        "N60", "D8", "D16",        # second D for one note
        "T4", "D3",                # D after a T: the note is in the past
        "N62", "D4",
    ])
    assert key(restored) == [(60, 0.0, 2.0), (62, 1.0, 1.0)]


def test_note_without_duration_gets_the_default_and_time_is_unchanged() -> None:
    encoder = make_encoder()
    restored = encoder.decode(["N60", "N64", "D4", "T4", "N62", "T2", "N65", "D1"])
    assert key(restored) == [
        (60, 0.0, 0.5),            # default: 2 steps, the ADL median
        (64, 0.0, 1.0),
        (62, 1.0, 0.5),
        (65, 1.5, 0.25),
    ]


def test_all_garbage_decodes_to_an_empty_piece() -> None:
    encoder = make_encoder()
    assert len(encoder.decode(GARBAGE + IGNORED)) == 0


def test_random_symbol_soup_never_raises_and_onsets_follow_t_only() -> None:
    encoder = make_encoder()
    vocab = list(encoder.vocabulary()) + GARBAGE + IGNORED
    rng = random.Random(7)
    for _ in range(50):
        symbols = [rng.choice(vocab) for _ in range(400)]
        restored = encoder.decode(symbols)
        clock, expected = 0, []
        for s in symbols:
            if s in encoder._lookup and s.startswith("T"):
                clock += int(s[1:])
            elif s in encoder._lookup and s.startswith("N"):
                expected.append((int(s[1:]), clock * 0.25))
        assert sorted(onsets(restored), key=lambda t: (t[1], t[0])) == sorted(
            expected, key=lambda t: (t[1], t[0])
        )


# -- vocabulary -------------------------------------------------------------


def test_vocabulary_is_fixed_and_complete() -> None:
    encoder = make_encoder()
    vocab = list(encoder.vocabulary())
    assert len(vocab) == 16 + 128 + 32 == 176
    assert len(set(vocab)) == len(vocab)
    assert not set(vocab) & set(RESERVED_SYMBOLS)
    assert len(list(make_encoder(**{"encoding.velocity_bins": 8}).vocabulary())) == 184


@pytest.mark.parametrize("bins", [0, 8])
def test_encoding_never_leaves_the_vocabulary(bins: int) -> None:
    encoder = make_encoder(**{"encoding.velocity_bins": bins})
    vocab = set(encoder.vocabulary())
    for seed in range(5):
        piece = random_piece(seed, max_dur_steps=80)   # durations past the cap
        for symbols in encoder.encode_transpositions(piece, [-6, 0, 6]):
            assert set(symbols) <= vocab


def _real_pieces(limit: int = 12):
    raw = ROOT / "data" / "raw" / "adl"
    cfg = load_config(str(ROOT / "configs" / "adl_events.yaml"))
    paths = []
    for style in ("Classical", "Jazz", "Blues"):
        found = sorted((raw / style).rglob("*.mid"))
        paths.extend(found[: limit // 3])
    if not paths:
        pytest.skip("ADL corpus not present")
    from src.data.parse import load_piece

    pieces = [p for p in (load_piece(path, cfg) for path in paths) if p is not None]
    if not pieces:
        pytest.skip("no ADL piece parsed")
    return cfg, pieces


def test_real_corpus_pieces_roundtrip_and_stay_in_vocabulary() -> None:
    cfg, pieces = _real_pieces()
    encoder = get_encoder(cfg)
    vocab = set(encoder.vocabulary())
    cap = encoder.max_duration_steps * encoder.grid
    for piece in pieces:
        symbols = encoder.encode(piece)
        assert set(symbols) <= vocab
        expected = [(e.pitch, e.start, min(e.duration, cap)) for e in piece.events]
        assert canonical(encoder.decode(symbols)) == sorted(expected, key=lambda t: (t[1], t[0]))


def test_vocab_build_keeps_every_declared_symbol_despite_pruning() -> None:
    from src.data.vocab import Vocab

    encoder = make_encoder()
    vocab = Vocab.build([encoder.encode(make_piece())], encoder=encoder, min_freq=1000)
    assert set(encoder.vocabulary()) <= set(vocab.itos)
    assert len(vocab) == 5 + 176


# -- determinism, transpositions, style tokens --------------------------------


def test_encoding_is_deterministic_and_order_independent() -> None:
    encoder = make_encoder()
    piece = random_piece(3)
    shuffled = list(piece.events)
    random.Random(0).shuffle(shuffled)
    first = encoder.encode(piece)
    assert first == encoder.encode(piece)
    assert first == make_encoder().encode(piece)
    assert first == encoder.encode(Piece(events=shuffled))


def test_chord_notes_are_in_ascending_pitch() -> None:
    encoder = make_encoder()
    piece = Piece(events=[NoteEvent(p, 0.0, 0.25) for p in (72, 48, 60, 55)])
    assert [s for s in encoder.encode(piece) if s.startswith("N")] == ["N48", "N55", "N60", "N72"]


@pytest.mark.parametrize("bins", [0, 8])
def test_encode_transpositions_equals_encoding_each_transposition(bins: int) -> None:
    encoder = make_encoder(**{"encoding.velocity_bins": bins})
    piece = random_piece(11)
    offsets = [0, -6, -1, 3, 6]
    assert encoder.encode_transpositions(piece, offsets) == [
        encoder.encode(piece.transposed(k)) for k in offsets
    ]


def test_style_tokens_pass_through() -> None:
    from src.data.dataset import pieces_to_symbols, style_symbol

    cfg = load_config(**{"encoding.scheme": "event", "encoding.style_tokens": True})
    encoder = get_encoder(cfg)
    piece = make_piece()
    piece.style = "Jazz"
    [symbols] = pieces_to_symbols([piece], encoder, cfg)
    assert symbols[0] == style_symbol("Jazz")
    assert symbols[1:] == encoder.encode(piece)
    assert key(encoder.decode(symbols)) == key(encoder.decode(symbols[1:]))


# -- velocity ---------------------------------------------------------------


@pytest.mark.parametrize("bins", [1, 2, 3, 8, 16, 32, 127])
def test_velocity_bins_are_idempotent(bins: int) -> None:
    encoder = make_encoder(**{"encoding.velocity_bins": bins})
    for b in range(bins):
        centre = encoder.bin_velocity(b)
        assert 1 <= centre <= 127
        assert encoder.velocity_bin(centre) == b
    seen = {encoder.velocity_bin(v) for v in range(1, 128)}
    assert seen == set(range(bins))


def test_velocity_roundtrips_to_its_bin() -> None:
    encoder = make_encoder(**{"encoding.velocity_bins": 8})
    piece = random_piece(5)
    restored = encoder.decode(encoder.encode(piece))
    assert canonical(restored) == canonical(piece)
    by_note = {(e.pitch, e.start): e.velocity for e in restored.events}
    for e in piece.events:
        assert encoder.velocity_bin(by_note[(e.pitch, e.start)]) == encoder.velocity_bin(e.velocity)
    assert "V" in "".join(s[0] for s in encoder.encode(piece))


def test_velocity_off_decodes_default_velocity_and_ignores_v_symbols() -> None:
    encoder = make_encoder()
    restored = encoder.decode(["N60", "D4", "V3", "T1", "N62", "D1"])
    assert key(restored) == [(60, 0.0, 1.0), (62, 0.25, 0.25)]
    assert {e.velocity for e in restored.events} == {80}


# -- shared onset grouping (touched by this change) ----------------------------


@pytest.mark.parametrize("scheme", ["note_chord", "interval", "pitch_duration", "event"])
def test_unsorted_input_encodes_like_sorted_input(scheme: str) -> None:
    encoder = get_encoder(load_config(**{"encoding.scheme": scheme}))
    piece = random_piece(9, n=80)
    shuffled = list(piece.events)
    random.Random(1).shuffle(shuffled)
    assert encoder.encode(Piece(events=shuffled)) == encoder.encode(piece)


def test_adl_events_config() -> None:
    cfg = load_config(str(ROOT / "configs" / "adl_events.yaml"))
    assert cfg.encoding.scheme == "event"
    assert cfg.name == "adl_events_lstm"
    assert cfg.model.window_stride * 2 == cfg.model.seq_len
    mixed = load_config(str(ROOT / "configs" / "adl_mixed.yaml"))
    assert cfg.data == mixed.data
    assert cfg.augment == mixed.augment
    assert cfg.encoding.style_tokens == mixed.encoding.style_tokens
