"""Splitting and transposition augmentation.

Order in the pipeline:

    parse -> quantize -> filter -> SPLIT -> augment -> encode -> vocab -> window

SPLIT COMES BEFORE AUGMENT. This is not a style preference, it is the one
mistake in this stage that silently inflates every number you report:
transposing first and splitting afterwards puts transpositions of the *same*
piece into train and val. The model then sees a val "piece" it has already
memorised twelve semitones away, val loss drops, early stopping fires late,
and the reported result is meaningless. Splitting by piece first makes that
impossible.
"""

from __future__ import annotations

import logging
import random
from typing import Any, List, Sequence, Tuple

from .types import Piece

__all__ = ["split_pieces", "transpose_augment", "transposition_offsets"]

LOGGER = logging.getLogger(__name__)


def split_pieces(
    pieces: Sequence[Piece], cfg: Any
) -> Tuple[List[Piece], List[Piece], List[Piece]]:
    """Split BY PIECE into (train, val, test).

    Call this BEFORE :func:`transpose_augment` -- see the module docstring for
    why. The split is a deterministic function of ``cfg.data.split_seed``, so
    re-running prepare_data reproduces the same partition and a checkpoint can
    still be evaluated against the split it was trained on.
    """
    total = len(pieces)
    if total == 0:
        return [], [], []

    val_frac = float(cfg.data.val_split)
    test_frac = float(cfg.data.test_split)

    order = list(range(total))
    random.Random(int(cfg.data.split_seed)).shuffle(order)

    n_val = int(round(total * val_frac))
    n_test = int(round(total * test_frac))
    # Tiny corpora: never let the held-out splits eat the whole training set.
    while n_val + n_test >= total and (n_val or n_test):
        if n_test >= n_val:
            n_test -= 1
        else:
            n_val -= 1

    val_idx = order[:n_val]
    test_idx = order[n_val : n_val + n_test]
    train_idx = order[n_val + n_test :]

    def take(indices: Sequence[int]) -> List[Piece]:
        return [pieces[i] for i in sorted(indices)]

    train, val, test = take(train_idx), take(val_idx), take(test_idx)

    # Splitting by piece does not stop a leak when the corpus contains the same
    # piece twice under different file names: ADL Classical+Jazz+Blues had 2
    # val and 3 test pieces identical to a training piece (and 1 test piece
    # identical to a val one). Drop held-out copies of anything already on the
    # training side; the key ignores transposition, because train is augmented
    # across keys and a copy in another key is the same leak. Train itself is
    # never touched, so the training tokens and vocabulary are unaffected.
    seen = {_content_key(p) for p in train}
    val, dropped_val = _drop_seen(val, seen)
    seen.update(_content_key(p) for p in val)
    test, dropped_test = _drop_seen(test, seen)
    if dropped_val or dropped_test:
        LOGGER.warning(
            "dropped %d val and %d test pieces that duplicate an earlier split "
            "(same notes, possibly transposed)",
            dropped_val,
            dropped_test,
        )
    LOGGER.info(
        "split %d pieces -> train=%d val=%d test=%d (seed=%s)",
        total,
        len(train),
        len(val),
        len(test),
        cfg.data.split_seed,
    )
    return train, val, test


def _content_key(piece: Piece) -> Tuple[Tuple[float, int], ...]:
    """Transposition-invariant identity of a piece's notes (onsets and pitches)."""
    if not piece.events:
        return ()
    lowest = min(e.pitch for e in piece.events)
    return tuple(sorted((float(e.start), e.pitch - lowest) for e in piece.events))


def _drop_seen(pieces: List[Piece], seen: set) -> Tuple[List[Piece], int]:
    kept: List[Piece] = []
    for piece in pieces:
        key = _content_key(piece)
        if key in seen:
            continue
        seen.add(key)  # also collapses duplicates within the same held-out split
        kept.append(piece)
    return kept, len(pieces) - len(kept)


def transposition_offsets(piece: Piece, cfg: Any) -> List[int]:
    """The non-zero offsets :func:`transpose_augment` would apply to ``piece``.

    Ascending, zero excluded, out-of-range transpositions already removed --
    exactly the copies transpose_augment appends after the original, in the
    same order. Exposed so prepare_data can encode the transpositions straight
    from the original's onset structure instead of materialising millions of
    transposed NoteEvents first (see NoteChordEncoder.encode_transpositions).
    """
    if not bool(cfg.augment.enabled) or not piece.events:
        return []
    lo_offset, hi_offset = (int(x) for x in cfg.augment.transpose_range)
    lo_pitch, hi_pitch = (int(x) for x in cfg.augment.pitch_range)
    pitches = piece.pitches
    piece_lo, piece_hi = min(pitches), max(pitches)
    return [
        offset
        for offset in range(lo_offset, hi_offset + 1)
        if offset != 0 and lo_pitch <= piece_lo + offset and piece_hi + offset <= hi_pitch
    ]


def transpose_augment(pieces: Sequence[Piece], cfg: Any) -> List[Piece]:
    """Return the originals plus every in-range transposition.

    Each offset in ``cfg.augment.transpose_range`` (inclusive at both ends) is
    applied to each piece. A transposition is dropped whole if *any* note would
    leave ``cfg.augment.pitch_range`` -- clamping instead would flatten the
    melodic contour at the extremes and teach the model a shape that is not in
    the data. Originals are always kept, even if they sit outside pitch_range.

    Apply to the TRAINING split only. Augmenting val/test measures nothing new
    and makes the metric depend on the augmentation settings.
    """
    originals = list(pieces)
    if not bool(cfg.augment.enabled):
        return originals

    lo_offset, hi_offset = (int(x) for x in cfg.augment.transpose_range)
    candidates = sum(1 for k in range(lo_offset, hi_offset + 1) if k != 0)

    out: List[Piece] = []
    dropped = 0
    for piece in originals:
        out.append(piece)
        if not piece.events:
            continue
        offsets = transposition_offsets(piece, cfg)
        dropped += candidates - len(offsets)
        out.extend(piece.transposed(offset) for offset in offsets)

    LOGGER.info(
        "augment: %d pieces -> %d (%d transpositions dropped as out of range)",
        len(originals),
        len(out),
        dropped,
    )
    return out
