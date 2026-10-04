"""Evaluation metrics for generated music.

Two families live here:

* ``perplexity`` -- the one likelihood metric, computed on a dataloader. Cheap
  enough to call once per epoch from the training loop.
* symbolic metrics over Pieces -- key adherence, repetition, range, interval
  statistics, pitch-class distribution distance. These are the ones that
  actually catch the two characteristic failure modes of this project:
  ``in_key_percentage`` catches the interval encoder wandering out of key, and
  ``note_repetition_rate`` catches greedy/low-temperature decoding collapsing
  onto a single repeated pitch.
* *set-level* diversity metrics -- ``distinct_n``, ``cross_sample_overlap``,
  ``rhythm_diversity``, ``rest_share``. Everything above scores one piece at a
  time, which is blind to the complaint "it produces the same tune again": N
  individually respectable pieces can still be N copies of each other, and no
  per-piece number moves when they are. These four turn "sounds samey" into a
  reportable number. See their docstrings for which failure each one catches --
  in particular ``rhythm_diversity``, because a real case in this project had
  pitch diversity that looked perfect (cross-sample overlap 0.00) while every
  sample shared one rigid rhythmic skeleton.

Everything except ``perplexity`` is pure numpy: no torch, no music21.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.types import Piece

__all__ = [
    "perplexity",
    "pitch_class_histogram",
    "histogram_distance",
    "key_normalized_histogram",
    "estimate_key",
    "key_name",
    "in_key_percentage",
    "note_repetition_rate",
    "pitch_range",
    "unique_pitch_count",
    "average_interval",
    "distinct_n",
    "cross_sample_overlap",
    "rhythm_diversity",
    "rest_share",
    "longest_common_run",
    "NGramIndex",
    "copy_metrics",
    "DEFAULT_GRID",
    "evaluate_pieces",
    "format_report",
]

# Grid used by the time-based metrics when the caller does not pass one. Matches
# configs/default.yaml's data.grid (0.25 = 16th notes); the metrics that use it
# take it as an argument so a config with a different grid stays measurable.
DEFAULT_GRID = 0.25

# Krumhansl-Schmuckler key profiles: perceived stability of each scale degree,
# indexed from the tonic. Rotating these to all 12 tonics and correlating with
# a piece's pitch-class distribution is the standard cheap key finder.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
# Natural minor. Harmonic/melodic raised degrees are deliberately NOT counted
# as in-key: including them would make the metric forgiving of exactly the
# chromatic drift it exists to detect.
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# --------------------------------------------------------------------------
# likelihood
# --------------------------------------------------------------------------


def perplexity(
    model: Any,
    dataloader: Any,
    device: Any = "cpu",
    ignore_index: int = -100,
) -> float:
    """Token-level perplexity of ``model`` over ``dataloader``.

    Expects batches of ``(x, y)`` as produced by src/data/dataset.py: logits
    ``[B, T, V]`` against shifted targets ``[B, T]``, flattened to
    ``[B*T, V]`` / ``[B*T]`` -- the same reshape src/train.py scores its loss
    with, so this number is directly comparable to ``val_ppl`` in history.csv.
    The older shapes (logits ``[B, V]`` with targets ``[B]``, or a sequence
    model scored against a single next token) still work.

    Pass ``ignore_index=vocab.pad_id`` to exclude padded positions, matching
    the trainer's ``CrossEntropyLoss(ignore_index=pad_id)``. The default
    excludes nothing, since windows built by MusicDataset never contain PAD.

    Returns ``inf`` for an empty dataloader rather than dividing by zero.
    """
    # Imported lazily so the symbolic metrics below stay usable without torch.
    import torch
    import torch.nn.functional as F

    was_training = bool(getattr(model, "training", False))
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    # Accumulated on the device: two .item() calls per batch were two host
    # syncs per batch for no reason. inference_mode keeps no graph.
    total_nll_t = None
    total_tokens_t = None
    with torch.inference_mode():
        for batch in dataloader:
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            own_loss = getattr(model, "loss", None)
            if callable(own_loss) and y.dim() == 2:
                # The model scores itself (see src/models/__init__.py): for
                # model.output=adaptive this never builds [B, T, V] logits.
                # Same per-token definition as below, so numbers stay comparable.
                flat_targets = y.reshape(-1).long()
                count = (flat_targets != ignore_index).sum()
                mean = own_loss(x, y.long(), ignore_index=ignore_index).double()
                loss = torch.where(count > 0, mean * count, torch.zeros_like(mean))
                total_nll_t = loss if total_nll_t is None else total_nll_t + loss
                total_tokens_t = count if total_tokens_t is None else total_tokens_t + count
                continue

            logits = model(x)
            if isinstance(logits, (tuple, list)):  # (logits, hidden) from LSTMs
                logits = logits[0]

            if logits.dim() == 3 and y.dim() == 1:
                # Sequence model scored against a single next token.
                logits = logits[:, -1, :]

            # Same flattening as src/train.py's _run_epoch: every position is
            # a scored token, so perplexity stays per-token no matter how many
            # targets a window carries.
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_targets = y.reshape(-1).long()

            # reduction="sum" already skips ignored entries, so the divisor
            # must skip them too or the mean is diluted toward zero.
            loss = F.cross_entropy(
                flat_logits, flat_targets, reduction="sum", ignore_index=ignore_index
            ).double()
            count = (flat_targets != ignore_index).sum()
            total_nll_t = loss if total_nll_t is None else total_nll_t + loss
            total_tokens_t = count if total_tokens_t is None else total_tokens_t + count

    total_nll = float(total_nll_t.item()) if total_nll_t is not None else 0.0
    total_tokens = int(total_tokens_t.item()) if total_tokens_t is not None else 0

    if was_training:
        model.train()
    if total_tokens == 0:
        return float("inf")
    return float(np.exp(total_nll / total_tokens))


# --------------------------------------------------------------------------
# pitch-class distribution
# --------------------------------------------------------------------------


def pitch_class_histogram(piece: Piece, weight_by_duration: bool = True) -> np.ndarray:
    """Normalized 12-bin pitch-class histogram (index 0 = C).

    Weighting by duration (the default) matches how the Krumhansl-Schmuckler
    profiles were derived: a whole note in the key counts more than a passing
    16th out of it. Returns an all-zero vector for an empty piece.
    """
    hist = np.zeros(12, dtype=float)
    for event in piece.events:
        weight = max(0.0, float(event.duration)) if weight_by_duration else 1.0
        hist[int(event.pitch) % 12] += weight
    total = hist.sum()
    return hist / total if total > 0 else hist


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (``1 - cosine similarity``) between two histograms.

    Cosine rather than JS divergence: it ignores overall magnitude, so a short
    generated excerpt can be compared against a whole reference corpus without
    normalizing sample counts. Range is 0 (identical shape) to 1 (disjoint);
    an all-zero histogram yields 1.0.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return float(1.0 - float(np.dot(a, b)) / denom)


# --------------------------------------------------------------------------
# key
# --------------------------------------------------------------------------


def estimate_key(piece: Piece) -> tuple[int, str, float]:
    """Krumhansl-Schmuckler key estimate.

    Correlates the piece's duration-weighted pitch-class histogram against all
    24 rotated major/minor profiles and returns the best match.

    Returns:
        ``(tonic_pitch_class, mode, correlation)`` where mode is "major" or
        "minor". Falls back to ``(0, "major", 0.0)`` for an empty piece.
    """
    hist = pitch_class_histogram(piece)
    if hist.sum() == 0:
        return 0, "major", 0.0

    best = (0, "major", -2.0)
    for tonic in range(12):
        rotated = np.roll(hist, -tonic)
        for mode, profile in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
            corr = _pearson(rotated, profile)
            if corr > best[2]:
                best = (tonic, mode, corr)
    return best[0], best[1], float(best[2])


def key_name(piece: Piece) -> str:
    """Human-readable estimated key, e.g. ``"D minor"``."""
    tonic, mode, _ = estimate_key(piece)
    return f"{_PITCH_NAMES[tonic]} {mode}"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom == 0.0:
        return -2.0
    return float(np.dot(xc, yc) / denom)


def in_key_percentage(piece: Piece) -> float:
    """Fraction of notes belonging to the estimated key's diatonic scale.

    This is the headline metric for the excessive-modulation failure mode: an
    interval encoder that integrates a drifting sequence of deltas still
    produces plausible local motion but scatters pitch classes uniformly, and
    this number collapses toward 7/12 (~0.58, the chance level) accordingly.
    """
    if not piece.events:
        return 0.0
    tonic, mode, _ = estimate_key(piece)
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    members = {(tonic + degree) % 12 for degree in scale}
    hits = sum(1 for e in piece.events if int(e.pitch) % 12 in members)
    return hits / len(piece.events)


# --------------------------------------------------------------------------
# surface statistics
# --------------------------------------------------------------------------


def note_repetition_rate(piece: Piece) -> float:
    """Fraction of adjacent note pairs with an identical pitch.

    Greedy decoding (or temperature near zero) collapses into hammering one
    note; that shows up here as a rate approaching 1.0 while perplexity still
    looks fine.
    """
    pitches = piece.pitches
    if len(pitches) < 2:
        return 0.0
    same = sum(1 for a, b in zip(pitches, pitches[1:]) if a == b)
    return same / (len(pitches) - 1)


def pitch_range(piece: Piece) -> int:
    """Semitones between the lowest and highest pitch. 0 if empty."""
    pitches = piece.pitches
    return (max(pitches) - min(pitches)) if pitches else 0


def unique_pitch_count(piece: Piece) -> int:
    """Number of distinct MIDI pitches used."""
    return len(set(piece.pitches))


def average_interval(piece: Piece) -> float:
    """Mean absolute semitone step between consecutive notes.

    Near 0 means stuck; very large means the melody is leaping incoherently.
    """
    pitches = piece.pitches
    if len(pitches) < 2:
        return 0.0
    steps = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    return float(np.mean(steps))


# --------------------------------------------------------------------------
# diversity -- metrics over a SET of pieces, not one piece
# --------------------------------------------------------------------------


def _ngrams(items: Sequence[Any], n: int) -> List[tuple]:
    """All contiguous n-grams of ``items``. Empty when the run is too short."""
    if n <= 0 or len(items) < n:
        return []
    return [tuple(items[i:i + n]) for i in range(len(items) - n + 1)]


def distinct_n(pieces: Sequence[Piece], n: int = 1) -> float:
    """Distinct-n: unique note n-grams / total note n-grams over ``pieces``.

    The standard diversity metric from text generation, applied to the pitch
    sequence. 1.0 means no n-gram is ever repeated; low values mean the same
    short figures come round again and again -- whether inside one piece or
    across the set, since n-grams from all pieces are pooled into one count.
    Read n=1 as "how much of the pitch alphabet is in use" and n=3/4 as "how
    much of the phrase vocabulary is".

    COMPARE ONLY AT EQUAL LENGTH. The denominator grows with every note while
    the numerator saturates, so distinct-n falls as the output gets longer even
    when nothing about the model got less diverse. Measured here: raising
    temperature from 0.9 to 1.3 raised the note count per batch from 530 to 789
    and *lowered* distinct_2 from 0.651 to 0.548 -- the opposite of what
    temperature actually did to the distribution. So a distinct-n column is only
    meaningful between runs of comparable ``n_notes``; ``cross_sample_overlap``
    is the safer number when lengths differ, being a ratio over set union.

    Returns 0.0 when no piece is long enough to contain an n-gram.
    """
    total = 0
    unique: set = set()
    for piece in pieces:
        grams = _ngrams(piece.pitches, n)
        total += len(grams)
        unique.update(grams)
    return (len(unique) / total) if total else 0.0


def cross_sample_overlap(pieces: Sequence[Piece], n: int = 3) -> float:
    """Mean pairwise Jaccard overlap of note n-gram SETS between pieces.

    Answers the actual complaint -- "it produces the same tune again" -- as one
    number: 1.0 means every sample is built from the same n-grams as every
    other, 0.0 means they share no figure of length n at all. Averaged over all
    ``P*(P-1)/2`` unordered pairs; 0.0 for fewer than two usable pieces.

    CHOSEN OVER SELF-BLEU deliberately. Self-BLEU measures the same thing but
    drags in a brevity penalty and a smoothing choice that both need defending
    in a write-up, and it costs a full BLEU evaluation per pair. Jaccard over
    n-gram sets is two set operations per pair, has no free parameters beyond
    ``n``, and is symmetric -- so the number means one thing and is cheap enough
    to run on every generation batch.

    Note what this does NOT see: it is a *pitch* metric. Samples can score 0.00
    here -- sharing no melodic figure whatsoever -- and still sound like one
    tune if they share a rhythm. That is ``rhythm_diversity``'s job.
    """
    sets = [set(_ngrams(p.pitches, n)) for p in pieces]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    scores: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            scores.append(len(sets[i] & sets[j]) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _ioi_pattern(piece: Piece, grid: float = DEFAULT_GRID) -> List[int]:
    """Inter-onset intervals of a piece, in whole grid steps.

    Simultaneous notes (a chord) are one onset: the rhythm of a piece is where
    attacks happen, not how many voices attack together. Zero-length gaps are
    dropped for the same reason.
    """
    if grid <= 0:
        grid = DEFAULT_GRID
    onsets = sorted({round(float(e.start) / grid) for e in piece.events})
    return [b - a for a, b in zip(onsets, onsets[1:]) if b > a]


def rhythm_diversity(
    pieces: Sequence[Piece], n: int = 3, grid: float = DEFAULT_GRID
) -> float:
    """Normalized Shannon entropy over inter-onset-interval n-gram patterns.

    THIS IS THE METRIC THAT WOULD HAVE CAUGHT THE REAL BUG IN THIS PROJECT.
    A batch of samples was diagnosed as "the same tune again" while every
    pitch-based metric said the opposite: zero duplicate note sequences,
    ``cross_sample_overlap`` 0.00 across all 15 pairs. What was identical was
    the *rhythm* -- every sample was 75-77% rest with the same
    one-note-then-three-rests skeleton, so the IOI pattern was (4, 4, 4)
    everywhere and this number sits near 0 while nothing else moves.

    Definition: reduce each piece to its inter-onset intervals in grid steps,
    take n-grams of those, pool over all pieces, and report the Shannon entropy
    of the resulting distribution divided by ``log(number of distinct patterns
    possible to observe)`` -- i.e. by ``log(total patterns)``, the entropy of a
    hypothetical run where no pattern ever repeated. So:

    * 0.0  -- one rhythmic figure, used everywhere. Rigid.
    * ~1.0 -- every rhythmic figure observed is unique. Free (possibly random).

    Normalizing makes the value comparable across runs of different length,
    which a raw entropy in bits is not. Returns 0.0 when there are fewer than
    two pooled patterns, since "diversity" is not defined on one observation.
    """
    patterns: List[tuple] = []
    for piece in pieces:
        patterns.extend(_ngrams(_ioi_pattern(piece, grid), n))
    if len(patterns) < 2:
        return 0.0

    counts = np.array(list(Counter(patterns).values()), dtype=float)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    ceiling = float(np.log(len(patterns)))
    if ceiling <= 0:
        return 0.0
    # Clamped rather than returned raw: a single repeated pattern gives
    # entropy -0.0, which prints as "-0.000" and reads like a bug in a report.
    return float(min(1.0, max(0.0, entropy / ceiling)))


def rest_share(piece: Piece, grid: float = DEFAULT_GRID) -> float:
    """Fraction of the piece's grid steps on which nothing is sounding.

    Measured on the piece's own span (first onset to last note-off), so it is
    about texture rather than about a trailing silence. Sparsity is the other
    half of the "sounds samey" story here: a sample that is three quarters rest
    has very little music in it to be diverse *with*, and a batch of such
    samples all sound like the same sparse plink regardless of which pitches
    they plink. Worth reading next to ``rhythm_diversity``.

    Returns 0.0 for a piece with no events (no span, hence no silence to
    measure) -- do not read that as "dense".
    """
    if not piece.events:
        return 0.0
    if grid <= 0:
        grid = DEFAULT_GRID

    origin = min(float(e.start) for e in piece.events)
    span_end = max(float(e.end) for e in piece.events)
    total_steps = int(round((span_end - origin) / grid))
    if total_steps <= 0:
        return 0.0

    sounding: set = set()
    for event in piece.events:
        start = int(round((float(event.start) - origin) / grid))
        # A note shorter than one grid step still occupies the step it lands on.
        length = max(1, int(round(float(event.duration) / grid)))
        sounding.update(range(max(0, start), min(total_steps, start + length)))
    return 1.0 - len(sounding) / total_steps


# --------------------------------------------------------------------------
# memorisation -- is the model continuing music, or reciting it?
# --------------------------------------------------------------------------
#
# Token-level, on vocabulary ids, because that is the level at which a model
# copies: a recited passage reproduces the exact symbol stream. Typical use
# (see the audit report): prime from a real piece at offset o, generate K
# tokens, and compare them with (a) that piece's TRUE next K tokens and (b)
# every n-gram of the training split. A model that generalises shares short
# figures with the corpus (4-grams: scales, arpeggios, rest patterns) but not
# long runs; a model that memorised reproduces 8-grams and long verbatim runs.


def longest_common_run(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Length of the longest contiguous run that appears in both sequences
    (longest common substring, O(len(a) * len(b)) -- fine at K ~ 500)."""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _ngram_hashes(seq: Any, n: int) -> np.ndarray:
    """uint64 polynomial hash of every contiguous n-gram of ``seq``.

    Wrap-around multiplication makes this a hash, not an encoding, so two
    different n-grams can collide; at 64 bits over ~1e8 n-grams the expected
    number of false matches is ~1e-3, i.e. none worth reporting.
    """
    arr = np.asarray(seq, dtype=np.uint64)
    if arr.size < n:
        return np.zeros(0, dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(arr, n)
    h = np.zeros(windows.shape[0], dtype=np.uint64)
    base = np.uint64(1_000_003)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * base + windows[:, i] + np.uint64(1)
    return h


class NGramIndex:
    """Every n-gram of a corpus (e.g. the training split), for membership tests.

    Built once as a sorted, de-duplicated uint64 array of n-gram hashes: about
    8 bytes per distinct n-gram, and lookups are a vectorised searchsorted. A
    Python set of tuples would cost ~20x the memory on a 13M-token corpus.
    """

    def __init__(self, sequences: Any, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        parts = [_ngram_hashes(seq, self.n) for seq in sequences]
        parts = [p for p in parts if p.size]
        self._hashes = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)

    def __len__(self) -> int:
        return int(self._hashes.size)

    def contains(self, seq: Sequence[int]) -> np.ndarray:
        """Boolean per n-gram of ``seq``: does it occur in the index?"""
        h = _ngram_hashes(seq, self.n)
        if h.size == 0 or self._hashes.size == 0:
            return np.zeros(h.size, dtype=bool)
        pos = np.searchsorted(self._hashes, h)
        pos = np.minimum(pos, self._hashes.size - 1)
        return self._hashes[pos] == h

    def overlap(self, seq: Sequence[int]) -> Optional[float]:
        """Fraction of ``seq``'s n-grams found in the index; None if too short."""
        hits = self.contains(seq)
        return float(hits.mean()) if hits.size else None


def copy_metrics(
    generated: Sequence[int],
    truth: Optional[Sequence[int]] = None,
    indexes: Optional[Dict[int, "NGramIndex"]] = None,
) -> Dict[str, Any]:
    """Memorisation numbers for one continuation.

    * ``prefix_match`` -- how many leading generated tokens equal the primer
      piece's true continuation, token for token.
    * ``lcs_run`` -- longest contiguous run shared with that true continuation,
      anywhere (a recitation that starts a few tokens late still counts).
    * ``train_<n>gram`` -- fraction of the generated n-grams that occur
      somewhere in the indexed corpus, one key per index.
    """
    generated = [int(t) for t in generated]
    row: Dict[str, Any] = {"tokens": len(generated)}
    if truth is not None:
        truth = [int(t) for t in truth]
        prefix = 0
        for g, t in zip(generated, truth):
            if g != t:
                break
            prefix += 1
        row["prefix_match"] = prefix
        row["lcs_run"] = longest_common_run(generated, truth)
    for n, index in sorted((indexes or {}).items()):
        row[f"train_{n}gram"] = index.overlap(generated)
    return row


# --------------------------------------------------------------------------
# bundling
# --------------------------------------------------------------------------


_DIVERSITY_KEYS = (
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "distinct_4",
    "cross_overlap",
    "rhythm_div",
    "rest_share",
)


def evaluate_pieces(
    generated: Sequence[Piece],
    reference: Optional[Sequence[Piece]] = None,
    label: str = "generated",
    grid: float = DEFAULT_GRID,
) -> Dict[str, Any]:
    """Aggregate every symbolic metric into a single results-table row.

    Per-piece metrics are averaged over ``generated``. The distribution
    distance ``pc_distance`` compares the KEY-NORMALISED pooled pitch-class
    histogram of ``generated`` against that of ``reference`` (usually the
    training split) -- see ``_pooled_histogram`` for why it is not a raw
    pooled histogram any more. It is ``None`` when no reference is supplied.
    Numbers from before this change are not comparable with new ones.

    The diversity block (``distinct_1``..``distinct_4``, ``cross_overlap``,
    ``rhythm_div``, ``rest_share``) is a property of the *set*, not of any one
    piece, so it is computed over ``generated`` as a whole rather than averaged
    -- except ``rest_share``, which is per-piece and therefore averaged.
    ``cross_overlap`` needs at least two pieces to mean anything; with one piece
    it reads 0.0, which is absence of evidence, not diversity.

    ``grid`` is the quantization grid in beats, for the two time-based metrics;
    pass ``cfg.data.grid`` when it is not the 0.25 default.
    """
    generated = list(generated)
    row: Dict[str, Any] = {
        "label": label,
        "n_pieces": len(generated),
        "n_notes": sum(len(p) for p in generated),
    }
    if not generated:
        row.update(
            {
                "in_key_pct": 0.0,
                "repetition_rate": 0.0,
                "pitch_range": 0.0,
                "unique_pitches": 0.0,
                "avg_interval": 0.0,
                "pc_distance": None,
            }
        )
        row.update({key: 0.0 for key in _DIVERSITY_KEYS})
        return row

    row["in_key_pct"] = float(np.mean([in_key_percentage(p) for p in generated]))
    row["repetition_rate"] = float(np.mean([note_repetition_rate(p) for p in generated]))
    row["pitch_range"] = float(np.mean([pitch_range(p) for p in generated]))
    row["unique_pitches"] = float(np.mean([unique_pitch_count(p) for p in generated]))
    row["avg_interval"] = float(np.mean([average_interval(p) for p in generated]))

    for n in (1, 2, 3, 4):
        row[f"distinct_{n}"] = distinct_n(generated, n)
    row["cross_overlap"] = cross_sample_overlap(generated, n=3)
    row["rhythm_div"] = rhythm_diversity(generated, n=3, grid=grid)
    row["rest_share"] = float(np.mean([rest_share(p, grid) for p in generated]))

    if reference:
        row["pc_distance"] = histogram_distance(
            _pooled_histogram(generated), _pooled_histogram(reference)
        )
    else:
        row["pc_distance"] = None
    return row


def key_normalized_histogram(piece: Piece) -> np.ndarray:
    """Pitch-class histogram rotated so the piece's key signature sits on C.

    Major keys are rotated by their tonic, minor keys by their relative major
    (tonic + 3), so C major and A minor -- and every transposition of either --
    land on the same seven white-key bins. Weighted by the piece's total
    duration so pooling it with others counts long pieces for more, as the
    unnormalised pooled histogram does.
    """
    hist = pitch_class_histogram(piece)
    if hist.sum() == 0:
        return hist
    tonic, mode, _ = estimate_key(piece)
    shift = tonic if mode == "major" else (tonic + 3) % 12
    weight = sum(max(0.0, float(e.duration)) for e in piece.events)
    return np.roll(hist, -shift) * weight


def _pooled_histogram(pieces: Sequence[Piece]) -> np.ndarray:
    """Key-normalised pitch-class histogram of a set of pieces.

    WHY KEY-NORMALISED. The previous version pooled raw pitch classes over the
    whole set. The reference is the training corpus, and with +/-6 semitone
    augmentation (or simply a corpus in many keys) that pool is close to FLAT
    -- every pitch class equally common. A cosine distance against a flat
    histogram is smallest for output that is itself flat, i.e. atonal: a
    uniformly chromatic sample scored 0.23 while a perfectly diatonic G major
    scale scored 0.43 (tests/test_model_audit.py). The metric rewarded exactly
    the failure it is meant to catch. Rotating every piece into a common key
    first compares *tonal shape* instead, which is what "pitch-class
    distribution like the corpus" is supposed to mean.
    """
    total = np.zeros(12, dtype=float)
    for piece in pieces:
        total += key_normalized_histogram(piece)
    s = total.sum()
    return total / s if s > 0 else total


def format_report(rows: Sequence[Dict[str, Any]]) -> str:
    """Render evaluate_pieces rows as a markdown table.

    Column order follows the union of keys in ``rows``, seeded by a preferred
    order so the identifying columns come first.
    """
    rows = list(rows)
    if not rows:
        return "_no results_"

    preferred = [
        "label",
        "n_pieces",
        "n_notes",
        "perplexity",
        "in_key_pct",
        # Diversity next to in_key_pct on purpose: the two trade off against
        # each other as temperature moves, and a reader needs to see both
        # columns at once to judge whether a run bought variety with noise.
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "distinct_4",
        "cross_overlap",
        "rhythm_div",
        "rest_share",
        "repetition_rate",
        "pc_distance",
        "pitch_range",
        "unique_pitches",
        "avg_interval",
    ]
    columns: List[str] = [c for c in preferred if any(c in r for r in rows)]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
