"""Windowing, DataLoaders, and the processed-data cache.

Last stage of the pipeline:

    parse -> quantize -> filter -> SPLIT -> augment -> encode -> vocab -> window

A training example is a fixed-length window of token ids and that same window
shifted one position to the left -- teacher forcing, so position t of x is
supervised by position t of y:

    x = seq[i     : i + seq_len]
    y = seq[i + 1 : i + seq_len + 1]                            stride 1

Both are LongTensor[seq_len]. The earlier form returned a scalar target
(``y = seq[i + seq_len]``), which supervised only the final position: at
seq_len=100 that discarded 99 of the 100 predictions the model computed on
every forward pass.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
from pathlib import Path
from collections import abc
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import config_hash, resolve_path
from .types import Piece
from .vocab import Vocab

__all__ = [
    "MusicDataset",
    "RaggedSequences",
    "make_dataloaders",
    "pieces_to_symbols",
    "pieces_to_ids",
    "style_symbol",
    "processed_dir",
    "save_processed",
    "load_processed",
    "load_split",
]

LOGGER = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
CACHE_SECTIONS = ("data", "augment", "encoding")
VOCAB_FILENAME = "vocab.json"
META_FILENAME = "meta.json"

# Bumped whenever a code change alters the tokens prepare_data produces for an
# unchanged config. The cache directory is keyed on the config alone, so
# without this a cache written by older code is indistinguishable from a fresh
# one. Recorded in meta.json; load_processed warns when a cache is older.
# Caches written before the field existed count as version 1.
#
#   v1  everything up to the 2026-09 data audit
#   v2  style tokens honoured by prepare_data; duplicate (onset, pitch) notes
#       merged; held-out pieces duplicating a training piece dropped
DATA_PIPELINE_VERSION = 2
PIPELINE_CHANGES = (
    "v2: encoding.style_tokens is honoured, duplicate same-onset notes are "
    "merged, held-out duplicates of training pieces are dropped"
)


def style_symbol(style: str) -> str:
    """The conditioning token for a style label, e.g. ``<STYLE:baroque>``."""
    return f"<STYLE:{style}>"


def pieces_to_symbols(
    pieces: Sequence[Piece], encoder: Any, cfg: Any
) -> List[List[str]]:
    """Encode pieces to symbol sequences, one list per piece.

    The piece boundary is preserved here and all the way to the window index --
    see MusicDataset. When ``cfg.encoding.style_tokens`` is on and a piece has a
    style, the sequence is prefixed with its style token; run this same function
    before building the Vocab so those tokens are in it.
    """
    use_style = bool(cfg.encoding.style_tokens)
    sequences: List[List[str]] = []
    for piece in pieces:
        symbols = list(encoder.encode(piece))
        if use_style and piece.style:
            symbols.insert(0, style_symbol(piece.style))
        sequences.append(symbols)
    return sequences


def pieces_to_ids(
    pieces: Sequence[Piece], encoder: Any, vocab: Vocab, cfg: Any
) -> List[List[int]]:
    """pieces -> per-piece token-id sequences."""
    return [vocab.encode(s) for s in pieces_to_symbols(pieces, encoder, cfg)]


class RaggedSequences(abc.Sequence):
    """Many variable-length token sequences in one flat numpy array.

    ``seqs[i]`` is a zero-copy numpy view of sequence i. This is the in-memory
    form of the processed cache. The previous list-of-lists-of-Python-ints
    cost one Python object per token (about 36 bytes each, ~0.5 GB transient
    for ADL's 12.75M tokens); this costs 4 bytes per token.

    It is a read-only ``Sequence``, so code written for a list of lists --
    ``len``, indexing, iteration, ``list(map(int, seq))`` -- keeps working.
    """

    def __init__(self, flat: np.ndarray, lengths: np.ndarray) -> None:
        self.flat = np.ascontiguousarray(flat)
        self.lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        self.starts = np.zeros(len(self.lengths) + 1, dtype=np.int64)
        np.cumsum(self.lengths, out=self.starts[1:])
        if int(self.starts[-1]) != len(self.flat):
            raise ValueError(
                f"lengths sum to {int(self.starts[-1])} but there are {len(self.flat)} tokens"
            )

    @classmethod
    def from_sequences(
        cls, sequences: Iterable[Sequence[int]], dtype: Any = np.int32
    ) -> "RaggedSequences":
        if isinstance(sequences, RaggedSequences):
            return sequences
        arrays = [np.asarray(s, dtype=dtype).reshape(-1) for s in sequences]
        lengths = np.asarray([len(a) for a in arrays], dtype=np.int64)
        flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=dtype)
        return cls(flat.astype(dtype, copy=False), lengths)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> np.ndarray:  # type: ignore[override]
        if isinstance(index, slice):
            raise TypeError("RaggedSequences does not support slicing")
        n = len(self.lengths)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(index)
        return self.flat[self.starts[index] : self.starts[index + 1]]

    def __iter__(self) -> Iterator[np.ndarray]:
        flat, starts = self.flat, self.starts
        for i in range(len(self.lengths)):
            yield flat[starts[i] : starts[i + 1]]

    @property
    def num_tokens(self) -> int:
        return int(self.starts[-1])

    def __repr__(self) -> str:
        return f"RaggedSequences({len(self)} sequences, {self.num_tokens} tokens)"


class MusicDataset(Dataset):
    """Sliding windows over token-id sequences, one sequence per piece.

    WINDOWS NEVER CROSS A PIECE BOUNDARY. Concatenating every piece into one
    long stream and windowing that is the obvious implementation and it is
    wrong: the model is trained on transitions from the end of Chopin into the
    start of Bach, which occur nowhere in the data and which it will happily
    reproduce at generation time. Windows are enumerated per piece and only the
    *indices* are concatenated.

    The index is stored as per-sequence window counts plus a running total and
    resolved with a binary search, rather than as a materialised list of
    (piece, offset) pairs -- an augmented corpus easily reaches millions of
    windows and that list would cost more memory than the token data itself.
    The tokens live in one flat numpy array (see RaggedSequences) and a window
    is a view into it, so nothing is copied until the tensor is made.

    ``cover_tail``: with a stride > 1 the last strided window can stop up to
    stride-1 tokens short of the end of a piece, and those final tokens are
    then never a training target. At seq_len 256 / stride 128 on ADL that was
    825k of 12.75M training tokens (6.5%) -- specifically every piece's
    ending, so the model never saw how music ends. With ``cover_tail`` one
    extra window per piece is aligned to the end of the piece whenever the
    strided windows miss it. It defaults off here so the enumeration of a bare
    MusicDataset is unchanged; make_dataloaders turns it on.
    """

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        seq_len: int,
        stride: int = 1,
        cover_tail: bool = False,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.seq_len = int(seq_len)
        # Stride 1 emits a window at every single token, so consecutive
        # windows overlap by seq_len-1 and carry almost the same gradient.
        # On a corpus of a few million tokens that is tens of hours per
        # epoch for very little extra signal. A stride of seq_len//8 or so
        # keeps plenty of overlap at a fraction of the cost.
        self.stride = int(stride)
        self.cover_tail = bool(cover_tail)

        self.data = RaggedSequences.from_sequences(sequences)
        lengths = self.data.lengths
        # A window reads seq_len inputs at offset o and seq_len targets at
        # offset o+1, so it needs seq_len+1 tokens: o + seq_len + 1 <= len(seq),
        # i.e. o <= len(seq) - seq_len - 1. With o = k*stride, the largest
        # usable k is floor((len(seq) - seq_len - 1) / stride), so the count is
        # that plus one == ceil((len(seq) - seq_len) / stride) -- exactly the
        # expression below. Pieces at or below seq_len contribute none, and
        # short pieces are dropped rather than padded: a window that is mostly
        # PAD teaches the model to predict PAD.
        regular = np.maximum(0, (lengths - self.seq_len + self.stride - 1) // self.stride)
        if self.cover_tail:
            # The last regular window starts at (regular-1)*stride; it already
            # reaches the end exactly when that equals len - seq_len - 1.
            last_start = (regular - 1) * self.stride
            tail = (regular > 0) & (last_start != lengths - self.seq_len - 1)
        else:
            tail = np.zeros(len(lengths), dtype=bool)
        self._regular = regular.astype(np.int64)
        counts = self._regular + tail.astype(np.int64)
        self._cumulative = np.cumsum(counts, dtype=np.int64)
        self._total: int = int(self._cumulative[-1]) if len(counts) else 0
        # Per-sequence index as plain Python lists for __getitem__: scalar
        # numpy indexing / searchsorted costs ~1 us per call each, and these
        # are one int per *piece* (16k on ADL), not per token.
        self._cum_list: List[int] = self._cumulative.tolist()
        self._regular_list: List[int] = self._regular.tolist()
        self._starts_list: List[int] = self.data.starts.tolist()

        dropped = int((regular == 0).sum())
        if dropped:
            LOGGER.info(
                "%d/%d sequences shorter than seq_len+1 (%d) and skipped",
                dropped,
                len(lengths),
                self.seq_len + 1,
            )

    @property
    def sequences(self) -> RaggedSequences:
        """The underlying per-piece sequences (numpy views)."""
        return self.data

    def __len__(self) -> int:
        return self._total

    def window_start(self, index: int) -> Tuple[int, int]:
        """(sequence index, offset within it) of window ``index``."""
        if index < 0:
            index += self._total
        if not 0 <= index < self._total:
            raise IndexError(index)
        seq_index = bisect.bisect_right(self._cum_list, index)
        base = self._cum_list[seq_index - 1] if seq_index else 0
        k = index - base
        if k < self._regular_list[seq_index]:
            offset = k * self.stride
        else:  # the end-aligned tail window
            length = self._starts_list[seq_index + 1] - self._starts_list[seq_index]
            offset = length - self.seq_len - 1
        return seq_index, offset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_index, offset = self.window_start(index)
        start = self._starts_list[seq_index] + offset
        # Teacher forcing: y is x shifted left by one, so y[t] is the token the
        # model must predict from position t. Both slices stay inside this one
        # piece -- y's extra token is sequence[offset + seq_len], which the
        # window count above already guarantees exists.
        window = torch.from_numpy(self.data.flat[start : start + self.seq_len + 1]).long()
        return window[:-1], window[1:]

    def get_batch(self, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Windows ``indices`` as (x, y) LongTensors of shape [len(indices), seq_len].

        Exactly ``default_collate([self[i] for i in indices])``, computed with
        one vectorised index lookup and one gather instead of a Python call,
        two tensor views and a stack per window.
        """
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size and (int(idx.min()) < -self._total or int(idx.max()) >= self._total):
            raise IndexError(f"window index out of range for {self._total} windows")
        idx = np.where(idx < 0, idx + self._total, idx)
        seq = np.searchsorted(self._cumulative, idx, side="right")
        base = np.where(seq > 0, self._cumulative[np.maximum(seq - 1, 0)], 0)
        k = idx - base
        starts = self.data.starts
        tail_offset = (starts[seq + 1] - starts[seq]) - self.seq_len - 1
        offset = np.where(k < self._regular[seq], k * self.stride, tail_offset)
        first = starts[seq] + offset
        windows = self.data.flat[first[:, None] + np.arange(self.seq_len + 1)]
        batch = torch.from_numpy(windows).long()
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


class _BatchedMusicDataset(MusicDataset):
    """MusicDataset whose DataLoader fetches a whole batch in one gather.

    torch's fetcher calls ``__getitems__(indices)`` when a dataset has one and
    hands the result to ``collate_fn``. Only make_dataloaders uses this class,
    paired with :func:`_collate_prebatched`; a bare MusicDataset keeps the
    per-item protocol, so a DataLoader built on it with the default collate
    still works. Measured on the ADL v4 train split (batch 48, seq_len 256,
    CPU): default per-window path 0.8-1.1 ms/batch, this path 0.22 ms/batch,
    i.e. ~1.1-1.6 s of a ~180 s epoch. Small, but free and exact.
    The sampler, and therefore the batch order and contents, are unchanged.
    """

    def __getitems__(self, indices: Sequence[int]) -> List[torch.Tensor]:
        x, y = self.get_batch(indices)
        return [x, y]


def _collate_prebatched(batch: List[torch.Tensor]) -> List[torch.Tensor]:
    """collate_fn for _BatchedMusicDataset: the batch is already assembled."""
    return batch


def make_dataloaders(
    cfg: Any,
    train: Sequence[Sequence[int]],
    val: Sequence[Sequence[int]] = (),
    test: Sequence[Sequence[int]] = (),
    seq_len: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from token-id sequences.

    Only the training loader shuffles: shuffling val/test changes nothing about
    the metric but makes per-batch debugging output non-reproducible.
    """
    length = int(seq_len if seq_len is not None else cfg.model.seq_len)
    stride = max(1, int(cfg.model.get("window_stride", 1)))
    # Window policy, like window_stride: applied when windows are built and
    # never part of the cached tokens. See MusicDataset for why the tail matters.
    cover_tail = bool(cfg.model.get("window_cover_tail", True))
    batch_size = int(cfg.train.batch_size)
    num_workers = int(cfg.train.num_workers)

    def loader(sequences: Sequence[Sequence[int]], shuffle: bool) -> DataLoader:
        dataset = _BatchedMusicDataset(sequences, length, stride=stride, cover_tail=cover_tail)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and len(dataset) > 0,
            collate_fn=_collate_prebatched,
            num_workers=num_workers,
            drop_last=False,
            # persistent_workers is invalid with num_workers=0, which is the
            # default on Windows.
            persistent_workers=num_workers > 0,
        )

    loaders = (loader(train, True), loader(val, False), loader(test, False))
    LOGGER.info(
        "dataloaders: train=%d val=%d test=%d windows (seq_len=%d, stride=%d, batch=%d)",
        len(loaders[0].dataset),  # type: ignore[arg-type]
        len(loaders[1].dataset),  # type: ignore[arg-type]
        len(loaders[2].dataset),  # type: ignore[arg-type]
        length,
        stride,
        batch_size,
    )
    return loaders


# --------------------------------------------------------------------------
# processed-data cache
#
# Parsing, quantizing and augmenting 300 MIDI files takes minutes; training
# reads the result in seconds. That asymmetry is the entire reason prepare_data
# is a separate step from train: the expensive stage runs once and writes here,
# and every subsequent experiment that shares the same data/augment/encoding
# settings reuses it. The directory is keyed by the hash of exactly those three
# sections, so changing the grid or the encoding scheme produces a *new*
# directory instead of silently training on a stale one.
# --------------------------------------------------------------------------


def processed_dir(cfg: Any) -> Path:
    """Cache directory for this config: processed_dir/<config_hash>/."""
    return resolve_path(cfg.data.processed_dir) / config_hash(cfg, *CACHE_SECTIONS)


def save_processed(
    cfg: Any,
    splits: Dict[str, Sequence[Sequence[int]]],
    vocab: Optional[Vocab] = None,
) -> Path:
    """Cache token-id sequences (and the vocabulary they were built with).

    Ragged sequences are stored as a flat token array plus a lengths array
    rather than an object array, so nothing needs ``allow_pickle`` on load.
    ``meta.json`` additionally records the data-pipeline version and a
    fingerprint of the raw MIDI files, which load_processed uses to warn about
    a cache that no longer matches the code or the corpus (see there).
    """
    out = processed_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    for name in SPLIT_NAMES:
        ragged = RaggedSequences.from_sequences(splits.get(name, []), dtype=np.int32)
        np.save(out / f"{name}_tokens.npy", ragged.flat.astype(np.int32, copy=False))
        np.save(out / f"{name}_lengths.npy", ragged.lengths.astype(np.int64, copy=False))

    if vocab is not None:
        vocab.save(out / VOCAB_FILENAME)

    meta = {
        "config_hash": config_hash(cfg, *CACHE_SECTIONS),
        "sections": list(CACHE_SECTIONS),
        "encoding": dict(cfg.encoding),
        "vocab_size": len(vocab) if vocab is not None else None,
        "sequences": {n: int(len(splits.get(n, []))) for n in SPLIT_NAMES},
        "pipeline_version": DATA_PIPELINE_VERSION,
        "raw_fingerprint": raw_fingerprint(cfg),
    }
    with open(out / META_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    LOGGER.info("wrote processed cache to %s", out)
    return out


def raw_fingerprint(cfg: Any) -> Optional[Dict[str, Any]]:
    """Cheap identity of the raw corpus: every file's path, size and mtime.

    The cache directory is keyed on the *config*, so adding, removing or
    re-exporting MIDI files under the same raw_dir leaves the key unchanged
    and the old cache would be reused without a word. Hashing file contents
    would catch that too but costs a full read of the corpus on every load;
    stat()-ing ~2,000 files costs ~0.1 s, which is cheap enough to do on every
    load_processed call and catches every realistic edit.
    """
    try:
        from .parse import discover_midi_files

        data = cfg.data
        root = resolve_path(data.raw_dir)
        paths = discover_midi_files(root, data.get("include_styles") or [])
    except Exception as exc:  # noqa: BLE001 - a fingerprint must never break prepare/load
        LOGGER.debug("raw fingerprint unavailable: %s", exc)
        return None
    if not paths:
        return None
    digest = hashlib.sha1()
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8"))
    return {"files": len(paths), "sha1": digest.hexdigest()}


def _check_cache_freshness(cfg: Any, directory: Path) -> None:
    """Warn -- never fail -- when a cache may not match the code or the corpus."""
    meta_path = directory / META_FILENAME
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return
    version = int(meta.get("pipeline_version", 1))
    if version < DATA_PIPELINE_VERSION:
        LOGGER.warning(
            "processed cache %s was built by data pipeline v%d; this code is v%d (%s). "
            "It still loads and checkpoints trained on it keep working, but a model "
            "trained from it does not get those fixes. Re-run prepare_data (and retrain) "
            "to pick them up.",
            directory, version, DATA_PIPELINE_VERSION, PIPELINE_CHANGES,
        )
    stored = meta.get("raw_fingerprint")
    if stored:
        current = raw_fingerprint(cfg)
        if current is not None and current != stored:
            LOGGER.warning(
                "raw MIDI under %s changed since the processed cache %s was built "
                "(%s files then, %s now). The cache is keyed on the config only, so it "
                "is being reused as-is; re-run prepare_data if the change was intended.",
                resolve_path(cfg.data.raw_dir), directory,
                stored.get("files"), current.get("files"),
            )


def load_processed(
    cfg: Any,
) -> Optional[Tuple[Dict[str, RaggedSequences], Optional[Vocab]]]:
    """Read the cache for this config, or None if it is absent/incomplete.

    Each split comes back as a :class:`RaggedSequences` -- one flat int32
    array plus offsets, indexable like a list of sequences -- rather than a
    list of lists of Python ints, which cost ~36 bytes per token and most of
    the load time on a 13M-token corpus.
    """
    directory = processed_dir(cfg)
    if not directory.is_dir():
        return None

    splits: Dict[str, RaggedSequences] = {}
    for name in SPLIT_NAMES:
        tokens_path = directory / f"{name}_tokens.npy"
        lengths_path = directory / f"{name}_lengths.npy"
        if not (tokens_path.exists() and lengths_path.exists()):
            LOGGER.warning("incomplete processed cache at %s (missing %s)", directory, name)
            return None
        splits[name] = RaggedSequences(np.load(tokens_path), np.load(lengths_path))

    vocab_path = directory / VOCAB_FILENAME
    vocab = Vocab.load(vocab_path) if vocab_path.exists() else None

    if vocab is not None:
        # An id at or past the vocabulary size means the token files and
        # vocab.json come from different builds. Training on that dies with an
        # opaque CUDA device-side assert in the embedding; say what it is here.
        for name, ragged in splits.items():
            if ragged.num_tokens and (
                int(ragged.flat.max()) >= len(vocab) or int(ragged.flat.min()) < 0
            ):
                raise ValueError(
                    f"processed cache {directory}: {name} token ids span "
                    f"[{int(ragged.flat.min())}, {int(ragged.flat.max())}] but "
                    f"{VOCAB_FILENAME} has {len(vocab)} symbols -- the files are from "
                    "different builds. Re-run prepare_data."
                )

    _check_cache_freshness(cfg, directory)

    LOGGER.info(
        "loaded processed cache from %s (train=%d val=%d test=%d sequences)",
        directory,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits, vocab


def load_split(cfg: Any, name: str) -> Optional[RaggedSequences]:
    """One split of the processed cache ("train", "val" or "test"), or None.

    Reads only that split's two .npy files -- no vocabulary, no other splits,
    no freshness checks -- so e.g. priming generation from val costs a few
    hundred KB of I/O. ``seqs[i]`` is a numpy int32 view of sequence i;
    ``list(map(int, seqs[i]))`` gives Python ints if a caller needs them.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    directory = processed_dir(cfg)
    tokens_path = directory / f"{name}_tokens.npy"
    lengths_path = directory / f"{name}_lengths.npy"
    if not (tokens_path.exists() and lengths_path.exists()):
        return None
    return RaggedSequences(np.load(tokens_path), np.load(lengths_path))


def dataloaders_from_cache(
    cfg: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    """Load the processed cache and build loaders in one call.

    Adapter between two reasonable-but-different shapes: make_dataloaders takes
    explicit splits (so tests and notebooks can feed it anything), while the
    trainer just wants "give me the loaders for this config". Everything the
    trainer needs -- including the Vocab, which must be the exact one the cache
    was built with -- comes back together.

    Raises if the cache is missing, rather than silently reparsing: prepare_data
    is a deliberate separate step and a missing cache usually means the config
    changed.
    """
    loaded = load_processed(cfg)
    if loaded is None:
        raise FileNotFoundError(
            f"no processed data at {processed_dir(cfg)}\n"
            f"run: python scripts/prepare_data.py --config <your config>"
        )
    splits, vocab = loaded
    if vocab is None:
        raise FileNotFoundError(
            f"processed cache at {processed_dir(cfg)} has no {VOCAB_FILENAME}; "
            "re-run prepare_data.py"
        )
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg, splits["train"], splits["val"], splits["test"]
    )
    return train_loader, val_loader, test_loader, vocab
