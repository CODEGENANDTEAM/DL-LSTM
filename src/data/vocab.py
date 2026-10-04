"""Symbol <-> integer id mapping.

The vocabulary is part of the model: an embedding row means whatever symbol had
that id at training time. Rebuild the vocabulary in a different order and an old
checkpoint keeps loading without complaint while playing nonsense. Hence the
determinism rules below, and hence Vocab.save() next to every checkpoint.
"""

from __future__ import annotations

import json
from collections import Counter
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .types import BOS, EOS, PAD, REST, RESERVED_SYMBOLS, UNK

__all__ = ["Vocab"]

LOGGER = logging.getLogger(__name__)

VOCAB_FORMAT_VERSION = 1


class Vocab:
    """Ordered symbol table. Reserved symbols always occupy ids 0-4.

        PAD=0  UNK=1  BOS=2  EOS=3  REST=4

    Those five come from types.py rather than being spelled out here so the
    encoders and the vocabulary can never disagree about what a rest is called.
    """

    def __init__(self, symbols: Sequence[str]) -> None:
        self.itos: List[str] = list(symbols)
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("duplicate symbols in vocabulary")
        for expected, symbol in enumerate(RESERVED_SYMBOLS):
            if self.itos[expected : expected + 1] != [symbol]:
                raise ValueError(
                    f"reserved symbol {symbol!r} must be at id {expected}"
                )

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        sequences: Iterable[Sequence[str]],
        encoder: Optional[Any] = None,
        min_freq: int = 1,
        max_size: int = 0,
        keep: Iterable[str] = (),
    ) -> "Vocab":
        """Build from encoded corpus sequences, plus an encoder's a priori set.

        If ``encoder.vocabulary()`` returns a symbol set (the interval encoder
        knows its own), it is unioned in so the model can emit legal symbols the
        training corpus happened not to contain.

        ``min_freq`` drops symbols occurring fewer than this many times and
        ``max_size`` caps the total, keeping the most frequent. Both matter on
        real performance data: chord-plus-duration symbols follow a brutal long
        tail where the majority of distinct symbols occur exactly once. Those
        rows can never be learned -- one gradient update each -- but they still
        cost a full embedding row and a full output-softmax column, and they
        inflate the loss denominator. Dropping them to UNK trades a handful of
        unreachable symbols for an output layer that fits in memory.

        Encoder-declared symbols are exempt: they are legal by construction, so
        a rare-but-valid interval must not be pruned. ``keep`` exempts further
        symbols *that occur in the corpus* -- prepare_data passes the style
        tokens, which occur once per sequence and so are exactly what a
        frequency filter removes first; pruned, a style's conditioning token
        would silently become <UNK>. ``max_size`` counts corpus symbols only:
        the result has up to 5 reserved + max_size + exempt symbols.
        """
        counts: Counter[str] = Counter()
        for sequence in sequences:
            counts.update(sequence)
        found: set[str] = set(counts)

        declared: set[str] = set()
        if encoder is not None:
            try:
                # Materialised once: vocabulary() may return any iterable, and
                # a generator consumed by the union below used to leave the
                # exemption set empty, so declared symbols got pruned.
                declared = set(encoder.vocabulary() or ())
            except Exception as exc:  # noqa: BLE001 - a bad encoder must not kill the build
                LOGGER.warning("encoder.vocabulary() failed: %s", exc)
            found.update(declared)

        found.difference_update(RESERVED_SYMBOLS)

        # Prune the long tail. Declared and kept symbols bypass both filters.
        exempt = declared | (set(keep) & found)
        if min_freq > 1:
            before = len(found)
            found = {s for s in found if counts[s] >= min_freq or s in exempt}
            LOGGER.info(
                "min_freq=%d: kept %d/%d symbols", min_freq, len(found), before
            )
        if max_size and len(found) > max_size:
            ranked = sorted(found, key=lambda s: (-counts[s], s))
            keep = set(ranked[:max_size]) | (exempt & found)
            LOGGER.info("max_size=%d: kept %d/%d symbols", max_size, len(keep), len(found))
            found = keep

        # sorted(), NOT set iteration order. Python randomises string hashing
        # per process (PYTHONHASHSEED), so iterating the set directly gives a
        # different id assignment on every run. A checkpoint reloaded against
        # such a vocabulary would map every embedding row to the wrong symbol
        # and produce plausible-looking garbage instead of an error.
        symbols = list(RESERVED_SYMBOLS) + sorted(found)
        LOGGER.info("built vocabulary of %d symbols", len(symbols))
        return cls(symbols)

    # -- mapping ---------------------------------------------------------

    def encode(self, symbols: Iterable[str]) -> List[int]:
        """Symbols -> ids. Anything unseen becomes UNK rather than raising."""
        unk = self.unk_id
        return [self.stoi.get(symbol, unk) for symbol in symbols]

    def decode(self, ids: Iterable[int]) -> List[str]:
        """Ids -> symbols. Out-of-range ids become UNK."""
        size = len(self.itos)
        return [self.itos[i] if 0 <= int(i) < size else UNK for i in ids]

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write as JSON. The id of a symbol is its index in ``symbols``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": VOCAB_FORMAT_VERSION, "symbols": self.itos}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        symbols = payload["symbols"] if isinstance(payload, dict) else payload
        return cls(symbols)

    # -- ids of the reserved symbols -------------------------------------

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def rest_id(self) -> int:
        return self.stoi[REST]

    # -- dunder ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self.stoi

    def __getitem__(self, symbol: str) -> int:
        return self.stoi.get(symbol, self.unk_id)

    def __repr__(self) -> str:
        return f"Vocab(size={len(self.itos)})"
