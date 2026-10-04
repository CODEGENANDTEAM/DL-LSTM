"""Core data types shared across the whole pipeline.

This module is the contract every other module builds against. Nothing here
imports torch, music21, or anything heavy -- keep it dependency-free so it can
be imported from any layer without circularity.

Pipeline order (non-negotiable):

    parse -> quantize -> filter -> SPLIT -> augment -> encode -> vocab -> window

Type flow:

    parse.py    -> Piece(events=[NoteEvent, ...])
    augment.py  -> Piece                      (pitches shifted)
    encode.py   -> list[str]                  (symbols)
    vocab.py    -> list[int]                  (ids)
    dataset.py  -> (x: LongTensor[B, seq_len], y: LongTensor[B, seq_len])

The model is trained with teacher forcing: ``y`` is ``x`` shifted one token
left, so position t of the output predicts token t+1 and a single window of
length L supplies L training targets rather than one. Models therefore return
logits of shape [B, L, V]; generation slices the last position itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Sequence

__all__ = [
    "NoteEvent",
    "Piece",
    "Encoder",
    "REST",
    "BOS",
    "EOS",
    "PAD",
    "UNK",
]

# Reserved vocabulary symbols. Kept here so encoders and the vocab builder
# cannot drift apart on spelling.
PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
REST = "<REST>"

RESERVED_SYMBOLS = (PAD, UNK, BOS, EOS, REST)


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """A single sounded note.

    Times are expressed in *quarter-note beats*, not seconds. Quantization
    happens in parse.py, so by the time a NoteEvent leaves the parser both
    ``start`` and ``duration`` are already snapped to the configured grid
    (e.g. multiples of 0.25 for a 16th-note grid).

    Attributes:
        pitch: MIDI note number, 0-127. Middle C is 60.
        start: Onset in quarter-note beats from the start of the piece.
        duration: Length in quarter-note beats. Always > 0.
        velocity: MIDI velocity 1-127. Retained for expressive rendering on
            decode; most encoders ignore it.
    """

    pitch: int
    start: float
    duration: float
    velocity: int = 80

    @property
    def end(self) -> float:
        return self.start + self.duration

    def __reduce__(self):  # type: ignore[override]
        # The default pickling of a frozen, slotted dataclass goes through
        # dataclasses.fields() for every object. Parsed pieces cross a process
        # boundary in parse.load_corpus (1.9M notes on ADL), where that cost
        # ~10 s; a plain constructor call is a fraction of it.
        return (NoteEvent, (self.pitch, self.start, self.duration, self.velocity))

    def transposed(self, semitones: int) -> "NoteEvent":
        """Return a copy shifted by ``semitones``. Does not clamp -- callers
        should drop pieces that fall outside 0-127 (see augment.py)."""
        # Direct construction, not dataclasses.replace: replace() re-inspects
        # the fields on every call and was the single largest cost in
        # prepare_data (15.7M calls, ~60 s of an ~2 min run on ADL).
        return NoteEvent(self.pitch + semitones, self.start, self.duration, self.velocity)


@dataclass(slots=True)
class Piece:
    """One musical work: an ordered list of NoteEvents plus provenance.

    ``events`` is always sorted by (start, pitch). Parsers guarantee this;
    every transform must preserve it.
    """

    events: List[NoteEvent] = field(default_factory=list)
    source: str = ""          # original file path, for debugging / provenance
    style: Optional[str] = None   # optional style label for conditioned mixing
    tempo: float = 120.0      # BPM, used only when writing MIDI back out

    def __len__(self) -> int:
        return len(self.events)

    @property
    def pitches(self) -> List[int]:
        return [e.pitch for e in self.events]

    def sorted(self) -> "Piece":
        return Piece(
            events=sorted(self.events, key=lambda e: (e.start, e.pitch)),
            source=self.source,
            style=self.style,
            tempo=self.tempo,
        )

    def transposed(self, semitones: int) -> "Piece":
        return Piece(
            events=[e.transposed(semitones) for e in self.events],
            source=self.source,
            style=self.style,
            tempo=self.tempo,
        )


class Encoder(Protocol):
    """SEAM 1.

    Any encoding scheme -- note/chord strings, interval-based, pitch+duration
    pairs -- implements exactly this. Swapping schemes must be a config change,
    never a rewrite.

    Contract: ``decode(encode(piece))`` must round-trip to a musically
    equivalent Piece. tests/test_roundtrip.py enforces this.
    """

    name: str

    def encode(self, piece: Piece) -> List[str]:
        """Piece -> flat list of vocabulary symbols."""
        ...

    def decode(self, symbols: Sequence[str]) -> Piece:
        """Vocabulary symbols -> Piece. Must tolerate malformed / unknown
        symbols from a sampling model by skipping them rather than raising."""
        ...

    def vocabulary(self) -> Optional[Iterable[str]]:
        """Optional: the full symbol set if it is known a priori (interval
        encoders know theirs; corpus-derived encoders return None and let
        vocab.py collect symbols from the data)."""
        ...
