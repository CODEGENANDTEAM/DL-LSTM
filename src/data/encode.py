"""SEAM 1: Piece <-> list[str] symbols.

Four interchangeable schemes, selected by ``cfg.encoding.scheme``:

    note_chord      "60"  "60.64.67"  "60.64.67_1.000"
    interval        "+2"  "-5"  "0"  "C+4"          (arXiv "Differential Music")
    pitch_duration  "P60" "D1.000"
    event           "T4"  "N60" "D8"                (MIDI-like / REMI-lite)

The ``event`` scheme has its own time model -- only ``T{k}`` moves the clock --
and is documented on EventEncoder. Everything below describes the other three.

THE TIME MODEL (shared by the first three, and the reason round-trips work):

    exactly one symbol per onset advances the clock by one grid step, and
    ``<REST>`` advances it by one grid step.

Everything else -- the duration suffix of a chord, the ``C+k`` chord members of
an interval group, the ``D...`` half of a pitch/duration pair -- attaches to the
onset before it and consumes no time. So a gap of k grid steps between two
onsets costs k-1 grid steps of silence, and decoding is a matter of walking the
symbol list while adding grid steps.

RUN-LENGTH RESTS (``cfg.encoding.run_length_rests``, default on):

    ``<REST:k>`` advances the clock by k grid steps.

One symbol per grid step is measurably ruinous: on the ADL corpus 56% of all
training tokens were ``<REST>`` and 75% of *generated* tokens were, so most of
the model's capacity went on predicting silence and every sample came out with
the same plodding note-rest-rest-rest pulse. Collapsing a run of silence into a
single token carrying its length cuts sequence length roughly in half and
leaves the rhythm to be learned rather than padded.

k is capped at ``cfg.encoding.max_rest_run`` (default 16, i.e. one bar on a
16th-note grid). A longer gap decomposes greedily -- largest available run
first, repeat -- so a 40-step gap is ``<REST:16> <REST:16> <REST:8>``. The cap
is what keeps the vocabulary bounded: without it one freak 400-step gap mints a
unique ``<REST:400>`` that occurs once, can never be learned, and still costs an
embedding row and a softmax column.

Bare ``<REST>`` keeps its old meaning of exactly one grid step -- it is a
reserved symbol at a fixed id (see types.py) and nothing here may change that.
Turning ``run_length_rests`` off restores the original one-token-per-step
encoding byte for byte, which is what makes the two comparable as ablation
arms.

Every ``decode`` skips symbols it cannot parse instead of raising: a sampling
model emits garbage, and one bad token must not lose the other 499. Reserved
tokens (PAD/UNK/BOS/EOS) and ``<STYLE:...>`` are ignored wherever they appear.
Skipped symbols do not advance the clock -- we have no idea how much time a
symbol we could not read was supposed to represent, and inventing one would
shear everything after it. Malformed rest runs (``<REST:>``, ``<REST:abc>``,
``<REST:0>``, ``<REST:-3>``, ``<REST:99999>``) fall under exactly that rule:
they are skipped and the clock does not move, because a guessed rest length
silently shifts every note after it, which is far worse than dropping one.
"""

from __future__ import annotations

import logging
import operator
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# RESERVED_SYMBOLS covers PAD/UNK/BOS/EOS; REST is handled separately because
# it is the only reserved symbol that carries timing.
from .types import REST, RESERVED_SYMBOLS, UNK, NoteEvent, Piece

__all__ = [
    "NoteChordEncoder",
    "IntervalEncoder",
    "PitchDurationEncoder",
    "EventEncoder",
    "get_encoder",
    "DURATION_DECIMALS",
    "DEFAULT_MAX_REST_RUN",
    "MAX_DECODABLE_REST_RUN",
    "format_rest_run",
    "parse_rest_run",
]

LOGGER = logging.getLogger(__name__)

# One fixed decimal format for every duration, in encode AND decode. If these
# two ever drift the vocabulary silently splits ("1.0" and "1.00" become two
# symbols for one duration) and the round-trip test fails in a confusing way.
DURATION_DECIMALS = 3

CHORD_SEP = "."
DURATION_SEP = "_"
CHORD_MEMBER_PREFIX = "C"
PITCH_PREFIX = "P"
DUR_PREFIX = "D"

MIN_PITCH = 0
MAX_PITCH = 127
DEFAULT_REFERENCE_PITCH = 60  # middle C, the interval decoder's starting point

# Run-length rests. ``<REST:4>`` is four grid steps of silence; bare ``<REST>``
# (from types.py, a reserved id) stays exactly one step and is never re-spelled
# as ``<REST:1>``, so the two encodings share the common case's token.
REST_RUN_PREFIX = "<REST:"
REST_RUN_SUFFIX = ">"

# Default cap on k. 16 is one bar on a 16th-note grid: long enough that almost
# every real gap is a single token, short enough that the rest tokens are a
# rounding error on the vocabulary (15 symbols) and every one of them occurs
# often enough to be learnable.
DEFAULT_MAX_REST_RUN = 16

# Decode-side sanity ceiling, deliberately far above any sane max_rest_run.
# Encoders never emit a k above their own cap, but decode also has to survive
# hand-written and sampled symbols, and it must stay usable across a config
# whose cap differs from the one that produced the tokens. Anything past this is
# not a rest length, it is noise, and is dropped without moving the clock.
MAX_DECODABLE_REST_RUN = 1024

# Event scheme (EventEncoder). One-letter prefixes; "D" is shared with
# pitch_duration's DUR_PREFIX above but the two never meet in one vocabulary
# (event durations are integer grid steps, "D8"; pitch_duration's are beats,
# "D2.000").
TIME_PREFIX = "T"
NOTE_PREFIX = "N"
VELOCITY_PREFIX = "V"
_EVENT_KINDS = frozenset((TIME_PREFIX, NOTE_PREFIX, DUR_PREFIX, VELOCITY_PREFIX))
# 16 = one 4/4 bar on a 16th grid. On ADL (1,654 pieces) 0.28% of onsets follow
# a longer gap, and splitting those costs +0.09% tokens.
DEFAULT_MAX_TIME_SHIFT = 16
# 32 = two 4/4 bars on a 16th grid. Clamps 0.077% of 1.88M ADL notes.
DEFAULT_MAX_DURATION_STEPS = 32
# Duration given to an N whose D never arrived (a sampling slip): 2 grid steps,
# the ADL median note length. A 16th is the mode (41% of notes), but the median
# is what minimises the expected absolute error of a guess.
DEFAULT_EVENT_DURATION_STEPS = 2
DEFAULT_VELOCITY = 80  # NoteEvent's default; used when velocity_bins is 0

_INT_RE = re.compile(r"^[+-]?\d+$")
_REST_RUN_RE = re.compile(r"^<REST:(\d+)>$")


def format_duration(beats: float) -> str:
    return f"{float(beats):.{DURATION_DECIMALS}f}"


def format_rest_run(steps: int) -> str:
    """``4 -> "<REST:4>"``. One symbol for ``steps`` grid steps of silence."""
    return f"{REST_RUN_PREFIX}{int(steps)}{REST_RUN_SUFFIX}"


def parse_rest_run(symbol: str) -> Optional[int]:
    """``"<REST:4>" -> 4``; None for anything that is not a legible rest run.

    Rejects every malformed variant a sampler or a hand-written test can
    produce -- ``<REST:>``, ``<REST:abc>``, ``<REST:0>``, ``<REST:-3>``,
    ``<REST: 4>``, ``<REST:2.5>`` -- and absurd lengths above
    ``MAX_DECODABLE_REST_RUN``. Callers treat None as "unparseable", which by
    the module-level rule means skip it and leave the clock alone.

    ``<REST:1>`` is accepted even though encode never writes it: it is
    unambiguous, so honouring it is strictly safer than dropping a grid step.
    """
    match = _REST_RUN_RE.match(symbol)
    if match is None:
        return None
    steps = int(match.group(1))
    if steps < 1 or steps > MAX_DECODABLE_REST_RUN:
        return None
    return steps


def _parse_duration(text: str) -> Optional[float]:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_pitch(text: str) -> Optional[int]:
    if not _INT_RE.match(text):
        return None
    value = int(text)
    return value if MIN_PITCH <= value <= MAX_PITCH else None


def _format_interval(semitones: int) -> str:
    return "0" if semitones == 0 else f"{semitones:+d}"


def _parse_interval(text: str) -> Optional[int]:
    if not _INT_RE.match(text):
        return None
    return int(text)


def _is_ignorable(symbol: str) -> bool:
    """Reserved tokens and style tokens: skipped by every decoder.

    ``<REST>`` and a well-formed ``<REST:k>`` are excluded here because they
    carry timing and are handled explicitly. The angle-bracket test also covers
    ``<STYLE:jazz>``, which dataset.py may prepend to a sequence -- and, by
    happy accident, every *malformed* rest run, which therefore gets the
    standard unparseable treatment of being skipped without moving the clock.
    """
    if symbol == REST or parse_rest_run(symbol) is not None:
        return False
    if symbol in RESERVED_SYMBOLS:
        return True
    return symbol.startswith("<") and symbol.endswith(">")


def _setting(section: Any, key: str, default: Any) -> Any:
    """Read one config field, tolerating a missing one.

    The two run-length knobs postdate every config file in configs/ except the
    default, and an experiment config is merged *over* the default rather than
    replacing it, so in practice the field is always there. Falling back keeps a
    hand-built cfg (tests, notebooks) from having to spell out both.
    """
    if hasattr(section, "get"):
        value = section.get(key, default)
        return default if value is None else value
    return getattr(section, key, default)


class _GridEncoder:
    """Shared grid arithmetic and onset grouping."""

    name = "base"

    def __init__(self, cfg: Any) -> None:
        self.grid: float = float(cfg.data.grid)
        self.include_duration: bool = bool(cfg.encoding.include_duration)
        if self.grid <= 0:
            raise ValueError(f"cfg.data.grid must be > 0, got {self.grid}")

        encoding = cfg.encoding
        self.run_length_rests: bool = bool(
            _setting(encoding, "run_length_rests", True)
        )
        self.max_rest_run: int = int(
            _setting(encoding, "max_rest_run", DEFAULT_MAX_REST_RUN)
        )
        if self.max_rest_run < 1:
            # Like an unknown scheme, this is a config typo and the user wants
            # to hear about it now rather than find silent zero-length rests.
            raise ValueError(
                f"cfg.encoding.max_rest_run must be >= 1, got {self.max_rest_run}"
            )

    # -- helpers ---------------------------------------------------------

    def _rest_tokens(self, steps: int) -> List[str]:
        """``steps`` grid steps of silence, as symbols.

        With run-length rests off this is the original ``[REST] * steps``. With
        them on it is a greedy largest-first decomposition, which for steps
        under the cap is a single token and above it is ceil(steps/cap) tokens.
        """
        if steps <= 0:
            return []
        if not self.run_length_rests or self.max_rest_run < 2:
            return [REST] * steps

        tokens: List[str] = []
        remaining = steps
        while remaining > 0:
            take = min(remaining, self.max_rest_run)
            tokens.append(REST if take == 1 else format_rest_run(take))
            remaining -= take
        return tokens

    def _rest_steps(self, symbol: str) -> Optional[int]:
        """Grid steps this symbol is silent for, or None if it is not a rest.

        Decoders call this before ``_is_ignorable`` so that rests -- the only
        timed symbols wearing angle brackets -- are never swallowed by the
        style/reserved filter. Rest runs are honoured regardless of
        ``run_length_rests``: with the flag off they never enter a vocabulary in
        the first place, so there is nothing to gain by refusing to read one and
        a whole piece's worth of sheared onsets to lose.
        """
        if symbol == REST:
            return 1
        return parse_rest_run(symbol)

    def _onset_groups(self, piece: Piece) -> List[Tuple[float, List[NoteEvent]]]:
        """Group events by quantized onset, ascending; each group sorted by pitch.

        Hot path of prepare_data (every training piece goes through here once).
        Piece.events is documented as sorted by (start, pitch) and parse.py
        guarantees it, so the fast path is one linear sweep that starts a new
        group whenever the onset changes, and a chord that is already strictly
        ascending in pitch (nearly all of them) skips the sort. Input that is not
        sorted by onset falls back to the dict grouping, which accepts any
        order. Both produce identical groups; sorting is stable, so notes of
        equal pitch keep their input order exactly as before.

        Measured on all 1,654 ADL Classical/Jazz/Blues pieces (1.88M notes),
        median of 7 with GC paused: 0.87 s -> 0.70 s for the grouping alone.
        """
        groups: List[Tuple[float, List[NoteEvent]]] = []
        append = groups.append
        previous: Optional[float] = None
        bucket: List[NoteEvent] = []
        for event in piece.events:
            key = round(float(event.start), 6)
            if key == previous:
                bucket.append(event)
                continue
            if previous is not None and key < previous:
                return self._onset_groups_unsorted(piece)
            bucket = [event]
            append((key, bucket))
            previous = key
        for index, (start, group) in enumerate(groups):
            if len(group) > 1:
                pitches = [e.pitch for e in group]
                if not all(map(operator.lt, pitches, pitches[1:])):
                    groups[index] = (start, _unique_pitches(group))
        return groups

    @staticmethod
    def _onset_groups_unsorted(piece: Piece) -> List[Tuple[float, List[NoteEvent]]]:
        """_onset_groups for events in arbitrary order (hand-built pieces)."""
        groups: Dict[float, List[NoteEvent]] = {}
        get = groups.get
        for event in piece.events:
            key = round(float(event.start), 6)
            bucket = get(key)
            if bucket is None:
                groups[key] = [event]
            else:
                bucket.append(event)
        return [
            (start, group if len(group) == 1 else _unique_pitches(group))
            for start, group in sorted(groups.items())
        ]

    def _steps(self, beats: float) -> int:
        return int(round(beats / self.grid))

    def _rests_between(self, previous: Optional[float], current: float) -> List[str]:
        """Filler tokens for the silence between two onsets.

        The onset symbol itself accounts for one grid step, so a k-step gap
        needs k-1 steps of silence. Negative/zero results produce nothing.
        """
        base = 0.0 if previous is None else previous
        gap_steps = self._steps(current - base) - (0 if previous is None else 1)
        return self._rest_tokens(max(0, gap_steps))

    def _piece_from(
        self, events: List[NoteEvent], template: Optional[Piece] = None
    ) -> Piece:
        events.sort(key=lambda e: (e.start, e.pitch))
        return Piece(
            events=events,
            source=template.source if template else "",
            style=template.style if template else None,
            tempo=template.tempo if template else 120.0,
        )

    def encode_transpositions(
        self, piece: Piece, offsets: Sequence[int]
    ) -> List[List[str]]:
        """``[encode(piece.transposed(k)) for k in offsets]``, possibly faster.

        The generic version is exactly that. Encoders whose output depends on
        pitch only through the symbol text override it to reuse one onset
        grouping for every offset.
        """
        return [
            self.encode(piece if k == 0 else piece.transposed(k)) for k in offsets
        ]

    def vocabulary(self) -> Optional[Iterable[str]]:
        """Note symbols are corpus-derived; rest runs are declared.

        vocab.py collects note and chord symbols from the data, but the rest
        runs are declared here so frequency pruning cannot reach them. That
        matters more than it looks: dropping a note symbol to <UNK> loses one
        note, while dropping a <REST:k> loses k grid steps of *time* and shifts
        every later note in the piece earlier. A long run is rare by nature --
        <REST:14> occurs a few hundred times in a 1600-piece corpus, well under
        a min_freq of 10 -- so without this it would be pruned exactly where it
        does the most damage. Declared symbols are exempt from both min_freq
        and max_size in Vocab.build.

        Returns None when run-length rests are off, so the ablation arm keeps
        its original fully corpus-derived vocabulary.
        """
        if not self.run_length_rests:
            return None
        return [format_rest_run(k) for k in range(2, self.max_rest_run + 1)]


class NoteChordEncoder(_GridEncoder):
    """Simultaneous notes collapse into one dot-joined chord symbol.

    ``"60"``, ``"60.64.67"``, and with ``include_duration`` ``"60.64.67_1.000"``.

    A chord carries a single duration (the longest note in it). Notes that start
    together but release apart therefore round-trip with a common release. That
    is the price of one symbol per onset; pitches and onsets -- what the contract
    in types.py actually requires -- survive exactly.
    """

    name = "note_chord"

    def encode(self, piece: Piece) -> List[str]:
        symbols: List[str] = []
        append = symbols.append
        previous: Optional[float] = None
        include_duration = self.include_duration
        for start, group in self._onset_groups(piece):
            # Inline fast path for the common case -- the next onset one grid
            # step after the last, so no rest -- and _rests_between otherwise.
            # Same arithmetic as _rests_between, so the symbols are identical.
            if previous is None or self._steps(start - previous) != 1:
                symbols.extend(self._rests_between(previous, start))
            if len(group) == 1:
                chord = _pitch_str(group[0].pitch)
            else:
                chord = CHORD_SEP.join([_pitch_str(e.pitch) for e in group])
            if include_duration:
                longest = max(e.duration for e in group)
                chord = f"{chord}{DURATION_SEP}{format_duration(longest)}"
            append(chord)
            previous = start
        return symbols

    def encode_transpositions(
        self, piece: Piece, offsets: Sequence[int]
    ) -> List[List[str]]:
        """Encode several transpositions of one piece in a single pass.

        Transposition moves pitches and nothing else, so the onset grouping,
        the rest tokens and the duration suffixes are shared by every offset;
        only the chord text changes. Computing them once and formatting per
        offset produces symbols identical to ``encode(piece.transposed(k))``
        (tests/test_data_audit.py checks this) without building the millions of
        transposed NoteEvents that dominated prepare_data's run time.
        """
        # The piece becomes one flat list of integer codes into a per-offset
        # symbol table: rest symbols first (they never change with the
        # offset), then every distinct (pitches, duration suffix) chord. A piece
        # reuses a few hundred distinct chords thousands of times, so each is
        # formatted once per offset, and the sequence itself is a single C-level
        # map over the codes. Measured on all 1,654 ADL Classical/Jazz/Blues
        # pieces at +/-6 (19,798 sequences, 16.0M tokens), median of 7 with GC
        # paused, against the previous per-onset extend/append loop: 2.81 s ->
        # 2.33 s including the faster _onset_groups, byte-identical output.
        rest_codes: Dict[str, int] = {}
        rest_table: List[str] = []
        distinct: Dict[Tuple[Tuple[int, ...], str], int] = {}
        chord_codes: List[int] = []   # distinct-chord index per onset, in order
        codes: List[int] = []
        append = codes.append
        include_duration = self.include_duration
        steps = self._steps
        previous: Optional[float] = None
        for start, group in self._onset_groups(piece):
            # Same fast path as encode(): a one-step gap needs no rest.
            if previous is None or steps(start - previous) != 1:
                for rest in self._rests_between(previous, start):
                    code = rest_codes.get(rest)
                    if code is None:
                        code = rest_codes[rest] = len(rest_table)
                        rest_table.append(rest)
                    append(code)
            suffix = ""
            if include_duration:
                longest = max(e.duration for e in group)
                suffix = f"{DURATION_SEP}{format_duration(longest)}"
            key = (tuple([e.pitch for e in group]), suffix)
            index = distinct.get(key)
            if index is None:
                index = distinct[key] = len(distinct)
            # Chord codes are offset by the final rest-table size below, which
            # is not known yet, so remember where they go.
            chord_codes.append(len(codes))
            append(index)
            previous = start

        shift = len(rest_table)
        for position in chord_codes:
            codes[position] += shift

        out: List[List[str]] = []
        for k in offsets:
            table = rest_table + [
                CHORD_SEP.join([_pitch_str(p + k) for p in pitches]) + suffix
                for pitches, suffix in distinct
            ]
            out.append(list(map(table.__getitem__, codes)))
        return out

    def decode(self, symbols: Sequence[str]) -> Piece:
        events: List[NoteEvent] = []
        # The clock is an integer count of grid steps; cursor is derived from
        # it. Accumulating round(cursor + grid, 6) drifts on any grid that is
        # not exact in 6 decimals (1/3 reads 0.999999 after three steps).
        grid_steps = 0
        cursor = 0.0
        for symbol in symbols:
            if not symbol:
                continue
            rest_steps = self._rest_steps(symbol)
            if rest_steps is not None:
                grid_steps += rest_steps
                cursor = round(grid_steps * self.grid, 6)
                continue
            if symbol == UNK:
                # <UNK> stands in for a chord that vocab pruning removed, and in
                # this scheme every chord symbol occupies exactly one grid step.
                # Skipping it without moving the clock would pull every later
                # note a step early -- one <UNK> knocks the rest of the piece off
                # the beat, which is far worse than losing the chord itself.
                # So: keep the time, drop the notes.
                grid_steps += 1
                cursor = round(grid_steps * self.grid, 6)
                continue
            if _is_ignorable(symbol):
                continue

            body, _, dur_text = symbol.partition(DURATION_SEP)
            duration = self.grid
            if dur_text:
                parsed = _parse_duration(dur_text)
                if parsed is None:
                    continue  # malformed duration -> drop the whole symbol
                duration = parsed

            pitches = [_parse_pitch(tok) for tok in body.split(CHORD_SEP) if tok]
            if not pitches or any(p is None for p in pitches):
                continue

            for pitch in pitches:
                events.append(
                    NoteEvent(
                        pitch=int(pitch),  # type: ignore[arg-type]
                        start=cursor,
                        duration=round(duration, 6),
                    )
                )
            grid_steps += 1
            cursor = round(grid_steps * self.grid, 6)
        return self._piece_from(events)


class IntervalEncoder(_GridEncoder):
    """Differential encoding: pitch relative to the previous note.

    Melody symbols are the signed interval from the previous onset's reference
    note (``"+2"``, ``"-5"``, ``"0"``); the remaining notes of a chord are
    written relative to that onset's lowest note as ``"C+4"``, ``"C+7"``.

    Transposition-invariant by construction, which is the whole appeal: the same
    phrase in every key is one symbol sequence. The cost is that durations are
    not represented at all (a duration would need its own symbol stream), so
    decoded notes all last one grid step.
    """

    name = "interval"

    def __init__(self, cfg: Any, reference_pitch: int = DEFAULT_REFERENCE_PITCH) -> None:
        super().__init__(cfg)
        self.reference_pitch = int(reference_pitch)

    def encode(self, piece: Piece) -> List[str]:
        symbols: List[str] = []
        previous_start: Optional[float] = None
        reference = self.reference_pitch
        for start, group in self._onset_groups(piece):
            symbols.extend(self._rests_between(previous_start, start))
            base = group[0].pitch  # lowest note of the onset
            symbols.append(_format_interval(base - reference))
            for event in group[1:]:
                symbols.append(f"{CHORD_MEMBER_PREFIX}{event.pitch - base:+d}")
            reference = base
            previous_start = start
        return symbols

    def decode(self, symbols: Sequence[str]) -> Piece:
        events: List[NoteEvent] = []
        # The clock is an integer count of grid steps; cursor is derived from
        # it. Accumulating round(cursor + grid, 6) drifts on any grid that is
        # not exact in 6 decimals (1/3 reads 0.999999 after three steps).
        grid_steps = 0
        cursor = 0.0
        current = self.reference_pitch
        base: Optional[int] = None  # lowest note of the onset being built
        onset_time = 0.0

        for symbol in symbols:
            if not symbol:
                continue
            rest_steps = self._rest_steps(symbol)
            if rest_steps is not None:
                grid_steps += rest_steps
                cursor = round(grid_steps * self.grid, 6)
                base = None
                continue
            if _is_ignorable(symbol):
                continue

            if symbol.startswith(CHORD_MEMBER_PREFIX):
                if base is None:
                    continue  # chord member with no onset to attach to
                offset = _parse_interval(symbol[len(CHORD_MEMBER_PREFIX) :])
                if offset is None:
                    continue
                events.append(
                    NoteEvent(
                        pitch=_clamp_pitch(base + offset),
                        start=onset_time,
                        duration=self.grid,
                    )
                )
                continue

            step = _parse_interval(symbol)
            if step is None:
                continue
            # Clamp rather than crash: a sampled sequence of forty "+11"s walks
            # off the top of the keyboard, and dropping the run entirely would
            # be worse than pinning it to 127.
            current = _clamp_pitch(current + step)
            base = current
            onset_time = cursor
            events.append(NoteEvent(pitch=current, start=cursor, duration=self.grid))
            grid_steps += 1
            cursor = round(grid_steps * self.grid, 6)

        return self._piece_from(events)

    def vocabulary(self) -> Optional[Iterable[str]]:
        """Known a priori: every interval representable on a 128-key range.

        Deliberately not derived from the corpus -- a leap the training data
        never contained is still a legal move for the model to make, and a
        fixed vocabulary keeps checkpoints comparable across corpora.

        The rest runs belong here for the same reason, and for a sharper one:
        this set is the *only* place they can come from for this encoder, and
        vocab.py exempts declared symbols from frequency pruning. Every k up to
        the cap is therefore legal and reachable even when the corpus happened
        to contain no 13-step gap. With run_length_rests off the set collapses
        to the old ``[REST]`` alone, keeping the ablation arms honest.
        """
        span = MAX_PITCH - MIN_PITCH
        symbols = [REST]
        if self.run_length_rests:
            symbols.extend(
                format_rest_run(k) for k in range(2, self.max_rest_run + 1)
            )
        symbols.extend(_format_interval(d) for d in range(-span, span + 1))
        symbols.extend(f"{CHORD_MEMBER_PREFIX}{d:+d}" for d in range(1, span + 1))
        return symbols


class PitchDurationEncoder(_GridEncoder):
    """Each note becomes two symbols: ``"P60"`` then ``"D1.000"``.

    Monophonic. Polyphonic input is reduced by keeping the highest note at each
    onset -- the melody line is what a listener tracks, and the top voice is the
    cheapest reliable proxy for it.
    """

    name = "pitch_duration"

    def encode(self, piece: Piece) -> List[str]:
        symbols: List[str] = []
        previous: Optional[float] = None
        for start, group in self._onset_groups(piece):
            symbols.extend(self._rests_between(previous, start))
            top = group[-1]  # groups are sorted ascending by pitch
            symbols.append(f"{PITCH_PREFIX}{top.pitch}")
            if self.include_duration:
                symbols.append(f"{DUR_PREFIX}{format_duration(top.duration)}")
            previous = start
        return symbols

    def decode(self, symbols: Sequence[str]) -> Piece:
        events: List[NoteEvent] = []
        # The clock is an integer count of grid steps; cursor is derived from
        # it. Accumulating round(cursor + grid, 6) drifts on any grid that is
        # not exact in 6 decimals (1/3 reads 0.999999 after three steps).
        grid_steps = 0
        cursor = 0.0
        pending: Optional[int] = None  # index into `events` awaiting its duration

        for symbol in symbols:
            if not symbol:
                continue
            rest_steps = self._rest_steps(symbol)
            if rest_steps is not None:
                grid_steps += rest_steps
                cursor = round(grid_steps * self.grid, 6)
                pending = None
                continue
            if _is_ignorable(symbol):
                continue

            if symbol.startswith(PITCH_PREFIX):
                pitch = _parse_pitch(symbol[len(PITCH_PREFIX) :])
                if pitch is None:
                    continue
                events.append(
                    NoteEvent(pitch=pitch, start=cursor, duration=self.grid)
                )
                pending = len(events) - 1
                grid_steps += 1
                cursor = round(grid_steps * self.grid, 6)
                continue

            if symbol.startswith(DUR_PREFIX):
                duration = _parse_duration(symbol[len(DUR_PREFIX) :])
                if duration is None or pending is None:
                    continue  # a duration with no pitch in front of it is noise
                note = events[pending]
                # NoteEvent is frozen, so replace rather than mutate.
                events[pending] = NoteEvent(
                    pitch=note.pitch,
                    start=note.start,
                    duration=round(duration, 6),
                    velocity=note.velocity,
                )
                pending = None

        return self._piece_from(events)


class EventEncoder(_GridEncoder):
    """MIDI-like event stream (REMI-lite) on the quantisation grid.

    ::

        T{k}   advance the clock by k grid steps (1 <= k <= max_time_shift)
        N{p}   a note onset at MIDI pitch p (0-127), at the current clock
        V{b}   velocity bin of the preceding note (only if velocity_bins > 0)
        D{k}   duration of the preceding note, k grid steps (1..max_duration_steps)

    A chord is several ``N [V] D`` groups with no ``T`` between them, in
    ascending pitch order. A gap longer than the cap decomposes greedily
    (``T16 T16 T8`` for 40 steps) exactly like ``<REST:k>`` runs; a note longer
    than the duration cap is clamped to it. That clamp is the scheme's only
    loss: pitch, onset and every duration up to the cap round-trip exactly.

    WHY. note_chord makes every distinct set of simultaneous pitches its own
    symbol: 37,020 symbols after pruning on ADL, ~4-5% of tokens still <UNK>,
    and no room for durations (363k symbols when tried). Here the vocabulary
    is fixed, declared up front and complete -- 128 + 16 + 32 = 176 symbols at
    the defaults -- so nothing is ever <UNK>, nothing is pruned, and durations
    are carried exactly. The price is length: 3.32 event tokens per note_chord
    token on ADL (9.14 vs 2.75 tokens per beat; see configs/adl_events.yaml).

    CAPS (measured on all 1,654 parsed ADL Classical/Jazz/Blues pieces, 1.88M
    notes, 893k onsets):

        max_time_shift 16      gaps > 16 steps: 0.28% of onsets, +0.09% tokens
                               (at 8: 1.56% of onsets)
        max_duration_steps 32  notes > 32 steps: 0.077% of notes clamped
                               (at 16: 0.70%; at 64: 0.007%)

    DECODE is a walk with an integer clock and one "pending" note -- the last
    ``N`` whose ``D`` has not arrived yet. Only ``T`` moves the clock, so no
    malformed or orphaned symbol can shift a later onset:

    * ``D`` or ``V`` with no pending note (after a ``T``, a second ``D``, or at
      the start): ignored.
    * ``N`` with no ``D``: keeps ``DEFAULT_EVENT_DURATION_STEPS`` (2 steps,
      the corpus median note length).
    * unparseable text, ``T0``/``T-3``/``T99999``, ``N200``, ``D0``: skipped
      without moving the clock; it also drops the pending note so that a later
      ``D`` cannot attach to the wrong note across the garbage.
    * reserved and ``<STYLE:x>`` tokens -- including ``<REST>`` and
      ``<REST:k>``, which this scheme never emits, so a sampled one is noise --
      and ``<UNK>``, which it cannot produce either: ignored, pending kept.
    """

    name = "event"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        encoding = cfg.encoding
        # Durations are part of the symbol stream by construction, so this is
        # always True whatever encoding.include_duration says: generate.py
        # reads it to skip decode.fill_durations, which would otherwise
        # overwrite real durations with next-onset guesses.
        self.include_duration = True
        self.max_time_shift = int(
            _setting(encoding, "max_time_shift", DEFAULT_MAX_TIME_SHIFT)
        )
        self.max_duration_steps = int(
            _setting(encoding, "max_duration_steps", DEFAULT_MAX_DURATION_STEPS)
        )
        self.velocity_bins = int(_setting(encoding, "velocity_bins", 0))
        if self.max_time_shift < 1:
            raise ValueError(
                f"cfg.encoding.max_time_shift must be >= 1, got {self.max_time_shift}"
            )
        if self.max_duration_steps < 1:
            raise ValueError(
                "cfg.encoding.max_duration_steps must be >= 1, got "
                f"{self.max_duration_steps}"
            )
        if not 0 <= self.velocity_bins <= 127:
            raise ValueError(
                f"cfg.encoding.velocity_bins must be 0-127, got {self.velocity_bins}"
            )
        self.default_duration_steps = min(
            DEFAULT_EVENT_DURATION_STEPS, self.max_duration_steps
        )

        # Every symbol this encoder can emit, formatted once. Encoding a piece
        # is then list lookups, and decode resolves the common case with one
        # dict hit instead of prefix tests and int() parsing.
        self._t_sym = [""] + [f"{TIME_PREFIX}{k}" for k in range(1, self.max_time_shift + 1)]
        self._d_sym = [""] + [
            f"{DUR_PREFIX}{k}" for k in range(1, self.max_duration_steps + 1)
        ]
        self._v_sym = [f"{VELOCITY_PREFIX}{b}" for b in range(self.velocity_bins)]
        # Code table for _plan: T1..Tmax, then D1..Dmax, then V0..V(bins-1).
        self._fixed: List[str] = self._t_sym[1:] + self._d_sym[1:] + self._v_sym
        self._lookup: Dict[str, Tuple[str, int]] = {}
        for k in range(1, self.max_time_shift + 1):
            self._lookup[self._t_sym[k]] = (TIME_PREFIX, k)
        for p in range(MIN_PITCH, MAX_PITCH + 1):
            self._lookup[_NOTE_STRINGS[p]] = (NOTE_PREFIX, p)
        for k in range(1, self.max_duration_steps + 1):
            self._lookup[self._d_sym[k]] = (DUR_PREFIX, k)
        for b in range(self.velocity_bins):
            self._lookup[self._v_sym[b]] = (VELOCITY_PREFIX, b)

    # -- velocity bins ---------------------------------------------------

    def velocity_bin(self, velocity: int) -> int:
        """1-127 -> 0..bins-1, equal-width bins."""
        v = max(1, min(127, int(velocity)))
        return min(self.velocity_bins - 1, (v - 1) * self.velocity_bins // 127)

    def bin_velocity(self, index: int) -> int:
        """Bin -> its centre velocity. ``velocity_bin(bin_velocity(b)) == b``."""
        lo = 1 + -(-index * 127 // self.velocity_bins)            # ceil
        hi = 1 + -(-(index + 1) * 127 // self.velocity_bins) - 1  # inclusive
        return (lo + hi) // 2

    # -- encode ----------------------------------------------------------

    def _plan(self, piece: Piece) -> Tuple[List[int], List[int]]:
        """The piece as integer codes, plus the distinct pitches it uses.

        Codes below ``len(self._fixed)`` index the pitch-independent symbols
        (every T, D and V); code ``len(self._fixed) + i`` is an ``N`` for
        ``pitches[i]``. Only the ``N`` symbols change under transposition, so a
        transposition by k is the same codes read through a table whose note
        entries are ``pitch + k`` -- the grouping, time shifts, durations and
        velocities are computed once per piece, not once per offset.
        """
        codes: List[int] = []
        append = codes.append
        grid = self.grid
        t_cap = self.max_time_shift
        d_cap = self.max_duration_steps
        d_base = t_cap - 1                    # D{k} is code d_base + k
        v_base = t_cap + d_cap                # V{b} is code v_base + b
        n_base = len(self._fixed)
        bins = self.velocity_bins
        pitch_index: Dict[int, int] = {}
        # A piece uses a handful of distinct durations (quantised to the grid)
        # thousands of times; one dict hit replaces a divide and a round.
        dur_code: Dict[float, int] = {}
        clock = 0
        for start, group in self._onset_groups(piece):
            step = int(round(start / grid))
            gap = step - clock
            if gap > 0:
                while gap > t_cap:
                    append(t_cap - 1)
                    gap -= t_cap
                append(gap - 1)
                clock = step
            for event in group:
                index = pitch_index.get(event.pitch)
                if index is None:
                    index = pitch_index[event.pitch] = len(pitch_index)
                append(n_base + index)
                if bins:
                    append(v_base + self.velocity_bin(event.velocity))
                code = dur_code.get(event.duration)
                if code is None:
                    k = int(round(event.duration / grid))
                    code = dur_code[event.duration] = d_base + (
                        1 if k < 1 else (d_cap if k > d_cap else k)
                    )
                append(code)
        return codes, list(pitch_index)

    def _render(self, codes: List[int], pitches: List[int], offset: int) -> List[str]:
        table = self._fixed + [
            _NOTE_STRINGS[p] if MIN_PITCH <= p <= MAX_PITCH else f"{NOTE_PREFIX}{p}"
            for p in (q + offset for q in pitches)
        ]
        return list(map(table.__getitem__, codes))

    def encode(self, piece: Piece) -> List[str]:
        return self._render(*self._plan(piece), 0)

    def encode_transpositions(
        self, piece: Piece, offsets: Sequence[int]
    ) -> List[List[str]]:
        """Identical to encoding each transposition, from one onset grouping."""
        codes, pitches = self._plan(piece)
        return [self._render(codes, pitches, k) for k in offsets]

    # -- decode ----------------------------------------------------------

    def _parse(self, symbol: str) -> Optional[Tuple[str, int]]:
        """Symbol -> (kind, value), or None if it is not a legal event.

        Declared symbols hit the lookup table. Values past the encoder's caps
        (``T40``, ``D48``) are still honoured up to MAX_DECODABLE_REST_RUN, so
        tokens stay readable under a config with different caps; anything else
        -- ``T0``, ``N128``, ``D-1``, ``N6O`` -- is None.
        """
        hit = self._lookup.get(symbol)
        if hit is not None:
            return hit
        kind = symbol[:1]
        body = symbol[1:]
        if kind not in _EVENT_KINDS or not body.isdigit() or not body.isascii():
            return None
        value = int(body)
        if kind == NOTE_PREFIX:
            return (kind, value) if value <= MAX_PITCH else None
        if kind == VELOCITY_PREFIX:
            return (kind, value) if value < self.velocity_bins else None
        return (kind, value) if 1 <= value <= MAX_DECODABLE_REST_RUN else None

    def decode(self, symbols: Sequence[str]) -> Piece:
        grid = self.grid
        default_duration = round(self.default_duration_steps * grid, 6)
        # Parallel lists instead of NoteEvents, because D and V rewrite the
        # pending note after it was opened and NoteEvent is frozen.
        pitches: List[int] = []
        starts: List[float] = []
        durations: List[float] = []
        velocities: List[int] = []
        clock = 0          # integer grid steps; start times derive from it
        start = 0.0
        pending = -1       # index of the note awaiting its D, or -1
        lookup = self._lookup.get
        for symbol in symbols:
            parsed = lookup(symbol)
            if parsed is None:
                if not symbol or _is_reserved_or_style(symbol):
                    continue
                parsed = self._parse(symbol)
                if parsed is None:
                    pending = -1
                    continue
            kind, value = parsed
            if kind == TIME_PREFIX:
                clock += value
                start = round(clock * grid, 6)
                pending = -1
            elif kind == NOTE_PREFIX:
                pending = len(pitches)
                pitches.append(value)
                starts.append(start)
                durations.append(default_duration)
                velocities.append(DEFAULT_VELOCITY)
            elif pending >= 0:
                if kind == DUR_PREFIX:
                    durations[pending] = round(value * grid, 6)
                    pending = -1
                else:  # VELOCITY_PREFIX
                    velocities[pending] = self.bin_velocity(value)
        events = [
            NoteEvent(p, s, d, v)
            for p, s, d, v in zip(pitches, starts, durations, velocities)
        ]
        return self._piece_from(events)

    def vocabulary(self) -> Optional[Iterable[str]]:
        """The complete symbol set: every T, N, D (and V) the encoder can emit.

        Declared, so vocab.py exempts all of it from min_freq/max_size pruning
        and a symbol the training split happens not to contain (``N3``, ``D29``)
        is still a legal output. Style tokens come from the data as usual.
        """
        symbols = list(self._t_sym[1:])
        symbols.extend(_NOTE_STRINGS)
        symbols.extend(self._d_sym[1:])
        symbols.extend(self._v_sym)
        return symbols


_ENCODERS: Dict[str, Any] = {
    NoteChordEncoder.name: NoteChordEncoder,
    IntervalEncoder.name: IntervalEncoder,
    PitchDurationEncoder.name: PitchDurationEncoder,
    EventEncoder.name: EventEncoder,
}


def get_encoder(cfg: Any) -> Any:
    """SEAM 1 dispatch: ``cfg.encoding.scheme`` -> Encoder instance.

    An unknown scheme raises: unlike a corrupt MIDI file, a typo in the config
    is a mistake the user wants to hear about immediately.
    """
    scheme = str(cfg.encoding.scheme)
    try:
        factory = _ENCODERS[scheme]
    except KeyError:
        raise ValueError(
            f"unknown encoding scheme {scheme!r}; expected one of "
            f"{sorted(_ENCODERS)}"
        ) from None
    return factory(cfg)


def _pitch_of(event: NoteEvent) -> int:
    return event.pitch


def _unique_pitches(group: List[NoteEvent]) -> List[NoteEvent]:
    """One onset's notes sorted by pitch, one note per pitch (the longest).

    parse.py already merges duplicate (onset, pitch) notes; this keeps the
    encoders honest for Pieces from anywhere else (generated, hand-built,
    audio transcription), so a chord symbol can never repeat a pitch.
    """
    ordered = sorted(group, key=_pitch_of)
    unique = [ordered[0]]
    for event in ordered[1:]:
        if event.pitch == unique[-1].pitch:
            if event.duration > unique[-1].duration:
                unique[-1] = event
        else:
            unique.append(event)
    return unique


# str(pitch) for the MIDI range, precomputed: note_chord encoding formats one
# of these per note of every training piece and transposition.
_PITCH_STRINGS = tuple(str(p) for p in range(MIN_PITCH, MAX_PITCH + 1))


# "N60" etc., precomputed for the event encoder's hot loop.
_NOTE_STRINGS = tuple(f"{NOTE_PREFIX}{p}" for p in range(MIN_PITCH, MAX_PITCH + 1))


def _is_reserved_or_style(symbol: str) -> bool:
    """PAD/UNK/BOS/EOS/REST, ``<REST:k>``, ``<STYLE:x>`` -- anything bracketed."""
    return symbol in RESERVED_SYMBOLS or (symbol[:1] == "<" and symbol[-1:] == ">")


def _pitch_str(pitch: int) -> str:
    if type(pitch) is int and MIN_PITCH <= pitch <= MAX_PITCH:
        return _PITCH_STRINGS[pitch - MIN_PITCH]
    return str(pitch)


def _clamp_pitch(pitch: int) -> int:
    return max(MIN_PITCH, min(MAX_PITCH, int(pitch)))
