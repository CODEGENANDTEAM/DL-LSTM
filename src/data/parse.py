"""MIDI -> Piece. Reading, tempo conversion, quantization and filtering.

This is the first stage of the pipeline:

    parse -> quantize -> filter -> SPLIT -> augment -> encode -> vocab -> window

Everything downstream assumes the guarantees established here: times are in
quarter-note beats, already snapped to ``cfg.data.grid``, durations are > 0,
and ``Piece.events`` is sorted by (start, pitch).

MIDI I/O is pretty_midi only. music21 is deliberately not used at this layer:
it is an order of magnitude slower on a 300-file corpus and its stream model
buys us nothing once we have flattened to NoteEvents.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import NoteEvent, Piece

try:  # pretty_midi is only needed to parse; keep the module importable without it
    import pretty_midi  # type: ignore
except ImportError:  # pragma: no cover - exercised only on a bare environment
    pretty_midi = None  # type: ignore[assignment]

__all__ = ["load_piece", "load_corpus", "discover_midi_files", "quantize", "MIDI_SUFFIXES"]

LOGGER = logging.getLogger(__name__)

MIDI_SUFFIXES = (".mid", ".midi")

# pretty_midi occasionally emits RuntimeWarnings on malformed files; those are
# handled by the broad except in load_piece rather than by silencing warnings.


def quantize(value: float, grid: float) -> float:
    """Snap ``value`` to the nearest multiple of ``grid``.

    The extra ``round(..., 6)`` matters: 3 * 0.25 is exact but 7 * 0.1 is not,
    and encoders group notes by using the start time as a dict key. Without the
    cleanup two notes that are musically simultaneous can hash apart.
    """
    if grid <= 0:
        return round(float(value), 6)
    return round(round(float(value) / grid) * grid, 6)


def load_piece(path: str | Path, cfg: Any) -> Optional[Piece]:
    """Parse one MIDI file into a quantized, filtered Piece.

    Returns None (with a warning) for anything unreadable or filtered out. A
    corrupt file in a 300-file corpus must never kill the run, so no exception
    from pretty_midi is allowed to escape.
    """
    return _load_piece(path, cfg)[0]


def _load_piece(path: str | Path, cfg: Any) -> Tuple[Optional[Piece], str]:
    """load_piece plus a machine-readable skip reason, for the corpus tally."""
    return _parse_file(path, _parse_settings(cfg))


def _parse_settings(cfg: Any) -> Dict[str, Any]:
    """The parse-relevant slice of ``cfg`` as a plain, picklable dict.

    Worker processes receive this rather than the whole config: it is small,
    it pickles regardless of what kind of object the caller's cfg is, and it
    makes explicit which settings can change a parsed Piece.
    """
    data = cfg.data
    if hasattr(data, "get"):
        tempo_map = data.get("tempo_map", False)
    else:
        tempo_map = getattr(data, "tempo_map", False)
    return {
        "grid": float(data.grid),
        "min_duration": float(data.min_duration),
        "min_events": int(data.min_events),
        "time_signatures": list(data.time_signatures or []),
        "track_strategy": str(data.track_strategy),
        "tempo_map": bool(tempo_map),
    }


def _parse_file(path: str | Path, settings: Dict[str, Any]) -> Tuple[Optional[Piece], str]:
    if pretty_midi is None:
        raise ImportError(
            "pretty_midi is required to parse MIDI. Install it with "
            "`pip install pretty_midi`."
        )

    path = Path(path)
    grid = settings["grid"]
    min_duration = settings["min_duration"]
    min_events = settings["min_events"]

    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:  # noqa: BLE001 - any parser error means "skip file"
        LOGGER.debug("unreadable MIDI %s: %s", path, exc)
        return None, "unreadable"

    if not _time_signature_ok(midi, settings["time_signatures"]):
        LOGGER.debug("time signature not allowed, skipping %s", path)
        return None, "time_signature"

    tempo = _first_tempo(midi)

    try:
        raw_notes = _collect_notes(midi, settings["track_strategy"])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not read instruments from %s: %s", path, exc)
        return None, "unreadable"

    if not raw_notes:
        LOGGER.warning("no pitched (non-drum) notes in %s", path)
        return None, "no_pitched_notes"

    # Seconds -> quarter-note beats. By default this uses a single tempo for
    # the whole file (the first one), on the premise that multi-tempo files
    # are rare. That premise does NOT hold for every corpus: on ADL
    # Classical+Jazz+Blues 273 of 1661 readable files (16%) carry tempo
    # changes, and in 263 of them some onset lands more than half a grid step
    # away from its notated beat (worst case: 600+ beats). ``data.tempo_map:
    # true`` switches to the exact piecewise tempo map. It is opt-in because it
    # changes the tokens of every multi-tempo file, which invalidates any
    # checkpoint trained on the single-tempo tokens.
    to_beats = _beat_mapper(midi, tempo, settings["tempo_map"])
    beats_per_second = tempo / 60.0

    events: List[NoteEvent] = []
    for note in raw_notes:
        if to_beats is None:
            # Exactly the historical arithmetic -- (end - start) * bps, not
            # end * bps - start * bps, which rounds differently -- so caches
            # built before this code existed reproduce byte for byte.
            start_beats = float(note.start) * beats_per_second
            dur_beats = (float(note.end) - float(note.start)) * beats_per_second
        else:
            start_beats = to_beats(float(note.start))
            dur_beats = to_beats(float(note.end)) - start_beats

        # min_duration is applied to the RAW duration, before quantization.
        # After quantization every duration is at least one grid step, so with
        # the default config (grid 0.25, min_duration 0.125) a post-quantization
        # filter could never fire and the knob would be dead.
        if dur_beats < min_duration:
            continue

        start_q = quantize(start_beats, grid)
        dur_q = quantize(dur_beats, grid)
        if dur_q < grid:  # never let a note collapse to zero length
            dur_q = round(grid, 6)

        pitch = int(note.pitch)
        if not 0 <= pitch <= 127:
            continue

        events.append(
            NoteEvent(
                pitch=pitch,
                start=start_q,
                duration=dur_q,
                velocity=max(1, min(127, int(note.velocity))),
            )
        )

    if len(events) < min_events:
        LOGGER.warning(
            "only %d events (min_events=%d), skipping %s", len(events), min_events, path
        )
        return None, "too_few_events"

    events.sort(key=lambda e: (e.start, e.pitch))
    events = _merge_duplicate_onsets(events)

    # Shift so the piece begins at beat 0. Many MIDI files carry a silent lead-in
    # that would otherwise become a long run of <REST> tokens at the head of
    # every sequence -- pure noise for the model to learn.
    offset = events[0].start
    if offset:
        events = [
            NoteEvent(e.pitch, round(e.start - offset, 6), e.duration, e.velocity)
            for e in events
        ]

    return Piece(events=events, source=str(path), style=None, tempo=tempo), "ok"


def _merge_duplicate_onsets(events: List[NoteEvent]) -> List[NoteEvent]:
    """Collapse notes of the same pitch at the same quantized onset into one.

    A merged multi-track file (piano parts doubled across tracks) or a note
    re-struck within one grid step yields two identical (onset, pitch)
    events. One piano key cannot sound twice at once, and left in, they made
    note_chord write "48.60.60" -- an alias of "48.60" with its own vocabulary
    row. On ADL that was 28,942 notes in 404 files, minting 1,881 alias
    symbols. The survivor keeps the longest duration and the loudest velocity.
    ``events`` must already be sorted by (start, pitch).
    """
    merged: List[NoteEvent] = []
    for event in events:
        if merged and merged[-1].start == event.start and merged[-1].pitch == event.pitch:
            last = merged[-1]
            merged[-1] = NoteEvent(
                last.pitch,
                last.start,
                max(last.duration, event.duration),
                max(last.velocity, event.velocity),
            )
        else:
            merged.append(event)
    return merged


def discover_midi_files(
    raw_dir: str | Path, include_styles: Optional[Sequence[str]] = None
) -> List[Path]:
    """Every MIDI file ``load_corpus`` would parse, in the order it parses them.

    The one definition of "which files are in the corpus", shared by
    load_corpus, prepare_data's file count and the processed cache's raw-data
    fingerprint, so the three can never disagree.

    The order is by path components compared case-insensitively. That is what
    ``sorted()`` of WindowsPath objects already did, so on Windows nothing
    changes; spelling it out makes the order -- and therefore the seeded
    train/val/test split -- identical on case-sensitive filesystems too. With
    the old ``sorted(paths)`` a Linux re-run of prepare_data could put
    different pieces in test than the Windows run a checkpoint was trained on,
    and evaluating that checkpoint would then score it on its training data.
    """
    root = Path(raw_dir)
    if not root.exists():
        return []
    wanted = {s.lower() for s in (include_styles or []) if s}
    if wanted:
        # Only descend into the selected top-level style directories. Walking
        # the whole corpus and filtering afterwards stat()ed all ~11k ADL files
        # three times per prepare_data run to keep ~2k of them.
        tops = [d for d in root.iterdir() if d.is_dir() and d.name.lower() in wanted]
    else:
        tops = [root]
    paths = [
        p
        for top in tops
        for p in top.rglob("*")
        # The suffix test is a string check; is_file() is a stat. Cheap first.
        if p.suffix.lower() in MIDI_SUFFIXES and p.is_file()
    ]
    paths.sort(key=lambda p: _sort_key(p, root))
    return paths


def _sort_key(path: Path, root: Path) -> Tuple[str, ...]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return tuple(part.lower() for part in parts)


def load_corpus(raw_dir: str | Path, cfg: Any, workers: Optional[int] = None) -> List[Piece]:
    """Parse every MIDI file under ``raw_dir`` (recursively).

    Files that fail to parse or are filtered out are counted, not raised. The
    per-reason tally is logged at the end so a corpus that silently loses 80%
    of its files is obvious rather than mysterious.

    Parsing is spread over worker processes once the corpus is more than a few
    dozen files (``workers=None`` picks a count; ``workers=1`` forces serial).
    Each file parses independently and results come back in input order, so
    the output is identical to a serial run.
    """
    root = Path(raw_dir)
    if not root.exists():
        LOGGER.warning("raw_dir does not exist: %s", root)
        return []

    # Genre selection happens here, before parsing, so excluded files cost
    # nothing. A corpus organised as raw_dir/<style>/... can then serve several
    # experiments without duplicating 100+ MB on disk per subset.
    include = [s for s in (cfg.data.get("include_styles") or []) if s]
    paths = discover_midi_files(root, include)
    if include:
        LOGGER.info(
            "include_styles %s: kept %d files", sorted(s.lower() for s in include), len(paths)
        )
        if not paths:
            available = sorted({_style_from_path(p, root) or "<root>"
                                for p in discover_midi_files(root)})
            raise ValueError(
                f"data.include_styles={include} matched no files under {root}. "
                f"available styles: {available}"
            )

    if not paths:
        LOGGER.warning("no MIDI files found under %s", root)
        return []

    settings = _parse_settings(cfg)
    pieces: List[Piece] = []
    skipped: Counter[str] = Counter()
    results = _parse_all(paths, settings, workers, desc=f"parsing {root.name}")
    for path, (piece, reason) in zip(paths, results):
        if piece is None:
            skipped[reason] += 1
            continue
        # Style label = the sub-directory the file lives in, when the corpus is
        # organised as raw_dir/<style>/*.mid. Only used when
        # cfg.encoding.style_tokens is on; harmless otherwise.
        piece.style = _style_from_path(path, root)
        pieces.append(piece)

    total_skipped = sum(skipped.values())
    detail = ", ".join(f"{reason}={count}" for reason, count in sorted(skipped.items()))
    LOGGER.info(
        "parsed %d/%d files from %s; %d skipped%s",
        len(pieces),
        len(paths),
        root,
        total_skipped,
        f" ({detail})" if detail else "",
    )
    if total_skipped:
        # Printed as well as logged: a corpus that silently loses most of its
        # files is the kind of thing that goes unnoticed until the loss curve
        # looks strange three hours later.
        print(f"  skipped {total_skipped}/{len(paths)} files ({detail})", flush=True)
    return pieces


# Below this many files a process pool costs more to start (Windows spawns a
# fresh interpreter per worker and each re-imports pretty_midi) than it saves.
_PARALLEL_MIN_FILES = 64


def _parse_all(
    paths: Sequence[Path],
    settings: Dict[str, Any],
    workers: Optional[int],
    desc: str,
) -> List[Tuple[Optional[Piece], str]]:
    import multiprocessing

    current = multiprocessing.current_process()
    if getattr(current, "_inheriting", False):
        # We are a spawned worker that is still importing __main__, i.e. the
        # calling script has no `if __name__ == "__main__":` guard and is
        # re-running itself in every worker. Die exactly as multiprocessing
        # itself would; the parent sees a broken pool and parses serially.
        # Swallowing this and parsing here instead would make every worker
        # redo the whole corpus.
        raise RuntimeError(
            "load_corpus called while a worker process was importing __main__; "
            "guard the calling script with `if __name__ == \"__main__\":`"
        )
    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 2) - 2))
        if len(paths) < _PARALLEL_MIN_FILES or multiprocessing.parent_process() is not None:
            workers = 1  # tiny corpus, or already inside a worker: stay serial
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as pool:
                mapped = pool.map(
                    _parse_file, paths, itertools.repeat(settings), chunksize=8
                )
                return list(_progress(mapped, desc=desc, total=len(paths)))
        except (OSError, RuntimeError, ImportError) as exc:
            # RuntimeError covers BrokenProcessPool: a pool that cannot start
            # (restricted sandbox, frozen app, an unguarded __main__ on
            # Windows). Serial parsing gives the same answer, just slower.
            LOGGER.warning(
                "parallel parse unavailable (%s: %s); parsing serially. If the calling "
                "script has no `if __name__ == \"__main__\":` guard, add one.",
                type(exc).__name__, str(exc).strip().splitlines()[0] if str(exc).strip() else "",
            )
    return [_parse_file(p, settings) for p in _progress(paths, desc=desc, total=len(paths))]


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _collect_notes(midi: Any, strategy: str) -> List[Any]:
    """Flatten instruments into one note list according to ``track_strategy``.

    Drum tracks are always dropped: channel-10 pitches are timbres, not notes,
    and would poison the pitch vocabulary.
    """
    tracks = [inst for inst in midi.instruments if not inst.is_drum and inst.notes]
    if not tracks:
        return []

    if strategy == "densest":
        densest = max(tracks, key=lambda inst: len(inst.notes))
        return list(densest.notes)

    if strategy != "merge":
        LOGGER.warning("unknown track_strategy %r, falling back to 'merge'", strategy)

    notes: List[Any] = []
    for inst in tracks:
        notes.extend(inst.notes)
    return notes


def _first_tempo(midi: Any) -> float:
    """Tempo in BPM. Uses the first tempo change; see the note in load_piece."""
    try:
        _times, tempi = midi.get_tempo_changes()
    except Exception:  # noqa: BLE001
        return 120.0
    if len(tempi) == 0:
        return 120.0
    tempo = float(tempi[0])
    return tempo if tempo > 0 else 120.0


def _time_signature_ok(midi: Any, allowed: Optional[Sequence[str]]) -> bool:
    """True if the file's time signatures are permitted. Empty list = keep all."""
    if not allowed:
        return True

    wanted = set()
    for spec in allowed:
        parsed = _parse_time_signature(str(spec))
        if parsed is not None:
            wanted.add(parsed)
    if not wanted:
        return True

    changes = getattr(midi, "time_signature_changes", None) or []
    if not changes:
        # No time signature event at all: the MIDI spec's default is 4/4.
        return (4, 4) in wanted

    # A piece is kept only if *every* signature it uses is allowed; a file that
    # wanders into 3/4 halfway through is not a clean 4/4 example.
    return all((int(ts.numerator), int(ts.denominator)) in wanted for ts in changes)


def _parse_time_signature(spec: str) -> Optional[Tuple[int, int]]:
    try:
        num, den = spec.strip().split("/")
        return int(num), int(den)
    except (ValueError, AttributeError):
        LOGGER.warning("ignoring malformed time signature in config: %r", spec)
        return None


def _style_from_path(path: Path, root: Path) -> Optional[str]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def _beat_mapper(
    midi: Any, first_tempo: float, use_tempo_map: bool
) -> Optional[Callable[[float], float]]:
    """seconds -> quarter-note beats through the file's full tempo map.

    None means "use the single first tempo" -- the default, and also the
    answer for a file with at most one tempo, where both agree. Otherwise the
    tempo map is integrated piecewise, which is exact for any number of tempo
    changes.
    """
    if not use_tempo_map:
        return None
    try:
        change_times, tempi = midi.get_tempo_changes()
    except Exception:  # noqa: BLE001
        return None
    times = [float(t) for t in change_times]
    if len(times) <= 1:
        return None
    fallback = first_tempo / 60.0
    rates = [float(t) / 60.0 if t > 0 else fallback for t in tempi]
    # Beat position at the start of each tempo segment.
    origin = [0.0]
    for i in range(1, len(times)):
        origin.append(origin[-1] + (times[i] - times[i - 1]) * rates[i - 1])

    def mapped(seconds: float) -> float:
        i = bisect.bisect_right(times, seconds) - 1
        if i < 0:
            return seconds * rates[0]
        return origin[i] + (seconds - times[i]) * rates[i]

    return mapped


def _progress(items: Iterable[Any], desc: str, total: Optional[int] = None) -> Iterable[Any]:
    """tqdm when available, otherwise a coarse printed counter."""
    if total is None:
        total = len(items)  # type: ignore[arg-type]
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(items, desc=desc, unit="file", total=total)
    except ImportError:
        return _counter(items, desc, total)


def _counter(items: Iterable[Any], desc: str, total: int) -> Iterable[Any]:
    step = max(1, total // 20)
    for index, item in enumerate(items, start=1):
        if index == 1 or index == total or index % step == 0:
            print(f"  {desc}: {index}/{total}", flush=True)
        yield item
