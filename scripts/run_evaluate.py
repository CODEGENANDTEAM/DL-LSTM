"""Score generated MIDI against a reference corpus and print a results table.

This is the script that builds the ablation table for the report. Point it at
one or more directories (or glob patterns) of generated MIDI and it emits a
markdown row per arm:

    python scripts/run_evaluate.py --config configs/default.yaml \
        --arm "note_chord/lstm=data/generated/baseline_lstm_*.mid" \
        --arm "interval/lstm=data/generated/interval_lstm_*.mid"

With no --arm, every *.mid under generate.output_dir is scored as one group.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config, parse_override_value, resolve_path  # noqa: E402
from src.data.parse import discover_midi_files, load_corpus, load_piece  # noqa: E402
from src.data.types import Piece  # noqa: E402
from src.evaluate import evaluate_pieces, format_report  # noqa: E402


def _parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = parse_override_value(raw.strip())
    return overrides


def _load_pieces(pattern: str, cfg: Config) -> List[Piece]:
    """Load MIDI matching a glob, or every MIDI under a directory.

    A directory is listed by parse.discover_midi_files -- .mid and .midi, any
    case, in a case-insensitive order -- rather than ``rglob("*.mid")``, which
    missed .midi everywhere and .MID on Linux only.
    """
    path = Path(pattern)
    files = discover_midi_files(path) if path.is_dir() else sorted(
        Path(path.parent or ".").glob(path.name)
    )
    pieces = []
    for f in files:
        piece = load_piece(f, cfg)
        if piece is not None:
            pieces.append(piece)
    return pieces


def _generated_config(config: str | None, overrides: Sequence[str]) -> Config:
    """The config generated MIDI is parsed with.

    Generated pieces are short and often sparse; the corpus filters exist to
    reject junk training files, and would silently drop valid outputs. That
    includes ``data.time_signatures``: decode.write_midi writes no time
    signature, which parses as 4/4, so under a config that does not list 4/4
    every generated file was filtered out and reported as "no MIDI matched".
    """
    return load_config(config, **{
        **_parse_overrides(overrides), "data.min_events": 1, "data.time_signatures": [],
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="LABEL=PATH_OR_GLOB",
        help="one arm of the comparison; repeatable",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="reference corpus dir (defaults to evaluate.reference, else data.raw_dir)",
    )
    parser.add_argument("--out", default=None, help="write the markdown table here")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, **_parse_overrides(args.overrides))

    eval_cfg = _generated_config(args.config, args.overrides)

    ref_dir = args.reference or cfg.evaluate.get("reference") or cfg.data.raw_dir
    reference = load_corpus(resolve_path(ref_dir), cfg)
    if not reference:
        raise SystemExit(f"no reference pieces parsed from {ref_dir}")

    arms = args.arm or [f"generated={cfg.generate.output_dir}"]
    rows: List[Dict[str, Any]] = []
    for spec in arms:
        label, _, pattern = spec.partition("=")
        if not pattern:
            label, pattern = Path(label).stem, label
        pieces = _load_pieces(pattern, eval_cfg)
        if not pieces:
            print(f"[evaluate] no MIDI matched {pattern!r}, skipping {label}")
            continue
        row = evaluate_pieces(pieces, reference, grid=float(cfg.data.grid))
        row["label"] = label
        rows.append(row)

    if not rows:
        raise SystemExit("nothing to evaluate")

    table = format_report(rows)
    print(f"\nreference: {len(reference)} pieces from {ref_dir}\n")
    print(table)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"\n[evaluate] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
