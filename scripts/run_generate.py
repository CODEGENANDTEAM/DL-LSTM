"""Generation entry point.

Usage:
    python scripts/run_generate.py --config configs/default.yaml
    python scripts/run_generate.py --temperature 1.1 --num-samples 3
    python scripts/run_generate.py --seed-midi data/raw/prelude.mid
    python scripts/run_generate.py --seed-audio hum.mp3
    python scripts/run_generate.py --seed 20250911     # reproduce an earlier run

Flags override the corresponding ``generate.*`` config fields; anything not
given keeps the config value.

``--seed`` is the RNG seed (``generate.seed``), not a musical seed -- the two
``--seed-midi`` / ``--seed-audio`` flags are what pick a starting melody. With
no ``--seed`` the run draws fresh entropy and prints the value it drew, so a
run you liked can be reproduced afterwards by passing that number back in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Any, Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, parse_override_value  # noqa: E402
from src.generate import generate  # noqa: E402


def _parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """Turn ["a.b=1"] into {"a.b": 1}. Values go through yaml.safe_load so
    "--set train.lr=1e-4" arrives as a float rather than the string."""
    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = parse_override_value(raw.strip())
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="directory holding best.pt and vocab.json (default: train.checkpoint_dir)",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed-midi", default=None, help="seed from this MIDI file")
    parser.add_argument("--seed-audio", default=None, help="seed from this audio file")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="RNG seed (generate.seed); omit for fresh entropy each run",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override; repeatable",
    )
    args = parser.parse_args(argv)

    if args.seed_midi and args.seed_audio:
        raise SystemExit("pass at most one of --seed-midi / --seed-audio")

    overrides: Dict[str, Any] = _parse_overrides(args.overrides)
    if args.temperature is not None:
        overrides["generate.temperature"] = args.temperature
    if args.num_samples is not None:
        overrides["generate.num_samples"] = args.num_samples
    if args.seed_midi:
        overrides["generate.seed_source"] = "midi"
        overrides["generate.seed_path"] = args.seed_midi
    if args.seed_audio:
        overrides["generate.seed_source"] = "audio"
        overrides["generate.seed_path"] = args.seed_audio
    if args.seed is not None:
        overrides["generate.seed"] = args.seed

    cfg = load_config(args.config, **overrides)
    paths = generate(cfg, checkpoint_dir=args.checkpoint)
    print(f"[run_generate] wrote {len(paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
