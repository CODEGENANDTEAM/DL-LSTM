"""Training entry point.

Usage:
    python scripts/run_train.py --config configs/default.yaml
    python scripts/run_train.py --set train.lr=0.0005 --set model.hidden_dim=256

``--set`` takes dotted keys and YAML-typed values, so ``train.epochs=5`` gives
an int and ``augment.enabled=false`` gives a bool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, parse_override_value  # noqa: E402
from src.train import train  # noqa: E402


def _parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        # yaml.safe_load gives ints/floats/bools/null their real types rather
        # than leaving everything a string.
        overrides[key.strip()] = parse_override_value(raw.strip())
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override; repeatable",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, **_parse_overrides(args.overrides))
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
