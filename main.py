"""One entry point for the whole pipeline: prepare -> train -> generate.

    python main.py                       # whole pipeline on configs/default.yaml
    python main.py demo                  # zero-setup smoke run, well under a minute
    python main.py train --config configs/transformer.yaml
    python main.py generate --temperature 1.1 --num-samples 3
    python main.py configs                # what experiments are shipped

This is a thin orchestrator. Every stage calls the same code the scripts in
``scripts/`` call -- nothing is reimplemented here, and running the scripts
directly stays equivalent.

``--set`` takes dotted keys with YAML-typed values, exactly as in
``scripts/run_train.py``: ``--set train.lr=0.0005 --set augment.enabled=false``.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
# So `python main.py` works from the project root without an install step.
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/default.yaml"
SMOKE_CONFIG = "configs/smoke.yaml"
DEMO_CORPUS = "data/raw/dummy"
DEMO_PIECES = 24

# Rough CPU throughput in "units" per second, where one unit is one window
# times seq_len times hidden_dim times num_layers. Measured on this machine at
# hidden 256 / 2 layers / seq_len 100, under the current teacher-forcing
# objective (the output head now runs at every position, so a window costs more
# than it did when only the last position was scored):
#
#     stride  16 -> 623,672 windows, 186 min/epoch -> 2.9e6 units/s
#     stride  50 -> 202,511 windows,  51 min/epoch -> 3.4e6 units/s
#     stride 100 -> 103,454 windows,  18 min/epoch -> 4.9e6 units/s
#
# The spread means this is a rough model, not a formula. 3.0e6 is deliberately
# at the pessimistic end: this value only gates a warning, and warning slightly
# early is much better than telling someone a 3-hour epoch will take 45 minutes.
CPU_UNITS_PER_SECOND = 3.0e6
SLOW_EPOCH_SECONDS = 3600.0

DEPENDENCY_HINT = (
    "missing dependency {name!r}.\n"
    "Install the requirements first:\n"
    "    D:/PYTH/python.exe -m pip install -r requirements.txt"
)


class PipelineError(RuntimeError):
    """A predictable failure that deserves a message rather than a traceback."""


try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - environment problem
    raise SystemExit(DEPENDENCY_HINT.format(name=exc.name))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _banner(title: str) -> None:
    """Print a stage header so a long run says where it is."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72, flush=True)


@contextmanager
def _stage(title: str, timings: List[Tuple[str, float]]) -> Iterator[None]:
    """Run a stage under a banner and record its wall-clock time."""
    _banner(title)
    started = time.time()
    yield
    # Deliberately not a finally: a failed stage did not "finish", and its
    # timing line would land between the failure and the error message.
    elapsed = time.time() - started
    timings.append((title, elapsed))
    print(f"\n[main] {title} finished in {_fmt_seconds(elapsed)}", flush=True)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 90:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """Turn ["a.b=1", "c=x"] into {"a.b": 1, "c": "x"} for load_config.

    Values go through ``config.parse_override_value`` (YAML typing, plus
    ``1e-4`` as a float, which plain YAML 1.1 leaves a string) so
    ``train.lr=1e-4`` arrives as a float and ``augment.enabled=false`` as a
    bool -- same convention as the scripts.
    """
    from src.config import parse_override_value

    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise PipelineError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = parse_override_value(raw.strip())
    return overrides


def _override_argv(items: Sequence[str]) -> List[str]:
    """The same --set pairs, as argv for a script's main()."""
    argv: List[str] = []
    for item in items:
        argv += ["--set", item]
    return argv


def _resolve_config(path: str) -> Path:
    """Accept a path relative to the cwd or to the project root."""
    candidate = Path(path)
    for option in (candidate, PROJECT_ROOT / candidate):
        if option.is_file():
            return option.resolve()
    available = sorted(p.name for p in (PROJECT_ROOT / "configs").glob("*.yaml"))
    raise PipelineError(
        f"no config at {path!r}.\n"
        f"Available in configs/: {', '.join(available) or '(none)'}\n"
        "Run `python main.py configs` for what each one does."
    )


def _display(path: Path) -> str:
    """A path worth pasting back into a command line."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load(config_path: Path, overrides: Sequence[str]) -> Any:
    """load_config with the CLI's dotted overrides applied."""
    from src.config import load_config

    return load_config(config_path, **_parse_overrides(overrides))


# --------------------------------------------------------------------------
# preflight checks -- turn the predictable failures into advice
# --------------------------------------------------------------------------


def _check_raw_corpus(cfg: Any) -> None:
    """Fail early and usefully when there is no MIDI to prepare."""
    from src.config import resolve_path
    from src.data.parse import MIDI_SUFFIXES

    raw_dir = resolve_path(cfg.data.raw_dir)
    found = 0
    if raw_dir.is_dir():
        found = sum(
            1
            for p in raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in MIDI_SUFFIXES
        )
    if found:
        return

    what = "does not exist" if not raw_dir.is_dir() else "contains no MIDI files"
    raise PipelineError(
        f"data.raw_dir {raw_dir} {what}.\n"
        f"Config {cfg.name!r} has nothing to train on. Either:\n"
        "  * download a corpus and unpack it there -- the README's 'Real data'\n"
        "    table lists four (Classical Piano MIDI is the easiest start), or\n"
        "  * point the config elsewhere: --set data.raw_dir=data/raw/<yours>, or\n"
        "  * run `python main.py demo`, which builds a synthetic corpus and\n"
        "    proves the install works without downloading anything."
    )


def _cache_is_ready(cfg: Any) -> bool:
    """True when the processed cache for this config looks complete.

    A file check rather than ``load_processed``: on a real corpus that call
    materialises millions of token ids, which is pure waste when the answer is
    only 'may prepare be skipped'. The trainer still goes through
    ``load_processed`` for the real thing.
    """
    from src.data.dataset import VOCAB_FILENAME, processed_dir

    cache = processed_dir(cfg)
    if not cache.is_dir():
        return False
    needed = [cache / VOCAB_FILENAME] + [
        cache / f"{split}_{kind}.npy"
        for split in ("train", "val", "test")
        for kind in ("tokens", "lengths")
    ]
    return all(p.exists() for p in needed)


def _count_train_windows(cfg: Any) -> Optional[int]:
    """Training windows this config will produce, or None if unknown.

    Mirrors the per-piece formula in ``dataset.MusicDataset``; reads only the
    lengths array so the estimate costs nothing on a large corpus.
    """
    import numpy as np

    from src.data.dataset import processed_dir

    lengths_path = processed_dir(cfg) / "train_lengths.npy"
    if not lengths_path.exists():
        return None
    lengths = np.load(lengths_path).astype("int64")
    seq_len = int(cfg.model.seq_len)
    stride = max(1, int(cfg.model.get("window_stride", 1)))
    counts = np.maximum(0, (lengths - seq_len + stride - 1) // stride)
    if bool(cfg.model.get("window_cover_tail", True)):
        # make_dataloaders adds one end-aligned window per piece whose strided
        # windows stop short of the final token (see dataset.MusicDataset).
        last_start = (counts - 1) * stride
        counts = counts + ((counts > 0) & (last_start < lengths - seq_len - 1))
    return int(counts.sum())


def _training_is_on_cpu(cfg: Any) -> bool:
    want = str(cfg.train.get("device", "auto"))
    if want == "cpu":
        return True
    if want == "cuda":
        return False
    import torch

    return not torch.cuda.is_available()


def _warn_if_slow_on_cpu(cfg: Any) -> None:
    """Warn before a CPU run that will take more than about an hour per epoch."""
    if not _training_is_on_cpu(cfg):
        return
    windows = _count_train_windows(cfg)
    if not windows:
        return

    model = cfg.model
    units = (
        windows
        * int(model.seq_len)
        * int(model.hidden_dim)
        * max(1, int(model.get("num_layers", 1)))
    )
    seconds = units / CPU_UNITS_PER_SECOND
    if seconds < SLOW_EPOCH_SECONDS:
        return

    stride = max(1, int(model.get("window_stride", 1)))
    epochs = int(cfg.train.get("epochs", 1))
    print(
        "\n"
        "WARNING: this looks like a long CPU run.\n"
        f"  {windows:,} training windows at stride {stride}, "
        f"seq_len {model.seq_len}, hidden_dim {model.hidden_dim} x "
        f"{model.get('num_layers', 1)} layers\n"
        f"  rough estimate: {_fmt_seconds(seconds)} per epoch, "
        f"{epochs} epochs configured\n"
        "  Two levers:\n"
        "    * raise model.window_stride -- consecutive windows overlap by\n"
        "      seq_len-1 tokens and carry nearly the same gradient, so stride 8\n"
        "      or 16 cuts the epoch by that factor for little lost signal\n"
        "      (--set model.window_stride=16)\n"
        "    * install the CUDA build of torch (https://pytorch.org/get-started/locally/)\n"
        "      and set train.device=cuda\n",
        flush=True,
    )


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def run_prepare(config_path: Path, overrides: Sequence[str], cfg: Any) -> None:
    """Parse, split, augment, encode and cache the corpus."""
    from scripts.prepare_data import main as prepare_main

    _check_raw_corpus(cfg)
    argv = ["--config", str(config_path), *_override_argv(overrides)]
    try:
        prepare_main(argv)
    except SystemExit as exc:  # prepare_data raises these with real advice
        if exc.code:
            raise PipelineError(str(exc.code)) from None


def run_train(config_path: Path, cfg: Any) -> None:
    """Train the configured model, checkpointing to train.checkpoint_dir."""
    from src.train import train

    _warn_if_slow_on_cpu(cfg)
    try:
        best = train(cfg)
    except FileNotFoundError as exc:
        raise PipelineError(
            f"{exc}\n"
            f"Build the dataset first: python main.py prepare --config {_display(config_path)}"
        ) from None
    print(f"[main] checkpoint: {best}")


def run_generate(
    config_path: Path,
    overrides: Sequence[str],
    cfg: Any,
    temperature: Optional[float] = None,
    num_samples: Optional[int] = None,
    seed_midi: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """Sample from the trained checkpoint and write MIDI."""
    from src.generate import generate

    extra: Dict[str, Any] = {}
    if temperature is not None:
        extra["generate.temperature"] = temperature
    if num_samples is not None:
        extra["generate.num_samples"] = num_samples
    if seed_midi is not None:
        extra["generate.seed_source"] = "midi"
        extra["generate.seed_path"] = seed_midi
    if seed is not None:
        extra["generate.seed"] = seed
    if extra:
        from src.config import load_config

        cfg = load_config(config_path, **{**_parse_overrides(overrides), **extra})

    try:
        paths = generate(cfg)
    except FileNotFoundError as exc:
        text = str(exc)
        if "best.pt" in text or "vocab.json" in text:
            raise PipelineError(
                f"{text}\n"
                "Nothing has been trained for this config yet. Run:\n"
                f"    python main.py train --config {_display(config_path)}\n"
                "or `python main.py demo` for a complete run on synthetic data."
            ) from None
        raise PipelineError(text) from None
    except RuntimeError as exc:
        # generate.py raises this when the checkpoint's config hash disagrees;
        # its message already says what to do.
        raise PipelineError(str(exc)) from None

    print(f"[main] wrote {len(paths)} file(s):")
    for path in paths:
        print(f"         {path}")


def run_evaluate(config_path: Path, overrides: Sequence[str]) -> None:
    """Score everything under generate.output_dir against the reference corpus."""
    from scripts.run_evaluate import main as evaluate_main

    argv = ["--config", str(config_path), *_override_argv(overrides)]
    try:
        evaluate_main(argv)
    except SystemExit as exc:
        if exc.code:
            raise PipelineError(str(exc.code)) from None


def run_all(
    config_path: Path,
    overrides: Sequence[str],
    cfg: Any,
    force_prepare: bool,
    timings: List[Tuple[str, float]],
) -> None:
    """prepare (unless cached) -> train -> generate."""
    from src.data.dataset import processed_dir

    if force_prepare or not _cache_is_ready(cfg):
        with _stage("STAGE 1/3  prepare data", timings):
            run_prepare(config_path, overrides, cfg)
    else:
        _banner("STAGE 1/3  prepare data (skipped)")
        print(f"[main] reusing the processed cache at {processed_dir(cfg)}")
        print("[main] the cache is keyed on data + augment + encoding, so it")
        print("       matches this config. Force a rebuild with --force-prepare.")

    with _stage("STAGE 2/3  train", timings):
        run_train(config_path, cfg)

    with _stage("STAGE 3/3  generate", timings):
        run_generate(config_path, overrides, cfg)


def run_demo(
    timings: List[Tuple[str, float]],
    overrides: Sequence[str] = (),
    force_prepare: bool = False,
) -> None:
    """Synthetic corpus -> prepare -> train -> generate, in under a minute.

    The zero-setup path: this is what you run to check the install works.
    ``--set`` and ``--force-prepare`` are honoured (they used to be parsed and
    then silently dropped).
    """
    config_path = _resolve_config(SMOKE_CONFIG)
    cfg = _load(config_path, overrides)

    from src.config import resolve_path
    from src.data.parse import MIDI_SUFFIXES

    corpus = resolve_path(DEMO_CORPUS)
    have = (
        any(
            p.suffix.lower() in MIDI_SUFFIXES for p in corpus.glob("*") if p.is_file()
        )
        if corpus.is_dir()
        else False
    )

    with _stage("STAGE 1/4  synthetic corpus", timings):
        if have:
            print(f"[main] reusing the synthetic corpus at {corpus}")
        else:
            from scripts.make_dummy_corpus import main as dummy_main

            # make_dummy_corpus.main() reads sys.argv directly; swap it rather
            # than duplicating its generation loop here.
            saved = sys.argv
            sys.argv = ["make_dummy_corpus.py", "--out", str(corpus),
                        "--n", str(DEMO_PIECES)]
            try:
                dummy_main()
            finally:
                sys.argv = saved

    with _stage("STAGE 2/4  prepare data", timings):
        if _cache_is_ready(cfg) and not force_prepare:
            from src.data.dataset import processed_dir

            print(f"[main] reusing the processed cache at {processed_dir(cfg)}")
        else:
            run_prepare(config_path, overrides, cfg)

    with _stage("STAGE 3/4  train", timings):
        run_train(config_path, cfg)

    with _stage("STAGE 4/4  generate", timings):
        run_generate(config_path, overrides, cfg)


def run_configs() -> None:
    """List the shipped configs with the two seams each one selects."""
    config_dir = PROJECT_ROOT / "configs"
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise PipelineError(f"no configs found in {config_dir}")

    rows = []
    for path in paths:
        try:
            cfg = _load(path, [])
            rows.append(
                (
                    path.name,
                    str(cfg.get("name", "?")),
                    str(cfg.encoding.get("scheme", "?")),
                    str(cfg.model.get("arch", "?")),
                    str(cfg.data.get("raw_dir", "?")),
                )
            )
        except Exception as exc:  # a broken YAML should not hide the others
            rows.append((path.name, f"<unreadable: {exc}>", "-", "-", "-"))

    headers = ("config", "name", "encoding", "arch", "data.raw_dir")
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(value.ljust(w) for value, w in zip(row, widths)))
    print()
    print(f"default: {DEFAULT_CONFIG}   (override any field with --set key=value)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _common_flags(set_dest: str = "overrides") -> argparse.ArgumentParser:
    """Flags accepted both before and after the subcommand.

    SUPPRESS defaults matter: without them the subparser would overwrite a
    value the top-level parser already read, so `main.py --config x train`
    would silently lose the config.

    ``--set`` needs more than SUPPRESS: it is an *append* list, and argparse
    copies the subparser's namespace over the parent's, so
    ``main.py --set a=1 generate --set b=2`` used to end up with only b=2.
    The top-level parser therefore stores into its own ``set_dest`` and
    ``main`` concatenates the two lists (global first, so a value repeated
    after the subcommand wins).
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=f"YAML config, layered over configs/default.yaml (default: {DEFAULT_CONFIG})",
    )
    common.add_argument(
        "--set",
        dest=set_dest,
        action="append",
        default=argparse.SUPPRESS,
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set train.lr=0.0005; repeatable",
    )
    common.add_argument(
        "--force-prepare",
        action="store_true",
        default=argparse.SUPPRESS,
        help="rebuild the processed cache even if it already exists",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """The full CLI. With no subcommand, `all` runs."""
    common = _common_flags()
    parser = argparse.ArgumentParser(
        prog="main.py",
        parents=[_common_flags(set_dest="global_overrides")],
        description=__doc__.splitlines()[0],
        epilog="With no subcommand, `all` runs on configs/default.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    subparsers.add_parser(
        "prepare", parents=[common], help="raw MIDI -> processed token cache"
    )
    subparsers.add_parser(
        "train", parents=[common], help="train the configured model"
    )

    gen = subparsers.add_parser(
        "generate", parents=[common], help="sample MIDI from the trained checkpoint"
    )
    gen.add_argument("--temperature", type=float, default=None,
                     help="override generate.temperature")
    gen.add_argument("--num-samples", type=int, default=None,
                     help="override generate.num_samples")
    gen.add_argument("--seed-midi", default=None, metavar="PATH",
                     help="seed the sampler from this MIDI file")
    # Distinct from --seed-midi: that is the musical starting point, this is the
    # RNG. generate prints the seed it used, so a sample you liked can be
    # reproduced exactly by passing it back.
    gen.add_argument("--seed", type=int, default=None, metavar="INT",
                     help="RNG seed for reproducible sampling (default: random)")

    subparsers.add_parser(
        "evaluate", parents=[common], help="score generated MIDI and print the table"
    )
    subparsers.add_parser(
        "all", parents=[common], help="prepare (if needed) + train + generate"
    )
    demo = subparsers.add_parser(
        "demo", help="synthetic corpus + full pipeline on configs/smoke.yaml"
    )
    # Deliberately not --config: demo IS the smoke config, so accepting one
    # would be a lie. But the other two are useful here -- --force-prepare
    # especially, since regenerating the synthetic corpus leaves the processed
    # cache stale (the cache key is the config, which has not changed).
    demo.add_argument(
        "--force-prepare", action="store_true", default=argparse.SUPPRESS,
        help="rebuild the processed cache even if it already exists",
    )
    demo.add_argument(
        "--set", dest="overrides", action="append", default=argparse.SUPPRESS,
        metavar="KEY=VALUE", help="dotted config override; repeatable",
    )
    subparsers.add_parser("configs", help="list the shipped configs")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command or "all"
    overrides: List[str] = list(getattr(args, "global_overrides", None) or []) + list(
        getattr(args, "overrides", None) or []
    )
    force_prepare: bool = bool(getattr(args, "force_prepare", False))
    timings: List[Tuple[str, float]] = []
    started = time.time()

    try:
        if command == "configs":
            run_configs()
            return 0
        if command == "demo":
            run_demo(timings, overrides, force_prepare)
        else:
            config_path = _resolve_config(getattr(args, "config", DEFAULT_CONFIG))
            cfg = _load(config_path, overrides)
            print(f"[main] config: {config_path}  (name={cfg.get('name', '?')}, "
                  f"encoding={cfg.encoding.get('scheme')}, arch={cfg.model.get('arch')})")

            if command == "prepare":
                with _stage("prepare data", timings):
                    run_prepare(config_path, overrides, cfg)
            elif command == "train":
                with _stage("train", timings):
                    run_train(config_path, cfg)
            elif command == "generate":
                with _stage("generate", timings):
                    run_generate(
                        config_path,
                        overrides,
                        cfg,
                        temperature=args.temperature,
                        num_samples=args.num_samples,
                        seed=getattr(args, "seed", None),
                        seed_midi=args.seed_midi,
                    )
            elif command == "evaluate":
                with _stage("evaluate", timings):
                    run_evaluate(config_path, overrides)
            elif command == "all":
                run_all(config_path, overrides, cfg, force_prepare, timings)
            else:  # pragma: no cover - argparse rejects anything else
                parser.error(f"unknown command {command!r}")

    except PipelineError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as exc:
        print(f"\nerror: {DEPENDENCY_HINT.format(name=exc.name)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    total = time.time() - started
    if len(timings) > 1:
        print()
        for title, seconds in timings:
            print(f"[main] {title:<28} {_fmt_seconds(seconds):>8}")
    print(f"[main] total {_fmt_seconds(total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
