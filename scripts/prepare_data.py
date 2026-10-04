"""Build the processed dataset: raw MIDI -> ids + vocabulary on disk.

    parse -> SPLIT (by piece) -> augment (train only) -> encode -> vocab -> save

Two orderings in there are not negotiable:

* The split happens BEFORE augmentation, and it splits whole pieces. Splitting
  windows, or augmenting first, puts transpositions of the same piece on both
  sides of the split -- the model then scores well on val by recognising music
  it has already memorised.
* The vocabulary is built from the TRAIN split only. See ``_build_vocab``.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config, parse_override_value, resolve_path  # noqa: E402
from src.data.augment import split_pieces, transposition_offsets  # noqa: E402
from src.data.dataset import (  # noqa: E402
    RaggedSequences,
    pieces_to_symbols,
    processed_dir,
    save_processed,
    style_symbol,
)
from src.data.encode import get_encoder  # noqa: E402
from src.data.parse import discover_midi_files, load_corpus  # noqa: E402
from src.data.types import Encoder, Piece  # noqa: E402
from src.data.vocab import Vocab  # noqa: E402

def _count_raw_files(raw_dir: Path, styles: Sequence[str] = ()) -> int:
    """Count MIDI files exactly as load_corpus discovers them.

    Delegates to parse.discover_midi_files, the single definition of the
    corpus file set, so the summary can never report a count the parser did
    not see (per-extension globbing used to double-count on case-insensitive
    Windows filesystems).
    """
    return len(discover_midi_files(raw_dir, styles))


def _set_aside_incompatible_cache(cache_dir: Path, vocab: Vocab) -> None:
    """Move an existing cache aside if its vocabulary differs from the new one.

    The cache directory is named after the config, not the code, so a data
    pipeline change can rebuild a *different* vocabulary into the same
    directory. Every checkpoint trained on the old one (train.py copies its
    vocab.json next to best.pt, but generate.py still seeds from this cache's
    token ids) would then read ids that mean other symbols. Rather than
    overwrite, the old directory is renamed to <hash>.replaced-<timestamp>;
    renaming it back restores the old checkpoints' seeds exactly.
    """
    old_vocab = cache_dir / "vocab.json"
    if not old_vocab.is_file():
        return
    try:
        previous = Vocab.load(old_vocab).itos
    except Exception:  # noqa: BLE001 - unreadable old vocab: nothing to protect
        previous = None
    if previous == vocab.itos:
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = cache_dir.with_name(f"{cache_dir.name}.replaced-{stamp}")
    cache_dir.rename(backup)
    print(
        f"[prepare] WARNING     : the vocabulary changed ({len(previous or [])} -> {len(vocab)} "
        f"symbols). The previous cache was moved to {backup} instead of being "
        "overwritten; checkpoints trained on it need it (rename it back to use "
        "them) or a retrain on the new cache.",
        flush=True,
    )


def _build_vocab(train_sequences: Sequence[List[str]], encoder: Encoder, cfg: Config) -> Vocab:
    """Build the vocabulary from the training split only.

    Including val/test symbols is a leak: at test time the model would be
    scored on a distribution whose rare symbols it was guaranteed to have an
    embedding for, which is not the situation it faces on unseen music. Symbols
    that only occur in val/test must map to <UNK>, exactly as they would in
    deployment.

    The encoder's a-priori vocabulary (interval encoders know their full symbol
    set by construction) is *not* a leak -- it is derived from the scheme, not
    from held-out data -- so Vocab.build unions it in.

    Style tokens are exempt from min_freq / max_size: each occurs once per
    sequence, so a rare style (or a tight max_size) would otherwise map its
    conditioning token to <UNK> and silently train that style unconditioned.
    Only styles present in the training split are kept.
    """
    sequences = list(train_sequences)
    prefix = style_symbol("")[:-1]          # "<STYLE:"
    styles = {s[0] for s in sequences if s and s[0].startswith(prefix)}
    return Vocab.build(
        sequences,
        encoder=encoder,
        min_freq=int(cfg.encoding.get("min_freq", 1)),
        max_size=int(cfg.encoding.get("max_size", 0)),
        keep=sorted(styles),
    )


def _parse_overrides(items: Sequence[str]) -> dict:
    """Turn ["a.b=1", "c=x"] into {"a.b": 1, "c": "x"} for load_config.

    Values are parsed as YAML so numbers and booleans arrive with the right
    type; a bare string stays a string.
    """
    overrides = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        # parse_override_value, exactly as run_train / run_generate do. Bare
        # yaml.safe_load left "1e-4"-style floats as strings here only, and the
        # processed cache is keyed on the config hash, so the trainer would
        # then look for a directory this script never wrote.
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

    cfg: Config = load_config(args.config, **_parse_overrides(args.overrides))
    raw_dir = resolve_path(cfg.data.raw_dir)
    cache_dir = processed_dir(cfg)

    n_raw = _count_raw_files(raw_dir, cfg.data.get("include_styles") or [])
    print(f"[prepare] config      : {cfg.name}")
    print(f"[prepare] raw dir     : {raw_dir}  ({n_raw} midi files)")
    print(f"[prepare] cache dir   : {cache_dir}")

    pieces: List[Piece] = list(load_corpus(raw_dir, cfg))
    if not pieces:
        raise SystemExit(
            f"no usable pieces parsed from {raw_dir}. Check data.raw_dir, "
            "data.min_events and data.time_signatures."
        )
    print(f"[prepare] parsed      : {len(pieces)} pieces kept of {n_raw} files")

    # SPLIT FIRST, by piece.
    train, val, test = split_pieces(pieces, cfg)
    print(f"[prepare] split       : train={len(train)} val={len(val)} test={len(test)}")

    encoder = get_encoder(cfg)

    # Augment (train only) and encode in one pass. transpose_augment would
    # first materialise every transposed Piece -- 12.6M NoteEvents on ADL, the
    # largest single cost of this script -- only for the encoder to walk them
    # once and drop them. encode_transpositions produces the identical symbol
    # sequences, in the identical order (original, then each in-range offset
    # ascending), from the original piece's onset structure.
    # encoding.style_tokens: every sequence of a piece with a style label starts
    # with its <STYLE:x> token. This used to be skipped here entirely (the
    # symbols came straight from encoder.encode), so a config asking for style
    # conditioning silently trained an unconditioned model. val/test go
    # through dataset.pieces_to_symbols, the same function generation-side
    # code is documented to use, and train mirrors it per transposition.
    use_style = bool(cfg.encoding.get("style_tokens", False))
    train_symbols: List[List[str]] = []
    for piece in train:
        offsets = [0] + transposition_offsets(piece, cfg)
        encoded = encoder.encode_transpositions(piece, offsets)
        if use_style and piece.style:
            token = style_symbol(piece.style)
            encoded = [[token] + symbols for symbols in encoded]
        train_symbols.extend(encoded)
    if cfg.augment.enabled:
        print(f"[prepare] augmented   : train {len(train)} -> {len(train_symbols)} pieces")
    else:
        print("[prepare] augmented   : disabled")

    symbol_splits: Dict[str, List[List[str]]] = {
        "train": train_symbols,
        "val": pieces_to_symbols(val, encoder, cfg),
        "test": pieces_to_symbols(test, encoder, cfg),
    }
    # Drop pieces that encoded to nothing; they would become empty rows.
    symbol_splits = {
        name: [s for s in seqs if s] for name, seqs in symbol_splits.items()
    }

    vocab = _build_vocab(symbol_splits["train"], encoder, cfg)

    # Straight to compact int32 arrays: a list of Python ints per sequence
    # costs ~9x the memory and is converted to exactly this on save anyway.
    id_splits = {
        name: RaggedSequences.from_sequences(vocab.encode(s) for s in seqs)
        for name, seqs in symbol_splits.items()
    }
    del symbol_splits

    _set_aside_incompatible_cache(cache_dir, vocab)

    # save_processed derives the cache directory from cfg itself, so the
    # hashing convention lives in exactly one place (src/data/dataset.py).
    cache_dir = save_processed(cfg, id_splits, vocab)

    print(f"[prepare] encoder     : {encoder.name}")
    print(f"[prepare] vocab size  : {len(vocab)}")
    for name in ("train", "val", "test"):
        seqs = id_splits[name]
        print(f"[prepare] {name:<11}: {len(seqs)} sequences, {seqs.num_tokens} tokens")
    grand_total = sum(seqs.num_tokens for seqs in id_splits.values())
    print(f"[prepare] total tokens: {grand_total}")
    print(f"[prepare] written to  : {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
