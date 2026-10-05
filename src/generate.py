"""Autoregressive sampling from a trained checkpoint into MIDI files.

The pipeline mirrors training in reverse:

    seed symbols -> ids -> [sample loop] -> ids -> symbols -> Piece -> .mid

RANDOMNESS. Two independent streams decide what comes out:

* one ``torch.Generator`` per sample, which ``sampling.sample_batch`` draws
  that sample's tokens from (torch's *global* RNG is never touched, so calling
  generate() from a notebook or a test does not reseed anything else);
* a Python ``random.Random``, which picks the seed window out of the dataset.

Both are controlled by ``generate.seed``. ``null`` (the default) means draw
fresh entropy per run and print it, so a run is nondeterministic but any output
can still be reproduced afterwards. An explicit seed makes the whole run
bit-reproducible while sample *i* still uses its own derived seed
(``seed + i``), so the N outputs of one run differ from each other instead of
being N copies of the same trajectory.

The one hard safety check is in ``_verify_config_hash``: a checkpoint carries
the config hash it was trained under, and the vocabulary that lives beside it
is the only vocabulary whose id ordering that model's output layer means
anything in. Pairing a model with a vocabulary built under different encoding
settings does not crash -- it produces confident, fluent garbage -- so it is
refused up front.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import torch

from src.config import Config, config_hash, resolve_path
from src.data.dataset import load_split, processed_dir, style_symbol
from src.data.encode import get_encoder
from src.data.parse import load_piece
from src.data.types import BOS, EOS, PAD, UNK, Piece
from src.data.vocab import Vocab
from src.decode import fill_durations, write_midi
from src.models import build_model
from src.sampling import sample_batch

__all__ = ["generate", "resolve_device", "resolve_seed", "derive_seed", "special_token_ids"]

# Python ints are unbounded; torch.manual_seed wants a 64-bit value, and
# random.Random is happiest with something modest. Keep derived seeds inside a
# 63-bit window so `seed + i` can never overflow either one.
_SEED_MODULUS = 2 ** 63 - 1
# Freshly drawn seeds come from a much smaller range than that: the whole point
# is that a human reads one off a log line and types it back in, and 2**31 still
# leaves far more distinct runs than anyone will make.
_FRESH_SEED_MAX = 2 ** 31 - 1

# Sections that determine the symbol set. Must match the hash train.py stores
# as `data_config_hash` and the one dataset.processed_dir keys the cache on.
_DATA_SECTIONS = ("data", "augment", "encoding")


def resolve_device(cfg: Config) -> torch.device:
    """Honour ``train.device``: auto | cuda | cpu."""
    want = str(cfg.train.get("device", "auto")).strip().lower()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("train.device is 'cuda' but no CUDA device is visible; "
                           "use --set train.device=cpu")
    return torch.device(want)


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------


def _find_checkpoint_dir(cfg: Config, checkpoint_dir: Optional[Any]) -> Path:
    """Locate the directory holding best.pt and vocab.json."""
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    root = resolve_path(cfg.train.checkpoint_dir)
    # Training may write either straight into checkpoint_dir or into a
    # per-experiment subdirectory named after the config.
    nested = root / str(cfg.name)
    if (nested / "best.pt").exists():
        return nested
    return root


def _verify_config_hash(cfg: Config, ckpt: dict) -> None:
    """Refuse to sample if the checkpoint was trained under other encoding settings.

    The model's output layer is a fixed-size distribution over vocabulary ids.
    Change the encoding scheme, the quantization grid or the augmentation range
    and the same id means a different symbol -- the model still samples happily
    and the result is silently meaningless. So this is a hard error, not a
    warning.
    """
    if not isinstance(ckpt, dict):
        return

    # `data_config_hash` covers only data/augment/encoding, so an unrelated
    # knob like train.epochs does not spuriously invalidate a checkpoint. Fall
    # back to the full-config hash for older checkpoints that lack it.
    stored = ckpt.get("data_config_hash")
    expected = config_hash(cfg, *_DATA_SECTIONS)
    which = "data_config_hash"
    if stored is None:
        stored = ckpt.get("config_hash")
        expected = config_hash(cfg)
        which = "config_hash"
    if stored is None:
        print(
            "[generate] warning: checkpoint carries no config hash; "
            "cannot verify it matches the current encoding config."
        )
        return

    if stored != expected:
        raise RuntimeError(
            f"checkpoint {which} {stored!r} does not match this config's {expected!r}.\n"
            "The checkpoint was trained under different data/encoding settings, so "
            "its vocabulary ids do not mean what this config thinks they mean.\n"
            "Re-run scripts/prepare_data.py and retrain, or point --config at the "
            "config this checkpoint was trained with."
        )


# Model fields that only shape the OUTPUT LAYER. They must match the weights,
# and a run trained with `--set model.output=adaptive` is naturally sampled
# with the plain config, so the checkpoint's values win for these.
_HEAD_FIELDS = ("output", "adaptive_cutoffs", "adaptive_div_value")


def _with_checkpoint_head(cfg: Config, ckpt: Any) -> Config:
    """``cfg`` with the checkpoint's output-layer fields, when they differ."""
    saved = ckpt.get("config", {}).get("model", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(saved, dict):
        return cfg
    changes = {
        f"model.{key}": saved[key]
        for key in _HEAD_FIELDS
        if key in saved and saved[key] != cfg.model.get(key, None)
    }
    # Pre-adaptive checkpoints carry no `output`: they are full-softmax.
    if "output" not in saved and str(cfg.model.get("output", "full")) != "full":
        changes["model.output"] = "full"
    if not changes:
        return cfg
    print(f"[generate] using the checkpoint's output layer "
          f"({', '.join(f'{k}={v}' for k, v in changes.items())})")
    new = copy.deepcopy(cfg)
    for dotted, value in changes.items():
        new["model"][dotted.split(".", 1)[1]] = value   # cfg.model is a copy
    return new


def _load_state_dict(model: torch.nn.Module, ckpt: Any) -> None:
    """Tolerate the usual checkpoint layouts for the weights payload."""
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            payload = ckpt.get(key)
            if isinstance(payload, dict):
                model.load_state_dict(payload)
                return
    model.load_state_dict(ckpt)


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------


def _seed_context(ids: Sequence[int], seq_len: int) -> List[int]:
    """The last ``seq_len`` ids of a seed -- and NO padding for a short one.

    A short seed used to be left-padded with <PAD> up to seq_len. That is
    out-of-distribution input: MusicDataset drops pieces shorter than a window
    rather than padding them, so the model never once saw <PAD> as an input
    during training. For the LSTM the pads are not a no-op either (a zero
    embedding still drives the gates through their biases), so a short MIDI
    seed was continued from a hidden state no real music produces. Both models
    are causal and accept any length >= 1, so the fix is simply not to pad.
    """
    return list(ids)[-seq_len:]


def _symbols_to_ids(symbols: Sequence[str], vocab: Vocab) -> List[int]:
    return [int(i) for i in vocab.encode(list(symbols))]


def _seed_from_piece(piece: Piece, encoder: Any, vocab: Vocab) -> List[int]:
    symbols = encoder.encode(piece)
    if not symbols:
        raise ValueError(f"seed piece {piece.source!r} encoded to zero symbols")
    return _symbols_to_ids(symbols, vocab)


_SEED_SPLITS = ("train", "val", "test")


def _check_cache_vocab_matches(cfg: Config, vocab: Vocab) -> None:
    """Refuse to seed from a processed cache built with a different vocabulary.

    The primer ids come from the processed cache; the model reads them through
    the checkpoint's vocabulary. The checkpoint's ``data_config_hash`` covers
    only the *config*, so a data-pipeline fix that changes the tokens -- as the
    duplicate-pitch and style-token fixes did -- leaves the hash matching while
    the id numbering moves (74.5% of ids were renumbered by those fixes). The
    seeds would then decode as unrelated notes with no error anywhere. Compare
    the symbol lists directly instead.
    """
    cache_vocab_path = processed_dir(cfg) / "vocab.json"
    if not cache_vocab_path.is_file():
        return  # _load_dataset_pool raises the clearer missing-cache error
    cache_vocab = Vocab.load(cache_vocab_path)
    if list(cache_vocab.itos) != list(vocab.itos):
        raise RuntimeError(
            f"the processed cache at {processed_dir(cfg)} was built with a different "
            f"vocabulary ({len(cache_vocab)} symbols) than this checkpoint "
            f"({len(vocab)} symbols), so its ids would decode as the wrong notes. "
            "Either re-run scripts/prepare_data.py with the config this checkpoint was "
            "trained on, retrain on the current cache, or seed from a file with "
            "--seed-midi."
        )


def _load_dataset_pool(cfg: Config, split: Optional[str] = None) -> List[List[int]]:
    """Read one processed split as a list of id sequences to seed from.

    Which split is ``generate.seed_split`` (default ``val``). It used to be
    hard-wired to ``train``, and v3 overfit after epoch 6 (train loss kept
    falling while val loss rose): primed with a training excerpt, the model can
    simply recite the piece it memorised, and the output then sounds great for
    the wrong reason. A held-out primer is the honest default; ``train`` stays
    available for measuring exactly that effect (evaluate.copy_metrics).
    """
    split = str(split or cfg.generate.get("seed_split", None) or "val").strip().lower()
    if split not in _SEED_SPLITS:
        raise ValueError(
            f"unknown generate.seed_split {split!r}; expected one of {', '.join(_SEED_SPLITS)}"
        )
    # load_split reads just this split's two .npy files -- not the 13M-token
    # train split plus a vocab and freshness check, as load_processed does.
    sequences = load_split(cfg, split)
    if sequences is None:
        raise FileNotFoundError(
            f"no processed {split} split at {processed_dir(cfg)}. Run "
            "scripts/prepare_data.py first, or set generate.seed_source to "
            "midi/audio."
        )
    pool = [[int(t) for t in seq] for seq in sequences if len(seq)]
    if not pool:
        raise ValueError(
            f"processed {split} split is empty; nothing to seed from "
            "(try --set generate.seed_split=train)"
        )
    return pool


def _usable_pool(pool: Sequence[Sequence[int]]) -> List[Sequence[int]]:
    return [s for s in pool if len(s) >= 2] or list(pool)


def _window_in(seq: Sequence[int], seq_len: int, rng: random.Random,
               pad: int) -> List[int]:
    """Take a random seq_len window out of one sequence.

    A sequence no longer than seq_len is returned whole (unpadded -- see
    ``_seed_context``; ``pad`` is kept only for call compatibility). Valid
    start offsets are 0..len-seq_len INCLUSIVE; the old
    ``randrange(0, len - seq_len)`` could never pick the final window.
    """
    if len(seq) <= seq_len:
        return _seed_context(seq, seq_len)
    start = rng.randrange(0, len(seq) - seq_len + 1)
    return list(seq[start:start + seq_len])


def _random_window(pool: Sequence[Sequence[int]], seq_len: int, rng: random.Random,
                   pad: int) -> List[int]:
    """Pick a random sequence from the pool and a random seq_len window in it.

    Kept for callers that want one window in isolation; ``generate`` uses
    ``_WindowPicker`` instead, which also keeps the windows distinct.
    """
    usable = _usable_pool(pool)
    return _window_in(usable[rng.randrange(len(usable))], seq_len, rng, pad)


class _WindowPicker:
    """Hands out one seed window per sample, from genuinely different contexts.

    WHY THIS EXISTS. The obvious implementation -- draw a random sequence and a
    random offset per sample, independently -- has two failure modes that both
    make the batch of outputs sound like one tune:

    1. Drawing *with replacement* from the pool: two samples can land on the
       same sequence, and with a small corpus that is likely rather than rare.
    2. The pool is post-augmentation, so it holds one entry per (piece,
       transposition). Two "different" sequences are routinely the same melody
       a few semitones apart -- musically the same context, and a model seeded
       with it continues it the same way.

    So sequences are dealt out of a shuffled permutation (no repeats until the
    pool is exhausted), and each candidate window is additionally checked
    against a transposition-invariant fingerprint of the windows already
    handed out. Distinctness is best-effort: after ``_MAX_TRIES`` rejections we
    accept a duplicate rather than spin, because a tiny corpus may genuinely
    not contain N distinct contexts.
    """

    _MAX_TRIES = 32

    def __init__(self, pool: Sequence[Sequence[int]], seq_len: int, pad: int,
                 rng: random.Random, fingerprint: Any = None) -> None:
        self._usable = _usable_pool(pool)
        self._seq_len = seq_len
        self._pad = pad
        self._fingerprint = fingerprint
        self._order = list(range(len(self._usable)))
        rng.shuffle(self._order)
        self._cursor = 0
        self._rng = rng
        self._seen: set = set()
        self.reused = 0   # how often distinctness had to be given up

    def _next_sequence(self) -> Sequence[int]:
        if self._cursor >= len(self._order):
            # Pool exhausted. Reshuffle and go round again; the offsets drawn
            # the second time differ, so the windows still are not identical.
            self._rng.shuffle(self._order)
            self._cursor = 0
        seq = self._usable[self._order[self._cursor]]
        self._cursor += 1
        return seq

    def take(self, rng: random.Random) -> List[int]:
        """One seed window. ``rng`` supplies the offset (per-sample derived)."""
        window: List[int] = []
        for _ in range(self._MAX_TRIES):
            window = _window_in(self._next_sequence(), self._seq_len, rng, self._pad)
            key = self._fingerprint(window) if self._fingerprint else tuple(window)
            if key not in self._seen:
                self._seen.add(key)
                return window
        self.reused += 1
        return window


def _make_fingerprint(vocab: Vocab, encoder: Any, pad: int) -> Any:
    """A transposition-invariant key for a window of ids.

    Decoding the window back to pitches and keeping only the *intervals* means
    the same phrase at five different transpositions -- which is exactly what
    data augmentation puts in the pool -- collapses to one key. Falls back to
    the raw ids for anything that will not decode, so a malformed window is
    treated as its own context rather than crashing the run.
    """

    def fingerprint(window: Sequence[int]) -> tuple:
        ids = [int(i) for i in window if int(i) != pad]
        # Tagged so an interval tuple can never collide with a raw-id tuple that
        # happens to hold the same numbers.
        raw = ("ids", tuple(int(i) for i in window))
        if not ids:
            return raw
        try:
            pitches = encoder.decode(list(vocab.decode(ids))).pitches
        except Exception:  # pragma: no cover - decoders are meant to be lenient
            return raw
        if len(pitches) < 2:
            return raw
        return ("iv", tuple(b - a for a, b in zip(pitches, pitches[1:])))

    return fingerprint


# --------------------------------------------------------------------------
# RNG seeding
# --------------------------------------------------------------------------


def resolve_seed(cfg: Config) -> int:
    """The seed this run will actually use, always a concrete int.

    ``generate.seed: null`` (the default) means "nondeterministic": fresh OS
    entropy is drawn per run. It is still materialised as an int and printed,
    which is the point -- a run nobody seeded can still be reproduced later by
    passing the number back in. Falling back to ``train.seed`` here would be
    worse than useless: every generate run would start from the same six
    dataset windows forever.
    """
    value = cfg.generate.get("seed", None)
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none")):
        return random.SystemRandom().randrange(_FRESH_SEED_MAX)
    return int(value) % _SEED_MODULUS


def derive_seed(seed: int, index: int) -> int:
    """Per-sample seed. ``seed + i``, kept in range."""
    return (int(seed) + int(index)) % _SEED_MODULUS


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def special_token_ids(vocab: Vocab, allow_unk: bool = False) -> List[int]:
    """Ids the sampler must never emit.

    * ``<PAD>``, ``<BOS>``, ``<EOS>`` -- never appear in a training sequence
      (MusicDataset windows real symbol streams; nothing inserts BOS/EOS), so
      any probability the model gives them is pure noise, and emitting one
      feeds the model an input it has never seen.
    * ``<STYLE:x>`` -- conditioning tokens that only ever occur at position 0
      of a piece. Sampled mid-stream they are silent in the decoder (which
      skips them) but still corrupt the context the model conditions on.
    * ``<UNK>`` unless ``allow_unk`` -- stands for "some chord that vocab
      pruning removed" (about 4% of tokens at min_freq=10). The decoder renders
      it as a silent grid step, so every sampled <UNK> is a hole in the music.
      Renormalising over real symbols instead is the standard choice; set
      ``generate.allow_unk: true`` to restore the old behaviour.
    """
    style_prefix = style_symbol("")[:-1]          # "<STYLE:"
    banned = []
    for index, symbol in enumerate(vocab.itos):
        if symbol in (PAD, BOS, EOS) or symbol.startswith(style_prefix):
            banned.append(index)
        elif symbol == UNK and not allow_unk:
            banned.append(index)
    return banned


def _supports_incremental(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "supports_incremental", False))


def _score(model: torch.nn.Module, x: torch.Tensor, **kwargs: Any) -> Any:
    """``model(x, **kwargs)``, scoring only the last position when the model can.

    Only the last row is ever sampled from; for the full-softmax LSTM that
    skips a [1, seq_len, V] projection while priming, and for
    ``model.output: adaptive`` it evaluates log_prob on one row instead of 256.
    """
    if getattr(model, "supports_last_only", False):
        kwargs["last_only"] = True
    return model(x, **kwargs)


def _last_logits(output: Any) -> torch.Tensor:
    """``model(x)`` -> the next-token row ``[B, V]``."""
    logits = output[0] if isinstance(output, (tuple, list)) else output
    # The model scores EVERY position, but only the last one predicts the
    # token after the context; rows 0..T-2 re-predict tokens we already have.
    return logits[:, -1, :] if logits.dim() == 3 else logits


@torch.no_grad()
def _sample_many(
    model: torch.nn.Module,
    seeds: Sequence[Sequence[int]],
    num_tokens: int,
    seq_len: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    banned_ids: Optional[Iterable[int]] = None,
    generators: Optional[Sequence[Optional[torch.Generator]]] = None,
    incremental: Optional[bool] = None,
) -> List[List[int]]:
    """Autoregressive loop for N samples. Returns only the new ids per sample.

    Two decoding paths, chosen by ``incremental`` (None = the model decides):

    INCREMENTAL (LSTM). Each seed is run through the model ONCE to build its
    (h, c); after that every step feeds a single token and carries the state.
    That is O(1) work per token instead of re-running a seq_len window, and
    the N samples step together as one batch. While the total context still
    fits in seq_len, this is mathematically the full-window path (an LSTM from
    zero state over the same tokens) -- tests/test_model_audit.py checks it
    token for token. Past seq_len the paths differ on purpose: the window path
    throws away the oldest token every step and re-reads the rest from a zero
    state, the incremental path keeps all history in (h, c), which is how a
    recurrent model is meant to run and what it measurably predicts better
    with (see the report in the audit notes / README).

    FULL WINDOW (Transformer, or ``incremental=False``). Attention has no
    carried state here, so each step re-scores the last seq_len tokens, one
    sample at a time -- identical to the original loop.

    ``generators[i]`` drives sample i's draws, so sample i's tokens do not
    depend on how many other samples share the batch.
    """
    n = len(seeds)
    if n == 0:
        return []
    contexts = [_seed_context(s, seq_len) for s in seeds]
    if any(not c for c in contexts):
        raise ValueError("every seed needs at least one token")
    gens: List[Optional[torch.Generator]] = list(generators) if generators is not None else [None] * n
    if len(gens) != n:
        raise ValueError(f"got {len(gens)} generators for {n} seeds")
    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        return [[] for _ in range(n)]

    banned: Optional[torch.Tensor] = None
    if banned_ids is not None:
        ids = sorted({int(i) for i in banned_ids})
        banned = torch.tensor(ids, dtype=torch.long, device=device) if ids else None

    use_state = _supports_incremental(model) if incremental is None else bool(incremental)
    if use_state and not _supports_incremental(model):
        raise ValueError(f"{type(model).__name__} does not support incremental decoding")

    if use_state:
        # Prime each seed on its own: seeds can differ in length, and padding
        # them into one batch is exactly the out-of-distribution input that
        # _seed_context exists to avoid. One cuDNN call per seed.
        rows, hs, cs = [], [], []
        for context in contexts:
            x = torch.tensor([context], dtype=torch.long, device=device)
            out, (h, c) = _score(model, x, return_hidden=True)
            rows.append(_last_logits(out))
            hs.append(h)
            cs.append(c)
        logits = torch.cat(rows, dim=0)                       # [N, V]
        hidden = (torch.cat(hs, dim=1), torch.cat(cs, dim=1))  # [layers, N, H]

        # Tokens stay on the device until the end: no host sync per step.
        generated = torch.empty(n, num_tokens, dtype=torch.long, device=device)
        for step in range(num_tokens):
            next_ids = sample_batch(logits, temperature=temperature, top_k=top_k,
                                    top_p=top_p, banned_ids=banned, generators=gens)
            generated[:, step] = next_ids
            if step + 1 < num_tokens:
                out, hidden = model(next_ids.unsqueeze(1), hidden=hidden)
                logits = _last_logits(out)
        return generated.tolist()

    results: List[List[int]] = []
    for context, gen in zip(contexts, gens):
        context = list(context)
        new_ids: List[int] = []
        for _ in range(num_tokens):
            x = torch.tensor([context[-seq_len:]], dtype=torch.long, device=device)
            logits = _last_logits(_score(model, x))
            next_id = int(sample_batch(logits, temperature=temperature, top_k=top_k,
                                       top_p=top_p, banned_ids=banned,
                                       generators=[gen])[0].item())
            new_ids.append(next_id)
            context.append(next_id)   # slide the window by appending
        results.append(new_ids)
    return results


def _sample_ids(
    model: torch.nn.Module,
    seed: Sequence[int],
    num_tokens: int,
    seq_len: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    banned_ids: Optional[Iterable[int]] = None,
    generator: Optional[torch.Generator] = None,
    incremental: Optional[bool] = None,
) -> List[int]:
    """One sample; see ``_sample_many``. Returns only the newly generated ids."""
    return _sample_many(model, [seed], num_tokens, seq_len, device, temperature, top_k,
                        top_p, banned_ids=banned_ids, generators=[generator],
                        incremental=incremental)[0]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def generate(cfg: Config, checkpoint_dir: Optional[Any] = None) -> List[Path]:
    """Sample ``cfg.generate.num_samples`` pieces and write them as MIDI.

    Args:
        cfg: Loaded config.
        checkpoint_dir: Directory containing ``best.pt`` and ``vocab.json``.
            Defaults to ``cfg.train.checkpoint_dir`` (optionally with a
            ``cfg.name`` subdirectory).

    Returns:
        Paths of the written .mid files. A matching .txt holding the raw symbol
        sequence is written next to each one for debugging.

    Reproducibility is governed by ``generate.seed``:

    * ``null`` (default) -- fresh entropy, so two runs differ. The drawn seed is
      printed, so any run can be reproduced after the fact.
    * an int -- the whole run is bit-reproducible. Sample *i* still runs on
      ``seed + i``, so the N outputs differ from each other.
    """
    gen_cfg = cfg.generate
    device = resolve_device(cfg)

    ckpt_dir = _find_checkpoint_dir(cfg, checkpoint_dir)
    ckpt_path = ckpt_dir / "best.pt"
    vocab_path = ckpt_dir / "vocab.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    if not vocab_path.exists():
        # The vocab must be the one saved alongside the weights, never one
        # rebuilt from the current data -- rebuilding can reorder ids.
        raise FileNotFoundError(
            f"no vocab.json beside the checkpoint at {vocab_path}; "
            "the model is unusable without the vocabulary it was trained on"
        )

    # Onto the CPU first: the checkpoint also carries Adam's two moment
    # buffers (2x the weights), which have no business occupying VRAM here.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _verify_config_hash(cfg, ckpt)

    vocab = Vocab.load(vocab_path)
    # Built on the CPU inside fork_rng: the random weight init (immediately
    # overwritten by the checkpoint) would otherwise advance torch's global
    # RNG, and generate() promises to leave global RNG state alone.
    with torch.random.fork_rng(devices=[]):
        model = build_model(_with_checkpoint_head(cfg, ckpt), len(vocab), pad_id=vocab.pad_id)
    _load_state_dict(model, ckpt)
    del ckpt
    model.to(device).eval()

    encoder = get_encoder(cfg)
    seq_len = int(cfg.model.seq_len)
    pad = int(vocab.pad_id)

    # One number reproduces the whole run; print it before anything is sampled
    # so it survives even if generation dies half way.
    run_seed = resolve_seed(cfg)
    explicit = cfg.generate.get("seed", None) is not None
    print(
        f"[generate] seed {run_seed} ({'from config' if explicit else 'fresh entropy'}); "
        f"reproduce with --seed {run_seed} (or --set generate.seed={run_seed})"
    )

    # Build the seed source once; `dataset` re-draws a fresh window per sample
    # so the outputs are not all continuations of the same excerpt.
    source = str(gen_cfg.get("seed_source", "dataset"))
    pool: Optional[List[List[int]]] = None
    fixed_seed: Optional[List[int]] = None

    if source == "dataset":
        _check_cache_vocab_matches(cfg, vocab)
        pool = _load_dataset_pool(cfg)
    elif source in ("midi", "audio"):
        seed_path = gen_cfg.get("seed_path")
        if not seed_path:
            raise ValueError(f"generate.seed_source is {source!r} but seed_path is unset")
        path = resolve_path(seed_path)
        if not path.exists():
            raise FileNotFoundError(f"seed file not found: {path}")
        if source == "audio":
            # The context manager deletes its temp dir; the piece must be parsed
            # inside it, before the MIDI file is gone.
            from app.transcribe import transcribed_seed
            with transcribed_seed(path, grid=float(cfg.data.grid)) as midi_path:
                piece = load_piece(midi_path, cfg)
        else:
            piece = load_piece(path, cfg)
        if piece is None or not piece.events:
            raise ValueError(f"seed file {path} parsed to an empty piece")
        fixed_seed = _seed_context(_seed_from_piece(piece, encoder, vocab), seq_len)
    else:
        raise ValueError(
            f"unknown generate.seed_source {source!r}; expected dataset|midi|audio"
        )

    out_dir = resolve_path(gen_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temperature = float(gen_cfg.temperature)
    program = gen_cfg.get("program", 0)
    num_samples = int(gen_cfg.num_samples)
    banned = special_token_ids(vocab, allow_unk=bool(gen_cfg.get("allow_unk", False)))
    written: List[Path] = []

    picker = (
        _WindowPicker(pool, seq_len, pad, random.Random(run_seed),
                      _make_fingerprint(vocab, encoder, pad))
        if pool is not None
        else None
    )

    # Seeds and RNGs for every sample up front, in sample order -- exactly the
    # draws the one-sample-at-a-time loop made, so the seed windows are
    # unchanged. Each sample gets its own torch.Generator on `seed + i`
    # instead of reseeding torch's global RNG: same token stream, and nothing
    # outside this function is disturbed.
    sample_seeds = [derive_seed(run_seed, index) for index in range(num_samples)]
    seeds: List[List[int]] = []
    generators: List[torch.Generator] = []
    for sample_seed in sample_seeds:
        sample_rng = random.Random(sample_seed)
        seeds.append(picker.take(sample_rng) if picker is not None else list(fixed_seed or []))
        generators.append(torch.Generator(device=device).manual_seed(sample_seed))

    incremental = gen_cfg.get("incremental", None)
    all_ids: List[List[int]] = []
    chunk = max(1, int(gen_cfg.get("batch_size", 64)))
    for start in range(0, num_samples, chunk):
        all_ids.extend(_sample_many(
            model,
            seeds[start:start + chunk],
            num_tokens=int(gen_cfg.num_tokens),
            seq_len=seq_len,
            device=device,
            temperature=temperature,
            top_k=int(gen_cfg.get("top_k", 0) or 0),
            top_p=float(gen_cfg.get("top_p", 0.0) or 0.0),
            banned_ids=banned,
            generators=generators[start:start + chunk],
            incremental=None if incremental is None else bool(incremental),
        ))

    for index, (ids, sample_seed) in enumerate(zip(all_ids, sample_seeds)):
        symbols = list(vocab.decode(ids))
        piece = encoder.decode(symbols)
        piece.source = f"generated:{cfg.name}:{index}"
        # Without durations in the encoding every note decodes as a 16th.
        # Filling to the next onset is what makes the output sound held
        # rather than plucked -- see decode.fill_durations for the numbers.
        fill = str(gen_cfg.get("duration_fill", "next_onset"))
        if fill == "next_onset" and not getattr(encoder, "include_duration", True):
            piece = fill_durations(piece, max_beats=float(gen_cfg.get("max_fill_beats", 4.0)))

        # The run seed makes the name unique per run (the old
        # "<name>_t0.90_00" was reused by every run, so each generate silently
        # overwrote the last one's samples) and is exactly what --seed needs
        # to reproduce this file.
        stem = f"{cfg.name}_T{temperature:.2f}_seed{run_seed}_{index:02d}"
        midi_path = write_midi(piece, out_dir / f"{stem}.mid", program=program)
        # Symbols next to the audio: when a sample sounds wrong this is the
        # only way to tell a decoding bug from a modelling one.
        (out_dir / f"{stem}.txt").write_text("\n".join(symbols), encoding="utf-8")

        written.append(midi_path)
        print(
            f"[generate] {midi_path.name}  ({len(piece)} notes, "
            f"{len(symbols)} symbols, seed {sample_seed})"
        )

    if picker is not None and picker.reused:
        print(
            f"[generate] warning: {picker.reused} of {len(written)} samples reused a "
            "seed context; the pool holds too few distinct musical windows "
            "(a bigger corpus, or a shorter model.seq_len, gives more)"
        )

    return written
