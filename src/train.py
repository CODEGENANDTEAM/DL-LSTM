"""Training loop. Architecture-agnostic by construction.

Everything below talks to a model only through SEAM 2 (``logits = model(x)``,
shape [B, seq_len, vocab_size] -- a prediction at every position), so
``model.arch: lstm`` and ``model.arch: transformer`` run through this identical
code path. That is what makes the two arms of the
comparison honest: same optimizer, same clipping, same early stopping, same
metric definitions.

Usage:
    D:/PYTH/python.exe -m src.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from src.config import Config, config_hash, load_config, resolve_path
from src.data.dataset import dataloaders_from_cache
from src.data.vocab import Vocab
from src.models import build_model

__all__ = ["train", "main", "resolve_device", "seed_everything"]

# Reserved-symbol ordering is fixed by the vocab builder: <PAD> is always id 0.
# It is both the embedding's padding_idx and the loss's ignore_index, so padded
# positions contribute neither gradient nor loss.
DEFAULT_PAD_ID = 0

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_ppl",
    "lr",
    "seconds",
    "improved",
)


# --------------------------------------------------------------------------
# setup helpers
# --------------------------------------------------------------------------

def resolve_device(spec: str) -> torch.device:
    """Turn ``train.device`` (auto | cuda | cpu) into a torch.device."""
    spec = str(spec).strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device is 'cuda' but no CUDA device is visible")
    return torch.device(spec)


def seed_everything(seed: int) -> None:
    """Seed every RNG that touches training.

    Not a guarantee of bit-identical runs (cuDNN kernel selection is still
    non-deterministic), but enough that two runs of the same config land in the
    same neighbourhood -- which is the bar for an ablation table to mean
    anything.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, train_cfg: Any) -> torch.optim.Optimizer:
    """Instantiate the optimizer named by ``train.optimizer``."""
    name = str(train_cfg.get("optimizer", "adam")).strip().lower()
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown train.optimizer {name!r}; valid options are: adam, adamw, rmsprop")


def _resolve_amp(setting: Any, device: torch.device) -> Optional[torch.dtype]:
    """Mixed-precision dtype for this run, or None for full fp32.

    Default is off, and that is a measured decision, not caution. On an RTX 5070
    with a 25k-symbol vocabulary, 256-token context, identical seed and data:

        after 700 steps       train loss   val loss (scored in fp32)   ms/step   VRAM
        fp32                     5.054           4.8016                  101     7.7 GB
        bf16                     5.469           5.3542                   74     4.1 GB
        bf16 + fp32 loss         5.454           5.3472                   82     7.4 GB

    bf16 plateaus from about step 350 at roughly the loss of predicting token
    frequencies with no context. Scoring its weights in fp32 gives the same
    number, so it is genuinely learning worse, not being mis-measured; and
    upcasting the loss does not help, so the damage is in the forward/backward
    pass. The likely mechanism is the logits: near magnitude 10, bf16 values
    are ~0.06 apart, which rounds away the small context-dependent adjustments
    the model has to learn. (cuDNN's LSTM is not affected -- autocast already
    leaves it in fp32.)

    The option stays because it may suit other architectures, but check that
    val loss tracks an fp32 run before trusting it.

    ``fp16`` (autocast float16 + GradScaler) has 3 more mantissa bits than bf16
    -- logit spacing ~0.008 near 10 instead of ~0.06 -- which is exactly the
    suspected failure mode above, so it was measured separately (v4, 37k
    vocab, RTX 5070, batch 48, val scored in fp32):

        from v4 epoch-5 best.pt, +700 steps   val      ms/step   VRAM alloc
        fp32 (baseline)                      3.6843      98       5.91 GB
        fp16 + GradScaler                    3.6852      76       4.85 GB
        bf16                                 3.6872      74       4.85 GB
        from scratch, 1 epoch (2 seeds)  fp32 4.389/4.441   fp16 4.151/4.487

    fp16 passes parity and is 22% faster; it stays opt-in (see
    configs/default.yaml). Loss and softmax run in fp32 either way: autocast
    runs cross_entropy / log_softmax in fp32, the adaptive softmax disables
    autocast, and the trainer upcasts the loss before accumulating it.
    """
    value = str(setting).lower() if setting is not None else "off"
    if value in ("off", "false", "none", "fp32", "0", "auto"):
        return None
    if device.type != "cuda":
        return None
    if value in ("bf16", "bfloat16", "true", "1"):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("train.amp=bf16 requested but this GPU does not support bfloat16")
        return torch.bfloat16
    if value in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"unknown train.amp {setting!r}; valid options are: off, bf16, fp16")


def apply_tf32(setting: Any) -> str:
    """Set TF32 math for fp32 matmuls and cuDNN (``train.tf32``); returns the mode.

    * ``default`` / null -- leave torch's own defaults, which are what every
      run so far used: cuDNN (so the LSTM) MAY use TF32, plain matmuls (the
      output projection, fc1) may NOT.
    * ``on``  -- TF32 for matmuls too (``torch.backends.cuda.matmul.allow_tf32``).
    * ``off`` -- strict IEEE fp32 everywhere, cuDNN included.

    TF32 keeps fp32's 8-bit exponent but only 10 mantissa bits in the matmul
    inputs, accumulating in fp32. Measured on v4 (ms/batch, val vs baseline):
    default 98; on 85, +0.0006 near convergence but +0.05 @700 steps and
    +0.14 / -0.02 @1 epoch from scratch; off 106, +0.0010. So ``on`` is
    opt-in, not default.
    """
    value = "default" if setting is None else str(setting).strip().lower()
    if value in ("default", "", "null", "none"):
        return "default"
    if value in ("on", "true", "1", "tf32"):
        flag = True
    elif value in ("off", "false", "0", "ieee", "fp32"):
        flag = False
    else:
        raise ValueError(f"unknown train.tf32 {setting!r}; valid options are: default, on, off")
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cudnn.allow_tf32 = flag
    return "on" if flag else "off"


def maybe_compile(fn: Any, enabled: bool, example: Optional[Tuple[Any, ...]] = None) -> Tuple[Any, bool]:
    """``torch.compile(fn)`` if asked and it actually works here, else ``fn``.

    Compilation is lazy, so a missing backend (no Triton -- the usual state of
    a Windows install) only surfaces on the first call. When ``example`` args
    are given the compiled function is run once on them right here, so the
    failure is caught before training starts instead of mid-epoch. Returns
    ``(callable, compiled?)``; any failure prints one warning and falls back to
    eager.
    """
    if not enabled:
        return fn, False
    try:
        compiled = torch.compile(fn)
        if example is not None:
            out = compiled(*example)
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.backward()
        return compiled, True
    except Exception as exc:  # noqa: BLE001 - any backend failure means "eager"
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        print(f"WARNING: train.compile requested but torch.compile failed ({first[:160]}); "
              "continuing in eager mode")
        try:
            torch._dynamo.reset()
        except Exception:  # pragma: no cover
            pass
        return fn, False


def make_loss_fn(model: nn.Module, criterion: nn.Module) -> Any:
    """``(x, y) -> mean loss``: the model's own ``loss`` if it has one.

    For the full-softmax LSTM that is numerically the old logits +
    CrossEntropyLoss path; for ``model.output: adaptive`` it is the only path,
    since full [B, L, V] logits are exactly what adaptive softmax avoids.
    """
    ignore_index = int(getattr(criterion, "ignore_index", -100))
    own = getattr(model, "loss", None)
    if callable(own):
        def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return own(x, y, ignore_index=ignore_index)
        return loss_fn

    def logits_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = model(x)                      # SEAM 2: [B, L, vocab_size]
        if isinstance(logits, (tuple, list)):  # (logits, hidden)
            logits = logits[0]
        # Teacher forcing: every position is a training target. Flattening
        # to [B*L, V] / [B*L] is what makes CrossEntropyLoss score all of
        # them at once; it is also why the loss is already a mean over
        # tokens rather than over windows.
        return criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return logits_loss


def target_counts(loader: Any, vocab_size: int) -> np.ndarray:
    """How often each id occurs in ``loader``'s training data (for adaptive softmax).

    Reads the flat token array of a MusicDataset directly (one bincount over
    12.8M tokens, ~0.05 s) and otherwise falls back to one pass over the
    loader's targets.
    """
    data = getattr(getattr(loader, "dataset", None), "data", None)
    flat = getattr(data, "flat", None)
    if flat is not None:
        return np.bincount(np.asarray(flat, dtype=np.int64), minlength=vocab_size)[:vocab_size]
    counts = np.zeros(vocab_size, dtype=np.int64)
    for batch in loader:
        y = batch[1].reshape(-1).long().numpy()
        counts += np.bincount(y[(y >= 0) & (y < vocab_size)], minlength=vocab_size)
    return counts


LR_SCHEDULES = ("none", "plateau", "cosine")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Any, steps_per_epoch: int) -> Tuple[Optional[Any], str]:
    """``train.lr_schedule``: none | plateau | cosine. Returns (scheduler, kind).

    * ``plateau`` -- ReduceLROnPlateau on val loss: multiply the LR by
      ``lr_factor`` (0.5) as soon as val has failed to improve for more than ``lr_patience`` (0)
      epochs. Aimed squarely at the observed failure: val peaks around epoch
      5-10 and then rises, at a fixed LR of 1e-3. Stepped once per epoch.
    * ``cosine`` -- cosine decay from ``lr`` to ``lr * min_lr_ratio`` over
      ``epochs`` epochs, stepped every batch. It decays over train.epochs, so
      with early stopping pick epochs close to the length you expect.
    """
    kind = str(train_cfg.get("lr_schedule", "none") or "none").strip().lower()
    if kind not in LR_SCHEDULES:
        raise ValueError(f"unknown train.lr_schedule {kind!r}; valid options are: {', '.join(LR_SCHEDULES)}")
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("lr_factor", 0.5)),
            patience=int(train_cfg.get("lr_patience", 0)),
            min_lr=float(train_cfg.get("min_lr", 0.0)),
        ), kind
    if kind == "cosine":
        total = max(1, int(train_cfg.get("epochs", 60)) * max(1, int(steps_per_epoch)))
        base = float(train_cfg.get("lr", 1e-3))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total, eta_min=base * float(train_cfg.get("min_lr_ratio", 0.05)),
        ), kind
    return None, kind

def _find_vocab_file(cfg: Config) -> Optional[Path]:
    """Locate the vocab JSON the data layer wrote, for copying into the run."""
    processed = resolve_path(cfg.data.get("processed_dir", "data/processed"))
    data_hash = config_hash(cfg, "data", "augment", "encoding")
    candidates = [
        processed / data_hash / "vocab.json",
        processed / "vocab.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    # No glob fallback. It used to return the first vocab.json found anywhere
    # under data/processed -- typically a different config's -- and copy it
    # beside the checkpoint, pairing the model with the wrong vocabulary.
    return None


def _write_vocab_beside_checkpoint(cfg: Config, vocab: Optional[Vocab], out_dir: Path) -> Optional[Path]:
    """Put ``vocab.json`` in the same directory as the checkpoints.

    A checkpoint separated from its vocabulary is useless: the model only ever
    emits integer ids, and those ids mean whatever the vocabulary that was
    built alongside the training data says they mean. Pair a checkpoint with a
    vocabulary built from a different corpus, encoding scheme, or even a
    different run of the same pipeline (dict iteration order is not a
    contract) and every generated token maps to the wrong note -- the model
    still "works", the output is just noise. So the vocabulary travels with
    the weights, always, and config_hash in the checkpoint lets a loader
    verify the pairing.
    """
    target = out_dir / "vocab.json"
    if vocab is not None and hasattr(vocab, "save"):
        vocab.save(target)
        return target
    source = _find_vocab_file(cfg)
    if source is not None:
        shutil.copyfile(source, target)
        return target
    return None


# --------------------------------------------------------------------------
# train / eval passes
# --------------------------------------------------------------------------

def _to_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalise a loader item to (x: [B, L], y: [B, L]) on ``device``.

    Targets are the shifted window, one per input position -- see
    ``MusicDataset.__getitem__``. A scalar-per-window target ([B]) is still
    accepted and lifted to [B, 1] so an old cached loader, or a caller that
    scores a single next token, keeps working through the same reshape below.
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise ValueError("dataloader must yield (x, y) pairs")
    x = x.to(device, non_blocking=True).long()
    y = y.to(device, non_blocking=True).long()
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    if y.dim() != 2:
        raise ValueError(f"expected targets of shape [B, L], got {tuple(y.shape)}")
    return x, y


def _run_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[Any] = None,
    loss_fn: Optional[Any] = None,
    step_scheduler: Optional[Any] = None,
) -> float:
    """One pass. Training pass when ``optimizer`` is given, else evaluation.

    ``scaler`` is a GradScaler (fp16 only); ``loss_fn(x, y)`` defaults to
    ``make_loss_fn(model, criterion)``; ``step_scheduler`` is stepped after
    every optimizer step (the per-batch cosine schedule).

    Returns the mean loss per (non-ignored) target token.
    """
    training = optimizer is not None
    model.train(training)
    if loss_fn is None:
        loss_fn = make_loss_fn(model, criterion)
    ignore_index = int(getattr(criterion, "ignore_index", -100))

    # Accumulated ON THE DEVICE. Calling loss.item() and (y != pad).sum().item()
    # every batch forced two device->host syncs per step, stalling the CPU
    # until the GPU drained its queue, so the next batch could not be queued
    # behind the current one. One sync at the end of the epoch instead.
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.int64, device=device)

    # inference_mode for evaluation: like no_grad, plus no version-counter /
    # view tracking. Validation keeps no graph either way.
    grad_ctx = torch.enable_grad() if training else torch.inference_mode()
    with grad_ctx:
        for batch in loader:
            x, y = _to_batch(batch, device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                loss = loss_fn(x, y)
            # Accumulate and back-propagate from an fp32 scalar whatever the
            # autocast dtype was.
            loss = loss.float()

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Unscale first so clipping sees the true gradient norm.
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if grad_clip and grad_clip > 0:
                    # Recurrent nets in particular hit occasional huge
                    # gradients; clipping keeps one bad batch from wiping out
                    # an otherwise healthy run.
                    clip_grad_norm_(model.parameters(), grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)       # skips the step on inf/NaN grads
                    scaler.update()
                else:
                    optimizer.step()
                if step_scheduler is not None:
                    step_scheduler.step()

            # CrossEntropyLoss's default reduction is the mean over the
            # NON-ignored entries it was given, so weighting each batch by that
            # same count and dividing by the total recovers the exact per-token
            # mean over the epoch. Weighting by B instead would mis-average the
            # final short batch, and perplexity is exp() of this number. A
            # batch with zero scored tokens gives loss NaN; mask it out rather
            # than letting it poison the sum.
            n = (y != ignore_index).sum()
            batch_loss = loss.detach().double() * n
            total_loss += torch.where(n > 0, batch_loss, torch.zeros_like(batch_loss))
            total_tokens += n

    tokens = int(total_tokens.item())
    return float(total_loss.item()) / tokens if tokens else float("nan")


def _perplexity(loss: float) -> float:
    """exp(loss), clamped so a diverged epoch prints a number instead of inf."""
    if not math.isfinite(loss):
        return float("nan")
    return math.exp(min(loss, 20.0))


# --------------------------------------------------------------------------
# checkpointing / history
# --------------------------------------------------------------------------

def _atomic_torch_save(payload: Any, path: Path) -> None:
    """torch.save to a temp file in the same directory, then os.replace.

    Writing best.pt in place meant that a crash, a full disk or a Ctrl-C
    during the write left a truncated best.pt -- destroying the one file the
    run exists to produce, in exchange for a checkpoint that was never
    finished. os.replace is atomic on both Windows and POSIX when source and
    destination are on the same volume, so readers see either the old file
    or the new one, never half of one.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: Config,
    vocab_size: int,
    epoch: int,
    val_loss: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    state: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a checkpoint atomically.

    ``epoch`` / ``val_loss`` keep their original meaning (the best epoch and
    its monitored loss) so every existing reader -- generate.py, notebooks,
    older checkpoints -- is unaffected. ``state`` adds the trainer's loop
    state (``last_epoch``, ``best_epoch``, ``best_val``, ``stale_epochs``) so a
    resume continues exactly where the run stopped.
    """
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": dict(cfg),
        "config_hash": config_hash(cfg),
        # Narrower hash over just the sections that determine the symbol set,
        # so a loader can check the vocabulary pairing independently of
        # unrelated knobs like train.epochs.
        "data_config_hash": config_hash(cfg, "data", "augment", "encoding"),
        "vocab_size": vocab_size,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if state:
        payload.update(state)
    # Adam's per-parameter moments take a few hundred steps to rebuild; a
    # resume that dropped them would visibly bump the loss on its first
    # epoch. Cheap to carry, so always carry it.
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    _atomic_torch_save(payload, path)


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    """Append one epoch to history.csv so ablation tables never need a re-run."""
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(HISTORY_FIELDS))
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})


def _truncate_history(path: Path, last_epoch: int) -> int:
    """Drop history rows for epochs after ``last_epoch``; return how many.

    On resume the run continues from the checkpoint's epoch. Any row after it
    describes work whose weights were never checkpointed (or were discarded),
    and appending the re-run epochs after it produced duplicate epoch numbers
    that silently corrupt any plot or table built from history.csv.
    """
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keep = []
    for row in rows:
        try:
            if int(row.get("epoch", "")) <= last_epoch:
                keep.append(row)
        except ValueError:
            continue
    dropped = len(rows) - len(keep)
    if dropped:
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(HISTORY_FIELDS))
            writer.writeheader()
            for row in keep:
                writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})
        os.replace(tmp, path)
    return dropped


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def train(cfg: Config) -> Path:
    """Train the configured model and return the path to ``best.pt``."""
    train_cfg = cfg.train
    device = resolve_device(train_cfg.get("device", "auto"))
    seed_everything(int(train_cfg.get("seed", 1337)))

    train_loader, val_loader, _test_loader, vocab = dataloaders_from_cache(cfg)
    if vocab is None:
        vocab_file = _find_vocab_file(cfg)
        if vocab_file is None:
            raise FileNotFoundError(
                "could not locate vocab.json under "
                f"{resolve_path(cfg.data.get('processed_dir', 'data/processed'))}; "
                "run the data preparation step first"
            )
        vocab = Vocab.load(vocab_file)

    vocab_size = len(vocab)
    pad_id = int(getattr(vocab, "pad_id", DEFAULT_PAD_ID))

    tf32_mode = apply_tf32(train_cfg.get("tf32", "default"))
    model = build_model(cfg, vocab_size, pad_id=pad_id)
    # Adaptive softmax orders its clusters by training-target frequency. On a
    # resume the checkpoint's saved permutation overwrites this below, so a
    # resumed run keeps the order it was trained with.
    if getattr(model, "output", "full") == "adaptive":
        model.set_token_frequencies(target_counts(train_loader, vocab_size))
    model = model.to(device)
    amp_dtype = _resolve_amp(train_cfg.get("amp", "auto"), device)
    if device.type == "cuda":
        # Window length is fixed, so cuDNN can benchmark its LSTM kernels once
        # and reuse the fastest for every batch.
        torch.backends.cudnn.benchmark = True
        # On Windows a near-full card makes the driver spill into system RAM
        # and epochs silently get 20-40x slower (seen twice on this project).
        # The estimate models the full [B, L, V] logits, so it only applies to
        # the full softmax; adaptive never materialises them.
        if getattr(model, "output", "full") == "full":
            from src.config import gpu_memory_warning
            total = torch.cuda.get_device_properties(device).total_memory
            warning = gpu_memory_warning(cfg, vocab_size, total)
            if warning:
                print(f"WARNING: {warning}", flush=True)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = build_optimizer(model, train_cfg)
    scheduler, schedule_kind = build_scheduler(optimizer, train_cfg, len(train_loader))
    # GradScaler only for fp16: bf16 has fp32's exponent range and never needs it.
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    # A dummy batch, not a real one: drawing from train_loader would advance
    # the shuffle RNG and change the data order relative to an eager run.
    probe_len = int(cfg.model.get("seq_len", 100))
    probe = torch.ones((int(train_cfg.get("batch_size", 64)), probe_len), dtype=torch.long, device=device)
    loss_fn, compiled = maybe_compile(
        make_loss_fn(model, criterion), bool(train_cfg.get("compile", False)), (probe, probe)
    )
    if compiled:
        optimizer.zero_grad(set_to_none=True)   # discard the probe's gradients

    epochs = int(train_cfg.get("epochs", 60))
    patience = int(train_cfg.get("patience", 3))
    # An epoch only counts as an improvement if it beats the best val loss by
    # more than this. 0 keeps the old "any decrease" rule.
    min_delta = float(train_cfg.get("min_delta", 0.0) or 0.0)
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    out_dir = resolve_path(train_cfg.get("checkpoint_dir", "runs")) / str(cfg.get("name", "run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    history_path = out_dir / "history.csv"

    vocab_copy = _write_vocab_beside_checkpoint(cfg, vocab, out_dir)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"{cfg.get('name', 'run')} | arch={cfg.model.arch} | device={device} | "
        f"amp={str(amp_dtype).replace('torch.', '') if amp_dtype else 'off'} | "
        f"tf32={tf32_mode} | output={getattr(model, 'output', 'full')} | "
        f"compile={'on' if compiled else 'off'} | lr_schedule={schedule_kind} | "
        f"vocab={vocab_size} | params={n_params:,} | out={out_dir}"
    )
    if vocab_copy is None:
        print("WARNING: no vocab.json saved next to the checkpoint; generation will not be reproducible")
    print(f"{'epoch':>7}  {'train':>9}  {'val':>9}  {'val_ppl':>10}  {'time':>7}")

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch = 0
    start_epoch = 1

    def loop_state() -> Dict[str, Any]:
        return {
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "stale_epochs": epochs_without_improvement,
            # Optional resume state; absent keys are simply not restored.
            **({"scheduler_state": scheduler.state_dict()} if scheduler is not None else {}),
            **({"scaler_state": scaler.state_dict()} if scaler is not None else {}),
        }

    # Resume. last.pt is now written atomically at the end of EVERY epoch and
    # records the epoch it actually holds (`last_epoch`), so it is always the
    # newest consistent state; best.pt is the fallback for runs that predate
    # that (or where last.pt was deleted). Previously last.pt was only written
    # at the very end or on Ctrl-C, and carried best_epoch as its epoch, so a
    # resume re-ran (and re-logged) every epoch after the best one on top of
    # weights that had already trained through them.
    if bool(train_cfg.get("resume", False)):
        source = last_path if last_path.exists() else best_path
        if not source.exists():
            print(f"train.resume set but no checkpoint in {out_dir}; starting fresh")
        else:
            ckpt = torch.load(source, map_location="cpu", weights_only=False)
            saved_hash = ckpt.get("data_config_hash")
            current_hash = config_hash(cfg, "data", "augment", "encoding")
            if saved_hash is not None and saved_hash != current_hash:
                raise RuntimeError(
                    f"cannot resume {source.name}: it was trained under a different "
                    f"data/encoding config ({saved_hash} vs {current_hash}), so its "
                    "vocabulary ids mean something else. Delete the run directory to "
                    "start fresh."
                )
            try:
                model.load_state_dict(ckpt["model_state"])
            except RuntimeError as exc:
                raise RuntimeError(
                    f"cannot resume {source}: its weights do not fit the model this "
                    "config builds (the model architecture -- arch, hidden_dim, "
                    "num_layers, embed_dim or vocab size -- changed since it was "
                    "saved). Use a new `name` to train a fresh run.\n"
                    f"{str(exc).splitlines()[0]}"
                ) from None
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            else:
                print("  (checkpoint has no optimizer state; Adam moments restart cold)")
            if scheduler is not None and "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if scaler is not None and "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])
            best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
            best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
            # Old checkpoints carry no loop state: their `epoch` is the best
            # epoch and the patience counter is unknown (assume fresh).
            last_epoch = int(ckpt.get("last_epoch", best_epoch))
            epochs_without_improvement = int(ckpt.get("stale_epochs", 0))
            start_epoch = last_epoch + 1
            del ckpt
            dropped = _truncate_history(history_path, last_epoch)
            print(f"resumed {source.name} at epoch {last_epoch} (best val {best_val:.4f} "
                  f"at epoch {best_epoch}); continuing from epoch {start_epoch}"
                  + (f"; dropped {dropped} history row(s) past the checkpoint" if dropped else ""))
            if start_epoch > epochs:
                print(f"already trained {last_epoch} epochs >= train.epochs={epochs}; nothing to do")
                return best_path
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"run already early-stopped ({epochs_without_improvement} epochs without "
                      f"improvement >= patience {patience}); nothing to do. Raise "
                      "train.patience to keep training it.")
                return best_path

    if start_epoch == 1 and history_path.exists():
        # A fresh (non-resumed) run owns its history. Appending to a previous
        # run's file interleaved two runs' epochs (1 2 3 1 2 3 ...).
        history_path.unlink()

    try:
        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            # The LR this epoch trains at (the cosine schedule moves it every
            # batch; this is its value at the start of the epoch).
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            train_loss = _run_epoch(
                model, train_loader, criterion, device, optimizer, grad_clip, amp_dtype,
                scaler=scaler, loss_fn=loss_fn,
                step_scheduler=scheduler if schedule_kind == "cosine" else None,
            )
            val_loss = (
                _run_epoch(model, val_loader, criterion, device, amp_dtype=amp_dtype, loss_fn=loss_fn)
                if val_loader is not None
                else float("nan")
            )
            elapsed = time.time() - started

            # With no validation split, fall back to train loss so early
            # stopping and checkpoint selection still have a signal.
            monitored = val_loss if math.isfinite(val_loss) else train_loss
            improved = monitored < best_val - min_delta
            if schedule_kind == "plateau" and math.isfinite(monitored):
                scheduler.step(monitored)

            last_epoch = epoch
            if improved:
                best_val = monitored
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if improved:
                _save_checkpoint(best_path, model, cfg, vocab_size, epoch, monitored,
                                 optimizer, loop_state())
            # Every epoch: the resume point. Written after best.pt so last.pt
            # is never older than best.pt.
            _save_checkpoint(last_path, model, cfg, vocab_size, best_epoch, best_val,
                             optimizer, loop_state())

            print(
                f"{epoch:>3}/{epochs:<3}  {train_loss:>9.4f}  {val_loss:>9.4f}  "
                f"{_perplexity(val_loss):>10.2f}  {elapsed:>6.1f}s"
                f"{'  *' if improved else ''}"
            )
            _append_history(
                history_path,
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_ppl": f"{_perplexity(val_loss):.6f}",
                    "lr": f"{epoch_lr:.6g}",
                    "seconds": f"{elapsed:.2f}",
                    "improved": int(improved),
                },
            )

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"early stop: no improvement for {patience} epochs (best epoch {best_epoch})")
                break

    except KeyboardInterrupt:
        # last.pt already holds the end of the last COMPLETE epoch. Saving the
        # half-trained weights of the interrupted epoch over it would make a
        # resume skip the rest of that epoch's data while claiming it ran, so
        # the partial epoch is simply discarded.
        print(f"\ninterrupted during epoch {last_epoch + 1}; resume point is epoch "
              f"{last_epoch} in {last_path.name}")
        return best_path if best_path.exists() else last_path

    print(f"done: best val loss {best_val:.4f} (epoch {best_epoch}) -> {best_path}")
    return best_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a music-generation model.")
    parser.add_argument("--config", default=None, help="path to a YAML config (layered over configs/default.yaml)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override, e.g. --set model.arch=transformer (repeatable)",
    )
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = yaml_scalar(raw.strip())

    cfg = load_config(args.config, **overrides)
    train(cfg)
    return 0


def yaml_scalar(text: str) -> Any:
    """Parse an ``--set`` value as YAML so 256 is an int and true is a bool."""
    from src.config import parse_override_value

    return parse_override_value(text)


if __name__ == "__main__":
    sys.exit(main())
