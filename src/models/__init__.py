"""SEAM 2 -- the model interface.

Every architecture in this package obeys exactly one contract:

    forward(x: LongTensor[B, seq_len])
        -> logits: FloatTensor[B, seq_len, vocab_size]

Logits at EVERY position: position t predicts the token at t+1. This is plain
teacher forcing -- one training target per position rather than one per window,
which is ~100x more supervision from the very same forward pass at the
project's 100-token window. Every architecture here is causal (the LSTM by
construction, the Transformer via its mask), so position t has only ever seen
tokens 0..t and no label leaks backwards.

Callers that want next-token-only -- the sampler, mainly -- slice
``logits[:, -1, :]`` themselves; the models never do it for them.

Because that is the *whole* interface, the trainer, the sampler and the
evaluator are written once and never branch on architecture -- which is what
makes the LSTM-vs-Transformer comparison a one-line config change
(``model.arch``) rather than a parallel codebase. Anything that would need to
know which model it holds belongs inside the model, not outside it.

Models may additionally accept ``hidden`` / ``return_hidden`` and return
``(logits, hidden)`` so incremental generation can avoid re-running the whole
window each step. That is strictly optional: callers that ignore it get the
plain logits tensor from every architecture.

Two further OPTIONAL hooks, used when present and never required:

* ``model.loss(x, y, ignore_index)`` -> mean NLL. A model whose output layer
  never materialises full [B, L, V] logits during training (the LSTM with
  ``model.output: adaptive``) scores itself; the trainer and
  ``evaluate.perplexity`` prefer this over logits + CrossEntropyLoss.
* ``forward(..., last_only=True)`` (flag ``supports_last_only``) -> scores only
  the final position; the sampler uses it while priming.
"""

from __future__ import annotations

from typing import Any, Dict, Type

import torch.nn as nn

from src.models.lstm import DEFAULT_ADAPTIVE_CUTOFFS, MusicLSTM
from src.models.transformer import MusicTransformer

__all__ = ["MusicLSTM", "MusicTransformer", "build_model", "ARCHITECTURES"]

ARCHITECTURES: Dict[str, Type[nn.Module]] = {
    "lstm": MusicLSTM,
    "transformer": MusicTransformer,
}


def build_model(cfg: Any, vocab_size: int, pad_id: int = 0) -> nn.Module:
    """Construct the model named by ``cfg.model.arch``.

    Args:
        cfg: Loaded Config (see src/config.py); reads the ``model`` section.
        vocab_size: Size of the vocabulary the model must predict over. Comes
            from the Vocab that was built alongside the data, never a constant.
        pad_id: Padding index, excluded from the embedding's gradient.

    Raises:
        ValueError: if ``model.arch`` is not a registered architecture.
    """
    model_cfg = cfg["model"] if isinstance(cfg, dict) else cfg.model
    arch = str(model_cfg["arch"]).strip().lower()

    if arch not in ARCHITECTURES:
        valid = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unknown model.arch {arch!r}; valid options are: {valid}")

    common = dict(
        vocab_size=vocab_size,
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        pad_id=pad_id,
    )

    output = str(model_cfg.get("output", "full") or "full").strip().lower()
    if arch == "lstm":
        cutoffs = model_cfg.get("adaptive_cutoffs", None)
        return MusicLSTM(
            output=output,
            adaptive_cutoffs=DEFAULT_ADAPTIVE_CUTOFFS if cutoffs is None else list(cutoffs),
            adaptive_div_value=float(model_cfg.get("adaptive_div_value", 4.0)),
            **common,
        )
    if output != "full":
        raise ValueError(
            f"model.output={output!r} is implemented for arch=lstm only; "
            "use model.output=full with the transformer"
        )

    # seq_len sizes the positional table; leave headroom so a longer prompt at
    # generation time does not trip the bounds check.
    seq_len = int(model_cfg.get("seq_len", 100))
    return MusicTransformer(
        num_heads=int(model_cfg.get("num_heads", 8)),
        max_len=max(512, seq_len * 4),
        **common,
    )
