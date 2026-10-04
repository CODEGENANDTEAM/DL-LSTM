"""Logit transforms for stochastic decoding.

WHY NOT ARGMAX. A trained model's most likely next token, given a bar of C
major, is very often another C. Greedy decoding takes that token, appends it,
and asks again -- from a context that now contains one more C, which makes C
even more likely. The loop is self-reinforcing and absorbing: within a few
dozen steps almost every greedy music LM collapses into a single repeated note
or a two-note oscillation, no matter how good its validation perplexity is.
This is the single most common failure mode in music-generation projects, and
it is a *decoding* bug, not a training bug -- retraining will not fix it.

Sampling from the distribution instead keeps the model in the region of the
space it was actually trained on. Bare sampling has the opposite problem: the
long tail of thousands of near-zero-probability symbols collectively holds
real mass, so every few steps a nonsense token slips through and derails the
context. Temperature, top-k and top-p are the standard middle ground --
temperature reshapes how peaked the distribution is, while top-k and top-p
truncate the tail before sampling.

Every function here is pure: a logits tensor in, a new logits tensor out. No
model, no config, no state -- which is what makes them unit-testable in
isolation (tests/test_model_audit.py).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "apply_temperature",
    "top_k_filter",
    "top_p_filter",
    "ban_tokens",
    "sample_next",
    "sample_batch",
    "NEG_INF",
]

# Masking with -inf rather than a large negative number: softmax maps it to
# exactly 0, so a filtered token can never be drawn.
NEG_INF = float("-inf")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by ``temperature``, sharpening (<1) or flattening (>1).

    t -> 0 approaches argmax, t = 1 leaves the model's own distribution
    untouched, t > 1 raises the chance of unlikely tokens. Values <= 0 are
    returned unchanged; ``sample_next`` treats those as a request for greedy
    decoding and never reaches a division.
    """
    if temperature is None or temperature <= 0:
        return logits
    if temperature == 1.0:
        return logits.clone()
    return logits / float(temperature)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` highest logits along the last dim, mask the rest.

    ``k <= 0`` disables the filter (config convention). ``k`` larger than the
    vocabulary is clamped rather than raising.
    """
    if k is None or k <= 0:
        return logits
    vocab_size = logits.size(-1)
    k = min(int(k), vocab_size)
    if k == vocab_size:
        return logits

    # kth_value is the smallest logit we keep; anything strictly below it goes.
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_value, NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens with cumulative
    probability >= ``p``, mask the rest.

    Unlike top-k this adapts to how confident the model is: a peaked step keeps
    two or three candidates, an ambiguous one keeps dozens. ``p <= 0`` disables
    the filter (config convention); ``p >= 1`` is a no-op.
    """
    if p is None or p <= 0.0 or p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    # Mass strictly BEFORE each token: a token is dropped once the tokens
    # ranked above it already reach p. That is the textbook "smallest set with
    # mass >= p" -- the token that crosses p is kept and the top-1 token always
    # survives (nothing precedes it). The earlier `cumulative > p` plus
    # shift-right version kept one token too many whenever the running total
    # landed on p exactly (probs [.5, .25, .25], p=.75 kept all three).
    preceding = torch.cumsum(probs, dim=-1) - probs
    remove = preceding >= p
    remove[..., 0] = False

    mask = torch.zeros_like(remove).scatter_(-1, sorted_idx, remove)
    return logits.masked_fill(mask, NEG_INF)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    banned_ids: Optional[Iterable[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Draw one token id from a single step's logits.

    Args:
        logits: FloatTensor of shape [vocab_size] or [1, vocab_size] -- ONE
            timestep's row. Models return [B, L, vocab_size] (logits at every
            position, for teacher forcing), so callers must slice the step they
            are sampling -- ``logits[:, -1, :]`` for the next token. Passing the
            full [B, L, V] tensor raises below rather than silently sampling
            from the wrong timestep.
        temperature: <= 0 means greedy argmax (top_k / top_p are then
            irrelevant and skipped).
        top_k: 0 disables.
        top_p: 0.0 disables.
        banned_ids: ids that may never be drawn (special / style tokens).
        generator: RNG to draw from; None uses torch's global RNG.

    Returns:
        A plain Python int, ready to append to a token list.

    Order matters: temperature first (it changes the shape of the
    distribution, hence which tokens fall inside the nucleus), then top-k as a
    hard cap, then top-p on what survives.
    """
    if logits.dim() == 2:
        if logits.size(0) != 1:
            raise ValueError(
                f"sample_next expects one row, got batch of {logits.size(0)}; "
                "index the batch before calling"
            )
        logits = logits[0]
    elif logits.dim() != 1:
        raise ValueError(f"expected [vocab_size] or [1, vocab_size], got {tuple(logits.shape)}")

    out = sample_batch(
        logits.unsqueeze(0), temperature=temperature, top_k=top_k, top_p=top_p,
        banned_ids=banned_ids, generators=[generator],
    )
    return int(out[0].item())


def ban_tokens(logits: torch.Tensor, banned_ids: Optional[Iterable[int]]) -> torch.Tensor:
    """Mask ``banned_ids`` to -inf along the last dim (returns a copy).

    Accepts any iterable of ints, or a LongTensor -- the generation loop passes
    one already on the device so a step costs no host->device index copy.
    """
    if banned_ids is None:
        return logits
    if isinstance(banned_ids, torch.Tensor):
        if banned_ids.numel() == 0:
            return logits
        return logits.index_fill(-1, banned_ids.to(logits.device, torch.long), NEG_INF)
    size = logits.size(-1)
    ids = [int(i) for i in banned_ids if 0 <= int(i) < size]
    if not ids:
        return logits
    out = logits.clone()
    out[..., ids] = NEG_INF
    return out


def sample_batch(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    banned_ids: Optional[Iterable[int]] = None,
    generators: Optional[Sequence[Optional[torch.Generator]]] = None,
) -> torch.Tensor:
    """Draw one token per row of ``logits [B, V]`` -> LongTensor ``[B]``.

    Stays on the logits' device and never synchronises with the host, so a
    generation loop can feed the result straight back into the model.

    ``generators[i]`` (optional; None = torch's global RNG) drives row i. Each
    row gets its own ``torch.multinomial`` call, so row i consumes exactly the
    random numbers a solo ``sample_next(logits[i], generator=generators[i])``
    would: batching N samples does not change any one sample's tokens.

    Robustness, in order:
      * NaN logits (a diverged model) count as -inf and are never chosen --
        previously torch.argmax picked the NaN index itself;
      * ``banned_ids`` are masked before anything else;
      * temperature <= 0 is greedy argmax over what remains;
      * a row whose filtered distribution is degenerate (everything masked,
        +inf logits, overflow at a near-zero temperature) falls back to that
        greedy choice instead of letting multinomial raise mid-run.
    """
    if logits.dim() != 2:
        raise ValueError(f"sample_batch expects [B, V], got {tuple(logits.shape)}")
    # Detach and lift to float32: sampling must not build graph, and
    # multinomial is unhappy with half precision on CPU.
    raw = torch.nan_to_num(logits.detach().float(), nan=NEG_INF)
    raw = ban_tokens(raw, banned_ids)
    greedy = torch.argmax(raw, dim=-1)

    if temperature is None or temperature <= 0:
        return greedy

    filtered = apply_temperature(raw, temperature)
    filtered = top_k_filter(filtered, top_k)
    filtered = top_p_filter(filtered, top_p)
    probs = F.softmax(filtered, dim=-1)

    # Degenerate rows become one-hot on the greedy pick. Tensor ops, not an
    # `if`, which would cost a device->host sync on every generated token.
    ok = torch.isfinite(probs).all(dim=-1, keepdim=True) & (probs.sum(dim=-1, keepdim=True) > 0)
    onehot = F.one_hot(greedy, probs.size(-1)).to(probs.dtype)
    probs = torch.where(ok, probs, onehot)

    rows = probs.size(0)
    gens = list(generators) if generators is not None else [None] * rows
    if len(gens) != rows:
        raise ValueError(f"got {len(gens)} generators for {rows} rows")
    picks = [torch.multinomial(probs[i], num_samples=1, generator=gens[i]) for i in range(rows)]
    return torch.cat(picks)
