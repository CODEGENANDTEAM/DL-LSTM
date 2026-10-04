"""LSTM language model over music symbols -- the primary architecture.

Implements SEAM 2: ``LongTensor[B, seq_len] -> FloatTensor[B, seq_len,
vocab_size]`` -- logits at EVERY position, where position t predicts the token
at t+1. See ``src/models/__init__.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MusicLSTM", "OUTPUT_MODES", "DEFAULT_ADAPTIVE_CUTOFFS", "sanitize_cutoffs"]

# (h_0, c_0) for nn.LSTM: each is [num_layers, B, hidden_dim].
HiddenState = Tuple[torch.Tensor, torch.Tensor]
Output = Union[torch.Tensor, Tuple[torch.Tensor, HiddenState]]

OUTPUT_MODES = ("full", "adaptive")

# Adaptive-softmax cluster boundaries, in FREQUENCY RANK (0 = most common
# symbol). Chosen from the v4 training split (12.83M tokens, 37,020 symbols):
#
#     top-k ranks     100    500   1000   2000   5000   8000  10000  20000
#     token mass     71.1%  81.3%  85.5%  89.1%  93.2%  95.0%  95.8%  98.2%
#
# * head = ranks 0..1999: the 2,000 symbols that carry 89% of all targets get a
#   full-width (256-d) projection, so the common case is scored exactly as the
#   full softmax scores it, and ~89% of positions never touch a tail at all.
# * tail 1 = ranks 2000..9999 (6.9% of mass; symbols seen 46..319 times) at
#   256/4 = 64 dims.
# * tail 2 = ranks 10000..37019 (4.2% of mass; symbols seen fewer than 46
#   times, i.e. under ~4 times per transposition) at 256/16 = 16 dims -- about
#   as much as so few examples can pin down.
# Per scored position that is ~0.6M multiply-adds instead of the full
# projection's 256 x 37,020 = 9.5M.
DEFAULT_ADAPTIVE_CUTOFFS = (2000, 10000)


def sanitize_cutoffs(cutoffs: Optional[Sequence[int]], vocab_size: int) -> list:
    """Cutoffs valid for ``vocab_size``: sorted, unique, inside (0, V-1).

    nn.AdaptiveLogSoftmaxWithLoss rejects any cutoff >= V-1 and needs at least
    one, so a small vocabulary (tests, the synthetic demo) keeps the cutoffs
    that fit and otherwise falls back to a single split at V//2.
    """
    kept = sorted({int(c) for c in (cutoffs or ()) if 0 < int(c) < vocab_size - 1})
    if not kept:
        if vocab_size < 3:
            raise ValueError(f"adaptive softmax needs vocab_size >= 3, got {vocab_size}")
        kept = [max(1, vocab_size // 2)]
    return kept


def _fp32(t: torch.Tensor):
    """Autocast off: the adaptive softmax runs in fp32.

    Under fp16 autocast nn.AdaptiveLogSoftmaxWithLoss crashes outright (its
    output buffer is created Half while log_softmax is autocast to Float, and
    index_copy_ refuses the mix). fp32 is also where the softmax belongs
    numerically, and with a 2,000-symbol head it is no longer the expensive
    part: the LSTM, which does run in reduced precision, is.
    """
    return torch.autocast(device_type=t.device.type, enabled=False)


class MusicLSTM(nn.Module):
    """Embedding -> stacked LSTM -> MLP head -> per-position logits.

    OUTPUT LAYER (``output``). ``full`` is a Linear to all V symbols -- the
    original model, whose state_dict (and so every existing checkpoint) is
    unchanged. ``adaptive`` replaces that last Linear with
    ``nn.AdaptiveLogSoftmaxWithLoss``: a small head over the most frequent
    symbols plus low-rank tail clusters for the rare ones. Its outputs are
    exact log-probabilities (they sum to 1 over the whole vocabulary; see
    tests/test_efficiency.py), not an approximation of the full softmax -- it
    is a different, cheaper parameterisation of the next-token distribution.

    Adaptive softmax needs the most frequent symbols at the LOWEST ids, but
    this vocabulary is alphabetical. Rather than renumber the vocabulary (which
    would break every cache and checkpoint), the model keeps a permutation as
    two persistent buffers, ``id_to_rank`` / ``rank_to_id``, filled from
    training-split counts by ``set_token_frequencies`` and saved with the
    weights. Everything outside the model keeps speaking vocabulary ids.

    Measured on v4: see ``model.output`` in configs/default.yaml.

    Args:
        vocab_size: Number of symbols in the vocabulary.
        embed_dim: Symbol embedding width.
        hidden_dim: LSTM hidden width.
        num_layers: Number of stacked LSTM layers.
        dropout: Applied between LSTM layers and before the head.
        pad_id: Index whose embedding is pinned to zero and never updated.
        output: ``full`` | ``adaptive`` (see above).
        adaptive_cutoffs: rank boundaries of the adaptive clusters.
        adaptive_div_value: each successive tail cluster's projection is this
            many times narrower than the last.
    """

    # Generation may prime once and then feed one token at a time, carrying
    # (h, c) -- see generate._sample_many. A plain class attribute: it is not
    # part of the state_dict, so old checkpoints load unchanged.
    supports_incremental = True
    # forward(..., last_only=True) scores just the final position -- all the
    # sampler needs, and it avoids a [B, L, V] tensor while priming.
    supports_last_only = True

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        pad_id: int = 0,
        output: str = "full",
        adaptive_cutoffs: Optional[Sequence[int]] = DEFAULT_ADAPTIVE_CUTOFFS,
        adaptive_div_value: float = 4.0,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim must be >= 2, got {hidden_dim}")
        output = str(output).strip().lower()
        if output not in OUTPUT_MODES:
            raise ValueError(
                f"unknown model.output {output!r}; valid options are: {', '.join(OUTPUT_MODES)}"
            )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.output = output

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM applies dropout *between* layers only, so it is a no-op
            # (and warns) with a single layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        if output == "full":
            self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        else:
            self.adaptive = nn.AdaptiveLogSoftmaxWithLoss(
                hidden_dim // 2,
                vocab_size,
                cutoffs=sanitize_cutoffs(adaptive_cutoffs, vocab_size),
                div_value=float(adaptive_div_value),
                # The full model's fc2 has a bias, which is where the unigram
                # frequencies live; keep one on the head for a like-for-like
                # comparison.
                head_bias=True,
            )
            # Identity until set_token_frequencies() runs. Persistent, so the
            # trained permutation travels inside the checkpoint's state_dict.
            self.register_buffer("id_to_rank", torch.arange(vocab_size, dtype=torch.long))
            self.register_buffer("rank_to_id", torch.arange(vocab_size, dtype=torch.long))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier for feed-forward paths, orthogonal for recurrent weights.

        The hidden-to-hidden matrix is applied once per timestep, so its
        spectrum is raised to the power of the sequence length. An orthogonal
        initialisation has all singular values equal to 1, which keeps that
        repeated product from exploding or vanishing over a 100-step window --
        the usual reason a music LSTM's loss stalls or blows up in epoch 1.
        """
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[self.pad_id].fill_(0.0)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Each of the 4 gates gets its own orthogonal block; running
                # orthogonal_ on the stacked [4H, H] matrix would not give
                # orthogonal gates.
                for i in range(0, param.size(0), self.hidden_dim):
                    nn.init.orthogonal_(param[i : i + self.hidden_dim])
            elif "bias" in name:
                nn.init.zeros_(param)

        # Forget-gate bias = 1 so the cell starts out *remembering* by default
        # and has to learn to forget. PyTorch's gate order is [i, f, g, o], and
        # b_ih and b_hh are summed, so only one of them is nudged.
        for name, param in self.lstm.named_parameters():
            if name.startswith("bias_ih"):
                with torch.no_grad():
                    param[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)

        linears = [self.fc1]
        if self.output == "full":
            linears.append(self.fc2)
        else:
            linears.append(self.adaptive.head)
            for module in self.adaptive.tail.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        for linear in linears:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    # ------------------------------------------------------------------
    # adaptive-softmax frequency permutation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_token_frequencies(self, counts: Union[Sequence[int], np.ndarray, torch.Tensor]) -> bool:
        """Order the adaptive clusters by these per-id counts (most common first).

        ``counts[i]`` is how often vocabulary id i occurs as a training target.
        Ties break by id, so the permutation is deterministic. A no-op that
        returns False for the full-softmax model, which does not care.
        """
        if self.output != "adaptive":
            return False
        if isinstance(counts, torch.Tensor):
            counts = counts.detach().cpu().numpy()
        counts = np.asarray(counts, dtype=np.int64)
        if counts.shape != (self.vocab_size,):
            raise ValueError(f"expected {self.vocab_size} counts, got shape {counts.shape}")
        order = np.lexsort((np.arange(self.vocab_size), -counts))      # rank -> id
        rank_to_id = torch.as_tensor(order, dtype=torch.long)
        id_to_rank = torch.empty_like(rank_to_id)
        id_to_rank[rank_to_id] = torch.arange(self.vocab_size, dtype=torch.long)
        self.rank_to_id.copy_(rank_to_id)
        self.id_to_rank.copy_(id_to_rank)
        return True

    # ------------------------------------------------------------------
    # forward / loss
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> HiddenState:
        """Zero (h_0, c_0) on the model's device, for incremental generation."""
        device = device or next(self.parameters()).device
        shape = (self.num_layers, batch_size, self.hidden_dim)
        zeros = torch.zeros(shape, device=device, dtype=torch.float32)
        return zeros, zeros.clone()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[HiddenState] = None,
        return_hidden: bool = False,
        last_only: bool = False,
    ) -> Output:
        """Predict the next token at *every* position of ``x``.

        Position t sees only tokens 0..t (the LSTM is causal by construction --
        it has no way to look ahead) and predicts the token at t+1. Emitting a
        logit per position is standard teacher forcing: the same forward pass
        that used to yield ONE training target now yields ``seq_len`` of them.
        At the project's 100-token window, taking only the last position threw
        away ~99% of the supervision the network had already computed, so the
        model saw ~100x fewer gradient signals per epoch for identical compute.

        Args:
            x: LongTensor[B, seq_len] of symbol ids.
            hidden: Optional (h, c) carried over from a previous call. Passing
                it lets generation feed one token at a time instead of
                re-running the whole window, an O(L) -> O(1) win per step.
            return_hidden: Force the tuple return even on the first
                incremental step, where ``hidden`` is still None.
            last_only: score only the final position (returns [B, 1, V]).
                The sampler needs nothing else.

        With ``output: adaptive`` the returned "logits" are exact
        log-probabilities in vocabulary-id order; softmax() of them is the
        model's distribution, so the sampler and temperature work unchanged.

        Returns:
            FloatTensor[B, seq_len, vocab_size], or ``(logits, hidden)`` when
            ``hidden`` was supplied or ``return_hidden`` is set. Callers that
            want next-token-only -- the sampler -- slice ``logits[:, -1, :]``
            themselves.
        """
        if x.dim() != 2:
            raise ValueError(f"expected LongTensor[B, seq_len], got shape {tuple(x.shape)}")

        h, new_hidden = self._features(x, hidden, last_only=last_only)
        if self.output == "full":
            logits = self.fc2(h)                   # [B, L, V]
        else:
            logits = self._adaptive_log_prob(h)    # [B, L, V] log-probs

        if hidden is not None or return_hidden:
            return logits, new_hidden
        return logits

    def _features(
        self, x: torch.Tensor, hidden: Optional[HiddenState] = None, last_only: bool = False
    ) -> Tuple[torch.Tensor, HiddenState]:
        """Everything before the output layer: [B, L, H//2] (L=1 if last_only)."""
        emb = self.embedding(x)                    # [B, L, E]
        out, new_hidden = self.lstm(emb, hidden)   # [B, L, H]
        if last_only:
            out = out[:, -1:, :]
        # The head runs over the whole sequence, not just out[:, -1, :].
        # nn.Linear maps only the trailing dimension and broadcasts over every
        # leading one, so [B, L, H] -> [B, L, H//2] -> [B, L, V] needs no
        # reshaping and uses exactly the same weights as the old [B, H] path.
        h = self.dropout(out)                      # [B, L, H]
        h = self.relu(self.fc1(h))                 # [B, L, H//2]
        return h, new_hidden

    def _adaptive_log_prob(self, h: torch.Tensor) -> torch.Tensor:
        """[..., H//2] features -> [..., V] log-probs in vocabulary-id order."""
        lead = h.shape[:-1]
        with _fp32(h):
            by_rank = self.adaptive.log_prob(h.reshape(-1, h.size(-1)).float())   # [N, V], rank order
        by_id = by_rank.index_select(1, self.id_to_rank)             # [:, i] = [:, rank(i)]
        return by_id.reshape(*lead, self.vocab_size)

    def loss(self, x: torch.Tensor, y: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """Mean next-token NLL over the targets in ``y`` that are not ``ignore_index``.

        The trainer and evaluate.perplexity call this instead of materialising
        ``[B, L, V]`` logits when a model provides it. For ``output: full`` it
        is exactly the old ``CrossEntropyLoss(logits.reshape(-1, V), y)``; for
        ``adaptive`` each position evaluates only the head and its target's own
        cluster, which is where the speed and memory saving comes from. Like
        CrossEntropyLoss it is NaN when nothing is scored, so callers mask it
        the same way.
        """
        if self.output == "full":
            logits = self.forward(x)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=ignore_index
            )
        h, _ = self._features(x)
        flat = h.reshape(-1, h.size(-1))
        target = y.reshape(-1)
        valid = target != ignore_index
        # Ignored positions still need an in-range target for the gather; their
        # contribution is zeroed below. Dropping those rows instead would need
        # a nonzero() and therefore a host sync.
        rank = self.id_to_rank[target.clamp(0, self.vocab_size - 1)]
        with _fp32(flat):
            target_logp = self.adaptive(flat.float(), rank).output        # [N]
        return -(target_logp * valid).sum() / valid.sum()

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, output={self.output}"
        )
