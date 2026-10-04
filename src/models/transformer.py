"""Causal Transformer encoder over music symbols -- the comparison arm.

Same SEAM 2 signature as MusicLSTM: ``LongTensor[B, seq_len]`` in,
``FloatTensor[B, seq_len, vocab_size]`` out -- logits at every position, where
position t predicts the token at t+1. Swapping architectures is one config line
precisely because nothing outside this file knows which one it got.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["MusicTransformer", "SinusoidalPositionalEncoding"]

Output = Union[torch.Tensor, Tuple[torch.Tensor, None]]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos position signal added to the token embeddings.

    Fixed rather than learned so a model trained at ``seq_len`` can still be
    fed a shorter prompt during generation without an out-of-range lookup.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        # An odd d_model leaves the cos block one column short of the sin block.
        pe[:, 1::2] = torch.cos(position * div)[:, : d_model // 2]
        # Not a parameter, but must follow .to(device) -- hence a buffer.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] -> [B, L, D]."""
        length = x.size(1)
        if length > self.pe.size(1):
            raise ValueError(
                f"sequence length {length} exceeds positional encoding table "
                f"({self.pe.size(1)}); raise max_len"
            )
        return self.dropout(x + self.pe[:, :length])


class MusicTransformer(nn.Module):
    """Embedding + positional encoding -> causal TransformerEncoder -> head.

    Args:
        vocab_size: Number of symbols in the vocabulary.
        embed_dim: Model width (d_model). Must be divisible by ``num_heads``.
        hidden_dim: Feed-forward width inside each block.
        num_layers: Number of encoder blocks.
        num_heads: Attention heads per block.
        dropout: Attention / feed-forward / embedding dropout.
        max_len: Longest sequence the positional table supports.
        pad_id: Index whose embedding is pinned to zero.
    """

    # No KV cache: generation re-scores the whole window each step.
    supports_incremental = False

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 4096,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"model.embed_dim ({embed_dim}) must be divisible by "
                f"model.num_heads ({num_heads})"
            )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_id = pad_id
        self._scale = math.sqrt(embed_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            # Pre-norm: on a small corpus this trains without a warmup schedule,
            # which post-norm transformers generally need.
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
            # The nested-tensor fast path is incompatible with pre-LN
            # (norm_first=True). Disable it explicitly rather than let
            # torch warn about it on every single run.
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Cached upper-triangular -inf mask, rebuilt only when a longer
        # sequence arrives. persistent=False keeps it out of the checkpoint.
        self.register_buffer("_causal_mask", self._build_causal_mask(max(1, max_len)), persistent=False)

        self._init_weights()

    @staticmethod
    def _build_causal_mask(size: int) -> torch.Tensor:
        """[size, size] float mask; -inf above the diagonal, 0 on and below.

        Additive mask semantics: entry (i, j) is added to the attention score
        of query i attending to key j, so -inf at j > i makes position i's
        view of the future exactly zero after the softmax. Without this the
        model reads the answer off its own input and val loss collapses to
        near zero while generation produces noise.
        """
        return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

    def _causal_mask_for(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._causal_mask.size(0) < length:
            self._causal_mask = self._build_causal_mask(length).to(device=device, dtype=dtype)
        return self._causal_mask[:length, :length].to(device=device, dtype=dtype)

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[self.pad_id].fill_(0.0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[object] = None,
        return_hidden: bool = False,
    ) -> Output:
        """Predict the next token at *every* position of ``x``.

        The encoder already computes a representation for all L positions; the
        old code threw L-1 of them away. The causal mask guarantees position t
        has attended only to tokens 0..t, so those representations are exactly
        what teacher forcing wants: one training target per position instead of
        one per window, i.e. ~100x more supervision from identical compute at
        the project's 100-token window.

        The mask is therefore load-bearing in a way it was not before. When
        only the last position was read, an off-by-one in the mask was mostly
        harmless; now, a mask that let position t see token t+1 would hand the
        model its own label. Loss would look excellent and generation would be
        garbage. ``_build_causal_mask`` uses ``diagonal=1``, which keeps the
        diagonal itself unmasked (t may see t) and -infs everything strictly
        above it -- see the causality test in the verification script.

        ``hidden`` is accepted and ignored: attention is stateless, so there is
        no cache to carry. It exists only so the sampler and trainer can call
        every architecture identically (SEAM 2).

        Returns:
            FloatTensor[B, seq_len, vocab_size], or ``(logits, None)`` when
            ``hidden`` was supplied or ``return_hidden`` is set. Callers that
            want next-token-only slice ``logits[:, -1, :]`` themselves.
        """
        if x.dim() != 2:
            raise ValueError(f"expected LongTensor[B, seq_len], got shape {tuple(x.shape)}")

        emb = self.embedding(x) * self._scale       # [B, L, D]
        h = self.pos_encoder(emb)

        mask = self._causal_mask_for(x.size(1), h.device, h.dtype)
        # Deliberately no src_key_padding_mask: combined with a causal mask, a
        # row whose entire prefix is PAD would have every key masked and the
        # softmax would return NaN. PAD is already a zero embedding and is
        # ignored on the loss side (ignore_index), so letting it through costs
        # nothing and keeps the forward pass numerically safe.
        out = self.encoder(h, mask=mask)            # [B, L, D]

        # Head over the whole sequence: nn.Linear maps the trailing dimension
        # and broadcasts over the leading ones, so [B, L, D] -> [B, L, V] with
        # the same weights the old out[:, -1, :] path used.
        logits = self.fc_out(self.dropout(out))     # [B, L, V]

        if hidden is not None or return_hidden:
            return logits, None
        return logits

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, embed_dim={self.embed_dim}"
