"""A readable PyTorch oracle for decoder-only transformer inference.

The implementation is intentionally explicit. It provides the semantics that the
C++/CUDA runtime must later match; it is not meant to compete with PyTorch's
optimized transformer kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .config import TransformerConfig


@dataclass(frozen=True)
class KVCache:
    """Keys and values for one layer, shaped [batch, heads, sequence, head_dim]."""

    keys: Tensor
    values: Tensor

    def __post_init__(self) -> None:
        if self.keys.shape != self.values.shape:
            raise ValueError("KV cache keys and values must have the same shape")
        if self.keys.ndim != 4:
            raise ValueError("KV cache tensors must have shape [B, H, S, D]")

    @property
    def sequence_length(self) -> int:
        return self.keys.shape[2]


class RMSNorm(nn.Module):
    """RMS normalization with a learnable per-channel scale."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_sequence_length = config.max_sequence_length
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def _causal_mask(self, query_length: int, key_length: int, past_length: int, device: torch.device) -> Tensor:
        query_positions = past_length + torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)
        return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    def forward(self, x: Tensor, past_key_value: KVCache | None = None, use_cache: bool = False) -> tuple[Tensor, KVCache | None]:
        batch, query_length, _ = x.shape
        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))

        past_length = 0
        if past_key_value is not None:
            if past_key_value.keys.shape[:2] != (batch, self.num_heads):
                raise ValueError("KV cache batch size and head count must match the input")
            if past_key_value.keys.shape[-1] != self.head_dim:
                raise ValueError("KV cache head dimension must match the model")
            past_length = past_key_value.sequence_length
            keys = torch.cat((past_key_value.keys, keys), dim=2)
            values = torch.cat((past_key_value.values, values), dim=2)

        key_length = keys.shape[2]
        if key_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {key_length} exceeds max_sequence_length {self.max_sequence_length}"
            )

        scale = self.head_dim ** -0.5
        scores = (queries @ keys.transpose(-2, -1)) * scale
        allowed = self._causal_mask(query_length, key_length, past_length, x.device)
        scores = scores.masked_fill(~allowed.view(1, 1, query_length, key_length), float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attended = probabilities @ values
        attended = attended.transpose(1, 2).contiguous().view(batch, query_length, -1)
        output = self.out_proj(attended)
        present = KVCache(keys=keys, values=values) if use_cache else None
        return output, present


class MLP(nn.Module):
    """A bias-free SwiGLU-style MLP using SiLU activation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, past_key_value: KVCache | None = None, use_cache: bool = False) -> tuple[Tensor, KVCache | None]:
        attention_output, present = self.attention(
            self.attention_norm(x), past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + attention_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class DecoderOnlyTransformer(nn.Module):
    """Small decoder-only model with an explicit per-layer KV cache API."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Sequence[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[KVCache, ...] | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        if past_key_values is not None and len(past_key_values) != self.config.num_layers:
            raise ValueError("past_key_values must contain one entry per layer")

        x = self.token_embedding(input_ids)
        next_cache: list[KVCache] = []
        for index, layer in enumerate(self.layers):
            past = past_key_values[index] if past_key_values is not None else None
            x, present = layer(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                assert present is not None
                next_cache.append(present)

        logits = self.output_projection(self.final_norm(x))
        return logits, tuple(next_cache) if use_cache else None

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Greedy decoding to exercise the same cache contract used by benchmarks."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.shape[1] + max_new_tokens > self.config.max_sequence_length:
            raise ValueError("generation would exceed max_sequence_length")

        tokens = input_ids
        logits, cache = self(tokens, use_cache=True)
        for _ in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            logits, cache = self(next_token, past_key_values=cache, use_cache=True)
        return tokens
