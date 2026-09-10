"""Configuration for the small, explicit reference transformer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 128
    max_sequence_length: int = 128
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads
