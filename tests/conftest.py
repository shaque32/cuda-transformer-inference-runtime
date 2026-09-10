import pytest
import torch

from hpti import DecoderOnlyTransformer, TransformerConfig


@pytest.fixture
def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=31,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        intermediate_size=32,
        max_sequence_length=12,
    )


@pytest.fixture
def tiny_model(tiny_config: TransformerConfig) -> DecoderOnlyTransformer:
    torch.manual_seed(7)
    return DecoderOnlyTransformer(tiny_config).eval()
