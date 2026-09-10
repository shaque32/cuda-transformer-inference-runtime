import pytest

from hpti import TransformerConfig


def test_head_dim_is_derived_from_valid_configuration() -> None:
    assert TransformerConfig(hidden_size=48, num_heads=6).head_dim == 8


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"hidden_size": 15, "num_heads": 4}, "divisible"),
        ({"max_sequence_length": 0}, "max_sequence_length"),
        ({"rms_norm_eps": 0}, "rms_norm_eps"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TransformerConfig(**kwargs)
