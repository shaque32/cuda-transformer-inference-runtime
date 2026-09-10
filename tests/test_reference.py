import torch

from hpti import RMSNorm


def test_rms_norm_matches_direct_formula() -> None:
    norm = RMSNorm(hidden_size=3, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 0.5]))
    x = torch.tensor([[[1.0, 2.0, 3.0]]])

    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * norm.weight
    torch.testing.assert_close(norm(x), expected)


def test_forward_returns_logits_with_expected_shape(tiny_model, tiny_config) -> None:
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    logits, cache = tiny_model(input_ids, use_cache=True)

    assert logits.shape == (2, 3, tiny_config.vocab_size)
    assert cache is not None
    assert len(cache) == tiny_config.num_layers
    assert all(layer_cache.keys.shape == (2, tiny_config.num_heads, 3, tiny_config.head_dim) for layer_cache in cache)


def test_causal_mask_prevents_future_tokens_from_affecting_earlier_logits(tiny_model) -> None:
    first = torch.tensor([[2, 3, 4, 5]])
    second = torch.tensor([[2, 3, 19, 20]])
    first_logits, _ = tiny_model(first)
    second_logits, _ = tiny_model(second)

    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2], rtol=1e-5, atol=1e-6)


def test_cached_decoding_matches_full_sequence_logits(tiny_model) -> None:
    token_ids = torch.tensor([[3, 1, 4, 1, 5]])
    full_logits, _ = tiny_model(token_ids)

    cache = None
    incremental_logits = []
    for position in range(token_ids.shape[1]):
        logits, cache = tiny_model(token_ids[:, position : position + 1], past_key_values=cache, use_cache=True)
        incremental_logits.append(logits)
    cached_logits = torch.cat(incremental_logits, dim=1)

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-6)


def test_greedy_generation_returns_prompt_plus_requested_tokens(tiny_model) -> None:
    prompt = torch.tensor([[1, 2, 3]])
    generated = tiny_model.generate(prompt, max_new_tokens=3)

    assert generated.shape == (1, 6)
    torch.testing.assert_close(generated[:, :3], prompt)
