"""Reproducible M1 PyTorch baseline for prefill and cached decode-step latency."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from .config import TransformerConfig
from .reference import DecoderOnlyTransformer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timing_stats(run: Callable[[], None], warmup: int, repeats: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        run()
    synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        run()
        synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1_000)

    sorted_samples = sorted(samples_ms)
    p95_index = min(len(sorted_samples) - 1, round(0.95 * (len(sorted_samples) - 1)))
    return {
        "median_ms": statistics.median(samples_ms),
        "p50_ms": sorted_samples[len(sorted_samples) // 2],
        "p95_ms": sorted_samples[p95_index],
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this PyTorch installation")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable in this PyTorch installation")
    if args.prompt_length <= 0 or args.batch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("batch size, prompt length, and repeats must be positive; warmup must be non-negative")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_sequence_length=args.prompt_length + 1,
    )
    model = DecoderOnlyTransformer(config).to(device).eval()
    prompt = torch.randint(0, config.vocab_size, (args.batch_size, args.prompt_length), device=device)
    next_token = torch.randint(0, config.vocab_size, (args.batch_size, 1), device=device)

    with torch.inference_mode():
        _, cache = model(prompt, use_cache=True)
        assert cache is not None

        def prefill() -> None:
            model(prompt)

        def decode_step() -> None:
            model(next_token, past_key_values=cache, use_cache=True)

        prefill_stats = timing_stats(prefill, args.warmup, args.repeats, device)
        decode_stats = timing_stats(decode_step, args.warmup, args.repeats, device)

    prefill_tokens = args.batch_size * args.prompt_length
    decode_tokens = args.batch_size
    result = {
        "implementation": "pytorch_reference_m1",
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
        },
        "configuration": {
            "batch_size": args.batch_size,
            "prompt_length": args.prompt_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "prefill": {
            **prefill_stats,
            "tokens_per_second": prefill_tokens / (prefill_stats["median_ms"] / 1_000),
        },
        "cached_decode_step": {
            **decode_stats,
            "tokens_per_second": decode_tokens / (decode_stats["median_ms"] / 1_000),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
