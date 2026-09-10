# Benchmarking and correctness protocol

## First benchmark: M1 PyTorch baseline

M1 establishes two end-to-end measurements with the same small, deterministic model fixture:

| Case | Measured work | Primary metric | Why it exists |
| --- | --- | --- | --- |
| Prefill | Forward pass over a prompt of `S` tokens | ms/request and prompt tokens/s | Baseline for sequence-length scaling and dense compute |
| Cached decode step | One new token using a prebuilt cache of length `S` | ms/token and decode tokens/s | Baseline for launch and KV-read-sensitive work |

The provided harness warm ups first, synchronizes CUDA around every timed region, and reports median, p50, p95, min, max, and throughput. It prints and optionally saves the model dimensions, dtype, device, PyTorch version, CUDA version, and sample count. On CPU, synchronization is a no-op. MPS may work for functional experimentation but is not a CUDA performance substitute.

Run it with:

```bash
python -m hpti.benchmark --device cpu
python -m hpti.benchmark --device cuda --output benchmarks/results/m1-pytorch.json
```

## Correctness gates

Before recording a runtime result, require the following:

1. Operator output agrees with the PyTorch oracle for a representative shape/dtype matrix.
2. Causal masking prevents future-token changes from affecting earlier logits.
3. Cached token-by-token logits agree with the equivalent full-sequence reference logits.
4. Report maximum absolute error and maximum relative error alongside timing.

For M1, tests run in float32 on CPU with strict but realistic tolerance. Later precision modes use explicit, documented tolerances rather than reusing float32 thresholds.

## Future comparison matrix

For every meaningful operator and the end-to-end model, record all available rows using the same input fixture:

| Implementation | Latency | Throughput | Max abs/rel error | Shape / dtype / device |
| --- | --- | --- | --- | --- |
| PyTorch reference | measured | measured | oracle | recorded |
| Naive custom CUDA | measured | measured | vs. reference | recorded |
| Optimized custom CUDA | measured | measured | vs. reference | recorded |

Never infer a speedup from a profiler screenshot or publish a number without its raw benchmark output and environment metadata.

## Profiling sequence

First use Nsight Systems to check end-to-end timeline behavior: launches, synchronizations, transfers, streams, and CPU gaps. Then choose a hot kernel and use Nsight Compute to assess occupancy, achieved memory bandwidth, arithmetic intensity, cache metrics, warp divergence, and synchronization behavior. A profiler metric is evidence for or against a hypothesis, not an optimization goal by itself.
