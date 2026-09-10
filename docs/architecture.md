# Architecture

## Scope and non-goals

The runtime is intentionally limited to inference for small decoder-only transformers. Training, distributed execution, model conversion pipelines, and a broad tensor-library API are outside the scope. Every layer has one job: preserve the inference semantics of the reference while exposing a concrete performance question.

## Layered design

```text
Python caller / benchmark / tests
             |
             v
PyTorch reference model  <-------------------- numerical oracle (M1)
             |
             v
pybind11 Python API -------------------------- C++ boundary (M3)
             |
             v
C++ Tensor / Storage / Allocator / Runtime --- ownership and dispatch (M2)
             |
             +---------------------------+
             v                           v
    cuBLAS GEMM wrapper (M3)      Custom CUDA operators (M4+)
                                             |
     RMSNorm -> softmax -> attention -> fused decode / MLP
                                             |
                                             v
                           CUDA streams, cache layout, profiling (M5+)
```

The Python benchmark calls either the reference path or a runtime path using an identical model fixture, input sequence, precision, and measurement protocol. It records correctness against the reference before presenting latency or throughput.

## Inference data flow

```text
token IDs
  -> token embedding
  -> [RMSNorm -> QKV projections -> causal attention -> residual
      RMSNorm -> gate/up MLP -> SiLU(gate) * up -> down projection -> residual] x N
  -> final RMSNorm
  -> output projection / logits
```

During **prefill**, all prompt tokens are processed and keys and values for every layer are written to the KV cache. During **decode**, a single new token reads the cached prefix and appends one key/value row per layer. This means prefill is dominated by larger matrix operations and attention over an increasing sequence, while decode becomes especially sensitive to launch overhead, cache reads, and memory traffic.

## Component ownership

| Component | First implementation | Responsibility |
| --- | --- | --- |
| Model semantics | Python/PyTorch | Correct logits, masks, cache behavior, deterministic fixtures |
| Benchmark harness | Python | Synchronization, warmup, statistics, JSON environment metadata |
| Tensor and storage metadata | C++ | Shape/stride/dtype/device and explicit ownership |
| Allocation/runtime dispatch | C++ | CPU/GPU allocation, streams, error handling, operator launch interface |
| Dense linear layers | C++ + cuBLAS | Correct GEMM baseline before custom tiling |
| Performance-critical operators | CUDA | RMSNorm, softmax, attention, fused residual/MLP paths |
| Python integration | pybind11 | Narrow test-and-benchmark-facing API, no duplicate tensor framework |

## Runtime contracts

- Tensors own or explicitly view contiguous storage; shape, stride, dtype, and device are never inferred from a raw pointer.
- Kernels use a specified layout and dtype; layout conversions are explicit and benchmarked.
- Each custom operation has a reference test, an allowed error tolerance by dtype, and an input-shape matrix.
- The KV cache records layer, batch, head, sequence position, and head-dimension ordering in one documented layout. Layout changes require cache-equivalence and decode benchmarks.
- Timing code synchronizes the measured device, performs warmups, reports distribution statistics, and does not include one-time initialization unless deliberately measuring it.

## Why PyTorch remains separate

The reference intentionally uses clear PyTorch operations rather than sharing runtime internals. It is slower to duplicate a little model logic, but it makes numerical comparisons independent: a bug in a custom kernel cannot be hidden by a shared implementation.
