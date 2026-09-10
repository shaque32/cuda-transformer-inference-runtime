# High-Performance CUDA Runtime for Transformer Inference

An incremental systems project for understanding decoder-only transformer inference from a correct PyTorch reference through a lightweight C++/CUDA runtime. The project deliberately favors measurable, explainable improvements over a large framework.

**Current status: Milestone 1 — PyTorch reference implementation.** There are no custom C++ or CUDA kernels yet, and this repository makes no performance claims. The reference model, cache semantics, tests, and baseline benchmark are the contract that later runtime layers must preserve.

## What this project will cover

- Decoder-only transformer inference: prefill and autoregressive decode
- RMSNorm, Q/K/V projections, causal attention, residual paths, SiLU MLP, token embeddings, final RMSNorm, and output projection
- KV-cache correctness and layout experiments
- Progressive comparison of PyTorch, naive CUDA, and optimized CUDA implementations
- Nsight Systems and Nsight Compute analysis tied to specific optimization hypotheses

The full design and dependency graph are in [docs/architecture.md](docs/architecture.md). The staged implementation plan is in [docs/milestones.md](docs/milestones.md), and the benchmarking protocol is in [docs/benchmarking.md](docs/benchmarking.md).

## Milestone 1 quick start

This milestone needs only Python and PyTorch; it is designed to work on macOS using CPU or MPS where supported. Use an NVIDIA CUDA machine later for the CUDA milestones.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python -m hpti.benchmark --device cpu
```

On a CUDA-capable environment, pass `--device cuda` and use a CUDA-enabled PyTorch installation appropriate for that machine. Benchmark output is intentionally measured locally and can optionally be persisted as JSON:

```bash
python -m hpti.benchmark --device cuda --output benchmarks/results/m1-pytorch.json
```

## Repository layout

```text
.
├── python/hpti/              # Milestone 1 reference model and benchmark harness
├── tests/                    # CPU correctness tests; later CUDA tests are added beside them
├── benchmarks/results/       # Local, git-ignored benchmark artifacts
├── docs/                     # Architecture, milestones, and profiling/benchmark protocol
├── include/                  # Public C++ headers (introduced in Milestone 2)
├── src/                      # C++ runtime, ops, and pybind11 bindings (Milestones 2–3)
└── cuda/                     # CUDA kernels by operator (introduced in Milestone 4)
```

## Guardrails

- The PyTorch reference remains available and is the numerical oracle.
- Tests must pass before a new optimized path is benchmarked.
- Each optimization records the hypothesis, measurement method, shape, precision, hardware/software environment, and result—including regressions.
- README and resume performance numbers are added only after benchmark artifacts exist in `benchmarks/results/`.

## Planned resume scope

The intended outcome is a lightweight C++/CUDA decoder-only transformer inference runtime with custom kernels, Python bindings, profiling evidence, and reproducible comparisons. That is a target state, not a claim about this Milestone 1 repository.
