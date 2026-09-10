# Milestones

Each milestone is independently runnable and has a narrow definition of done. Do not begin an optimization milestone until its predecessor's tests and baseline measurements are checked in.

## M1 — Python/PyTorch correctness reference (current)

Implement a compact decoder-only model with RMSNorm, separate Q/K/V projections, causal attention, SiLU-gated MLP, residuals, output projection, and append-only KV-cache semantics. Add CPU tests for masking and cached decoding equivalence. Establish the first PyTorch prefill and one-token decode baseline.

**Done when:** `pytest` passes and the benchmark produces a local JSON artifact. No C++/CUDA performance comparison is claimed.

## M2 — Minimal C++ tensor/runtime substrate

Add `Tensor`, `Storage`, `Device`, `DType`, and allocator abstractions with CPU tests. Keep the API deliberately small: contiguous tensors and only the metadata needed by the next operators. Introduce explicit error handling and lifetime tests.

**Done when:** C++ unit tests validate metadata, ownership, and allocation behavior; no GPU kernels are required.

## M3 — Python bindings and GEMM-backed execution

Expose narrow pybind11 bindings for tensor creation and a linear operation. Use a clear C++ baseline and cuBLAS (on CUDA) for GEMM rather than prematurely writing a matmul kernel. Compare linear correctness with PyTorch and measure binding/launch overhead separately.

**Done when:** Python tests cover bindings and end-to-end linear outputs match the reference within dtype tolerance.

## M4 — Naive custom CUDA operators

Implement readable, one-operation-at-a-time CUDA versions of RMSNorm, softmax, and causal attention. First establish correct grid/block mappings and memory layouts; add a basic tiled GEMM only as a learning baseline. Keep each kernel independently benchmarked against PyTorch and cuBLAS where applicable.

**Done when:** operator tests pass across a small shape matrix and benchmark results distinguish PyTorch, naive custom, and any library baseline.

## M5 — Kernel-level optimization and profiling

Use Nsight Systems to locate launch and synchronization cost, then Nsight Compute to validate a specific bottleneck hypothesis. Apply shared-memory tiling, coalesced/vectorized memory access, warp reductions, and reduced synchronization one technique at a time. Record changes and regressions.

**Done when:** every retained optimization has a before/after benchmark and profiler evidence explaining why it helps for its target shapes.

## M6 — End-to-end runtime path

Compose the runtime operators into one decoder layer and then the small transformer. Add weight loading from a deterministic test fixture, reference equivalence tests, and a runtime model API that has the same prefill/decode contract as M1.

**Done when:** runtime prefill and cached decode logits meet agreed tolerances versus the PyTorch oracle.

## M7 — KV cache, fusion, precision, and streams

Compare documented KV-cache layouts; optimize decode reads/appends. Investigate fused residual/norm and MLP paths, FP16/BF16 accumulation policies, and streams only when workload dependencies permit useful overlap. Test both prefill and decode separately.

**Done when:** retained choices are backed by shape- and precision-specific measurements, with accuracy limits documented.

## M8 — Reproducibility and portfolio evidence

Run the benchmark matrix on a named NVIDIA environment; archive raw JSON/CSV summaries and Nsight reports outside source control as appropriate. Publish only measured numbers, setup commands, kernel design notes, and a concise systems-focused project narrative.

**Done when:** a new machine can reproduce the documented method and every public number has a traceable artifact.
