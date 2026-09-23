# InferKernelLab

InferKernelLab is a small, auditable LLM inference kernel laboratory. It is
intentionally narrower than a serving framework: the project focuses on the
three pieces that are easiest to benchmark and explain in an interview:

```text
Paged KV cache -> Decode attention -> Scheduling / workload replay
```

The project has a pure PyTorch reference path that runs on CPU or CUDA, plus
an optional Triton backend for CUDA experiments. Every optimized path is
checked against the reference implementation before benchmarking.

## Current scope

- Physical KV-cache blocks with logical block tables.
- Allocation, release, slot mapping, and cache read/write tests.
- Reference Paged Decode Attention with MHA, GQA, and MQA support.
- A small token-budget scheduler for prefill/decode workload replay.
- Optional Triton paged decode attention kernel.
- Triton KV-cache append kernel.
- Dense, paged-reference, and Triton benchmark comparison with P50/P95 latency.
- JSON benchmark output suitable for later plotting and regression checks.

This is not a replacement for vLLM. The goal is to make the memory layout,
kernel behavior, and performance trade-offs small enough to inspect end to end.

## Quick start

```bash
cd /data/InferKernelLab
python3 -m pip install -e '.[test]'
pytest -q
python3 -m inferkernellab.benchmark --device cpu --batch-size 4 --context-len 128
```

For a CUDA/Triton run:

```bash
python3 -m inferkernellab.benchmark \
  --device cuda \
  --backend auto \
  --batch-size 8 \
  --context-len 2048 \
  --num-kv-heads 8 \
  --num-heads 32
```

The command prints a JSON record containing correctness error, latency, and
throughput. It does not claim a performance win until the same shapes and
environment are measured against the dense reference.

For a reproducible CUDA environment:

```bash
./scripts/docker_build.sh
./scripts/docker_test.sh

# On the current DGX host, use the already cached CUDA/PyTorch image:
BASE_IMAGE=nano-vllm:latest ./scripts/docker_build.sh
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 -m inferkernellab.benchmark --device cuda --backend auto \
  --batch-size 8 --context-len 2048 --num-heads 32 --num-kv-heads 8
```

Run a JSONL parameter sweep:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 scripts/run_benchmark_sweep.py
```

## Architecture

```text
src/inferkernellab/
  cache.py       Physical block allocator and paged KV storage
  attention.py   Device-independent reference attention
  triton_ops.py  Optional Triton KV store and decode attention
  scheduler.py   Token-budget prefill/decode scheduler
  benchmark.py   Reproducible microbenchmark CLI
```

The cache layout is:

```text
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

Given a logical token position `p` and a request block table:

```text
logical_block = p // block_size
physical_block = block_table[logical_block]
slot = physical_block * block_size + p % block_size
```

The reference attention deliberately uses explicit indexing. That makes it
easy to compare a kernel against a correct implementation and to inspect the
cost of non-contiguous KV access.

## Development roadmap

1. Add benchmark plots and Nsight Compute reports for the Triton kernel.
2. Add prefix-cache reference semantics and LRU eviction.
3. Add a model adapter for a small Hugging Face causal LM.
4. Add a fused prefill attention path and compare it with decode behavior.
5. Add continuous-batching replay traces and TTFT/TPOT measurements.

## Resume-worthy evidence

Do not report a speedup without recording:

- GPU, driver, CUDA, PyTorch, and Triton versions.
- dtype, batch size, context length, head count, head dimension, and block size.
- warmup count, measured iterations, and synchronization method.
- reference correctness error and latency distribution.
- kernel metrics from Nsight Compute when making a kernel-level claim.
