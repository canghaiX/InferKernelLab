# Benchmark Report

This report records one reproducible smoke configuration. It is not a claim
that the kernel wins for every shape.

## Environment

- GPU: NVIDIA A100-SXM4-40GB
- Container base: local `nano-vllm:latest` CUDA image
- Python: 3.12.13
- PyTorch: 2.11.0+cu130
- Triton: 3.6.0
- Dtype: FP16

## Configuration

```text
batch_size=2
context_len=64
num_heads=4
num_kv_heads=2
head_dim=32
block_size=16
warmup=20
iterations=50
```

| Implementation | Mean (ms) | P50 (ms) | P95 (ms) | Max error | Speedup vs dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense contiguous reference | 0.504 | 0.477 | 0.836 | 0 | 1.00x |
| Paged PyTorch reference | 0.921 | 0.904 | 0.932 | 0 | 0.55x |
| Triton paged attention | 0.159 | 0.158 | 0.164 | 6.7e-4 | 3.17x |

The Triton result is faster than the Python reference because the latter
contains an explicit per-request loop and tensor indexing. The dense baseline
is also a mathematical reference rather than a production fused attention
implementation. A stronger comparison should add FlashAttention/FlashInfer
and collect Nsight Compute counters under an exclusive GPU allocation.

Reproduce the measurement with:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 -m inferkernellab.benchmark --device cuda --backend auto \
  --dtype float16 --batch-size 2 --context-len 64 \
  --num-heads 4 --num-kv-heads 2 --head-dim 32 \
  --block-size 16 --num-blocks 128 --warmup 20 --iterations 50
```

