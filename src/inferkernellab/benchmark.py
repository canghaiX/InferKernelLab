from __future__ import annotations

import argparse
import json
import time

import torch

from .attention import paged_decode_attention
from .cache import PagedKVCache
from .triton_ops import paged_decode_attention_triton, triton_available


def _measure(fn, warmup: int, iterations: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
    return elapsed_ms, 1000 / elapsed_ms


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cache = PagedKVCache(
        args.num_blocks, args.block_size, args.num_kv_heads, args.head_dim,
        dtype=dtype, device=device,
    )
    tables = [cache.allocate_request(i, args.context_len) for i in range(args.batch_size)]
    keys = torch.randn(args.context_len, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    for i, table in enumerate(tables):
        cache.write(0, table, 0, keys, values)
    query = torch.randn(args.batch_size, args.num_heads, args.head_dim, device=device, dtype=dtype)
    context_lens = [args.context_len] * args.batch_size
    reference = lambda: paged_decode_attention(query, cache, tables, context_lens)
    backend = args.backend
    if backend == "auto":
        backend = "triton" if triton_available() and device.type == "cuda" else "reference"
    fn = reference if backend == "reference" else lambda: paged_decode_attention_triton(query, cache, tables, context_lens)
    output = fn()
    reference_output = reference()
    max_error = (output.float() - reference_output.float()).abs().max().item()
    latency_ms, calls_per_second = _measure(fn, args.warmup, args.iterations, device)
    return {
        "backend": backend,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "latency_ms": latency_ms,
        "calls_per_second": calls_per_second,
        "max_abs_error": max_error,
        "kv_cache_bytes": cache.numel_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reference or Triton paged decode attention")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("auto", "reference", "triton"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

