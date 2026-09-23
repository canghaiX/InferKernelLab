from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .attention import dense_decode_attention_batch, paged_decode_attention
from .cache import PagedKVCache
from .triton_ops import paged_decode_attention_triton, triton_available


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    p50_ms: float
    p95_ms: float


def _percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _measure(fn, warmup: int, iterations: int, device: torch.device) -> Timing:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - start) * 1000)
    return Timing(
        mean_ms=sum(samples) / len(samples),
        p50_ms=_percentile(samples, 0.50),
        p95_ms=_percentile(samples, 0.95),
    )


def _make_case(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    cache = PagedKVCache(
        args.num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    tables = [cache.allocate_request(i, args.context_len) for i in range(args.batch_size)]
    keys = []
    values = []
    for i, table in enumerate(tables):
        key = torch.randn(args.context_len, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        value = torch.randn_like(key)
        keys.append(key)
        values.append(value)
        cache.write(0, table, 0, key, value)
    query = torch.randn(args.batch_size, args.num_heads, args.head_dim, device=device, dtype=dtype)
    lengths = [args.context_len] * args.batch_size
    return device, cache, tables, keys, values, query, lengths


def run(args: argparse.Namespace) -> dict:
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    device, cache, tables, keys, values, query, lengths = _make_case(args)
    reference = lambda: dense_decode_attention_batch(query, keys, values)
    paged_reference = lambda: paged_decode_attention(query, cache, tables, lengths)
    results = {}
    reference_output = reference()
    paged_output = paged_reference()
    results["dense"] = {
        "timing": _measure(reference, args.warmup, args.iterations, device).__dict__,
        "max_abs_error": 0.0,
    }
    results["paged_reference"] = {
        "timing": _measure(paged_reference, args.warmup, args.iterations, device).__dict__,
        "max_abs_error": (paged_output.float() - reference_output.float()).abs().max().item(),
    }
    if args.backend in ("auto", "triton") and triton_available() and device.type == "cuda":
        triton_fn = lambda: paged_decode_attention_triton(query, cache, tables, lengths)
        triton_output = triton_fn()
        results["triton_paged"] = {
            "timing": _measure(triton_fn, args.warmup, args.iterations, device).__dict__,
            "max_abs_error": (triton_output.float() - reference_output.float()).abs().max().item(),
        }
    kv_read_bytes = (
        2
        * args.batch_size
        * args.context_len
        * args.num_kv_heads
        * args.head_dim
        * torch.tensor([], dtype=getattr(torch, args.dtype)).element_size()
    )
    dense_mean = results["dense"]["timing"]["mean_ms"]
    for name, result in results.items():
        mean_ms = result["timing"]["mean_ms"]
        result["speedup_vs_dense"] = dense_mean / mean_ms
        result["estimated_kv_read_gbps"] = kv_read_bytes / (mean_ms / 1000) / 1e9
    return {
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "config": {
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "context_len": args.context_len,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "kv_cache_bytes": cache.numel_bytes,
        "estimated_kv_read_bytes": kv_read_bytes,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense, paged-reference, and Triton decode attention")
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
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run(args)
    text = json.dumps(record, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
