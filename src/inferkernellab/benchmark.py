from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .attention import (
    dense_decode_attention_batch,
    dense_decode_attention_sdpa_tensor_batch,
    paged_decode_attention,
    paged_decode_attention_sdpa,
)
from .cache import PagedKVCache
from .triton_ops import (
    paged_decode_attention_triton,
    paged_decode_autotune_config,
    triton_available,
)


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
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(iterations):
            if device.type == "cuda":
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            else:
                start_time = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - start_time) * 1000)
    return Timing(
        mean_ms=sum(samples) / len(samples),
        p50_ms=_percentile(samples, 0.50),
        p95_ms=_percentile(samples, 0.95),
    )


def _make_case(args: argparse.Namespace):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    required_blocks = (args.context_len + args.block_size - 1) // args.block_size
    tables = [[] for _ in range(args.batch_size)]
    cache = PagedKVCache(
        args.num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    for logical_block in range(required_blocks):
        for request_id in range(args.batch_size):
            tables[request_id].extend(cache.allocator.allocate(1, owner=request_id))
        if logical_block + 1 < required_blocks:
            cache.allocator.allocate(1, owner=-1)
    keys = []
    values = []
    for table in tables:
        key = torch.randn(args.context_len, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        value = torch.randn_like(key)
        keys.append(key)
        values.append(value)
        cache.write(0, table, 0, key, value)
    query = torch.randn(args.batch_size, args.num_heads, args.head_dim, device=device, dtype=dtype)
    lengths = [args.context_len] * args.batch_size
    tables_tensor = torch.tensor(tables, dtype=torch.int64, device=device)
    if device.type == "cuda":
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    else:
        lengths_tensor = None
    dense_keys = torch.stack(keys)
    dense_values = torch.stack(values)
    return (
        device,
        cache,
        tables,
        keys,
        values,
        query,
        lengths,
        tables_tensor,
        lengths_tensor,
        dense_keys,
        dense_values,
    )


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = difference.max().item()
    max_rel = (difference / expected.float().abs().clamp_min(1e-8)).max().item()
    return max_abs, max_rel


def _correctness_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    if dtype == torch.float16:
        return 3e-3, 3e-3
    return 1e-4, 1e-4


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()[0].strip()
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError):
        return None


def _environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": _triton_version(),
        "driver": _driver_version() if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
    }


def run(args: argparse.Namespace) -> dict:
    if args.num_heads <= 0 or args.num_kv_heads <= 0 or args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be positive and divisible by num_kv_heads")
    if min(args.batch_size, args.context_len, args.head_dim, args.block_size, args.num_blocks) <= 0:
        raise ValueError("batch, context, and cache dimensions must be positive")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    required_blocks = (args.context_len + args.block_size - 1) // args.block_size
    minimum_blocks = args.batch_size * required_blocks + required_blocks - 1
    if args.num_blocks < minimum_blocks:
        raise ValueError("num_blocks is insufficient for interleaved paged KV allocation")
    if args.backend == "triton" and (torch.device(args.device).type != "cuda" or not triton_available()):
        raise RuntimeError("backend=triton requires a CUDA device and an installed Triton backend")

    (
        device,
        cache,
        tables,
        keys,
        values,
        query,
        lengths,
        tables_tensor,
        lengths_tensor,
        dense_keys,
        dense_values,
    ) = _make_case(args)

    if getattr(args, "profile_kernel", False):
        if device.type != "cuda" or not triton_available():
            raise RuntimeError("--profile-kernel requires CUDA and Triton")
        paged_decode_attention_triton(query, cache, tables_tensor, lengths_tensor, validate_inputs=True)
        torch.cuda.synchronize(device)
        torch.cuda.profiler.start()
        try:
            paged_decode_attention_triton(
                query, cache, tables_tensor, lengths_tensor, validate_inputs=False
            )
            torch.cuda.synchronize(device)
        finally:
            torch.cuda.profiler.stop()
        return {
            "schema_version": 2,
            "environment": _environment(device),
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
            },
            "profile_mode": "single autotuned Triton paged-decode launch",
            "kernel_config": paged_decode_autotune_config(),
        }

    reference = lambda: dense_decode_attention_batch(query, keys, values)
    paged_reference = lambda: paged_decode_attention(query, cache, tables, lengths)
    dense_sdpa = lambda: dense_decode_attention_sdpa_tensor_batch(query, dense_keys, dense_values)
    paged_sdpa = lambda: paged_decode_attention_sdpa(query, cache, tables_tensor, lengths)
    results = {}
    reference_output = reference()
    atol, rtol = _correctness_tolerance(query.dtype)

    for name, fn in (
        ("dense", reference),
        ("paged_reference", paged_reference),
        ("dense_sdpa", dense_sdpa),
        ("paged_sdpa", paged_sdpa),
    ):
        output = fn()
        max_abs, max_rel = _max_errors(output, reference_output)
        matches_reference = torch.allclose(output, reference_output, atol=atol, rtol=rtol)
        if not matches_reference:
            raise RuntimeError(f"{name} output differs from the PyTorch reference")
        results[name] = {
            "timing": _measure(fn, args.warmup, args.iterations, device).__dict__,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "matches_reference": matches_reference,
        }

    run_triton = args.backend in ("auto", "triton") and device.type == "cuda" and triton_available()
    if run_triton:
        checked_output = paged_decode_attention_triton(
            query, cache, tables_tensor, lengths_tensor, validate_inputs=True
        )
        triton_fn = lambda: paged_decode_attention_triton(
            query, cache, tables_tensor, lengths_tensor, validate_inputs=False
        )
        max_abs, max_rel = _max_errors(checked_output, reference_output)
        matches_reference = torch.allclose(checked_output, reference_output, atol=atol, rtol=rtol)
        if not matches_reference:
            raise RuntimeError("triton_paged output differs from the PyTorch reference")
        results["triton_paged"] = {
            "timing": _measure(triton_fn, args.warmup, args.iterations, device).__dict__,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "matches_reference": matches_reference,
        }
    kernel_config = paged_decode_autotune_config() if run_triton else None

    estimated_kv_read_bytes = (
        2
        * args.batch_size
        * args.context_len
        * args.num_kv_heads
        * args.head_dim
        * torch.tensor([], dtype=getattr(torch, args.dtype)).element_size()
    )
    dense_mean = results["dense"]["timing"]["mean_ms"]
    dense_sdpa_mean = results["dense_sdpa"]["timing"]["mean_ms"]
    paged_sdpa_mean = results["paged_sdpa"]["timing"]["mean_ms"]
    for result in results.values():
        mean_ms = result["timing"]["mean_ms"]
        result["speedup_vs_dense"] = dense_mean / mean_ms
        result["speedup_vs_dense_sdpa"] = dense_sdpa_mean / mean_ms
        result["speedup_vs_paged_sdpa"] = paged_sdpa_mean / mean_ms
        result["estimated_kv_read_gbps"] = estimated_kv_read_bytes / (mean_ms / 1000) / 1e9

    return {
        "schema_version": 2,
        "environment": _environment(device),
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
            "block_table_layout": "interleaved requests with reserved blocks between logical blocks",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "correctness_atol": atol,
            "correctness_rtol": rtol,
            "timing": "CUDA events with per-sample synchronization" if device.type == "cuda" else "time.perf_counter",
        },
        "kv_cache_bytes": cache.numel_bytes,
        "estimated_kv_read_bytes": estimated_kv_read_bytes,
        "kernel_config": kernel_config,
        "results": results,
    }


def _triton_version() -> str | None:
    try:
        import triton

        return triton.__version__
    except ImportError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reference, PyTorch SDPA, and Triton paged decode")
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
    parser.add_argument("--profile-kernel", action="store_true")
    args = parser.parse_args()
    record = run(args)
    text = json.dumps(record, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
