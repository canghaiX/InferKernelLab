#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from statistics import median

import torch

from inferkernellab.decode import DecodeRunResult, SyntheticDecodeRunner, SyntheticDecoderConfig
from inferkernellab.triton_ops import (
    paged_decode_autotune_config,
    paged_decode_grouped_autotune_config,
)


def _dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)


def _triton_version() -> str | None:
    try:
        import triton
    except ImportError:
        return None
    return getattr(triton, "__version__", None)


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


def _provenance() -> dict:
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
    }


def _prompts(batch_size: int, context_len: int, vocab_size: int) -> list[list[int]]:
    return [
        [((request_id + 3) * 17 + position * 13) % vocab_size for position in range(context_len)]
        for request_id in range(batch_size)
    ]


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _compare_logits(
    actual: DecodeRunResult,
    expected: DecodeRunResult,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float, bool]:
    if len(actual.logits) != len(expected.logits):
        raise RuntimeError("backend runs produced different numbers of decode steps")
    differences = []
    relative = []
    allclose = True
    for actual_step, expected_step in zip(actual.logits, expected.logits):
        if actual_step.shape != expected_step.shape:
            raise RuntimeError("backend runs produced logits with different shapes")
        difference = (actual_step.float() - expected_step.float()).abs()
        differences.append(float(difference.max().item()))
        relative.append(
            float((difference / expected_step.float().abs().clamp_min(atol)).max().item())
        )
        allclose = allclose and torch.allclose(actual_step, expected_step, atol=atol, rtol=rtol)
    actual_tokens = [token for request in actual.generated_tokens for token in request]
    expected_tokens = [token for request in expected.generated_tokens for token in request]
    if len(actual_tokens) != len(expected_tokens):
        raise RuntimeError("backend runs produced different numbers of generated tokens")
    token_match_rate = (
        sum(actual_token == expected_token for actual_token, expected_token in zip(actual_tokens, expected_tokens))
        / len(expected_tokens)
        if expected_tokens
        else 1.0
    )
    return max(differences, default=0.0), max(relative, default=0.0), token_match_rate, allclose


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    if dtype == torch.float16:
        return 3e-3, 3e-3
    return 1e-4, 1e-4


def _run_once(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    context_len: int,
    max_new_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    vocab_size: int,
    seed: int,
    max_num_batched_tokens: int,
    backend: str,
    append_backend: str,
) -> DecodeRunResult:
    config = SyntheticDecoderConfig(
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        device=device,
        seed=seed,
        backend=backend,
        append_backend=append_backend,
        max_num_seqs=max(batch_size, 32),
        max_num_batched_tokens=max_num_batched_tokens,
    )
    return SyntheticDecodeRunner(
        config,
        _prompts(batch_size, context_len, vocab_size),
        max_new_tokens=max_new_tokens,
    ).run()


def _repeat_summary(result: DecodeRunResult) -> dict:
    return {
        "ttft_ms": result.ttft_wall_ms,
        "prefill_ms": result.prefill_wall_ms,
        "decode_step_p50_ms": result.decode_step_p50_ms,
        "decode_step_p95_ms": result.decode_step_p95_ms,
        "tpot_ms": result.tpot_wall_ms,
        "tokens_per_sec": result.tokens_per_sec,
        "end_to_end_tokens_per_sec": result.end_to_end_tokens_per_sec,
        "ttft_device_ms": result.ttft_device_ms,
        "decode_step_device_p50_ms": result.decode_step_device_p50_ms,
        "decode_step_device_p95_ms": result.decode_step_device_p95_ms,
        "tpot_device_ms": result.tpot_device_ms,
        "device_tokens_per_sec": result.device_tokens_per_sec,
        "end_to_end_device_tokens_per_sec": result.end_to_end_device_tokens_per_sec,
        "peak_memory_allocated_bytes": result.peak_memory_allocated_bytes,
        "used_blocks_peak": result.used_blocks_peak,
    }


def _aggregate(repeats: list[dict]) -> dict:
    fields = (
        "ttft_ms",
        "prefill_ms",
        "decode_step_p50_ms",
        "decode_step_p95_ms",
        "tpot_ms",
        "tokens_per_sec",
        "end_to_end_tokens_per_sec",
        "ttft_device_ms",
        "decode_step_device_p50_ms",
        "decode_step_device_p95_ms",
        "tpot_device_ms",
        "device_tokens_per_sec",
        "end_to_end_device_tokens_per_sec",
        "peak_memory_allocated_bytes",
        "used_blocks_peak",
    )
    return {field: _median([float(repeat[field]) for repeat in repeats]) for field in fields}


def _run_case(args: argparse.Namespace, backend: str, append_backend: str) -> dict:
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    common = {
        "device": device,
        "dtype": dtype,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "max_new_tokens": args.max_new_tokens,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "vocab_size": args.vocab_size,
        "seed": args.seed,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }

    for _ in range(args.warmup):
        _run_once(**common, backend=backend, append_backend=append_backend)

    reference_repeats = [
        _run_once(**common, backend="paged_reference", append_backend="torch")
        for _ in range(args.repeats)
    ]
    actual_repeats = [
        _run_once(**common, backend=backend, append_backend=append_backend)
        for _ in range(args.repeats)
    ]
    atol, rtol = _tolerance(dtype)
    comparisons = []
    for actual, reference in zip(actual_repeats, reference_repeats):
        max_abs, max_rel, token_match_rate, matches_reference = _compare_logits(
            actual,
            reference,
            atol=atol,
            rtol=rtol,
        )
        comparisons.append(
            {
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "token_match_rate": token_match_rate,
                "matches_reference": matches_reference,
            }
        )

    summary = _aggregate([_repeat_summary(result) for result in actual_repeats])
    result = {
        "schema_version": 1,
        "benchmark": "synthetic_single_layer_decode",
        "environment": _environment(device),
        "provenance": _provenance(),
        "config": {
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "context_len": args.context_len,
            "max_new_tokens": args.max_new_tokens,
            "vocab_size": args.vocab_size,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "seed": args.seed,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "block_table_layout": "logical blocks are allocated round-robin across requests with reserved physical blocks between rounds",
        },
        "attention_backend": backend,
        "append_backend": append_backend,
        "kernel_config": (
            {
                "kernel_variant": "grouped",
                "query_group_size": min(8, args.num_heads // args.num_kv_heads),
                "kv_reuse_factor": args.num_heads // args.num_kv_heads,
                **(paged_decode_grouped_autotune_config() or {}),
            }
            if backend == "triton_paged_grouped"
            else paged_decode_autotune_config()
            if backend == "triton_paged"
            else None
        ),
        "comparison_reference": "paged_reference + torch append",
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "wall_clock": "perf_counter around each runtime step",
            "device_clock": "CUDA Events around each runtime step; CPU uses wall clock",
            "aggregation": "median across independent fresh runner repeats",
            "ttft_definition": "all scheduled prefill steps plus the first decode step",
            "tpot_definition": "mean runtime decode step time per generated token; P50/P95 are also reported",
            "tokens_per_sec_definition": "generated tokens divided by decode-step device/wall time",
        },
        "metrics": summary,
        "correctness": {
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max(item["max_abs_error"] for item in comparisons),
            "max_rel_error": max(item["max_rel_error"] for item in comparisons),
            "token_match_rate": min(item["token_match_rate"] for item in comparisons),
            "matches_reference": all(item["matches_reference"] for item in comparisons),
            "repeats": comparisons,
        },
        "generated_tokens": [list(tokens) for tokens in actual_repeats[0].generated_tokens],
        "limitations": [
            "single-layer synthetic decoder; not a real pretrained model",
            "no network, queueing, sampling, or production serving overhead",
        ],
    }
    if not result["correctness"]["matches_reference"]:
        raise RuntimeError(f"{backend}/{append_backend} failed correctness: {result['correctness']}")
    return result


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic single-layer decode sweep")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark_results/synthetic_decode.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 4, 16])
    parser.add_argument("--context-lens", type=_parse_int_list, default=[128, 512])
    parser.add_argument("--num-kv-heads-list", type=_parse_int_list, default=[32, 8, 1])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backends",
        default="paged_reference,paged_sdpa,triton_paged",
        help="comma-separated attention backends; triton_paged_grouped enables GQA/MQA reuse",
    )
    parser.add_argument("--include-triton-append", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("max-new-tokens and repeats must be positive; warmup cannot be negative")
    if args.num_heads <= 0 or args.head_dim <= 0 or args.block_size <= 0:
        parser.error("model and cache dimensions must be positive")
    if any(args.num_heads % num_kv_heads for num_kv_heads in args.num_kv_heads_list):
        parser.error("num-kv-heads-list values must divide num-heads")
    backends = [backend for backend in args.backends.split(",") if backend]
    allowed = {
        "paged_reference",
        "paged_sdpa",
        "triton_paged",
        "triton_paged_grouped",
    }
    if not backends or any(backend not in allowed for backend in backends):
        parser.error(f"backends must be selected from {sorted(allowed)}")
    dtype_names = [args.dtype] if args.dtype is not None else [item for item in args.dtypes.split(",") if item]
    if not dtype_names or any(name not in {"float16", "bfloat16", "float32"} for name in dtype_names):
        parser.error("dtypes must contain float16, bfloat16, or float32")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dtype_name in dtype_names:
        for batch_size in args.batch_sizes:
            for context_len in args.context_lens:
                for num_kv_heads in args.num_kv_heads_list:
                    for backend in backends:
                        case_args = argparse.Namespace(**vars(args))
                        case_args.dtype = dtype_name
                        case_args.batch_size = batch_size
                        case_args.context_len = context_len
                        case_args.num_kv_heads = num_kv_heads
                        case_args.output = None
                        print(
                            f"running backend={backend} append=torch batch={batch_size} "
                            f"context={context_len} kv_heads={num_kv_heads} dtype={dtype_name}",
                            flush=True,
                        )
                        rows.append(_run_case(case_args, backend, "torch"))
                    if args.include_triton_append:
                        for triton_backend in (
                            backend
                            for backend in backends
                            if backend in {"triton_paged", "triton_paged_grouped"}
                        ):
                            case_args = argparse.Namespace(**vars(args))
                            case_args.dtype = dtype_name
                            case_args.batch_size = batch_size
                            case_args.context_len = context_len
                            case_args.num_kv_heads = num_kv_heads
                            case_args.output = None
                            print(
                                f"running backend={triton_backend} append=triton batch={batch_size} "
                                f"context={context_len} kv_heads={num_kv_heads} dtype={dtype_name}",
                                flush=True,
                            )
                            rows.append(_run_case(case_args, triton_backend, "triton"))

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
