#!/usr/bin/env python3
"""Evaluate a local tokenizer/model served by SGLang on a prepared GSM8K split."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

from inferkernellab.gsm8k import format_gsm8k_prompt, gsm8k_exact_match
from inferkernellab.rollout import (
    _git_dirty,
    _git_revision,
    model_fingerprint,
    tokenizer_fingerprint,
)
from inferkernellab.sglang_client import SGLangClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch-size and max-new-tokens must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    prompts = [
        tokenizer.encode(
            format_gsm8k_prompt(tokenizer, str(row["question"])),
            add_special_tokens=False,
        )
        for row in rows
    ]
    generation = GenerationConfig.from_pretrained(
        args.model_path, local_files_only=True
    )
    eos = generation.eos_token_id or tokenizer.eos_token_id
    stop_ids = (
        list(eos) if isinstance(eos, list) else ([int(eos)] if eos is not None else [])
    )
    client = SGLangClient(args.base_url, timeout=300)
    server_info = client.server_info()
    if server_info["version"] != "0.5.18":
        raise ValueError("GSM8K evaluator requires SGLang 0.5.18")
    correct = 0
    latencies = []
    generated = 0
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        tick = time.perf_counter()
        outputs = client.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seeds=[args.seed + index for index in range(start, start + len(batch))],
            stop_token_ids=stop_ids,
            return_logprob=False,
        )
        latencies.append((time.perf_counter() - tick) * 1000)
        for row, output in zip(rows[start : start + len(batch)], outputs):
            correct += int(gsm8k_exact_match(output.text, row["answer"]))
            generated += len(output.token_ids)
    elapsed = time.perf_counter() - started
    result = {
        "task": "gsm8k",
        "model_path": args.model_path,
        "model_fingerprint": model_fingerprint(args.model_path),
        "tokenizer_fingerprint": tokenizer_fingerprint(args.model_path, len(tokenizer)),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "sglang_server": server_info,
        "provenance": {"git_revision": _git_revision(), "git_dirty": _git_dirty()},
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "temperature": 0.0,
        "prompt_format": "gsm8k_chat_no_thinking_v1",
        "count": len(rows),
        "exact_match": correct / max(len(rows), 1),
        "correct": correct,
        "generated_tokens": generated,
        "throughput_samples_per_sec": len(rows) / max(elapsed, 1e-9),
        "batch_latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "batch_latency_p95_ms": sorted(latencies)[
            max(0, int(len(latencies) * 0.95) - 1)
        ]
        if latencies
        else 0.0,
        "seed": args.seed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
