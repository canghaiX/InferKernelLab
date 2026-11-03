#!/usr/bin/env python3
"""Compare stored rollout, SGLang sampler, and HF teacher-forced token scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from inferkernellab.rollout import model_fingerprint
from inferkernellab.sglang_client import SGLangClient


def _read_rollout(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if not rows or rows[0].get("type") != "manifest":
        raise ValueError("rollout must begin with a manifest")
    return rows[0], rows[1:]


def _select_completions(groups: list[dict[str, Any]], args):
    if args.cases:
        cases = json.loads(Path(args.cases).read_text())
        if not isinstance(cases, list):
            raise ValueError("cases file must contain a JSON list")
        by_id = {str(group["group_id"]): group for group in groups}
        selected = []
        seen = set()
        for case in cases:
            group = by_id[str(case["group_id"])]
            index = int(case["completion"])
            key = (str(group["group_id"]), index)
            if key not in seen:
                selected.append((group, index, group["completions"][index]))
                seen.add(key)
        return selected[: args.max_completions], cases
    selected = []
    for group in groups:
        for index, completion in enumerate(group["completions"]):
            selected.append((group, index, completion))
            if len(selected) >= args.max_completions:
                return selected, None
    return selected, None


def _hf_scores(
    model,
    input_ids: list[int],
    prompt_length: int,
    response_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
) -> list[float]:
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
        ),
    ):
        logits = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits
    start = prompt_length - 1
    response_logits = logits[0, start : start + response_length].float() / temperature
    labels = ids[0, prompt_length : prompt_length + response_length]
    scores = (
        torch.log_softmax(response_logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    )
    return [float(value) for value in scores.cpu()]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "max_abs_error": max(values) if values else None,
        "mean_abs_error": statistics.fmean(values) if values else None,
        "p99_abs_error": float(torch.quantile(torch.tensor(values), 0.99))
        if values
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sglang-base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cases", help="JSON list of group_id/completion/response_position records"
    )
    parser.add_argument(
        "--preflight-json",
        type=Path,
        help="derive the worst-token case from a MiniTrainBench preflight report",
    )
    parser.add_argument("--max-completions", type=int, default=8)
    parser.add_argument("--max-positions-per-completion", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--tolerance", type=float, default=0.5)
    args = parser.parse_args()
    if (
        min(
            args.max_completions,
            args.max_positions_per_completion,
            args.score_batch_size,
        )
        <= 0
    ):
        parser.error("completion, position, and score batch limits must be positive")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("tolerance must be finite and positive")

    manifest, groups = _read_rollout(args.rollouts)
    if manifest.get("engine") != "sglang":
        raise ValueError("the diagnostic requires an SGLang rollout artifact")
    temperature = float(manifest["sampling"]["temperature"])
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    if manifest.get("model_fingerprint") != model_fingerprint(args.model_path):
        raise ValueError("rollout model fingerprint does not match --model-path")
    if manifest.get("model_type") != config.model_type:
        raise ValueError("rollout model type does not match --model-path")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation=args.attention_implementation,
        )
        .to(device)
        .eval()
    )
    client = SGLangClient(args.sglang_base_url, timeout=600)
    server_info = client.server_info()
    if server_info["version"] != "0.5.18":
        raise ValueError("diagnostic requires SGLang 0.5.18")

    if args.cases and args.preflight_json:
        parser.error("choose either --cases or --preflight-json")
    if args.preflight_json:
        preflight = json.loads(args.preflight_json.read_text())
        rollout_sha = hashlib.sha256(args.rollouts.read_bytes()).hexdigest()
        if preflight.get("rollout_sha256") != rollout_sha:
            raise ValueError("preflight report refers to a different rollout artifact")
        worst = preflight.get("logprob_probe", {}).get("worst")
        if not isinstance(worst, dict):
            raise ValueError("preflight report has no worst-token diagnostic")
        args.cases = str(args.output.with_suffix(".cases.json"))
        Path(args.cases).write_text(
            json.dumps(
                [
                    {
                        key: worst[key]
                        for key in ("group_id", "completion", "response_position")
                    }
                ],
                indent=2,
            )
            + "\n"
        )
    selected, requested_cases = _select_completions(groups, args)
    prefixes, targets, rows = [], [], []
    for group, completion_index, completion in selected:
        prompt = [int(token) for token in group["prompt_tokens"]]
        response = [int(token) for token in completion["response_tokens"]]
        hf = _hf_scores(
            model,
            prompt + response,
            len(prompt),
            len(response),
            device=device,
            dtype=dtype,
            temperature=temperature,
        )
        requested_positions = None
        if requested_cases is not None:
            requested_positions = sorted(
                {
                    int(case["response_position"])
                    for case in requested_cases
                    if str(case["group_id"]) == str(group["group_id"])
                    and int(case["completion"]) == completion_index
                }
            )
        if requested_positions is None:
            if len(response) <= args.max_positions_per_completion:
                positions = list(range(len(response)))
            else:
                positions = sorted(
                    {
                        round(
                            i
                            * (len(response) - 1)
                            / (args.max_positions_per_completion - 1)
                        )
                        for i in range(max(args.max_positions_per_completion, 2))
                    }
                )
        else:
            positions = requested_positions
        for position in positions:
            if position < 0 or position >= len(response):
                raise ValueError("case response_position is out of range")
            prefixes.append(prompt + response[:position])
            targets.append(response[position])
            rows.append(
                {
                    "group_id": str(group["group_id"]),
                    "completion": completion_index,
                    "response_position": position,
                    "target_token_id": response[position],
                    "is_eos": response[position]
                    in manifest["generation"]["eos_token_ids"],
                    "rollout_logprob": float(completion["old_logprobs"][position]),
                    "hf_logprob": hf[position],
                }
            )

    scores = []
    for start in range(0, len(prefixes), args.score_batch_size):
        end = min(start + args.score_batch_size, len(prefixes))
        scores.extend(
            client.score_next_tokens(
                prefixes[start:end],
                targets[start:end],
                temperature=temperature,
                seeds=range(start, end),
            )
        )
    for row, score in zip(rows, scores):
        row["sglang_teacher_forced_logprob"] = score.logprob
        row["sglang_sampled_token_id"] = score.sampled_token_id
        row["rollout_vs_sglang_abs_error"] = abs(row["rollout_logprob"] - score.logprob)
        row["sglang_vs_hf_abs_error"] = abs(score.logprob - row["hf_logprob"])
        row["rollout_vs_hf_abs_error"] = abs(row["rollout_logprob"] - row["hf_logprob"])

    result = {
        "schema_version": 1,
        "task": "grpo_token_logprob_alignment",
        "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        "rollout_seed": manifest.get("sampling", {}).get("seed"),
        "diagnostic_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "model_fingerprint": model_fingerprint(args.model_path),
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
        "sglang_server": server_info,
        "hf": {
            "torch": torch.__version__,
            "dtype": args.dtype,
            "attention_implementation": args.attention_implementation,
            "temperature": temperature,
        },
        "sampling_logprob_contract": manifest["logprob_convention"],
        "position_count": len(rows),
        "tolerance": args.tolerance,
        "comparisons": {
            "rollout_vs_sglang_teacher_forced": _stats(
                [r["rollout_vs_sglang_abs_error"] for r in rows]
            ),
            "sglang_teacher_forced_vs_hf": _stats(
                [r["sglang_vs_hf_abs_error"] for r in rows]
            ),
            "rollout_vs_hf": _stats([r["rollout_vs_hf_abs_error"] for r in rows]),
        },
        "worst_rollout_vs_sglang": max(
            rows, key=lambda r: r["rollout_vs_sglang_abs_error"], default=None
        ),
        "worst_sglang_vs_hf": max(
            rows, key=lambda r: r["sglang_vs_hf_abs_error"], default=None
        ),
        "worst_rollout_vs_hf": max(
            rows, key=lambda r: r["rollout_vs_hf_abs_error"], default=None
        ),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"position_count": len(rows), "comparisons": result["comparisons"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
