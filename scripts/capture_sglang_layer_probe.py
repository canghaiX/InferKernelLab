#!/usr/bin/env python3
"""Capture and compare one SGLang/HF Qwen3 decoder token layer by layer."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from inferkernellab.sglang_client import SGLangClient


def _read_rollout(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if not records or records[0].get("type") != "manifest":
        raise ValueError("rollout must begin with a manifest")
    return records[0], records[1:]


def _hf_layer_vectors(model, input_ids: torch.Tensor, dtype: torch.dtype):
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        for stage, module in (
            ("input_norm", layer.input_layernorm),
            ("attention", layer.self_attn),
            ("post_attention_norm", layer.post_attention_layernorm),
            ("gate_proj", layer.mlp.gate_proj),
            ("up_proj", layer.mlp.up_proj),
            ("down_proj", layer.mlp.down_proj),
            ("mlp", layer.mlp),
            ("block", layer),
        ):

            def save(_module, _inputs, output, *, index=layer_index, name=stage):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if isinstance(value, torch.Tensor):
                    captured[f"layer{index:03d}_{name}"] = (
                        value.reshape(-1, value.shape[-1])[-1].detach().float().cpu()
                    )

            handles.append(module.register_forward_hook(save))
    try:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=input_ids.device.type,
                dtype=dtype,
                enabled=dtype != torch.float32,
            ),
        ):
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return captured, output.logits[0, -1].float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sglang-base-url", required=True)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--completion", required=True, type=int)
    parser.add_argument("--response-position", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    if args.completion < 0 or args.response_position < 0:
        parser.error("completion and response position must be nonnegative")

    manifest, groups = _read_rollout(args.rollouts)
    group = next(
        (item for item in groups if str(item["group_id"]) == args.group_id), None
    )
    if group is None:
        raise ValueError(f"rollout has no group {args.group_id}")
    completion = group["completions"][args.completion]
    response = [int(token) for token in completion["response_tokens"]]
    position = args.response_position
    if position >= len(response):
        raise ValueError("response position is out of range")
    prefix = [int(token) for token in group["prompt_tokens"]] + response[:position]
    target = response[position]
    temperature = float(args.temperature or manifest["sampling"]["temperature"])
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    args.capture_dir.mkdir(parents=True, exist_ok=True)
    for path in args.capture_dir.glob("pass*_layer*_*.pt"):
        path.unlink()
    active = args.capture_dir / "capture.active"
    active.write_text(secrets.token_hex(16), encoding="utf-8")
    try:
        score = SGLangClient(args.sglang_base_url, timeout=600).score_next_tokens(
            [prefix], [target], temperature=temperature, return_hidden_states=True
        )[0]
    finally:
        active.unlink(missing_ok=True)
    if score.hidden_state is None:
        raise ValueError("SGLang did not return final hidden state")

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError("layer probe currently supports Qwen3 checkpoints only")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation=args.attention_implementation,
        )
        .to(torch.device(args.device))
        .eval()
    )
    input_ids = torch.tensor([prefix], dtype=torch.long, device=args.device)
    hf_layers, hf_logits = _hf_layer_vectors(model, input_ids, dtype)
    hf_logprob = float(torch.log_softmax(hf_logits / temperature, dim=-1)[target])

    sglang_files = sorted(args.capture_dir.glob("pass*_layer*_*.pt"))
    if not sglang_files:
        raise ValueError("no SGLang layer outputs captured; check server forward_hooks")
    sglang_vectors = {
        path.stem.split("_", 1)[1].replace("down_projection", "down_proj"): torch.load(
            path, map_location="cpu", weights_only=True
        )
        for path in sglang_files
        if path.name.startswith("pass000_")
    }
    mlp_width = int(config.intermediate_size)
    for path in sglang_files:
        if not path.name.startswith("pass000_") or not path.name.endswith(
            "_gate_up_projection.pt"
        ):
            continue
        layer = int(path.name.split("_layer", 1)[1][:3])
        fused = torch.load(path, map_location="cpu", weights_only=True)
        if fused.numel() != 2 * mlp_width:
            raise ValueError("SGLang gate_up projection has unexpected width")
        sglang_vectors[f"layer{layer:03d}_gate_proj"] = fused[:mlp_width]
        sglang_vectors[f"layer{layer:03d}_up_proj"] = fused[mlp_width:]
    comparisons = []
    for key, sglang_vector in sorted(sglang_vectors.items()):
        hf_vector = hf_layers.get(key)
        if hf_vector is None:
            continue
        delta = (sglang_vector.float() - hf_vector.float()).abs()
        comparisons.append(
            {
                "stage": key,
                "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean()),
                "relative_l2_error": float(
                    torch.linalg.vector_norm(delta)
                    / torch.linalg.vector_norm(sglang_vector.float()).clamp_min(1e-12)
                ),
            }
        )
    if not comparisons:
        raise ValueError("captured SGLang module names did not match HF layers")

    block_comparisons = [
        item for item in comparisons if item["stage"].endswith("_block")
    ]
    attention_comparisons = [
        item for item in comparisons if item["stage"].endswith("_attention")
    ]
    result = {
        "schema_version": 1,
        "task": "qwen3_sglang_hf_layer_probe",
        "rollout_seed": manifest.get("sampling", {}).get("seed"),
        "group_id": args.group_id,
        "completion": args.completion,
        "response_position": position,
        "target_token_id": target,
        "temperature": temperature,
        "sglang_logprob": score.logprob,
        "rollout_logprob": float(completion["old_logprobs"][position]),
        "hf_prefix_logprob": hf_logprob,
        "sglang_vs_hf_logprob_abs_error": abs(score.logprob - hf_logprob),
        "sglang_sampled_token_id": score.sampled_token_id,
        "captured_layer_stage_count": len(comparisons),
        "comparisons": comparisons,
        "first_stage_over_0_1": next(
            (
                item["stage"]
                for item in block_comparisons
                if item["max_abs_error"] > 0.1
            ),
            None,
        ),
        "first_attention_stage_over_0_1": next(
            (
                item["stage"]
                for item in attention_comparisons
                if item["max_abs_error"] > 0.1
            ),
            None,
        ),
        "first_mlp_projection_over_0_1": next(
            (
                item["stage"]
                for item in comparisons
                if item["stage"].endswith(("_gate_proj", "_up_proj", "_down_proj"))
                and item["max_abs_error"] > 0.1
            ),
            None,
        ),
        "limitations": [
            "SGLang hook tensors are sampled at the final token of each module output.",
            "The block output combines SGLang's hidden and FP32 residual streams.",
            "A hook-based first divergence is diagnostic evidence, not proof that one operator alone causes the final logprob error.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "sglang_vs_hf_logprob_abs_error",
                    "first_stage_over_0_1",
                    "comparisons",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
