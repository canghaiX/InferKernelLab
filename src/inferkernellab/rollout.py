from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .gsm8k import format_gsm8k_prompt, gsm8k_exact_match
from .hf_benchmark import LlamaPagedDecodeRunner, _dtype, _load_model, _resolve_device
from .sglang_client import SGLangClient

_MODEL_FILE_SUFFIXES = (".safetensors", ".bin", ".index.json")
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
}


def _fingerprint_files(root: Path, predicate) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and predicate(path)
    )
    if not files:
        raise ValueError(f"no fingerprintable files found in {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_path: str | Path) -> str:
    root = Path(model_path)
    return _fingerprint_files(
        root,
        lambda path: (
            path.name == "config.json" or path.name.endswith(_MODEL_FILE_SUFFIXES)
        ),
    )


def tokenizer_fingerprint(tokenizer_path: str | Path, vocab_size: int) -> str:
    root = Path(tokenizer_path)
    has_tokenizer = any(
        path.is_file()
        and (path.name in _TOKENIZER_FILES or path.name.endswith(".model"))
        for path in root.rglob("*")
    )
    if not has_tokenizer:
        return hashlib.sha256(f"token_ids:vocab={vocab_size}".encode()).hexdigest()
    return _fingerprint_files(
        root,
        lambda path: path.name in _TOKENIZER_FILES or path.name.endswith(".model"),
    )


def _load_prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").split("\n"), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid prompt JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise TypeError(f"prompt line {line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("prompt file contains no records")
    return rows


def _encode_prompts(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any | None,
    vocab_size: int,
) -> tuple[
    list[tuple[int, ...]], list[tuple[int, ...]], list[str | None], list[str | None]
]:
    prompts = []
    targets = []
    prompt_texts = []
    target_texts = []
    for index, row in enumerate(rows):
        if "prompt_tokens" in row and "target_tokens" in row:
            if any(
                not isinstance(row[key], list)
                or any(type(value) is not int for value in row[key])
                for key in ("prompt_tokens", "target_tokens")
            ):
                raise ValueError(
                    f"prompt line {index + 1} token IDs must be lists of integers"
                )
            prompt = tuple(row["prompt_tokens"])
            target = tuple(row["target_tokens"])
            prompt_text = target_text = None
        elif "question" in row and "answer" in row and tokenizer is not None:
            prompt_text = format_gsm8k_prompt(tokenizer, str(row["question"]))
            target_text = str(row["answer"])
            prompt = tuple(tokenizer.encode(prompt_text, add_special_tokens=False))
            target = tuple(tokenizer.encode(target_text, add_special_tokens=False))
        elif "prompt" in row and "target" in row and tokenizer is not None:
            prompt_text = str(row["prompt"])
            target_text = str(row["target"])
            prompt = tuple(tokenizer.encode(prompt_text, add_special_tokens=False))
            target = tuple(tokenizer.encode(target_text, add_special_tokens=False))
        else:
            raise ValueError(
                f"prompt line {index + 1} needs prompt_tokens/target_tokens, or text fields "
                "with --tokenizer-path"
            )
        if not prompt or not target:
            raise ValueError(f"prompt line {index + 1} has an empty prompt or target")
        if any(token < 0 or token >= vocab_size for token in (*prompt, *target)):
            raise ValueError(
                f"prompt line {index + 1} contains an out-of-vocabulary token"
            )
        prompts.append(prompt)
        targets.append(target)
        prompt_texts.append(prompt_text)
        target_texts.append(target_text)
    return prompts, targets, prompt_texts, target_texts


def _write_manifest(
    output: str, manifest: dict[str, Any], groups: list[dict[str, Any]]
) -> dict[str, Any]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        for group in groups:
            stream.write(json.dumps(group, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)
    return manifest


def _run_sglang(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_source != "path" or not args.model_path:
        raise ValueError("SGLang rollout requires --model-source path and --model-path")
    if not args.sglang_base_url:
        raise ValueError("SGLang rollout requires --sglang-base-url")
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError(
            "SGLang rollout requires 'transformers' for the local tokenizer"
        ) from error
    model_path = Path(args.model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = _load_tokenizer(tokenizer_path)
    rows = _load_prompt_rows(Path(args.prompts))
    is_gsm8k = all("question" in row and "answer" in row for row in rows)
    if is_gsm8k and not args.data_manifest:
        raise ValueError("GSM8K rollout requires --data-manifest from prepare_gsm8k.py")
    if args.data_manifest:
        # Fail before generating when the manifest is missing or malformed.
        json.loads(Path(args.data_manifest).read_text())
    prompts, targets, prompt_texts, target_texts = _encode_prompts(
        rows, tokenizer=tokenizer, vocab_size=int(config.vocab_size)
    )
    expanded_prompts = [
        prompt for prompt in prompts for _ in range(args.num_generations)
    ]
    client = SGLangClient(args.sglang_base_url, timeout=args.sglang_timeout)
    server_info = client.server_info()
    if server_info["version"] != "0.5.18":
        raise ValueError("SGLang rollout requires server version 0.5.18")
    eos_ids = (
        config.eos_token_id
        if isinstance(config.eos_token_id, list)
        else [config.eos_token_id]
    )
    try:
        from transformers import GenerationConfig

        generation_config = GenerationConfig.from_pretrained(
            model_path, local_files_only=True
        )
        configured_eos = generation_config.eos_token_id
        if configured_eos is not None:
            eos_ids = (
                configured_eos if isinstance(configured_eos, list) else [configured_eos]
            )
    except (OSError, TypeError, ValueError):
        pass
    eos_ids = [int(token) for token in eos_ids if token is not None]
    started = time.perf_counter()
    generations = client.generate(
        expanded_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seeds=_request_sampling_seeds(args.seed, len(expanded_prompts)),
        stop_token_ids=eos_ids,
    )
    generation_seconds = time.perf_counter() - started
    groups = []
    for group_index, (prompt, target) in enumerate(zip(prompts, targets)):
        completions = []
        for offset in range(args.num_generations):
            generation = generations[group_index * args.num_generations + offset]
            reward = (
                gsm8k_exact_match(generation.text, target_texts[group_index] or "")
                if is_gsm8k
                else _prefix_accuracy(generation.token_ids, target)
            )
            completions.append(
                {
                    "response_tokens": list(generation.token_ids),
                    "response_text": generation.text,
                    "old_logprobs": list(generation.token_logprobs),
                    "sampling_logprobs": list(generation.token_logprobs),
                    "reward": reward,
                    "finish_reason": generation.finish_reason,
                }
            )
        group = {
            "type": "prompt_group",
            "group_id": str(
                rows[group_index].get(
                    "id", rows[group_index].get("group_id", group_index)
                )
            ),
            "prompt_tokens": list(prompt),
            "target_tokens": list(target),
            "prompt_text": prompt_texts[group_index],
            "target_text": target_texts[group_index],
            "completions": completions,
        }
        if is_gsm8k:
            group["task"] = "gsm8k"
            group["answer"] = str(rows[group_index]["answer"])
        groups.append(group)
    manifest = {
        "type": "manifest",
        "schema_version": 1,
        "architecture": (config.architectures or ["AutoModelForCausalLM"])[0],
        "model_type": config.model_type,
        "model_source": str(model_path),
        "model_fingerprint": model_fingerprint(model_path),
        "tokenizer_fingerprint": tokenizer_fingerprint(
            tokenizer_path, int(config.vocab_size)
        ),
        "tokenizer_mode": "text",
        "vocab_size": int(config.vocab_size),
        "task": "gsm8k" if is_gsm8k else "generic",
        "prompt_format": "gsm8k_chat_no_thinking_v1" if is_gsm8k else None,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "request_seed_scheme": "sha256_base_seed_and_flat_index_v1",
        },
        "logprob_convention": "temperature_scaled_untruncated_policy",
        "logprob_tolerance": 0.5
        if config.model_type == "qwen3" and args.dtype == "bfloat16"
        else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "num_generations": args.num_generations,
            "eos_token_ids": eos_ids,
        },
        "reward_function": "gsm8k_exact_match_v1"
        if is_gsm8k
        else "token_prefix_accuracy_v1",
        "prompt_group_count": len(groups),
        "backend": "sglang",
        "engine": "sglang",
        "sglang_base_url": args.sglang_base_url,
        "rollout_metrics": {
            "elapsed_seconds": generation_seconds,
            "generated_tokens": sum(len(item.token_ids) for item in generations),
            "tokens_per_sec": sum(len(item.token_ids) for item in generations)
            / max(generation_seconds, 1e-9),
        },
        "sglang_server": server_info,
        "sglang_server_fingerprint": hashlib.sha256(
            json.dumps(server_info, sort_keys=True).encode()
        ).hexdigest(),
        "dtype": args.dtype,
        "cache_released": None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "sglang",
        },
        "provenance": {
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "command": " ".join(sys.argv),
            "source_sha256": _fingerprint_files(
                Path(__file__).parent, lambda path: path.suffix == ".py"
            ),
        },
    }
    if args.data_manifest:
        manifest["data_manifest"] = str(Path(args.data_manifest).resolve())
        manifest["data_fingerprint"] = hashlib.sha256(
            Path(args.data_manifest).read_bytes()
        ).hexdigest()
        manifest["data_metadata"] = json.loads(Path(args.data_manifest).read_text())
        manifest["prompt_file"] = str(Path(args.prompts).resolve())
        manifest["prompt_file_sha256"] = hashlib.sha256(
            Path(args.prompts).read_bytes()
        ).hexdigest()
    return _write_manifest(args.output, manifest, groups)


def _hf_incremental_generate(
    model: torch.nn.Module,
    prompt: tuple[int, ...],
    *,
    max_new_tokens: int,
    temperature: float,
    eos_token_ids: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], list[float], str]:
    prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(prompt_ids)
    response: list[int] = []
    logprobs: list[float] = []
    eos = set(eos_token_ids)

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
        ),
    ):
        output = model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits[:, -1].float()
        cache = output.past_key_values
        for index in range(max_new_tokens):
            policy_logprobs = torch.log_softmax(logits / temperature, dim=-1)
            token = int(
                torch.multinomial(
                    policy_logprobs.exp(), num_samples=1, generator=generator
                ).item()
            )
            response.append(token)
            logprobs.append(float(policy_logprobs[0, token]))
            if token in eos:
                finish_reason = "eos"
                break
            if index + 1 < max_new_tokens:
                prompt_ids = torch.tensor([[token]], dtype=torch.long, device=device)
                attention_mask = torch.ones(
                    (1, len(prompt) + index + 1), dtype=torch.long, device=device
                )
                output = model(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                logits = output.logits[:, -1].float()
                cache = output.past_key_values
        else:
            finish_reason = "length"
    return response, logprobs, finish_reason


def _run_hf_reference(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_source != "path" or not args.model_path:
        raise ValueError("HF reference rollout requires a local --model-path")
    if args.max_new_tokens <= 0:
        raise ValueError("HF reference rollout requires max_new_tokens > 0")
    if args.top_p != 1.0:
        raise ValueError("HF reference GRPO rollout requires top_p=1.0")
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "HF reference rollout requires the optional Transformers dependency"
        ) from error

    model_path = Path(args.model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError(
            "HF reference rollout currently supports Qwen3 checkpoints only"
        )
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = _load_tokenizer(tokenizer_path)
    rows = _load_prompt_rows(Path(args.prompts))
    is_gsm8k = all("question" in row and "answer" in row for row in rows)
    if is_gsm8k and not args.data_manifest:
        raise ValueError("GSM8K rollout requires --data-manifest from prepare_gsm8k.py")
    if args.data_manifest:
        json.loads(Path(args.data_manifest).read_text())
    prompts, targets, prompt_texts, target_texts = _encode_prompts(
        rows, tokenizer=tokenizer, vocab_size=int(config.vocab_size)
    )
    if max(map(len, prompts)) + args.max_new_tokens > int(
        config.max_position_embeddings
    ):
        raise ValueError("prompt plus generation length exceeds model position limit")

    device = _resolve_device(args.device)
    dtype = _dtype(args.dtype)
    config._attn_implementation = args.attention_implementation
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation=args.attention_implementation,
        )
        .to(device)
        .eval()
    )
    generation_config = None
    try:
        from transformers import GenerationConfig

        generation_config = GenerationConfig.from_pretrained(
            model_path, local_files_only=True
        )
    except (OSError, TypeError, ValueError):
        pass
    eos_value = (
        generation_config.eos_token_id
        if generation_config is not None and generation_config.eos_token_id is not None
        else config.eos_token_id
    )
    eos_token_ids = tuple(
        int(token)
        for token in (eos_value if isinstance(eos_value, list) else [eos_value])
        if token is not None
    )
    seeds = _request_sampling_seeds(args.seed, len(prompts) * args.num_generations)
    groups = []
    generated_tokens = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for group_index, (prompt, target) in enumerate(zip(prompts, targets, strict=True)):
        completions = []
        for offset in range(args.num_generations):
            flat_index = group_index * args.num_generations + offset
            generator = torch.Generator(device=device)
            generator.manual_seed(seeds[flat_index])
            response, old_logprobs, finish_reason = _hf_incremental_generate(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eos_token_ids=eos_token_ids,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            reward = (
                gsm8k_exact_match(response_text, target_texts[group_index] or "")
                if is_gsm8k
                else _prefix_accuracy(tuple(response), target)
            )
            completions.append(
                {
                    "response_tokens": response,
                    "response_text": response_text,
                    "old_logprobs": old_logprobs,
                    "sampling_logprobs": old_logprobs,
                    "reward": reward,
                    "finish_reason": finish_reason,
                }
            )
            generated_tokens += len(response)
        group = {
            "type": "prompt_group",
            "group_id": str(
                rows[group_index].get(
                    "id", rows[group_index].get("group_id", group_index)
                )
            ),
            "prompt_tokens": list(prompt),
            "target_tokens": list(target),
            "prompt_text": prompt_texts[group_index],
            "target_text": target_texts[group_index],
            "completions": completions,
        }
        if is_gsm8k:
            group["task"] = "gsm8k"
            group["answer"] = str(rows[group_index]["answer"])
        groups.append(group)
        print(
            f"hf_reference rollout groups={group_index + 1}/{len(prompts)} "
            f"tokens={generated_tokens} elapsed_s={time.perf_counter() - started:.1f}",
            file=sys.stderr,
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    del model
    tokenizer_fp = tokenizer_fingerprint(tokenizer_path, int(config.vocab_size))
    manifest = {
        "type": "manifest",
        "schema_version": 1,
        "architecture": (config.architectures or ["AutoModelForCausalLM"])[0],
        "model_type": config.model_type,
        "model_source": str(model_path),
        "model_fingerprint": model_fingerprint(model_path),
        "tokenizer_fingerprint": tokenizer_fp,
        "tokenizer_mode": "text",
        "vocab_size": int(config.vocab_size),
        "task": "gsm8k" if is_gsm8k else "generic",
        "prompt_format": "gsm8k_chat_no_thinking_v1" if is_gsm8k else None,
        "sampling": {
            "temperature": args.temperature,
            "top_p": 1.0,
            "seed": args.seed,
            "request_seed_scheme": "sha256_base_seed_and_flat_index_v1",
        },
        "logprob_convention": "temperature_scaled_untruncated_policy",
        "logprob_tolerance": 0.5 if args.dtype == "bfloat16" else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "num_generations": args.num_generations,
            "eos_token_ids": list(eos_token_ids),
        },
        "reward_function": "gsm8k_exact_match_v1"
        if is_gsm8k
        else "token_prefix_accuracy_v1",
        "prompt_group_count": len(groups),
        "backend": "hf_transformers_incremental",
        "engine": "hf_reference",
        "forward_contract": {
            "name": "qwen3_hf_incremental_cache_v1",
            "attention_implementation": args.attention_implementation,
            "weights_dtype": "float32",
            "autocast_dtype": args.dtype,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "rollout_metrics": {
            "elapsed_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": generated_tokens / max(elapsed, 1e-9),
            "peak_memory_allocated_bytes": peak_memory,
        },
        "dtype": args.dtype,
        "cache_released": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "provenance": {
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "command": " ".join(sys.argv),
            "source_sha256": _fingerprint_files(
                Path(__file__).resolve().parents[1], lambda path: path.suffix == ".py"
            ),
        },
    }
    if args.data_manifest:
        manifest["data_manifest"] = str(Path(args.data_manifest).resolve())
        manifest["data_fingerprint"] = hashlib.sha256(
            Path(args.data_manifest).read_bytes()
        ).hexdigest()
        manifest["data_metadata"] = json.loads(Path(args.data_manifest).read_text())
        manifest["prompt_file"] = str(Path(args.prompts).resolve())
        manifest["prompt_file_sha256"] = hashlib.sha256(
            Path(args.prompts).read_bytes()
        ).hexdigest()
    return _write_manifest(args.output, manifest, groups)


def _request_sampling_seeds(seed: int, count: int) -> list[int]:
    # Adding the index to adjacent base seeds would reuse 7/8 draws per group
    # across seeds 2026/2027/2028. Hash each pair into an independent stream.
    return [
        int.from_bytes(hashlib.sha256(f"{seed}:{index}".encode()).digest()[:4], "big")
        & 0x7FFFFFFF
        for index in range(count)
    ]


def _prefix_accuracy(response: tuple[int, ...], target: tuple[int, ...]) -> float:
    matched = 0
    for actual, expected in zip(response, target):
        if actual != expected:
            break
        matched += 1
    return matched / max(len(response), len(target), 1)


def _load_tokenizer(path: str | None) -> Any | None:
    if path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "text prompts require the optional Transformers dependency"
        ) from error
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_source=args.model_source,
        model_path=args.model_path,
        seed=args.seed,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_generations < 2:
        raise ValueError("GRPO rollout requires at least two generations per prompt")
    if (
        not math.isfinite(args.temperature)
        or args.temperature <= 0
        or not 0 < args.top_p <= 1
    ):
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")
    if args.model_source == "random" and not args.export_model_path:
        raise ValueError("random models require --export-model-path for later training")
    if args.model_source == "path" and not args.model_path:
        raise ValueError("--model-path is required for model-source=path")
    if getattr(args, "engine", "paged_hf") == "sglang":
        return _run_sglang(args)
    if getattr(args, "engine", "paged_hf") == "hf_reference":
        return _run_hf_reference(args)

    device = _resolve_device(args.device)
    dtype = _dtype(args.dtype)
    model, source = _load_model(_model_args(args), device, dtype)
    model_path = Path(args.export_model_path or args.model_path)
    if args.export_model_path:
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_path)
    rows = _load_prompt_rows(Path(args.prompts))
    needs_text_tokenizer = any("prompt" in row and "target" in row for row in rows)
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    elif needs_text_tokenizer and args.model_source == "path":
        tokenizer_path = args.model_path
    else:
        tokenizer_path = None
    tokenizer = _load_tokenizer(tokenizer_path)
    prompts, targets, prompt_texts, target_texts = _encode_prompts(
        rows,
        tokenizer=tokenizer,
        vocab_size=int(model.config.vocab_size),
    )
    max_prompt = max(map(len, prompts))
    if max_prompt + args.max_new_tokens > int(model.config.max_position_embeddings):
        raise ValueError("prompt plus generation length exceeds model position limit")
    expanded_prompts = [
        prompt for prompt in prompts for _ in range(args.num_generations)
    ]
    runner = LlamaPagedDecodeRunner(
        model,
        expanded_prompts,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
        append_backend=args.append_backend,
        block_size=args.block_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_mode="batched",
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        eos_token_ids=(
            [int(model.config.eos_token_id)]
            if isinstance(model.config.eos_token_id, int)
            else model.config.eos_token_id
        ),
    )
    with torch.inference_mode():
        result = runner.run()

    groups = []
    for group_index, (prompt, target) in enumerate(zip(prompts, targets)):
        start = group_index * args.num_generations
        completions = []
        for offset in range(args.num_generations):
            item_index = start + offset
            response = result.generated_tokens[item_index]
            completions.append(
                {
                    "response_tokens": list(response),
                    "old_logprobs": list(result.token_logprobs[item_index]),
                    "sampling_logprobs": list(result.sampling_logprobs[item_index]),
                    "reward": _prefix_accuracy(response, target),
                    "finish_reason": result.finish_reasons[item_index],
                }
            )
        groups.append(
            {
                "type": "prompt_group",
                "group_id": str(rows[group_index].get("group_id", group_index)),
                "prompt_tokens": list(prompt),
                "target_tokens": list(target),
                "prompt_text": prompt_texts[group_index],
                "target_text": target_texts[group_index],
                "completions": completions,
            }
        )

    tokenizer_fp = (
        tokenizer_fingerprint(tokenizer_path, int(model.config.vocab_size))
        if tokenizer_path
        else hashlib.sha256(
            f"token_ids:vocab={int(model.config.vocab_size)}".encode()
        ).hexdigest()
    )
    manifest = {
        "type": "manifest",
        "schema_version": 1,
        "architecture": "LlamaForCausalLM",
        "model_type": "llama",
        "model_source": source,
        "model_fingerprint": model_fingerprint(model_path),
        "tokenizer_fingerprint": tokenizer_fp,
        "tokenizer_mode": "text" if tokenizer is not None else "token_ids",
        "vocab_size": int(model.config.vocab_size),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "logprob_convention": "temperature_scaled_untruncated_policy",
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "num_generations": args.num_generations,
            "eos_token_ids": list(runner.eos_token_ids),
        },
        "reward_function": "token_prefix_accuracy_v1",
        "prompt_group_count": len(groups),
        "backend": args.backend,
        "dtype": args.dtype,
        "append_backend": args.append_backend,
        "prefill_mode": "batched",
        "prefill_latency_ms": result.prefill_wall_ms,
        "ttft_ms": result.ttft_wall_ms,
        "tpot_ms": result.tpot_wall_ms,
        "decode_p50_ms": result.decode_p50_ms,
        "decode_p95_ms": result.decode_p95_ms,
        "peak_kv_memory_bytes": result.peak_kv_memory_bytes,
        "peak_memory_allocated_bytes": result.peak_memory_allocated_bytes,
        "response_tokens": sum(map(len, result.generated_tokens)),
        "decode_tokens_per_sec": sum(map(len, result.generated_tokens))
        / max(sum(result.decode_step_wall_ms) / 1000, 1e-9),
        "cache_released": result.cache_released,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "provenance": {
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "source_sha256": _fingerprint_files(
                Path(__file__).resolve().parents[1], lambda path: path.suffix == ".py"
            ),
            "command": " ".join(sys.argv),
        },
    }
    return _write_manifest(args.output, manifest, groups)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[2]),
                    "status",
                    "--porcelain",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate local GRPO rollouts")
    parser.add_argument(
        "--engine", choices=("paged_hf", "sglang", "hf_reference"), default="paged_hf"
    )
    parser.add_argument("--sglang-base-url", default=None)
    parser.add_argument("--sglang-timeout", type=float, default=120.0)
    parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--data-manifest", default=None)
    parser.add_argument("--model-source", choices=("random", "path"), default="path")
    parser.add_argument("--model-path")
    parser.add_argument("--export-model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--backend",
        choices=(
            "paged_reference",
            "paged_sdpa",
            "triton_paged",
            "triton_paged_grouped",
        ),
        default="paged_sdpa",
    )
    parser.add_argument(
        "--append-backend", choices=("torch", "triton"), default="torch"
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
