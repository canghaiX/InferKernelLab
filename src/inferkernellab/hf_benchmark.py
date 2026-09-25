from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .append import append_kv_triton, triton_append_available
from .attention import paged_decode_attention, paged_decode_attention_sdpa
from .cache import PagedKVCache
from .runtime import InferenceRuntime
from .scheduler import InferenceRequest, TokenBudgetScheduler
from .triton_ops import (
    paged_decode_attention_triton,
    paged_decode_autotune_config,
    triton_available,
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _dtype(name: str) -> torch.dtype:
    try:
        value = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unsupported dtype: {name}") from error
    if value not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(f"unsupported dtype: {name}")
    return value


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values[..., : values.shape[-1] // 2], values[..., values.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


@dataclass(frozen=True)
class LlamaPagedRunResult:
    backend: str
    append_backend: str
    prompt_lengths: tuple[int, ...]
    max_new_tokens: int
    generated_tokens: tuple[tuple[int, ...], ...]
    prompt_logits: torch.Tensor
    decode_logits: tuple[torch.Tensor, ...]
    prefill_wall_ms: float
    decode_step_wall_ms: tuple[float, ...]
    ttft_wall_ms: float
    tpot_wall_ms: float
    peak_memory_allocated_bytes: int
    peak_kv_memory_bytes: int
    used_blocks_peak: int
    cache_released: bool

    @property
    def decode_p50_ms(self) -> float:
        return _percentile(self.decode_step_wall_ms, 0.50)

    @property
    def decode_p95_ms(self) -> float:
        return _percentile(self.decode_step_wall_ms, 0.95)


class LlamaPagedDecodeRunner:
    """Execute a Transformers Llama decoder one token at a time over paged KV."""

    def __init__(
        self,
        model: Any,
        prompts: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        backend: str,
        block_size: int,
        append_backend: str = "torch",
        max_num_batched_tokens: int = 2048,
    ) -> None:
        if not prompts:
            raise ValueError("at least one prompt is required")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if block_size <= 0 or max_num_batched_tokens <= 0:
            raise ValueError("block-size and max-num-batched-tokens must be positive")
        if backend not in {"paged_reference", "paged_sdpa", "triton_paged"}:
            raise ValueError(f"unsupported backend: {backend}")
        if append_backend not in {"torch", "triton"}:
            raise ValueError(f"unsupported append backend: {append_backend}")
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.prompts = tuple(tuple(int(token) for token in prompt) for prompt in prompts)
        if any(not prompt for prompt in self.prompts):
            raise ValueError("prompts must be non-empty")
        vocab_size = int(model.config.vocab_size)
        if any(token < 0 or token >= vocab_size for prompt in self.prompts for token in prompt):
            raise ValueError("prompt token is outside the configured vocabulary")
        if backend == "triton_paged" and not triton_available():
            raise RuntimeError("triton_paged requires CUDA and an installed Triton backend")
        if append_backend == "triton" and not triton_append_available():
            raise RuntimeError("triton append requires CUDA and an installed Triton backend")

        self.backend = backend
        self.append_backend = append_backend
        self.max_new_tokens = max_new_tokens
        self.prompt_lengths = tuple(len(prompt) for prompt in self.prompts)
        self.num_layers = int(model.config.num_hidden_layers)
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(model.config.head_dim)
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        total_blocks = sum(
            math.ceil((length + max_new_tokens) / block_size)
            for length in self.prompt_lengths
        )
        self.cache = PagedKVCache(
            total_blocks,
            block_size,
            self.num_kv_heads,
            self.head_dim,
            num_layers=self.num_layers,
            dtype=self.dtype,
            device=self.device,
        )
        scheduler = TokenBudgetScheduler(
            max_num_seqs=len(self.prompts),
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self.runtime = InferenceRuntime(
            self.cache,
            scheduler,
            prefill_fn=self._prefill,
            decode_fn=self._decode,
        )
        self.requests = [
            self.runtime.submit(request_id, len(prompt), max_new_tokens)
            for request_id, prompt in enumerate(self.prompts)
        ]
        self._prompt_logits: dict[int, torch.Tensor] = {}
        self._next_logits: dict[int, torch.Tensor] = {}
        self._decode_logits: list[torch.Tensor] = []
        self._generated: dict[int, list[int]] = {
            request.request_id: [] for request in self.requests
        }
        self._has_run = False

    def _block_tables(self, requests: Sequence[InferenceRequest]) -> torch.Tensor:
        width = max(len(request.block_table) for request in requests)
        tables = torch.zeros((len(requests), width), dtype=torch.int32, device=self.device)
        for row, request in enumerate(requests):
            table = torch.tensor(request.block_table, dtype=torch.int32, device=self.device)
            tables[row, : table.numel()] = table
        return tables

    def _attention(
        self,
        query: torch.Tensor,
        requests: Sequence[InferenceRequest],
        context_lengths: Sequence[int],
        layer: int,
    ) -> torch.Tensor:
        tables = self._block_tables(requests)
        if self.backend == "paged_reference":
            return paged_decode_attention(
                query,
                self.cache,
                tables,
                context_lengths,
                layer=layer,
            )
        if self.backend == "paged_sdpa":
            return paged_decode_attention_sdpa(
                query,
                self.cache,
                tables,
                context_lengths,
                layer=layer,
            )
        lengths = torch.tensor(context_lengths, dtype=torch.int32, device=self.device)
        return paged_decode_attention_triton(
            query,
            self.cache,
            tables,
            lengths,
            layer=layer,
            validate_inputs=True,
        )

    def _write_kv(
        self,
        layer: int,
        requests: Sequence[InferenceRequest],
        positions: Sequence[int],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self.append_backend == "torch":
            for row, request in enumerate(requests):
                self.cache.write(
                    layer,
                    request.block_table,
                    positions[row],
                    key[row : row + 1],
                    value[row : row + 1],
                )
            return
        slots = torch.cat(
            [
                self.cache.slot_mapping(request.block_table, positions[row], 1)
                for row, request in enumerate(requests)
            ]
        ).to(torch.int32)
        append_kv_triton(key, value, self.cache, slots, layer=layer)

    def _forward_batch(
        self,
        token_ids: torch.Tensor,
        requests: Sequence[InferenceRequest],
        positions: Sequence[int],
    ) -> torch.Tensor:
        hidden = self.model.model.embed_tokens(token_ids.to(dtype=torch.long)).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)
        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            query = layer.self_attn.q_proj(hidden).view(
                hidden.shape[0], 1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key = layer.self_attn.k_proj(hidden).view(
                hidden.shape[0], 1, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            value = layer.self_attn.v_proj(hidden).view(
                hidden.shape[0], 1, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            cos, sin = self.model.model.rotary_emb(hidden, position_ids)
            query, key = _apply_rotary(query, key, cos, sin)
            self._write_kv(
                layer_index,
                requests,
                positions,
                key[:, :, 0, :].contiguous(),
                value[:, :, 0, :].contiguous(),
            )
            context_lengths = [position + 1 for position in positions]
            attended = self._attention(
                query[:, :, 0, :].contiguous(),
                requests,
                context_lengths,
                layer_index,
            )
            attended = attended.reshape(hidden.shape[0], 1, self.num_heads * self.head_dim)
            hidden = residual + layer.self_attn.o_proj(attended)
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp.down_proj(
                layer.mlp.act_fn(layer.mlp.gate_proj(hidden))
                * layer.mlp.up_proj(hidden)
            )
            hidden = residual + hidden
        hidden = self.model.model.norm(hidden)
        return self.model.lm_head(hidden).squeeze(1)

    def _prefill(self, request: InferenceRequest, start_token: int, end_token: int) -> None:
        prompt = self.prompts[request.request_id]
        for position in range(start_token, end_token):
            token_ids = torch.tensor([prompt[position]], dtype=torch.long, device=self.device)
            logits = self._forward_batch(token_ids, (request,), (position,))[0]
            if position == len(prompt) - 1:
                self._prompt_logits[request.request_id] = logits.detach()
                self._next_logits[request.request_id] = logits.detach()

    def _decode(self, requests: tuple[InferenceRequest, ...]) -> None:
        used_logits = torch.stack(
            [self._next_logits[request.request_id] for request in requests], dim=0
        )
        next_tokens = used_logits.argmax(dim=-1).to(device=self.device, dtype=torch.long)
        positions = [request.prompt_tokens + request.generated_tokens for request in requests]
        next_logits = self._forward_batch(next_tokens, requests, positions)
        self._decode_logits.append(used_logits.detach().float().cpu())
        for row, request in enumerate(requests):
            token = int(next_tokens[row].item())
            self._generated[request.request_id].append(token)
            self._next_logits[request.request_id] = next_logits[row].detach()

    def _timed_step(self) -> tuple[object, float]:
        started = time.perf_counter()
        batch = self.runtime.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return batch, (time.perf_counter() - started) * 1000

    def run(self) -> LlamaPagedRunResult:
        if self._has_run:
            raise RuntimeError("a LlamaPagedDecodeRunner instance can only be run once")
        self._has_run = True
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        prefill_ms = 0.0
        decode_ms: list[float] = []
        used_blocks_peak = self.cache.allocator.num_used_blocks
        try:
            with torch.inference_mode():
                while self.runtime.scheduler.waiting or self.runtime.scheduler.running:
                    batch, elapsed_ms = self._timed_step()
                    used_blocks_peak = max(used_blocks_peak, self.cache.allocator.num_used_blocks)
                    if batch.phase == "prefill":
                        prefill_ms += elapsed_ms
                    elif batch.phase == "decode":
                        decode_ms.append(elapsed_ms)
                    elif batch.phase == "idle":
                        raise RuntimeError("scheduler became idle before all requests finished")
        finally:
            cache_released = self.cache.allocator.num_used_blocks == 0

        if len(self._prompt_logits) != len(self.requests):
            raise RuntimeError("prefill did not produce one final logits tensor per request")
        generated_tokens = tuple(
            tuple(self._generated[request.request_id]) for request in self.requests
        )
        expected = len(self.requests) * self.max_new_tokens
        if sum(len(tokens) for tokens in generated_tokens) != expected:
            raise RuntimeError("decoder finished with an unexpected number of generated tokens")
        if len(decode_ms) != self.max_new_tokens:
            raise RuntimeError("decoder did not produce one batched step per generated token")
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        peak_kv_memory = (
            used_blocks_peak
            * self.cache.layout.block_size
            * self.cache.layout.num_layers
            * self.cache.layout.num_kv_heads
            * self.cache.layout.head_dim
            * 2
            * element_size
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return LlamaPagedRunResult(
            backend=self.backend,
            append_backend=self.append_backend,
            prompt_lengths=self.prompt_lengths,
            max_new_tokens=self.max_new_tokens,
            generated_tokens=generated_tokens,
            prompt_logits=torch.stack(
                [self._prompt_logits[request.request_id] for request in self.requests]
            ).float().cpu(),
            decode_logits=tuple(self._decode_logits),
            prefill_wall_ms=prefill_ms,
            decode_step_wall_ms=tuple(decode_ms),
            ttft_wall_ms=prefill_ms + decode_ms[0],
            tpot_wall_ms=sum(decode_ms) / len(decode_ms),
            peak_memory_allocated_bytes=peak_memory,
            peak_kv_memory_bytes=peak_kv_memory,
            used_blocks_peak=used_blocks_peak,
            cache_released=cache_released and self.cache.allocator.num_used_blocks == 0,
        )


def _load_model(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, dict[str, Any]]:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "Transformers adapter requires the optional dependency: "
            "pip install 'inferkernellab[hf]'"
        ) from error

    if args.model_source == "random":
        torch.manual_seed(args.seed)
        config = LlamaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_kv_heads,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
        source = {"type": "random", "download": False}
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --model-source path")
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        model = LlamaForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=dtype,
        )
        source = {"type": "path", "path": str(model_path), "download": False}
    model.config._attn_implementation = "eager"
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, source


def _make_prompts(
    *,
    batch_size: int,
    prompt_length: int,
    vocab_size: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if batch_size <= 0 or prompt_length <= 0:
        raise ValueError("batch-size and prompt-length must be positive")
    base = torch.arange(prompt_length, dtype=torch.long)
    return tuple(
        tuple(((base + seed + row * 17) % vocab_size).tolist())
        for row in range(batch_size)
    )


def _oracle_logits(
    model: Any,
    prompts: Sequence[Sequence[int]],
    generated_tokens: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    outputs = []
    for step in range(len(generated_tokens[0]) + 1):
        sequences = [
            tuple(prompt) + tuple(tokens[:step])
            for prompt, tokens in zip(prompts, generated_tokens)
        ]
        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids, use_cache=False, return_dict=True).logits[:, -1, :]
        outputs.append(logits.float().cpu())
    return tuple(outputs)


def _correctness(
    model: Any,
    prompts: Sequence[Sequence[int]],
    result: LlamaPagedRunResult,
    device: torch.device,
) -> dict[str, Any]:
    references = _oracle_logits(model, prompts, result.generated_tokens, device)
    prompt_error = float(
        (result.prompt_logits.float() - references[0].float()).abs().max().item()
    )
    decode_errors = [
        float((actual.float() - expected.float()).abs().max().item())
        for actual, expected in zip(result.decode_logits, references[:-1])
    ]
    max_errors = [prompt_error, *decode_errors]
    manual_tokens = torch.stack(
        [logits.argmax(dim=-1) for logits in result.decode_logits], dim=1
    )
    reference_tokens = torch.stack(
        [logits.argmax(dim=-1) for logits in references[:-1]], dim=1
    )
    matches = int((manual_tokens == reference_tokens).sum().item())
    total_tokens = int(reference_tokens.numel())
    return {
        "logits_max_error": max(max_errors),
        "prefill_logits_max_error": max_errors[0],
        "decode_logits_max_error": max(max_errors[1:]),
        "token_match_rate": matches / max(total_tokens, 1),
        "token_matches": matches,
        "token_count": total_tokens,
        "status": "ok" if max(max_errors) <= 5e-3 and matches == total_tokens else "failed",
        "oracle": "Transformers Llama eager full-sequence path",
    }


def _environment(device: torch.device) -> dict[str, Any]:
    driver = None
    if device.type == "cuda":
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0].strip()
        except (FileNotFoundError, IndexError, subprocess.CalledProcessError):
            driver = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    model_source = (
        "path" if args.model_path and args.model_source == "random" else args.model_source
    )
    if model_source not in {"random", "path"}:
        raise ValueError("model-source must be random or path")
    args.model_source = model_source
    device = _resolve_device(args.device)
    dtype = _dtype(args.dtype)
    model, source = _load_model(args, device, dtype)
    if args.prompt_length + args.max_new_tokens > int(model.config.max_position_embeddings):
        raise ValueError("prompt-length + max-new-tokens exceeds max-position-embeddings")
    prompts = _make_prompts(
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        vocab_size=int(model.config.vocab_size),
        seed=args.seed,
    )

    def execute() -> LlamaPagedRunResult:
        runner = LlamaPagedDecodeRunner(
            model,
            prompts,
            max_new_tokens=args.max_new_tokens,
            backend=args.backend,
            block_size=args.block_size,
            append_backend=args.append_backend,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        return runner.run()

    with torch.inference_mode():
        for _ in range(args.warmup):
            execute()
        runs = [execute() for _ in range(args.iterations)]
    first = runs[0]
    correctness = _correctness(model, prompts, first, device)
    if correctness["status"] != "ok":
        raise RuntimeError(f"Llama paged correctness failed: {correctness}")
    decode_samples = [sample for item in runs for sample in item.decode_step_wall_ms]
    kv_memory = max(item.peak_kv_memory_bytes for item in runs)
    peak_memory = max(item.peak_memory_allocated_bytes for item in runs)
    record = {
        "schema_version": 1,
        "benchmark": "llama_paged_decode",
        "architecture": "LlamaForCausalLM",
        "model_source": source,
        "number_of_layers": int(model.config.num_hidden_layers),
        "number_of_attention_heads": int(model.config.num_attention_heads),
        "number_of_kv_heads": int(model.config.num_key_value_heads),
        "hidden_size": int(model.config.hidden_size),
        "head_dim": int(model.config.head_dim),
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "device": str(device),
        "backend": args.backend,
        "append_backend": args.append_backend,
        "logits_max_error": correctness["logits_max_error"],
        "token_match_rate": correctness["token_match_rate"],
        "prefill_latency_ms": sum(item.prefill_wall_ms for item in runs) / len(runs),
        "decode_p50_ms": _percentile(decode_samples, 0.50),
        "decode_p95_ms": _percentile(decode_samples, 0.95),
        "ttft_ms": sum(item.ttft_wall_ms for item in runs) / len(runs),
        "tpot_ms": sum(item.tpot_wall_ms for item in runs) / len(runs),
        "peak_kv_memory_bytes": kv_memory,
        "peak_memory_allocated_bytes": peak_memory,
        "used_blocks_peak": max(item.used_blocks_peak for item in runs),
        "cache_released": all(item.cache_released for item in runs),
        "correctness": correctness,
        "config": {
            "block_size": args.block_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "rope_theta": float(model.config.rope_theta),
            "no_implicit_download": True,
            "oracle_path": "Transformers eager full-sequence reference",
        },
        "kernel_config": (
            paged_decode_autotune_config() if args.backend == "triton_paged" else None
        ),
        "environment": _environment(device),
        "limitations": [
            "Llama/Llama-like decoder only",
            "no quantization, tensor parallel, or streaming server",
            "random source is deterministic and does not download weights",
        ],
    }
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a real Transformers Llama over paged KV cache"
    )
    parser.add_argument("--model-source", choices=("random", "path"), default="random")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--backend", choices=("paged_reference", "paged_sdpa", "triton_paged"), default="paged_sdpa"
    )
    parser.add_argument("--append-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=704)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
