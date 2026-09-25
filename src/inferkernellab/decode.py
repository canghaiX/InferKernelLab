from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .append import append_kv_triton, triton_append_available
from .attention import paged_decode_attention, paged_decode_attention_sdpa
from .cache import PagedKVCache
from .runtime import InferenceRuntime
from .scheduler import InferenceRequest, TokenBudgetScheduler
from .triton_ops import paged_decode_attention_triton, triton_available


_ATTENTION_BACKENDS = {"paged_reference", "paged_sdpa", "triton_paged"}
_APPEND_BACKENDS = {"torch", "triton"}


@dataclass(frozen=True)
class SyntheticDecoderConfig:
    vocab_size: int = 256
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 64
    block_size: int = 16
    dtype: torch.dtype = torch.float16
    device: torch.device | str = "cpu"
    seed: int = 0
    backend: str = "paged_reference"
    append_backend: str = "torch"
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    num_blocks: int | None = None

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        object.__setattr__(self, "device", device)
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if min(self.num_heads, self.num_kv_heads, self.head_dim, self.block_size) <= 0:
            raise ValueError("model and cache dimensions must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.backend not in _ATTENTION_BACKENDS:
            raise ValueError(f"unsupported attention backend: {self.backend}")
        if self.append_backend not in _APPEND_BACKENDS:
            raise ValueError(f"unsupported append backend: {self.append_backend}")
        if self.max_num_seqs <= 0 or self.max_num_batched_tokens <= 0:
            raise ValueError("scheduler limits must be positive")
        if self.num_blocks is not None and self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive when provided")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.backend == "triton_paged" and device.type != "cuda":
            raise ValueError("triton_paged requires a CUDA device")
        if self.append_backend == "triton" and device.type != "cuda":
            raise ValueError("triton append requires a CUDA device")


@dataclass(frozen=True)
class DecodeRunResult:
    backend: str
    append_backend: str
    prompt_lengths: tuple[int, ...]
    max_new_tokens: int
    generated_tokens: tuple[tuple[int, ...], ...]
    logits: tuple[torch.Tensor, ...]
    prefill_wall_ms: float
    decode_step_wall_ms: tuple[float, ...]
    ttft_wall_ms: float
    tpot_wall_ms: float
    tokens_per_sec: float
    end_to_end_tokens_per_sec: float
    prefill_device_ms: float
    decode_step_device_ms: tuple[float, ...]
    ttft_device_ms: float
    tpot_device_ms: float
    device_tokens_per_sec: float
    end_to_end_device_tokens_per_sec: float
    peak_memory_allocated_bytes: int
    used_blocks_peak: int

    @property
    def decode_step_p50_ms(self) -> float:
        return _percentile(self.decode_step_wall_ms, 0.50)

    @property
    def decode_step_p95_ms(self) -> float:
        return _percentile(self.decode_step_wall_ms, 0.95)

    @property
    def decode_step_device_p50_ms(self) -> float:
        return _percentile(self.decode_step_device_ms, 0.50)

    @property
    def decode_step_device_p95_ms(self) -> float:
        return _percentile(self.decode_step_device_ms, 0.95)


@dataclass(frozen=True)
class _SyntheticWeights:
    embedding: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    out_proj: torch.Tensor
    lm_head: torch.Tensor


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


class SyntheticDecodeRunner:
    """Run a deterministic single-layer decoder through the paged runtime."""

    def __init__(
        self,
        config: SyntheticDecoderConfig,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int = 16,
    ):
        if not prompts:
            raise ValueError("at least one prompt is required")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if len(prompts) > config.max_num_seqs:
            raise ValueError("the scheduler max_num_seqs is smaller than the batch")

        self.config = config
        self.device = config.device
        self.max_new_tokens = max_new_tokens
        self.prompts = tuple(tuple(int(token) for token in prompt) for prompt in prompts)
        if any(not prompt for prompt in self.prompts):
            raise ValueError("prompts must be non-empty")
        if any(token < 0 or token >= config.vocab_size for prompt in self.prompts for token in prompt):
            raise ValueError("prompt token is outside the configured vocabulary")
        self.prompt_lengths = tuple(len(prompt) for prompt in self.prompts)

        self.weights = self._make_weights()
        initial_block_counts = tuple(
            math.ceil(length / config.block_size) for length in self.prompt_lengths
        )
        total_blocks = sum(
            math.ceil((length + max_new_tokens) / config.block_size)
            for length in self.prompt_lengths
        )
        reserved_blocks = max(0, max(initial_block_counts, default=0) - 1)
        minimum_blocks = total_blocks + reserved_blocks
        num_blocks = config.num_blocks if config.num_blocks is not None else minimum_blocks
        if num_blocks < minimum_blocks:
            raise ValueError(
                f"num_blocks is insufficient: requested={num_blocks}, minimum={minimum_blocks}"
            )

        self.cache = PagedKVCache(
            num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
            dtype=config.dtype,
            device=self.device,
        )
        scheduler = TokenBudgetScheduler(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
        )
        self.runtime = InferenceRuntime(
            self.cache,
            scheduler,
            prefill_fn=self._prefill,
            decode_fn=self._decode,
        )
        self.requests: list[InferenceRequest] = [
            self.runtime.submit(request_id, len(prompt), max_new_tokens)
            for request_id, prompt in enumerate(self.prompts)
        ]
        self._reserved_blocks: list[int] = []
        self._repack_initial_block_tables(initial_block_counts)

        self._last_tokens: dict[int, torch.Tensor] = {}
        self._generated_token_tensors: dict[int, list[torch.Tensor]] = {
            request.request_id: [] for request in self.requests
        }
        self._step_logits: list[torch.Tensor] = []
        self._has_run = False

    def _repack_initial_block_tables(self, block_counts: Sequence[int]) -> None:
        initial_tables = [list(request.block_table) for request in self.requests]
        for request, table in zip(self.requests, initial_tables):
            self.cache.release_request(table, request_id=request.request_id)
            request.block_table = []

        max_blocks = max(block_counts, default=0)
        for logical_block in range(max_blocks):
            for request, block_count in zip(self.requests, block_counts):
                if logical_block < block_count:
                    request.block_table.extend(
                        self.cache.allocator.allocate(1, owner=request.request_id)
                    )
            if logical_block + 1 < max_blocks:
                self._reserved_blocks.extend(self.cache.allocator.allocate(1, owner=-1))

    def _make_weights(self) -> _SyntheticWeights:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        hidden_size = self.config.num_heads * self.config.head_dim
        kv_size = self.config.num_kv_heads * self.config.head_dim

        def random_weight(shape: tuple[int, ...], scale: float) -> torch.Tensor:
            value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
            return value.to(device=self.device, dtype=self.config.dtype)

        hidden_scale = hidden_size ** -0.5
        return _SyntheticWeights(
            embedding=random_weight((self.config.vocab_size, hidden_size), 0.02),
            q_proj=random_weight((hidden_size, hidden_size), hidden_scale),
            k_proj=random_weight((kv_size, hidden_size), hidden_scale),
            v_proj=random_weight((kv_size, hidden_size), hidden_scale),
            out_proj=random_weight((hidden_size, hidden_size), hidden_scale),
            lm_head=random_weight((self.config.vocab_size, hidden_size), hidden_scale),
        )

    def _project(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.embedding(token_ids.to(dtype=torch.long), self.weights.embedding)
        query = F.linear(hidden, self.weights.q_proj)
        key = F.linear(hidden, self.weights.k_proj)
        value = F.linear(hidden, self.weights.v_proj)
        return (
            query.reshape(-1, self.config.num_heads, self.config.head_dim),
            key.reshape(-1, self.config.num_kv_heads, self.config.head_dim),
            value.reshape(-1, self.config.num_kv_heads, self.config.head_dim),
        )

    def _project_kv(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.embedding(token_ids.to(dtype=torch.long), self.weights.embedding)
        key = F.linear(hidden, self.weights.k_proj)
        value = F.linear(hidden, self.weights.v_proj)
        return (
            key.reshape(-1, self.config.num_kv_heads, self.config.head_dim),
            value.reshape(-1, self.config.num_kv_heads, self.config.head_dim),
        )

    def _prefill(self, request: InferenceRequest, start_token: int, end_token: int) -> None:
        token_ids = torch.tensor(
            self.prompts[request.request_id][start_token:end_token],
            dtype=torch.long,
            device=self.device,
        )
        key, value = self._project_kv(token_ids)
        self._append_to_cache(request, start_token, key, value)

    def _block_tables(self, requests: Sequence[InferenceRequest]) -> torch.Tensor:
        width = max(1, max(len(request.block_table) for request in requests))
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
    ) -> torch.Tensor:
        tables = [request.block_table for request in requests]
        if self.config.backend == "paged_reference":
            return paged_decode_attention(query, self.cache, tables, context_lengths)
        if self.config.backend == "paged_sdpa":
            return paged_decode_attention_sdpa(query, self.cache, tables, context_lengths)
        if not triton_available():
            raise RuntimeError("triton_paged requires a CUDA device and an installed Triton backend")
        table_tensor = self._block_tables(requests)
        length_tensor = torch.tensor(context_lengths, dtype=torch.int32, device=self.device)
        return paged_decode_attention_triton(
            query,
            self.cache,
            table_tensor,
            length_tensor,
            validate_inputs=True,
        )

    def _append_to_cache(
        self,
        request: InferenceRequest,
        start_token: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self.config.append_backend == "torch":
            self.cache.write(0, request.block_table, start_token, key, value)
            return
        if not triton_append_available():
            raise RuntimeError("triton append requires a CUDA device and an installed Triton backend")
        slots = self.cache.slot_mapping(request.block_table, start_token, key.shape[0]).to(torch.int32)
        append_kv_triton(key, value, self.cache, slots)

    def _append_batch(
        self,
        requests: Sequence[InferenceRequest],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self.config.append_backend == "torch":
            for row, request in enumerate(requests):
                position = request.prompt_tokens + request.generated_tokens
                self._append_to_cache(request, position, key[row : row + 1], value[row : row + 1])
            return
        if not triton_append_available():
            raise RuntimeError("triton append requires a CUDA device and an installed Triton backend")
        slots = torch.cat(
            [
                self.cache.slot_mapping(
                    request.block_table,
                    request.prompt_tokens + request.generated_tokens,
                    1,
                )
                for request in requests
            ]
        ).to(torch.int32)
        append_kv_triton(key, value, self.cache, slots)

    def _decode(self, requests: tuple[InferenceRequest, ...]) -> None:
        current_ids = []
        for request in requests:
            if request.generated_tokens == 0:
                current_ids.append(self.prompts[request.request_id][-1])
            else:
                current_ids.append(self._last_tokens[request.request_id])
        if any(isinstance(token, torch.Tensor) for token in current_ids):
            current_tensor = torch.stack([
                token if isinstance(token, torch.Tensor) else torch.tensor(token, device=self.device)
                for token in current_ids
            ]).to(dtype=torch.long)
        else:
            current_tensor = torch.tensor(current_ids, dtype=torch.long, device=self.device)

        query, _, _ = self._project(current_tensor)
        context_lengths = [request.prompt_tokens + request.generated_tokens for request in requests]
        attended = self._attention(query, requests, context_lengths)
        hidden = F.linear(attended.reshape(attended.shape[0], -1), self.weights.out_proj)
        logits = F.linear(hidden, self.weights.lm_head)
        next_tokens = logits.argmax(dim=-1)
        _, next_key, next_value = self._project(next_tokens)
        self._append_batch(requests, next_key, next_value)

        self._step_logits.append(logits.detach().float().cpu())
        for row, request in enumerate(requests):
            token = next_tokens[row]
            self._last_tokens[request.request_id] = token
            self._generated_token_tensors[request.request_id].append(token)

    def _timed_step(self) -> tuple[object, float, float]:
        wall_start = time.perf_counter()
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            batch = self.runtime.step()
            end.record()
            end.synchronize()
            device_ms = float(start.elapsed_time(end))
        else:
            batch = self.runtime.step()
            device_ms = 0.0
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        if self.device.type != "cuda":
            device_ms = wall_ms
        return batch, wall_ms, device_ms

    def _release_reserved_blocks(self) -> None:
        if self._reserved_blocks:
            self.cache.allocator.free(self._reserved_blocks, owner=-1)
            self._reserved_blocks = []

    def run(self) -> DecodeRunResult:
        if self._has_run:
            raise RuntimeError("a SyntheticDecodeRunner instance can only be run once")
        self._has_run = True
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        prefill_wall_ms = 0.0
        prefill_device_ms = 0.0
        decode_wall_ms: list[float] = []
        decode_device_ms: list[float] = []
        used_blocks_peak = self.cache.allocator.num_used_blocks
        try:
            with torch.inference_mode():
                while self.runtime.scheduler.waiting or self.runtime.scheduler.running:
                    batch, wall_ms, device_ms = self._timed_step()
                    used_blocks_peak = max(used_blocks_peak, self.cache.allocator.num_used_blocks)
                    if batch.phase == "prefill":
                        prefill_wall_ms += wall_ms
                        prefill_device_ms += device_ms
                    elif batch.phase == "decode":
                        decode_wall_ms.append(wall_ms)
                        decode_device_ms.append(device_ms)
                    elif batch.phase == "idle":
                        raise RuntimeError("scheduler became idle before all requests finished")
        finally:
            self._release_reserved_blocks()

        generated_tokens = tuple(
            tuple(int(token) for token in torch.stack(self._generated_token_tensors[request.request_id]).cpu().tolist())
            for request in self.requests
        )
        expected_tokens = len(self.requests) * self.max_new_tokens
        if sum(len(tokens) for tokens in generated_tokens) != expected_tokens:
            raise RuntimeError("decoder finished with an unexpected number of generated tokens")
        if not decode_wall_ms:
            raise RuntimeError("decoder produced no decode steps")

        total_wall_ms = prefill_wall_ms + sum(decode_wall_ms)
        total_device_ms = prefill_device_ms + sum(decode_device_ms)
        tpot_wall_ms = sum(decode_wall_ms) / len(decode_wall_ms)
        tpot_device_ms = sum(decode_device_ms) / len(decode_device_ms)
        generated_count = float(expected_tokens)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0
        )
        return DecodeRunResult(
            backend=self.config.backend,
            append_backend=self.config.append_backend,
            prompt_lengths=self.prompt_lengths,
            max_new_tokens=self.max_new_tokens,
            generated_tokens=generated_tokens,
            logits=tuple(self._step_logits),
            prefill_wall_ms=prefill_wall_ms,
            decode_step_wall_ms=tuple(decode_wall_ms),
            ttft_wall_ms=prefill_wall_ms + decode_wall_ms[0],
            tpot_wall_ms=tpot_wall_ms,
            tokens_per_sec=generated_count / (sum(decode_wall_ms) / 1000.0),
            end_to_end_tokens_per_sec=generated_count / (total_wall_ms / 1000.0),
            prefill_device_ms=prefill_device_ms,
            decode_step_device_ms=tuple(decode_device_ms),
            ttft_device_ms=prefill_device_ms + decode_device_ms[0],
            tpot_device_ms=tpot_device_ms,
            device_tokens_per_sec=generated_count / (sum(decode_device_ms) / 1000.0),
            end_to_end_device_tokens_per_sec=generated_count / (total_device_ms / 1000.0),
            peak_memory_allocated_bytes=peak_memory,
            used_blocks_peak=used_blocks_peak,
        )
