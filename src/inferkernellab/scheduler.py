from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class InferenceRequest:
    request_id: int
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    cached_tokens: int = 0
    status: str = "waiting"
    block_table: list[int] = field(default_factory=list)

    @property
    def prompt_remaining(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def finished(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens


@dataclass(frozen=True)
class ScheduleBatch:
    requests: tuple[InferenceRequest, ...]
    phase: str
    scheduled_tokens: int


class TokenBudgetScheduler:
    """Small FCFS scheduler demonstrating prefill/decode token accounting."""

    def __init__(self, max_num_seqs: int = 32, max_num_batched_tokens: int = 2048):
        if max_num_seqs <= 0 or max_num_batched_tokens <= 0:
            raise ValueError("scheduler limits must be positive")
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[InferenceRequest] = deque()
        self.running: list[InferenceRequest] = []
        self.finished: list[InferenceRequest] = []

    def add(self, request: InferenceRequest) -> None:
        if request.status != "waiting":
            raise ValueError("only waiting requests can be added")
        self.waiting.append(request)

    def schedule(self) -> ScheduleBatch:
        prefill = []
        tokens = 0
        while self.waiting and len(prefill) < self.max_num_seqs:
            request = self.waiting[0]
            remaining = self.max_num_batched_tokens - tokens
            if remaining <= 0:
                break
            needed = request.prompt_remaining
            if needed == 0:
                self.waiting.popleft()
                request.status = "running"
                self.running.append(request)
                continue
            if prefill and needed > remaining:
                break
            self.waiting.popleft()
            scheduled = min(needed, remaining)
            request.cached_tokens += scheduled
            tokens += scheduled
            request.status = "running" if request.prompt_remaining == 0 else "waiting"
            prefill.append(request)
            if request.status == "running":
                self.running.append(request)
            else:
                # Preserve FCFS order when a long prompt is split across
                # multiple token-budget windows.
                self.waiting.appendleft(request)
        if prefill:
            return ScheduleBatch(tuple(prefill), "prefill", tokens)

        decode = tuple(self.running[: self.max_num_seqs])
        if not decode:
            return ScheduleBatch((), "idle", 0)
        return ScheduleBatch(decode, "decode", len(decode))

    def mark_decode_step(self, requests: tuple[InferenceRequest, ...]) -> None:
        active = set(id(request) for request in requests)
        for request in self.running[:]:
            if id(request) not in active:
                continue
            request.generated_tokens += 1
            if request.finished:
                request.status = "finished"
                self.running.remove(request)
                self.finished.append(request)
