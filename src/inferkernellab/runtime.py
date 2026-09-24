from __future__ import annotations

from dataclasses import dataclass

from .cache import PagedKVCache
from .scheduler import InferenceRequest, ScheduleBatch, TokenBudgetScheduler


@dataclass(frozen=True)
class RuntimeStats:
    waiting: int
    running: int
    finished: int
    free_blocks: int
    used_blocks: int


class InferenceRuntime:
    """A model-independent runtime for replaying cache and scheduler behavior.

    The runtime intentionally does not generate model logits. It owns the
    resource lifecycle around a model: requests enter a queue, consume token
    budget, grow their KV block tables, and release blocks on completion.
    """

    def __init__(self, cache: PagedKVCache, scheduler: TokenBudgetScheduler):
        self.cache = cache
        self.scheduler = scheduler
        self._requests: dict[int, InferenceRequest] = {}

    def submit(self, request_id: int, prompt_tokens: int, max_new_tokens: int) -> InferenceRequest:
        if request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request_id}")
        request = InferenceRequest(request_id, prompt_tokens, max_new_tokens)
        request.block_table = self.cache.allocate_request(request_id, max(1, prompt_tokens))
        self._requests[request_id] = request
        self.scheduler.add(request)
        return request

    def _ensure_capacity(self, request: InferenceRequest, num_tokens: int) -> None:
        required = (num_tokens + self.cache.layout.block_size - 1) // self.cache.layout.block_size
        missing = required - len(request.block_table)
        if missing > 0:
            request.block_table.extend(self.cache.allocator.allocate(missing, owner=request.request_id))

    def step(self) -> ScheduleBatch:
        batch = self.scheduler.schedule()
        if batch.phase == "prefill":
            for request in batch.requests:
                self._ensure_capacity(request, max(1, request.cached_tokens))
        elif batch.phase == "decode":
            for request in batch.requests:
                # Reserve the KV slot that the next model step will write.
                self._ensure_capacity(request, request.prompt_tokens + request.generated_tokens + 1)
            self.scheduler.mark_decode_step(batch.requests)
            for request in batch.requests:
                if request.finished:
                    self.cache.release_request(request.block_table, request_id=request.request_id)
                    request.block_table = []
        return batch

    def stats(self) -> RuntimeStats:
        return RuntimeStats(
            waiting=len(self.scheduler.waiting),
            running=len(self.scheduler.running),
            finished=len(self.scheduler.finished),
            free_blocks=self.cache.allocator.num_free_blocks,
            used_blocks=self.cache.allocator.num_used_blocks,
        )
