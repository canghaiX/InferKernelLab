"""Small reference implementations for LLM inference optimization."""

from .attention import (
    dense_decode_attention,
    dense_decode_attention_batch,
    dense_decode_attention_sdpa,
    dense_decode_attention_sdpa_batch,
    dense_decode_attention_sdpa_tensor_batch,
    paged_decode_attention,
    paged_decode_attention_sdpa,
)
from .append import append_kv_triton
from .cache import BlockAllocator, PagedKVCache
from .decode import DecodeRunResult, SyntheticDecodeRunner, SyntheticDecoderConfig
from .runtime import InferenceRuntime, RuntimeStats
from .scheduler import InferenceRequest, TokenBudgetScheduler

__all__ = [
    "BlockAllocator",
    "InferenceRequest",
    "PagedKVCache",
    "SyntheticDecoderConfig",
    "SyntheticDecodeRunner",
    "DecodeRunResult",
    "InferenceRuntime",
    "RuntimeStats",
    "TokenBudgetScheduler",
    "paged_decode_attention",
    "dense_decode_attention",
    "dense_decode_attention_batch",
    "dense_decode_attention_sdpa",
    "dense_decode_attention_sdpa_batch",
    "dense_decode_attention_sdpa_tensor_batch",
    "paged_decode_attention_sdpa",
    "append_kv_triton",
]
