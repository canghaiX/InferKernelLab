from __future__ import annotations

import torch

from .cache import PagedKVCache


def _expand_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Expand MQA/GQA heads without changing the reference cache layout."""
    if x.shape[1] == num_heads:
        return x
    if num_heads % x.shape[1] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    return x.repeat_interleave(num_heads // x.shape[1], dim=1)


def dense_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference attention for [batch, heads, dim] and [tokens, kv_heads, dim]."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query must be [B,H,D], key/value must be [T,Hkv,D]")
    if key.shape != value.shape or query.shape[2] != key.shape[2]:
        raise ValueError("incompatible attention shapes")
    q = query.float()
    k = _expand_kv(key, query.shape[1]).float().transpose(0, 1)
    v = _expand_kv(value, query.shape[1]).float().transpose(0, 1)
    scores = torch.einsum("bhd,htd->bht", q, k)
    scores = scores * (scale if scale is not None else query.shape[-1] ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,htd->bhd", probs, v).to(query.dtype)


def dense_decode_attention_batch(
    query: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense contiguous-KV baseline for variable-length request batches."""
    if len(keys) != query.shape[0] or len(values) != query.shape[0]:
        raise ValueError("one key/value tensor is required per query")
    return torch.stack([
        dense_decode_attention(query[i:i + 1], keys[i], values[i], scale=scale)[0]
        for i in range(query.shape[0])
    ])


def paged_decode_attention(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: list[list[int]] | torch.Tensor,
    context_lens: list[int] | torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Correctness-first Paged Decode Attention implementation.

    The explicit batch loop is deliberate. It is a reference for the Triton
    kernel and makes the logical-to-physical block mapping easy to inspect.
    """
    if query.ndim != 3:
        raise ValueError("query must be [B,H,D]")
    batch, num_heads, head_dim = query.shape
    if head_dim != cache.layout.head_dim:
        raise ValueError("query head_dim does not match cache")
    if isinstance(context_lens, torch.Tensor):
        lengths = context_lens.detach().cpu().tolist()
    else:
        lengths = list(context_lens)
    if len(lengths) != batch:
        raise ValueError("context_lens must have one entry per query")
    if isinstance(block_tables, torch.Tensor):
        tables = block_tables.detach().cpu().tolist()
    else:
        tables = block_tables
    if len(tables) != batch:
        raise ValueError("block_tables must have one row per query")
    outputs = []
    for index, length in enumerate(lengths):
        if length <= 0:
            raise ValueError("context lengths must be positive")
        key, value = cache.read(0, list(tables[index]), int(length))
        outputs.append(dense_decode_attention(query[index:index + 1], key, value, scale=scale)[0])
    return torch.stack(outputs, dim=0)
