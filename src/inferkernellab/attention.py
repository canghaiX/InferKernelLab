from __future__ import annotations

import torch
import torch.nn.functional as F

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


def dense_decode_attention_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense contiguous-KV baseline using PyTorch scaled-dot-product attention."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query must be [B,H,D], key/value must be [T,Hkv,D]")
    if key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("incompatible attention shapes")
    if min(query.shape) <= 0 or key.shape[0] == 0 or key.shape[1] == 0:
        raise ValueError("attention batch, heads, dimensions, and context must be non-empty")
    if query.device != key.device or key.device != value.device:
        raise ValueError("query/key/value must be on the same device")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("query/key/value must have the same dtype")

    key_heads = key.permute(1, 0, 2).unsqueeze(0)
    value_heads = value.permute(1, 0, 2).unsqueeze(0)
    attended = _call_sdpa(query.unsqueeze(2), key_heads, value_heads, scale)
    return attended.squeeze(2)


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


def dense_decode_attention_sdpa_batch(
    query: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Run the SDPA baseline for a batch with potentially different KV lengths."""
    if query.ndim != 3 or len(keys) != query.shape[0] or len(values) != query.shape[0]:
        raise ValueError("one key/value tensor is required per query")
    if not keys:
        raise ValueError("the attention batch must be non-empty")
    if len({key.shape[0] for key in keys}) == 1:
        return dense_decode_attention_sdpa_tensor_batch(
            query, torch.stack(keys), torch.stack(values), scale=scale
        )
    return torch.stack([
        dense_decode_attention_sdpa(query[i:i + 1], keys[i], values[i], scale=scale)[0]
        for i in range(query.shape[0])
    ])


def dense_decode_attention_sdpa_tensor_batch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Run SDPA on pre-stacked dense KV with shape [B,T,Hkv,D]."""
    if query.ndim != 3 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query must be [B,H,D], key/value must be [B,T,Hkv,D]")
    if (
        min(query.shape) <= 0
        or key.shape != value.shape
        or key.shape[0] != query.shape[0]
        or min(key.shape[1:]) <= 0
    ):
        raise ValueError("incompatible attention shapes")
    if query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[2] != 0:
        raise ValueError("incompatible query and KV head dimensions")
    if query.device != key.device or key.device != value.device:
        raise ValueError("query/key/value must be on the same device")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("query/key/value must have the same dtype")
    attended = _call_sdpa(
        query.unsqueeze(2),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        scale,
    )
    return attended.squeeze(2)


def paged_decode_attention(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: list[list[int]] | torch.Tensor,
    context_lens: list[int] | torch.Tensor,
    *,
    scale: float | None = None,
    layer: int = 0,
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
    if num_heads % cache.layout.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if query.device != cache.device:
        raise ValueError("query and cache must be on the same device")
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
        key, value = cache.read(layer, list(tables[index]), int(length))
        outputs.append(dense_decode_attention(query[index:index + 1], key, value, scale=scale)[0])
    return torch.stack(outputs, dim=0)


def paged_decode_attention_sdpa(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: list[list[int]] | torch.Tensor,
    context_lens: list[int] | torch.Tensor,
    *,
    scale: float | None = None,
    layer: int = 0,
) -> torch.Tensor:
    """Gather paged KV into contiguous tensors, then call PyTorch SDPA."""
    if query.ndim != 3:
        raise ValueError("query must be [B,H,D]")
    if query.shape[-1] != cache.layout.head_dim:
        raise ValueError("query head_dim does not match cache")
    if query.shape[1] % cache.layout.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if query.device != cache.device:
        raise ValueError("query and cache must be on the same device")
    lengths = context_lens.detach().cpu().tolist() if isinstance(context_lens, torch.Tensor) else list(context_lens)
    if len(lengths) != query.shape[0] or len(block_tables) != query.shape[0]:
        raise ValueError("block_tables and context_lens must have one entry per query")

    if lengths and all(int(length) == int(lengths[0]) for length in lengths):
        context_len = int(lengths[0])
        if context_len <= 0:
            raise ValueError("context lengths must be positive")
        table_tensor = torch.as_tensor(block_tables, dtype=torch.long, device=cache.device)
        if table_tensor.ndim != 2:
            raise ValueError("block_tables must have shape [B, max_blocks]")
        required_blocks = (context_len + cache.layout.block_size - 1) // cache.layout.block_size
        if required_blocks > table_tensor.shape[1]:
            raise ValueError("context length exceeds block table capacity")
        positions = torch.arange(context_len, dtype=torch.long, device=cache.device)
        logical_blocks = positions // cache.layout.block_size
        offsets = positions.remainder(cache.layout.block_size)
        physical_blocks = table_tensor[:, logical_blocks]
        key = cache.k[layer, physical_blocks, offsets.unsqueeze(0)]
        value = cache.v[layer, physical_blocks, offsets.unsqueeze(0)]
        return dense_decode_attention_sdpa_tensor_batch(query, key, value, scale=scale)

    keys = []
    values = []
    tables = block_tables.detach().cpu().tolist() if isinstance(block_tables, torch.Tensor) else block_tables
    for index, length in enumerate(lengths):
        if int(length) <= 0:
            raise ValueError("context lengths must be positive")
        key, value = cache.read(layer, list(tables[index]), int(length))
        keys.append(key)
        values.append(value)
    return dense_decode_attention_sdpa_batch(query, keys, values, scale=scale)


def paged_prefill_attention_sdpa(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: list[list[int]] | torch.Tensor,
    query_starts: list[int] | torch.Tensor,
    context_lens: list[int] | torch.Tensor,
    *,
    scale: float | None = None,
    layer: int = 0,
) -> torch.Tensor:
    """Run causal SDPA for a ragged token batch backed by paged KV.

    ``query`` has shape ``[B, H, T, D]``.  Each row contains a contiguous
    prefill range ``[query_starts[i], context_lens[i])``; padded query rows are
    ignored by the causal mask.  The cache is expected to contain the complete
    range before this function is called.
    """
    if query.ndim != 4:
        raise ValueError("query must be [B,H,T,D]")
    batch, num_heads, query_width, head_dim = query.shape
    if head_dim != cache.layout.head_dim:
        raise ValueError("query head_dim does not match cache")
    if num_heads % cache.layout.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if query.device != cache.device:
        raise ValueError("query and cache must be on the same device")

    starts_list = (
        query_starts.detach().cpu().tolist()
        if isinstance(query_starts, torch.Tensor)
        else list(query_starts)
    )
    lengths_list = (
        context_lens.detach().cpu().tolist()
        if isinstance(context_lens, torch.Tensor)
        else list(context_lens)
    )
    if len(starts_list) != batch or len(lengths_list) != batch:
        raise ValueError("query_starts and context_lens must have one entry per query batch")
    local_lengths = [int(end) - int(start) for start, end in zip(starts_list, lengths_list)]
    if any(start < 0 or length <= 0 for start, length in zip(starts_list, local_lengths)):
        raise ValueError("prefill ranges must be non-empty and non-negative")
    if max(local_lengths) > query_width:
        raise ValueError("query width is smaller than a prefill range")
    if not 0 <= layer < cache.layout.num_layers:
        raise IndexError("layer is out of range")

    if isinstance(block_tables, torch.Tensor):
        table_tensor = block_tables.to(device=cache.device, dtype=torch.long)
    else:
        table_tensor = torch.as_tensor(block_tables, dtype=torch.long, device=cache.device)
    if table_tensor.ndim != 2 or table_tensor.shape[0] != batch:
        raise ValueError("block_tables must have one row per query batch")
    max_context = max(int(length) for length in lengths_list)
    if max_context <= 0:
        raise ValueError("context lengths must be positive")
    required_blocks = (max_context + cache.layout.block_size - 1) // cache.layout.block_size
    if required_blocks > table_tensor.shape[1]:
        raise ValueError("context length exceeds block table capacity")

    key_positions = torch.arange(max_context, dtype=torch.long, device=cache.device)
    logical_blocks = key_positions // cache.layout.block_size
    offsets = key_positions.remainder(cache.layout.block_size)
    physical_blocks = table_tensor[:, logical_blocks]
    key = cache.k[layer, physical_blocks, offsets.unsqueeze(0)]
    value = cache.v[layer, physical_blocks, offsets.unsqueeze(0)]

    query_positions = torch.arange(query_width, dtype=torch.long, device=cache.device)
    start_tensor = torch.as_tensor(starts_list, dtype=torch.long, device=cache.device)
    length_tensor = torch.as_tensor(lengths_list, dtype=torch.long, device=cache.device)
    key_length_mask = key_positions.view(1, 1, -1) < length_tensor.view(-1, 1, 1)
    valid_query = query_positions.view(1, -1, 1) < (
        length_tensor - start_tensor
    ).view(-1, 1, 1)
    causal = key_positions.view(1, 1, -1) <= (
        start_tensor.view(-1, 1, 1) + query_positions.view(1, -1, 1)
    )
    allow = valid_query & key_length_mask & causal
    allow = allow | ((~valid_query) & (key_positions.view(1, 1, -1) == 0))
    attended = _call_sdpa(
        query,
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        scale,
        attn_mask=allow.unsqueeze(1),
    )
    return attended


def _call_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    *,
    attn_mask: torch.Tensor | None = None,
):
    if scale is not None:
        query = query * (scale * query.shape[-1] ** 0.5)
    if query.shape[1] == key.shape[1]:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            enable_gqa=True,
        )
    except TypeError:
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
