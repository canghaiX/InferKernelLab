from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton installed
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        table_ptr,
        len_ptr,
        out_ptr,
        stride_q_b,
        stride_q_h,
        stride_k_b,
        stride_k_t,
        stride_k_h,
        stride_v_b,
        stride_v_t,
        stride_v_h,
        stride_table_b,
        stride_out_b,
        stride_out_h,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        scale,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offs_d = tl.arange(0, block_d)
        mask_d = offs_d < head_dim
        q = tl.load(q_ptr + batch_id * stride_q_b + head_id * stride_q_h + offs_d, mask=mask_d, other=0.0)
        context_len = tl.load(len_ptr + batch_id)
        kv_head = head_id * num_kv_heads // num_heads
        max_score = -float("inf")
        denom = 0.0
        acc = tl.zeros((block_d,), dtype=tl.float32)
        offs_n = tl.arange(0, block_n)
        for start_n in tl.range(0, context_len, block_n, num_stages=2):
            positions = start_n + offs_n
            active = positions < context_len
            logical_blocks = positions // block_size
            offsets = positions - logical_blocks * block_size
            physical_blocks = tl.load(
                table_ptr + batch_id * stride_table_b + logical_blocks,
                mask=active,
                other=0,
            )
            k = tl.load(
                k_ptr
                + physical_blocks[:, None] * stride_k_b
                + offsets[:, None] * stride_k_t
                + kv_head * stride_k_h
                + offs_d[None, :],
                mask=active[:, None] & mask_d[None, :],
                other=0.0,
            )
            v = tl.load(
                v_ptr
                + physical_blocks[:, None] * stride_v_b
                + offsets[:, None] * stride_v_t
                + kv_head * stride_v_h
                + offs_d[None, :],
                mask=active[:, None] & mask_d[None, :],
                other=0.0,
            )
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(active, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            new_max = tl.maximum(max_score, tile_max)
            alpha = tl.exp(max_score - new_max)
            probabilities = tl.exp(scores - new_max)
            acc = acc * alpha + tl.sum(probabilities[:, None] * v, axis=0)
            denom = denom * alpha + tl.sum(probabilities, axis=0)
            max_score = new_max
        out = acc / denom
        tl.store(out_ptr + batch_id * stride_out_b + head_id * stride_out_h + offs_d, out, mask=mask_d)


def paged_decode_attention_triton(query, cache, block_tables, context_lens, scale=None, layer=0):
    """Run the optional Triton kernel; imports are kept optional for CPU tests."""
    if not triton_available():
        raise RuntimeError("Triton CUDA backend is unavailable")
    from .attention import paged_decode_attention

    if query.ndim != 3 or query.device.type != "cuda":
        raise ValueError("query must be a CUDA tensor with shape [B,H,D]")
    if isinstance(block_tables, list):
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=query.device)
    if isinstance(context_lens, list):
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device=query.device)
    output = torch.empty_like(query)
    layout = cache.layout
    if not 0 <= layer < layout.num_layers:
        raise IndexError("layer is out of range")
    if layout.head_dim > 128:
        raise ValueError("the current Triton kernel supports head_dim <= 128")
    k_cache = cache.k[layer]
    v_cache = cache.v[layer]
    block_d = triton.next_power_of_2(layout.head_dim)
    grid = (query.shape[0], query.shape[1])
    _paged_decode_kernel[grid](
        query, k_cache, v_cache, block_tables, context_lens, output,
        query.stride(0), query.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0), output.stride(0), output.stride(1),
        query.shape[1], layout.num_kv_heads, layout.head_dim,
        block_size=layout.block_size,
        block_n=64,
        block_d=block_d,
        scale=scale if scale is not None else layout.head_dim ** -0.5,
        num_warps=4,
    )
    return output
