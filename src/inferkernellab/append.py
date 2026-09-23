from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def triton_append_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _append_kv_kernel(
        key_ptr,
        value_ptr,
        slot_ptr,
        k_ptr,
        v_ptr,
        stride_key_t,
        stride_key_h,
        stride_value_t,
        stride_value_h,
        stride_k_block,
        stride_k_token,
        stride_k_head,
        stride_v_block,
        stride_v_token,
        stride_v_head,
        head_dim,
        block_size: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offs_d = tl.arange(0, block_d)
        mask = offs_d < head_dim
        slot = tl.load(slot_ptr + token)
        block = slot // block_size
        offset = slot % block_size
        key = tl.load(
            key_ptr + token * stride_key_t + head * stride_key_h + offs_d,
            mask=mask,
            other=0.0,
        )
        value = tl.load(
            value_ptr + token * stride_value_t + head * stride_value_h + offs_d,
            mask=mask,
            other=0.0,
        )
        tl.store(
            k_ptr + block * stride_k_block + offset * stride_k_token + head * stride_k_head + offs_d,
            key,
            mask=mask,
        )
        tl.store(
            v_ptr + block * stride_v_block + offset * stride_v_token + head * stride_v_head + offs_d,
            value,
            mask=mask,
        )


def append_kv_triton(key: torch.Tensor, value: torch.Tensor, cache, slots: torch.Tensor, layer: int = 0) -> None:
    """Append [tokens, kv_heads, head_dim] into paged cache slots."""
    if not triton_append_available():
        raise RuntimeError("Triton CUDA backend is unavailable")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must have shape [T, Hkv, D]")
    if key.device.type != "cuda" or slots.device != key.device:
        raise ValueError("key/value/slots must be on the same CUDA device")
    layout = cache.layout
    if key.shape[1:] != (layout.num_kv_heads, layout.head_dim):
        raise ValueError("key/value shape does not match cache layout")
    if slots.numel() != key.shape[0]:
        raise ValueError("slots and key/value token counts must match")
    if not 0 <= layer < layout.num_layers:
        raise IndexError("layer is out of range")
    block_d = triton.next_power_of_2(layout.head_dim)
    k_cache = cache.k[layer]
    v_cache = cache.v[layer]
    _append_kv_kernel[(key.shape[0], layout.num_kv_heads)](
        key,
        value,
        slots,
        k_cache,
        v_cache,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        layout.head_dim,
        block_size=layout.block_size,
        block_d=block_d,
    )

