import pytest
import torch

from inferkernellab.attention import (
    dense_decode_attention,
    dense_decode_attention_batch,
    dense_decode_attention_sdpa_batch,
    paged_decode_attention,
    paged_decode_attention_sdpa,
)
from inferkernellab.cache import PagedKVCache
from inferkernellab.append import append_kv_triton, triton_append_available
from inferkernellab.triton_ops import (
    paged_decode_attention_triton,
    paged_decode_attention_triton_grouped,
    triton_available,
)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (4, 2), (4, 1)])
def test_paged_attention_matches_dense(num_heads, num_kv_heads):
    torch.manual_seed(0)
    cache = PagedKVCache(16, 4, num_kv_heads, 8, dtype=torch.float32)
    tables = [cache.allocate_request(i, 7 + i) for i in range(2)]
    query = torch.randn(2, num_heads, 8)
    outputs = []
    for i, table in enumerate(tables):
        length = 7 + i
        key = torch.randn(length, num_kv_heads, 8)
        value = torch.randn_like(key)
        cache.write(0, table, 0, key, value)
        outputs.append(dense_decode_attention(query[i:i + 1], key, value)[0])
    actual = paged_decode_attention(query, cache, tables, [7, 8])
    expected = torch.stack(outputs)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sdpa_baselines_match_reference_with_ragged_paged_kv(num_kv_heads, dtype):
    torch.manual_seed(3)
    block_size = 4
    cache = PagedKVCache(16, block_size, num_kv_heads, 16, dtype=dtype)
    tables = [[11, 3], [7, 1]]
    lengths = [5, 7]
    query = torch.randn(2, 4, 16, dtype=dtype)
    keys = []
    values = []
    for table, length in zip(tables, lengths):
        key = torch.randn(length, num_kv_heads, 16, dtype=dtype)
        value = torch.randn_like(key)
        cache.write(0, table, 0, key, value)
        keys.append(key)
        values.append(value)

    expected = dense_decode_attention_batch(query, keys, values)
    dense_output = dense_decode_attention_sdpa_batch(query, keys, values)
    paged_output = paged_decode_attention_sdpa(query, cache, tables, lengths)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(dense_output, expected, atol=tolerance, rtol=tolerance)
    assert torch.allclose(paged_output, expected, atol=tolerance, rtol=tolerance)
    assert torch.allclose(paged_decode_attention(query, cache, tables, lengths), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not triton_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_attention_matches_reference(num_kv_heads, dtype):
    torch.manual_seed(1)
    cache = PagedKVCache(16, 8, num_kv_heads, 16, dtype=dtype, device="cuda")
    tables = [[13, 2, 9], [5, 14, 1]]
    lengths = [19, 23]
    for table, length in zip(tables, lengths):
        key = torch.randn(length, num_kv_heads, 16, device="cuda", dtype=dtype)
        value = torch.randn_like(key)
        cache.write(0, table, 0, key, value)
    query = torch.randn(2, 4, 16, device="cuda", dtype=dtype)
    expected = paged_decode_attention(query, cache, tables, lengths)
    block_tables = torch.tensor(tables, dtype=torch.int32, device="cuda")
    context_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    actual = paged_decode_attention_triton(query, cache, block_tables, context_lens)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 1e-4
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not triton_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("num_kv_heads", [8, 4, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_triton_attention_matches_reference(num_kv_heads, dtype):
    torch.manual_seed(7)
    cache = PagedKVCache(
        32,
        8,
        num_kv_heads,
        16,
        num_layers=2,
        dtype=dtype,
        device="cuda",
    )
    tables = [[13, 2, 9], [5, 14, 1]]
    lengths = [19, 23]
    for table, length in zip(tables, lengths):
        key = torch.randn(length, num_kv_heads, 16, device="cuda", dtype=dtype)
        value = torch.randn_like(key)
        cache.write(1, table, 0, key, value)
    query = torch.randn(2, 32, 16, device="cuda", dtype=dtype)
    block_tables = torch.tensor(tables, dtype=torch.int32, device="cuda")
    context_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    expected = paged_decode_attention(query, cache, tables, lengths, layer=1)
    actual = paged_decode_attention_triton_grouped(
        query,
        cache,
        block_tables,
        context_lens,
        layer=1,
        query_group_size=8,
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not triton_available(), reason="requires CUDA and Triton")
def test_triton_attention_rejects_invalid_metadata():
    cache = PagedKVCache(8, 8, 2, 16, dtype=torch.float16, device="cuda")
    query = torch.randn(1, 4, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="positive"):
        paged_decode_attention_triton(
            query,
            cache,
            torch.tensor([[0]], dtype=torch.int32, device="cuda"),
            torch.tensor([0], dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(ValueError, match="physical block"):
        paged_decode_attention_triton(
            query,
            cache,
            torch.tensor([[8]], dtype=torch.int32, device="cuda"),
            torch.tensor([1], dtype=torch.int32, device="cuda"),
        )


@pytest.mark.skipif(not triton_append_available(), reason="requires CUDA and Triton")
def test_triton_append_matches_torch_write():
    torch.manual_seed(2)
    cache = PagedKVCache(8, 8, 2, 16, dtype=torch.float16, device="cuda")
    table = cache.allocate_request(0, 4)
    key = torch.randn(4, 2, 16, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    slots = cache.slot_mapping(table, 0, 4).to(torch.int32)
    append_kv_triton(key, value, cache, slots)
    actual_k, actual_v = cache.read(0, table, 4)
    assert torch.equal(actual_k, key)
    assert torch.equal(actual_v, value)
