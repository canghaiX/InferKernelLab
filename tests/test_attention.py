import pytest
import torch

from inferkernellab.attention import dense_decode_attention, paged_decode_attention
from inferkernellab.cache import PagedKVCache
from inferkernellab.append import append_kv_triton, triton_append_available
from inferkernellab.triton_ops import paged_decode_attention_triton, triton_available


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


@pytest.mark.skipif(not triton_available(), reason="requires CUDA and Triton")
def test_triton_attention_matches_reference():
    torch.manual_seed(1)
    cache = PagedKVCache(16, 8, 2, 16, dtype=torch.float16, device="cuda")
    tables = [cache.allocate_request(i, 19 + i) for i in range(2)]
    for i, table in enumerate(tables):
        length = 19 + i
        key = torch.randn(length, 2, 16, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        cache.write(0, table, 0, key, value)
    query = torch.randn(2, 4, 16, device="cuda", dtype=torch.float16)
    expected = paged_decode_attention(query, cache, tables, [19, 20])
    actual = paged_decode_attention_triton(query, cache, tables, [19, 20])
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


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
