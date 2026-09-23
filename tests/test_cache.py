import torch

from inferkernellab.cache import PagedKVCache


def test_slot_mapping_crosses_physical_blocks():
    cache = PagedKVCache(8, 4, 2, 3, dtype=torch.float32)
    table = [6, 2, 5]
    slots = cache.slot_mapping(table, 2, 6)
    assert slots.tolist() == [26, 27, 8, 9, 10, 11]


def test_write_and_read_preserve_logical_order():
    cache = PagedKVCache(8, 4, 2, 3, dtype=torch.float32)
    table = [5, 1]
    key = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    value = key + 100
    cache.write(0, table, 2, key, value)
    read_key, read_value = cache.read(0, table, 4)
    assert torch.equal(read_key[2:], key)
    assert torch.equal(read_value[2:], value)


def test_allocator_releases_blocks():
    cache = PagedKVCache(4, 4, 1, 2, dtype=torch.float32)
    table = cache.allocate_request(7, 5)
    assert cache.allocator.num_free_blocks == 2
    cache.release_request(table)
    assert cache.allocator.num_free_blocks == 4


def test_layers_are_independent():
    cache = PagedKVCache(4, 4, 1, 2, num_layers=2, dtype=torch.float32)
    table = [0]
    key0 = torch.ones(2, 1, 2)
    key1 = torch.full((2, 1, 2), 3.0)
    cache.write(0, table, 0, key0, key0)
    cache.write(1, table, 0, key1, key1)
    assert torch.equal(cache.read(0, table, 2)[0], key0)
    assert torch.equal(cache.read(1, table, 2)[0], key1)
