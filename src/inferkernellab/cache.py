from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheLayout:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int


class BlockAllocator:
    """A deterministic physical-block allocator used by the reference runtime."""

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self._free = list(range(num_blocks - 1, -1, -1))
        self._owners: dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return len(self._owners)

    def allocate(self, count: int, owner: int | None = None) -> list[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self._free):
            raise RuntimeError(f"out of KV blocks: requested={count}, free={len(self._free)}")
        blocks = [self._free.pop() for _ in range(count)]
        if owner is not None:
            for block_id in blocks:
                self._owners[block_id] = owner
        return blocks

    def free(self, block_ids: list[int] | tuple[int, ...]) -> None:
        for block_id in block_ids:
            if block_id not in self._owners and block_id in self._free:
                raise ValueError(f"block {block_id} is already free")
            self._owners.pop(block_id, None)
            self._free.append(block_id)

    def owner(self, block_id: int) -> int | None:
        return self._owners.get(block_id)


class PagedKVCache:
    """KV storage with physical blocks and logical request block tables.

    Storage uses [block, token-in-block, kv-head, head-dim]. The class keeps
    request metadata outside the tensor so the same storage can be exercised by
    both the PyTorch reference path and a Triton kernel.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        if min(num_layers, num_blocks, block_size, num_kv_heads, head_dim) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.layout = CacheLayout(num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.device = torch.device(device)
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=self.device)
        self.v = torch.empty_like(self.k)
        self.allocator = BlockAllocator(num_blocks)

    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype

    @property
    def numel_bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    def allocate_request(self, request_id: int, num_tokens: int) -> list[int]:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        count = (num_tokens + self.layout.block_size - 1) // self.layout.block_size
        return self.allocator.allocate(count, owner=request_id)

    def release_request(self, block_table: list[int]) -> None:
        self.allocator.free(block_table)

    def slot_mapping(self, block_table: list[int], start: int, length: int) -> torch.Tensor:
        if start < 0 or length < 0:
            raise ValueError("start and length must be non-negative")
        positions = torch.arange(start, start + length, dtype=torch.long, device=self.device)
        logical = torch.div(positions, self.layout.block_size, rounding_mode="floor")
        offsets = positions.remainder(self.layout.block_size)
        table = torch.as_tensor(block_table, dtype=torch.long, device=self.device)
        if logical.numel() and int(logical.max()) >= table.numel():
            raise ValueError("block_table is shorter than the requested range")
        return table[logical] * self.layout.block_size + offsets

    def write(
        self,
        layer: int,
        block_table: list[int],
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write [tokens, kv_heads, head_dim] into logical positions."""
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key/value must have the same [T, H, D] shape")
        if key.shape[1:] != (self.layout.num_kv_heads, self.layout.head_dim):
            raise ValueError("key/value shape does not match cache layout")
        slots = self.slot_mapping(block_table, start, key.shape[0])
        if key.device != self.device:
            key = key.to(self.device)
        if value.device != self.device:
            value = value.to(self.device)
        block_ids = torch.div(slots, self.layout.block_size, rounding_mode="floor")
        offsets = slots.remainder(self.layout.block_size)
        if not 0 <= layer < self.layout.num_layers:
            raise IndexError("layer is out of range")
        self.k[layer, block_ids, offsets, :, :] = key
        self.v[layer, block_ids, offsets, :, :] = value

    def read(self, layer: int, block_table: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the first ``length`` logical tokens in order."""
        if length < 0:
            raise ValueError("length must be non-negative")
        slots = self.slot_mapping(block_table, 0, length)
        block_ids = torch.div(slots, self.layout.block_size, rounding_mode="floor")
        offsets = slots.remainder(self.layout.block_size)
        if not 0 <= layer < self.layout.num_layers:
            raise IndexError("layer is out of range")
        return self.k[layer, block_ids, offsets], self.v[layer, block_ids, offsets]
