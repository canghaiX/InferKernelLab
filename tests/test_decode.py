import pytest
import torch

from inferkernellab.decode import SyntheticDecodeRunner, SyntheticDecoderConfig


def _run(backend: str, num_kv_heads: int):
    config = SyntheticDecoderConfig(
        vocab_size=32,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        head_dim=8,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
        seed=19,
        backend=backend,
        max_num_seqs=2,
        max_num_batched_tokens=4,
    )
    return SyntheticDecodeRunner(config, [[1, 2, 3], [7, 8, 9, 10]], max_new_tokens=3).run()


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
def test_synthetic_decoder_reference_and_sdpa_match(num_kv_heads):
    reference = _run("paged_reference", num_kv_heads)
    sdpa = _run("paged_sdpa", num_kv_heads)

    assert reference.generated_tokens == sdpa.generated_tokens
    assert len(reference.logits) == len(sdpa.logits) == 3
    for expected, actual in zip(reference.logits, sdpa.logits):
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert reference.used_blocks_peak > 0
    assert sdpa.used_blocks_peak > 0


def test_synthetic_decoder_releases_all_blocks_after_generation():
    config = SyntheticDecoderConfig(
        vocab_size=16,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
        seed=3,
        backend="paged_reference",
        max_num_seqs=2,
        max_num_batched_tokens=3,
    )
    runner = SyntheticDecodeRunner(config, [[1, 2, 3, 4], [5, 6]], max_new_tokens=4)
    result = runner.run()

    assert all(len(tokens) == 4 for tokens in result.generated_tokens)
    assert runner.runtime.stats().used_blocks == 0
    assert runner.runtime.stats().finished == 2


def test_synthetic_decoder_uses_non_contiguous_initial_block_tables_and_reports_tpot():
    config = SyntheticDecoderConfig(
        vocab_size=16,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
        seed=3,
        backend="paged_reference",
        max_num_seqs=2,
        max_num_batched_tokens=8,
    )
    runner = SyntheticDecodeRunner(config, [[1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10]], max_new_tokens=2)

    assert all(
        len(request.block_table) > 1
        and any(right - left != 1 for left, right in zip(request.block_table, request.block_table[1:]))
        for request in runner.requests
    )
    result = runner.run()

    assert result.tpot_wall_ms > 0
    assert result.tpot_device_ms > 0


def test_synthetic_decoder_rejects_second_run():
    config = SyntheticDecoderConfig(
        vocab_size=16,
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
        seed=3,
        backend="paged_reference",
    )
    runner = SyntheticDecodeRunner(config, [[1, 2]], max_new_tokens=1)
    runner.run()
    with pytest.raises(RuntimeError, match="only be run once"):
        runner.run()
