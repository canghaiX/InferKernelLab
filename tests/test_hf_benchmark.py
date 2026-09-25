from argparse import Namespace

import pytest
import torch

pytest.importorskip("transformers")

from inferkernellab.hf_benchmark import LlamaPagedDecodeRunner, _load_model, run
from inferkernellab.triton_ops import triton_available


def _args(tmp_path, *, num_kv_heads: int, num_layers: int = 2) -> Namespace:
    return Namespace(
        model_source="random",
        model_path=None,
        device="cpu",
        dtype="float32",
        backend="paged_sdpa",
        append_backend="torch",
        batch_size=2,
        prompt_length=5,
        max_new_tokens=2,
        warmup=0,
        iterations=1,
        block_size=4,
        max_num_batched_tokens=2048,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=num_layers,
        num_attention_heads=4,
        num_kv_heads=num_kv_heads,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        seed=2026,
        output=str(tmp_path / f"llama-{num_kv_heads}.json"),
    )


@pytest.mark.parametrize("num_kv_heads,num_layers", [(4, 1), (2, 2)])
def test_random_llama_paged_decode_matches_eager_and_releases_cache(
    tmp_path,
    num_kv_heads,
    num_layers,
):
    record = run(
        _args(tmp_path, num_kv_heads=num_kv_heads, num_layers=num_layers)
    )

    assert record["schema_version"] == 1
    assert record["benchmark"] == "llama_paged_decode"
    assert record["architecture"] == "LlamaForCausalLM"
    assert record["number_of_layers"] == num_layers
    assert record["number_of_kv_heads"] == num_kv_heads
    assert record["correctness"]["status"] == "ok"
    assert record["logits_max_error"] < 1e-4
    assert record["token_match_rate"] == 1.0
    assert record["cache_released"] is True
    assert record["decode_p50_ms"] > 0
    assert record["decode_p95_ms"] >= record["decode_p50_ms"]


def test_random_llama_runner_prompt_crosses_block_boundary(tmp_path):
    args = _args(tmp_path, num_kv_heads=2)
    device = torch.device("cpu")
    dtype = torch.float32
    model, _ = _load_model(args, device, dtype)
    prompts = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    runner = LlamaPagedDecodeRunner(
        model,
        prompts,
        max_new_tokens=2,
        backend="paged_reference",
        block_size=4,
    )
    result = runner.run()

    assert result.used_blocks_peak >= 4
    assert result.cache_released
    assert runner.runtime.stats().used_blocks == 0


def test_local_checkpoint_mode_rejects_missing_path(tmp_path):
    args = _args(tmp_path, num_kv_heads=2)
    args.model_source = "path"
    args.model_path = str(tmp_path / "missing-checkpoint")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("backend", ["paged_sdpa", "triton_paged"])
def test_random_llama_cuda_backends_report_latency_and_correctness(
    tmp_path,
    dtype,
    backend,
):
    if backend == "triton_paged" and not triton_available():
        pytest.skip("requires Triton")
    args = _args(tmp_path, num_kv_heads=2, num_layers=2)
    args.device = "cuda"
    args.dtype = dtype
    args.backend = backend
    args.append_backend = "triton" if backend == "triton_paged" else "torch"
    record = run(args)

    assert record["correctness"]["status"] == "ok"
    assert record["token_match_rate"] == 1.0
    assert record["decode_p50_ms"] > 0
    assert record["decode_p95_ms"] >= record["decode_p50_ms"]
    assert record["cache_released"] is True
