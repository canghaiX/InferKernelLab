from argparse import Namespace

from inferkernellab.benchmark import run


def test_benchmark_reports_reference_and_sdpa_baselines():
    record = run(
        Namespace(
            device="cpu",
            backend="reference",
            dtype="float32",
            batch_size=2,
            context_len=7,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            block_size=4,
            num_blocks=5,
            warmup=0,
            iterations=2,
            seed=4,
        )
    )

    assert record["schema_version"] == 2
    assert "interleaved requests" in record["config"]["block_table_layout"]
    assert set(record["results"]) == {"dense", "paged_reference", "dense_sdpa", "paged_sdpa"}
    for result in record["results"].values():
        assert result["max_abs_error"] < 1e-5
        assert result["max_rel_error"] < 1e-3
        assert result["matches_reference"]
        assert result["timing"]["p50_ms"] > 0
        assert "speedup_vs_dense_sdpa" in result
        assert "speedup_vs_paged_sdpa" in result
