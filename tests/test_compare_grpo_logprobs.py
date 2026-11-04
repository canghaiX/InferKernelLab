from __future__ import annotations

import pytest
import torch


def test_group_batch_scores_match_individual_cpu_qwen3_scores():
    transformers = pytest.importorskip("transformers")
    from scripts.compare_grpo_logprobs import _hf_group_batch_scores, _hf_scores

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    group = {
        "prompt_tokens": [1, 4, 7, 2],
        "completions": [
            {"response_tokens": [9, 10, 11]},
            {"response_tokens": [12]},
            {"response_tokens": [13, 14]},
        ],
    }

    batched = _hf_group_batch_scores(
        model,
        group,
        device=torch.device("cpu"),
        dtype=torch.float32,
        temperature=0.8,
    )
    individual = [
        _hf_scores(
            model,
            group["prompt_tokens"] + completion["response_tokens"],
            len(group["prompt_tokens"]),
            len(completion["response_tokens"]),
            device=torch.device("cpu"),
            dtype=torch.float32,
            temperature=0.8,
        )
        for completion in group["completions"]
    ]

    for batched_scores, individual_scores in zip(batched, individual, strict=True):
        assert batched_scores == pytest.approx(individual_scores, abs=1e-5)
