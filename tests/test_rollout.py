import json
from argparse import Namespace

import pytest
import torch

pytest.importorskip("transformers")

from inferkernellab.hf_benchmark import LlamaPagedDecodeRunner, _load_model
from inferkernellab.rollout import run


def _rollout_args(tmp_path, *, output_name="rollouts.jsonl"):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        json.dumps({"group_id": "a", "prompt_tokens": [1, 4], "target_tokens": [3, 5]})
        + "\n"
        + json.dumps(
            {"group_id": "b", "prompt_tokens": [2, 6], "target_tokens": [7, 8]}
        )
        + "\n"
    )
    return Namespace(
        model_source="random",
        model_path=None,
        export_model_path=str(tmp_path / "model"),
        tokenizer_path=None,
        prompts=str(prompts_path),
        output=str(tmp_path / output_name),
        device="cpu",
        dtype="float32",
        backend="paged_sdpa",
        append_backend="torch",
        num_generations=2,
        max_new_tokens=3,
        temperature=0.8,
        top_p=0.95,
        block_size=4,
        max_num_batched_tokens=16,
        seed=2026,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        max_position_embeddings=16,
        rope_theta=10_000.0,
    )


def test_rollout_writes_grouped_logprobs_and_model_fingerprints(tmp_path):
    args = _rollout_args(tmp_path)

    manifest = run(args)
    records = [
        json.loads(line)
        for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()
    ]

    assert manifest["schema_version"] == 1
    assert manifest["reward_function"] == "token_prefix_accuracy_v1"
    assert manifest["cache_released"] is True
    assert len(manifest["model_fingerprint"]) == 64
    assert len(records) == 3
    for group in records[1:]:
        assert len(group["completions"]) == 2
        for completion in group["completions"]:
            assert len(completion["response_tokens"]) == len(completion["old_logprobs"])
            assert completion["finish_reason"] in {"eos", "length"}


def test_runner_stops_on_eos_and_releases_blocks(tmp_path):
    args = _rollout_args(tmp_path)
    model, _ = _load_model(
        Namespace(
            model_source="random",
            model_path=None,
            seed=args.seed,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_kv_heads=args.num_kv_heads,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
        ),
        torch.device("cpu"),
        torch.float32,
    )
    prompts = ((1, 4), (2, 6))
    baseline = LlamaPagedDecodeRunner(
        model,
        prompts,
        max_new_tokens=3,
        backend="paged_sdpa",
        block_size=4,
        temperature=0.8,
        seed=args.seed,
    ).run()
    eos_ids = {tokens[0] for tokens in baseline.generated_tokens}
    stopped = LlamaPagedDecodeRunner(
        model,
        prompts,
        max_new_tokens=3,
        backend="paged_sdpa",
        block_size=4,
        temperature=0.8,
        seed=args.seed,
        eos_token_ids=tuple(eos_ids),
    )
    result = stopped.run()

    assert all(len(tokens) == 1 for tokens in result.generated_tokens)
    assert result.finish_reasons == ("eos", "eos")
    assert result.cache_released
    assert stopped.runtime.stats().used_blocks == 0


def test_seeded_sampling_logprobs_match_hf_eager(tmp_path):
    args = _rollout_args(tmp_path)
    model, _ = _load_model(
        Namespace(
            model_source="random",
            model_path=None,
            seed=args.seed,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_kv_heads=args.num_kv_heads,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
        ),
        torch.device("cpu"),
        torch.float32,
    )
    prompts = ((1, 4), (2, 6, 9))

    def sample():
        return LlamaPagedDecodeRunner(
            model,
            prompts,
            max_new_tokens=3,
            backend="paged_sdpa",
            block_size=4,
            temperature=0.8,
            top_p=0.95,
            seed=91,
        ).run()

    first = sample()
    second = sample()
    assert first.generated_tokens == second.generated_tokens
    assert first.token_logprobs == second.token_logprobs
    assert first.sampling_logprobs == second.sampling_logprobs

    for prompt, response, actual in zip(
        prompts, first.generated_tokens, first.token_logprobs
    ):
        full = torch.tensor([(*prompt, *response)])
        with torch.no_grad():
            logits = model(input_ids=full).logits[0]
        for offset, (token, logprob) in enumerate(zip(response, actual)):
            policy = torch.log_softmax(
                logits[len(prompt) - 1 + offset].float() / 0.8, dim=-1
            )
            torch.testing.assert_close(
                policy[token], torch.tensor(logprob), atol=1e-6, rtol=1e-6
            )
    assert first.cache_released
