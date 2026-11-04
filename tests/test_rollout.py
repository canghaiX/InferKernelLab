import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from inferkernellab.gsm8k import extract_gsm8k_answer, gsm8k_exact_match
from inferkernellab.hf_benchmark import LlamaPagedDecodeRunner, _load_model
from inferkernellab.rollout import _hf_incremental_generate, run
from inferkernellab.sglang_client import parse_generate_response


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


def test_hf_reference_qwen3_rollout_is_seeded_and_records_contract(
    tmp_path, monkeypatch
):
    transformers = pytest.importorskip("transformers")
    model_path = tmp_path / "qwen3"
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        eos_token_id=31,
        attention_dropout=0.0,
    )
    transformers.Qwen3ForCausalLM(config).save_pretrained(model_path)

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(map(str, tokens))

    monkeypatch.setattr("inferkernellab.rollout._load_tokenizer", lambda _: Tokenizer())
    args = _rollout_args(tmp_path)
    args.engine = "hf_reference"
    args.model_source = "path"
    args.model_path = str(model_path)
    args.export_model_path = None
    args.tokenizer_path = str(model_path)
    args.data_manifest = None
    args.attention_implementation = "sdpa"
    args.top_p = 1.0
    args.max_new_tokens = 3

    first_path = tmp_path / "hf-first.jsonl"
    args.output = str(first_path)
    manifest = run(args)
    first_records = [json.loads(line) for line in first_path.read_text().splitlines()]
    second_path = tmp_path / "hf-second.jsonl"
    args.output = str(second_path)
    run(args)
    second_records = [json.loads(line) for line in second_path.read_text().splitlines()]

    assert manifest["engine"] == "hf_reference"
    assert manifest["forward_contract"]["name"] == "qwen3_hf_incremental_cache_v1"
    assert manifest["forward_contract"]["weights_dtype"] == "float32"
    assert manifest["forward_contract"]["attention_implementation"] == "sdpa"
    assert manifest["cache_released"]
    assert [row["completions"] for row in first_records[1:]] == [
        row["completions"] for row in second_records[1:]
    ]
    for group in first_records[1:]:
        for completion in group["completions"]:
            assert len(completion["response_tokens"]) == len(completion["old_logprobs"])
            assert completion["finish_reason"] in {"eos", "length"}


def test_hf_reference_generation_keeps_eos_and_its_logprob():
    class EosModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[-10.0, 10.0]], device=input_ids.device)
            return SimpleNamespace(logits=logits[:, None, :], past_key_values=None)

    generator = torch.Generator(device="cpu").manual_seed(5)
    tokens, logprobs, finish_reason = _hf_incremental_generate(
        EosModel(),
        (1,),
        max_new_tokens=4,
        temperature=0.8,
        eos_token_ids=(1,),
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert tokens == [1]
    assert len(logprobs) == 1
    assert logprobs[0] == pytest.approx(0.0)
    assert finish_reason == "eos"


def test_hf_reference_rejects_truncated_policy(tmp_path):
    args = _rollout_args(tmp_path)
    args.engine = "hf_reference"
    args.model_source = "path"
    args.model_path = "/not/loaded/because/top-p-is-invalid"
    args.top_p = 0.9

    with pytest.raises(ValueError, match="top_p=1.0"):
        run(args)


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


def test_gsm8k_answer_parser_prefers_marked_answer():
    assert extract_gsm8k_answer("work 2 + 2 = 4\n#### 4") == "4"
    assert extract_gsm8k_answer("The result is 1,024") == "1024"
    assert gsm8k_exact_match("reasoning\n#### 18", "#### 18") == 1.0
    assert gsm8k_exact_match("#### 17", "#### 18") == 0.0


def test_sglang_logprob_response_parser_supports_meta_info():
    payload = {
        "text": "42",
        "meta_info": {
            "output_token_ids": [42, 43],
            "output_token_logprobs": [[-0.5, 42], [-0.25, 43]],
            "finish_reason": "stop",
            "prompt_tokens": 3,
        },
    }
    result = parse_generate_response(payload, 1)[0]
    assert result.token_ids == (42, 43)
    assert result.token_logprobs == (-0.5, -0.25)
    assert result.finish_reason == "eos"
    assert result.prompt_tokens == 3


def test_sglang_logprob_response_parser_rejects_missing_logprobs():
    with pytest.raises(ValueError, match="logprobs"):
        parse_generate_response({"text": "42"}, 1)
