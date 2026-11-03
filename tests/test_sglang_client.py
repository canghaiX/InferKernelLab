import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from inferkernellab.gsm8k import format_gsm8k_prompt, gsm8k_exact_match
from inferkernellab.rollout import _load_prompt_rows, _request_sampling_seeds
from inferkernellab.sglang_client import SGLangClient, parse_generate_response


def test_jsonl_preserves_unicode_line_separator(tmp_path):
    path = tmp_path / "prompts.jsonl"
    row = {"question": "first\u2028second", "answer": "#### 42"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert _load_prompt_rows(path) == [row]


def test_rollout_seed_streams_do_not_overlap_for_adjacent_base_seeds():
    first = _request_sampling_seeds(2026, 2048)
    second = _request_sampling_seeds(2027, 2048)
    assert first == _request_sampling_seeds(2026, 2048)
    assert not set(first) & set(second)
    assert len(set(first)) == 2048
    assert all(0 <= seed < 2**31 for seed in first + second)


def test_qwen_prompt_uses_local_template_and_disables_thinking():
    class Tokenizer:
        chat_template = "local template"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            assert messages[0]["role"] == "user"
            assert "#### " in messages[0]["content"]
            return messages[0]["content"]

    assert format_gsm8k_prompt(Tokenizer(), "2+2?").endswith("2+2?")


@pytest.mark.parametrize(
    "response,target,reward",
    [
        ("#### 42.00", "#### 42", 1),
        ("#### -1,024", "#### -1024", 1),
        ("no answer", "#### 42", 0),
        ("#### 42.1", "#### 42", 0),
    ],
)
def test_numeric_reward(response, target, reward):
    assert gsm8k_exact_match(response, target) == reward


def test_native_http_batch_keeps_eos_and_sampling_contract():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            result = [
                {
                    "text": "42",
                    "meta_info": {
                        "output_token_logprobs": [[-0.3, 42], [-0.2, 2]],
                        "finish_reason": {"type": "stop", "matched": 2},
                    },
                }
                for _ in payload["input_ids"]
            ]
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = SGLangClient(f"http://127.0.0.1:{server.server_port}")
        outputs = client.generate(
            [[1, 4], [1, 5]],
            max_new_tokens=2,
            temperature=0.8,
            top_p=1.0,
            seeds=[7, 8],
            stop_token_ids=[2],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    path, payload = received[0]
    assert path == "/generate"
    assert payload["input_ids"] == [[1, 4], [1, 5]]
    assert payload["return_logprob"] is True
    assert payload["sampling_params"][1]["sampling_seed"] == 8
    assert payload["sampling_params"][0]["top_k"] == -1
    assert payload["sampling_params"][0]["top_p"] == 1
    assert outputs[0].token_ids == (42, 2)
    assert outputs[0].token_logprobs == (-0.3, -0.2)
    assert outputs[0].finish_reason == "eos"


@pytest.mark.parametrize("logprobs", [None, []])
def test_greedy_response_without_logprobs(logprobs):
    item = {
        "output_ids": [42, 2],
        "meta_info": {
            "output_token_logprobs": logprobs,
            "finish_reason": {"type": "stop"},
        },
    }
    assert parse_generate_response(item, 1, require_logprobs=False)[0].token_ids == (
        42,
        2,
    )


def test_batch_count_and_logprob_length_are_checked():
    with pytest.raises(ValueError, match="responses"):
        parse_generate_response([], 2)
    with pytest.raises(ValueError, match="different lengths"):
        parse_generate_response(
            {"output_ids": [1, 2], "output_token_logprobs": [[-0.1, 1]]}, 1
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.2])
def test_invalid_logprob_is_rejected(value):
    with pytest.raises(ValueError, match="logprob value"):
        parse_generate_response({"output_token_logprobs": [[value, 2]]}, 1)


def test_logprob_token_ids_must_match_output_ids():
    with pytest.raises(ValueError, match="disagree"):
        parse_generate_response(
            {"output_ids": [2], "output_token_logprobs": [[-0.1, 3]]}, 1
        )


def test_teacher_forced_candidate_score_uses_sampling_endpoint():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(request)
            outputs = [
                {
                    "output_ids": [99],
                    "meta_info": {
                        "output_token_ids_logprobs": [[[-0.125, target[0], None]]]
                    },
                }
                for target in request["token_ids_logprob"]
            ]
            body = json.dumps(outputs).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = SGLangClient(f"http://127.0.0.1:{server.server_port}")
        scores = client.score_next_tokens([[1, 2], [3, 4]], [7, 8], temperature=0.8)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert [score.logprob for score in scores] == [-0.125, -0.125]
    assert [score.target_token_id for score in scores] == [7, 8]
    assert all(score.sampled_token_id == 99 for score in scores)
    assert received[0]["sampling_params"][0]["temperature"] == 0.8
    assert received[0]["sampling_params"][0]["max_new_tokens"] == 1
    assert received[0]["token_ids_logprob"] == [[7], [8]]


def test_teacher_forced_candidate_score_rejects_target_mismatch():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(
                [
                    {
                        "output_ids": [99],
                        "meta_info": {
                            "output_token_ids_logprobs": [[[-0.125, 3, None]]]
                        },
                    }
                    for _ in request["input_ids"]
                ]
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match="target token"):
            SGLangClient(f"http://127.0.0.1:{server.server_port}").score_next_tokens(
                [[1, 2]], [7], temperature=0.8
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
