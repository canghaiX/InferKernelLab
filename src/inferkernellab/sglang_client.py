from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SGLangGeneration:
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    text: str
    finish_reason: str
    prompt_tokens: int | None = None


def _logprob_entry(entry: Any) -> tuple[float, int | None]:
    if isinstance(entry, dict):
        value = entry.get("logprob", entry.get("log_prob"))
        token = entry.get("token_id", entry.get("token"))
    elif isinstance(entry, (list, tuple)) and entry:
        value = entry[0]
        token = entry[1] if len(entry) > 1 else None
    else:
        value, token = entry, None
    if not isinstance(value, (float, int)):
        raise TypeError(f"invalid SGLang token logprob entry: {entry!r}")
    if not math.isfinite(value) or value > 1e-5:
        raise ValueError(f"invalid SGLang token logprob value: {value!r}")
    if token is not None and not isinstance(token, int):
        raise ValueError(f"invalid SGLang token id in logprob entry: {entry!r}")
    return float(value), token


def parse_generate_response(
    payload: Any, expected_count: int, *, require_logprobs: bool = True
) -> tuple[SGLangGeneration, ...]:
    items = payload if isinstance(payload, list) else [payload]
    if len(items) != expected_count:
        raise ValueError(
            f"SGLang returned {len(items)} responses for {expected_count} inputs"
        )
    parsed = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("SGLang response items must be JSON objects")
        meta = item.get("meta_info", {})
        if not isinstance(meta, dict):
            meta = {}
        raw_ids = item.get("output_ids", meta.get("output_token_ids"))
        raw_lps = item.get("output_token_logprobs", meta.get("output_token_logprobs"))
        if raw_lps is None:
            raw_lps = meta.get("completion_token_logprobs")
        if not require_logprobs and not raw_lps:
            raw_lps = (
                [[0.0, token] for token in raw_ids] if isinstance(raw_ids, list) else []
            )
        if not isinstance(raw_lps, list) or (require_logprobs and not raw_lps):
            raise ValueError("SGLang response is missing output token logprobs")
        values = [_logprob_entry(entry) for entry in raw_lps]
        token_ids = (
            tuple(int(value) for value in raw_ids) if isinstance(raw_ids, list) else ()
        )
        if not token_ids:
            inferred = [token for _, token in values]
            if any(token is None for token in inferred):
                raise ValueError("SGLang response is missing output token ids")
            token_ids = tuple(int(token) for token in inferred if token is not None)
        if len(token_ids) != len(values):
            raise ValueError("SGLang token ids and logprobs have different lengths")
        if any(
            token is not None and token != actual
            for actual, (_, token) in zip(token_ids, values)
        ):
            raise ValueError("SGLang output ids disagree with logprob token ids")
        reason = item.get("finish_reason", meta.get("finish_reason"))
        if isinstance(reason, dict):
            reason = reason.get("type", reason.get("reason"))
        if reason in {"stop", "eos_token"}:
            reason = "eos"
        elif reason in {None, "length", "length_limit"}:
            reason = "length"
        elif reason not in {"eos", "length"}:
            raise ValueError(f"unsupported SGLang finish reason: {reason}")
        parsed.append(
            SGLangGeneration(
                token_ids=token_ids,
                token_logprobs=tuple(value for value, _ in values),
                text=str(item.get("text", "")),
                finish_reason=reason,
                prompt_tokens=meta.get("prompt_tokens")
                if isinstance(meta.get("prompt_tokens"), int)
                else None,
            )
        )
    return tuple(parsed)


class SGLangClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def server_info(self) -> dict[str, Any]:
        with urllib.request.urlopen(
            self.base_url + "/get_server_info", timeout=self.timeout
        ) as response:
            info = json.loads(response.read())
        if not isinstance(info, dict):
            raise TypeError("SGLang server info must be a JSON object")
        if not info.get("version"):
            raise ValueError("SGLang server info is missing its version")
        return {
            key: info.get(key)
            for key in (
                "version",
                "model_path",
                "dtype",
                "attention_backend",
                "sampling_backend",
                "enable_fp32_lm_head",
                "enable_deterministic_inference",
                "rl_on_policy_target",
            )
        }

    def generate(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: Sequence[int],
        stop_token_ids: Sequence[int],
        return_logprob: bool = True,
    ) -> tuple[SGLangGeneration, ...]:
        if len(input_ids) != len(seeds):
            raise ValueError("input_ids and seeds must have the same length")
        sampling = [
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": -1,
                "n": 1,
                "sampling_seed": int(seed),
                "stop_token_ids": [int(token) for token in stop_token_ids],
                "ignore_eos": False,
            }
            for seed in seeds
        ]
        payload = self._post(
            "/generate",
            {
                "input_ids": [list(map(int, row)) for row in input_ids],
                "sampling_params": sampling,
                "return_logprob": return_logprob,
                "logprob_start_len": -1,
                "return_text_in_logprobs": False,
            },
        )
        return parse_generate_response(
            payload, len(input_ids), require_logprobs=return_logprob
        )

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                f"SGLang request failed at {self.base_url + path}: {error}"
            ) from error
