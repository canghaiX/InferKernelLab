from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def format_gsm8k_prompt(tokenizer, question: str) -> str:
    content = (
        "Solve the math problem step by step. Put the final numerical answer "
        "on the last line after '#### '.\n\n" + question
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"Question: {content}\nAnswer:"


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the canonical numeric answer, preferring GSM8K's #### marker."""
    marked = _ANSWER_RE.findall(text)
    candidates = marked or _NUMBER_RE.findall(text)
    if not candidates:
        return None
    return candidates[-1].replace(",", "")


def gsm8k_exact_match(response: str, target: str) -> float:
    actual = extract_gsm8k_answer(response)
    expected = extract_gsm8k_answer(target) or target.strip().replace(",", "")
    if actual is None:
        return 0.0
    try:
        return float(Decimal(actual) == Decimal(expected))
    except InvalidOperation:
        return 0.0
