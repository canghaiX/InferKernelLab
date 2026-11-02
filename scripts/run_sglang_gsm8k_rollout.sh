#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to a local Qwen3 checkpoint}"
: "${PROMPTS:?set PROMPTS to prepared GSM8K train.jsonl}"
: "${OUTPUT:?set OUTPUT for the rollout artifact}"
: "${SGLANG_BASE_URL:?set SGLANG_BASE_URL, for example http://127.0.0.1:30000}"
: "${DATA_MANIFEST:?set DATA_MANIFEST to prepared manifest.json}"

cmd=(python3 -m inferkernellab.rollout \
  --engine sglang \
  --sglang-base-url "$SGLANG_BASE_URL" \
  --sglang-timeout "${SGLANG_TIMEOUT:-600}" \
  --model-source path \
  --model-path "$MODEL_PATH" \
  --prompts "$PROMPTS" \
  --data-manifest "$DATA_MANIFEST" \
  --output "$OUTPUT" \
  --dtype bfloat16 \
  --num-generations "${NUM_GENERATIONS:-8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --temperature 0.8 \
  --top-p 1.0 \
  --seed "${SEED:-2026}")
PYTHONPATH="${PYTHONPATH:-src}" "${cmd[@]}"
