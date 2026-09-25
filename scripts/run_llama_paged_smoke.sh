#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-inferkernellab:cuda}"
OUT="${OUT:-docs/benchmark_results/llama_paged.json}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BACKEND="${BACKEND:-paged_sdpa}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"

mkdir -p "$(dirname "${OUT}")"

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace "${IMAGE}" \
  python3 -m inferkernellab.hf_benchmark \
  --model-source random \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --backend "${BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --prompt-length "${PROMPT_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --output "/workspace/${OUT}"

printf 'Llama paged benchmark result: %s\n' "${OUT}"
