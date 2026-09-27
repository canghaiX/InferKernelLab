#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-inferkernellab:cuda}"
OUT_DIR="${OUT_DIR:-docs/benchmark_results/prefill_ab}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BACKEND="${BACKEND:-paged_sdpa}"
SHAPES="${SHAPES:-2:128 4:512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

mkdir -p "${OUT_DIR}"

for shape in ${SHAPES}; do
  IFS=: read -r batch_size prompt_length <<< "${shape}"
  for mode in token batched; do
    docker run --rm --gpus all --ipc=host --shm-size=16g \
      -v "$PWD:/workspace" -w /workspace "${IMAGE}" \
      python3 -m inferkernellab.hf_benchmark \
      --model-source random \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --backend "${BACKEND}" \
      --batch-size "${batch_size:-$BATCH_SIZE}" \
      --prompt-length "${prompt_length:-$PROMPT_LENGTH}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --warmup "${WARMUP}" \
      --iterations "${ITERATIONS}" \
      --prefill-mode "${mode}" \
      --output "/workspace/${OUT_DIR}/b${batch_size:-$BATCH_SIZE}_c${prompt_length:-$PROMPT_LENGTH}_${mode}.json"
  done
done

printf 'Prefill A/B results: %s\n' "${OUT_DIR}"
