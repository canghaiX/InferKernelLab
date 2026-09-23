#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "Nsight Compute (ncu) is required for this script." >&2
  exit 1
fi

ncu --set full --target-processes all \
  python3 -m inferkernellab.benchmark \
  --device cuda --backend triton --dtype float16 \
  --batch-size "${BATCH_SIZE:-2}" \
  --context-len "${CONTEXT_LEN:-256}" \
  --num-heads "${NUM_HEADS:-8}" \
  --num-kv-heads "${NUM_KV_HEADS:-2}" \
  --head-dim "${HEAD_DIM:-64}" \
  --block-size "${BLOCK_SIZE:-16}" \
  --num-blocks "${NUM_BLOCKS:-256}" \
  --warmup 5 --iterations 10

