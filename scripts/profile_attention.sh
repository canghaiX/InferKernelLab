#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "Nsight Compute (ncu) must be available in the container image." >&2
  exit 1
fi

output="${PROFILE_OUTPUT:-profile/paged_decode}"
mkdir -p "$(dirname "$output")"
batch_size="${BATCH_SIZE:-2}"
context_len="${CONTEXT_LEN:-256}"
block_size="${BLOCK_SIZE:-16}"
required_blocks=$(((context_len + block_size - 1) / block_size))
num_blocks="${NUM_BLOCKS:-$((batch_size * required_blocks + required_blocks - 1))}"

ncu --set basic --profile-from-start off \
  --kernel-name regex:_paged_decode_kernel --launch-count 1 \
  --target-processes all --export "$output" \
  python3 -m inferkernellab.benchmark \
  --device cuda --backend triton --dtype float16 \
  --batch-size "$batch_size" \
  --context-len "$context_len" \
  --num-heads "${NUM_HEADS:-8}" \
  --num-kv-heads "${NUM_KV_HEADS:-2}" \
  --head-dim "${HEAD_DIM:-64}" \
  --block-size "$block_size" \
  --num-blocks "$num_blocks" \
  --profile-kernel
