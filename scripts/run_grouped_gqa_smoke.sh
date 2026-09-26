#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${1:-docs/benchmark_results/grouped_gqa_smoke.jsonl}

python3 scripts/run_decode_sweep.py \
  --device cuda \
  --dtypes float16,bfloat16 \
  --batch-sizes 4 \
  --context-lens 128,512 \
  --num-kv-heads-list 8,1 \
  --num-heads 32 \
  --head-dim 64 \
  --block-size 16 \
  --max-new-tokens 8 \
  --warmup 3 \
  --repeats 3 \
  --backends paged_sdpa,triton_paged,triton_paged_grouped \
  --output "${OUTPUT}"
