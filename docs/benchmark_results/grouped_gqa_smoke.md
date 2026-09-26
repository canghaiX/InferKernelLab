# Grouped GQA/MQA Smoke

测量日期：2026-09-26。硬件为 NVIDIA A100-SXM4-40GB，PyTorch `2.6.0+cu124`，
Triton `3.2.0`，FP16，batch `4`，query heads `32`，head dim `64`，block size `16`，
生成 `8` tokens，warmup `2`，独立重复 `3`。原始 JSONL 由
`scripts/run_grouped_gqa_smoke.sh` 生成；记录中的 provenance 为 clean commit
`f841db19c70fa0152ce4e47331efb28c9e93de24`。

| KV heads | Context | paged SDPA P50 (ms) | old Triton P50 (ms) | grouped Triton P50 (ms) | grouped/old | Token match |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 128 | 1.5002 | 1.8004 | 1.7994 | 0.999x | 100% |
| 8 | 512 | 1.5305 | 1.8171 | 1.8358 | 1.010x | 100% |
| 1 | 128 | 1.4966 | 1.8108 | 1.8841 | 1.041x | 100% |
| 1 | 512 | 1.5305 | 1.8474 | 2.2650 | 1.226x | 100% |

## Interpretation

- 12/12 FP16 sweep records passed logits correctness against `paged_reference`; the
  largest reported absolute error was `3.82e-6` and token match was `100%`.
- A direct multi-layer cache check also passed FP16/BF16 MHA, GQA and MQA. The largest
  BF16 output error was `7.81e-3`, within the existing dtype-aware tolerance.
- Grouped mapping is functionally correct, but this implementation does not yet provide
  a stable speedup on the measured A100 shapes. It remains an explicit opt-in backend;
  the default `triton_paged` path is unchanged.
- The result does not claim DRAM bandwidth or occupancy. Those counters remain subject
  to the host `ERR_NVGPUCTRPERM` restriction documented in the benchmark report.
