# Grouped GQA/MQA 与 Decode Runtime 优化结果

测量日期：2026-09-27。硬件为 NVIDIA A100-SXM4-40GB，8 卡机器上的单卡测试；PyTorch
`2.6.0+cu124`、CUDA runtime `12.4`、Triton `3.2.0`、driver `580.173.02`。最终代码为
clean commit `5f7fc4f253630b3f19e6e85c294d58a098781698`。

## Synthetic A/B

配置：FP16，query heads `32`，head dim `64`，block size `16`，batch `4/16`，context
`128/512`，KV heads `8/1`，生成 `16` tokens，warmup `3`，独立重复 `3`；每个 backend
都使用 Torch KV append。原始记录见同目录的
`grouped_gqa_tensorcore_20260927.jsonl`。

| Batch | Context | KV heads | Implementation | Old Triton P50 (ms) | Grouped P50 (ms) | Grouped/old | 变化 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 128 | 8 | tensorcore_short_context | 1.5200 | 1.5286 | 1.006x | -0.56% |
| 4 | 128 | 1 | tensorcore_short_context | 1.5252 | 1.5297 | 1.003x | -0.29% |
| 4 | 512 | 8 | scalar_grouped | 1.5333 | 1.5826 | 1.032x | -3.22% |
| 4 | 512 | 1 | scalar_grouped | 1.5498 | 1.9977 | 1.289x | -28.90% |
| 16 | 128 | 8 | tensorcore_short_context | 3.9175 | 3.8852 | 0.992x | +0.82% |
| 16 | 128 | 1 | tensorcore_short_context | 3.8972 | 3.8994 | 1.001x | -0.06% |
| 16 | 512 | 8 | scalar_grouped | 3.9750 | 4.1464 | 1.043x | -4.31% |
| 16 | 512 | 1 | scalar_grouped | 3.9524 | 4.5973 | 1.163x | -16.32% |

24/24 records pass reference logits correctness and token match is `100%`. MHA was not part
of this grouped table; its public path still falls back to the old Triton kernel.

## Runtime Metadata 优化

与前一 clean commit `0b1a087bb1aa125d04a762df98c2d1ffc4c799f4` 的相同 24-shape 结果
相比，预分配并复用 block-table/context metadata 后：

- `triton_paged` decode-step P50 平均下降约 `12.2%`，中位数下降约 `12.1%`；
- `triton_paged_grouped` decode-step P50 平均下降约 `11.7%`，中位数下降约 `11.6%`；
- `paged_sdpa` 基本持平，说明收益主要来自 Triton runtime metadata 构造，而不是把 SDPA
  结果误算为 kernel 加速。

## HF Llama 对照

2-layer deterministic random Llama，batch `2`，prompt `128`，生成 `8` tokens，FP16，
GQA `8/2`，head dim `16`，Triton KV append，warmup `2`，iterations `3`：

| Backend | Decode P50 (ms) | Decode P95 (ms) | TPOT (ms) | TTFT (ms) | Logits max error | Token match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `triton_paged` | 3.653 | 3.794 | 3.677 | 856.744 | 7.93e-4 | 100% |
| `triton_paged_grouped` | 3.666 | 3.795 | 3.680 | 851.718 | 7.93e-4 | 100% |

HF 路径 correctness 通过，但 grouped 与旧 kernel 在该端到端 shape 上基本持平，不能宣称
稳定加速。所有 benchmark 都保留 `provenance.git_dirty=false` 和实际
`kernel_config.implementation`，便于复现与面试复盘。
