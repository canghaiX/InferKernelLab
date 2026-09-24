# 基准测试报告

本报告记录可复现的 decode attention microbenchmark。数据仅适用于列出的硬件、形状和实现，不代表整模型 serving 吞吐，也不承诺 Triton 对所有 baseline 都更快。

## 环境

- 运行日期：2026-09-23
- GPU：NVIDIA A100-SXM4-40GB，Compute Capability 8.0
- Driver：580.173.02
- 容器基础镜像：`nano-vllm:optimized`
- Python：3.12.13；PyTorch：2.11.0+cu130；CUDA runtime：13.0；Triton：3.6.0
- profiler：容器内有 Nsight Compute；本机驱动拒绝访问 GPU performance counters，错误为 `ERR_NVGPUCTRPERM`

## 比较对象

- `dense`、`paged_reference`：PyTorch 数学参考实现，用于验证，不作为优化性能基线。
- `dense_sdpa`：KV 在计时前已堆叠为连续 batch，计时期间执行一次 batched PyTorch SDPA。
- `paged_sdpa`：计时期间按 block table 批量 gather K/V，再执行 batched SDPA；时间包含 gather 与 attention。
- `triton_paged`：Triton kernel 直接通过 block table 读取 paged K/V。

物理 block 在不同请求的 logical block 之间交错分配，并插入保留 block，避免 sweep 退化为逻辑顺序与连续物理地址完全一致的特例。所有实现使用相同 query、K/V、context 和 GQA 配置。

## Sweep 配置与结果

共 54 组：batch 为 1/4/16，context 为 128/512/2048，query heads 固定为 32，KV heads 为 32/8/1（MHA/GQA/MQA），dtype 为 FP16/BF16；head dim=64、block size=16，每组 warmup 10 次、测量 50 次。CUDA Event 每个样本同步一次，报告 mean/P50/P95。

54 组所有实现均通过 dtype-aware reference 比对。容差为 FP16 `atol/rtol=3e-3`、BF16 `3e-2`；benchmark 发现不匹配会直接失败。全量原始 JSONL：`docs/benchmark_results/a100_cuda13_triton36.jsonl`。

| 对比项 | Triton 结果 |
| --- | ---: |
| 相对 dense SDPA 更快 | 0/54 组；`dense_mean / triton_mean` 中位数 0.671×，范围 0.402–0.922× |
| 相对 paged gather + SDPA 更快 | 54/54 组；`paged_sdpa_mean / triton_mean` 中位数 2.179×，范围 1.246–5.579× |
| Triton 最大绝对误差 | 0.02344（BF16）；全组均通过上文 dtype-aware `allclose` |
| Autotuner 选择 | 51 组为 `block_n=128, num_warps=4, num_stages=3`；3 组为 `64,4,2` |

以上 ratio 由每组 mean latency 计算，跨不同形状的中位数只用于概括整个矩阵，不应当作单一 workload 的 speedup。结果表明当前 kernel 能避免显式 gather 的成本，但当前实现没有胜过连续 KV 的 PyTorch SDPA；不能据此宣称优于成熟的生产级 paged attention kernel。

## 独立复测

以下三组各独立运行 3 次，每次 warmup 10 次、测量 100 次。表中 latency 为 3 次运行各自统计值的中位数；ratio 为 `paged_sdpa mean / triton mean`。完整数据见 `docs/benchmark_results/a100_selected_repeats.jsonl`。

| 配置 | 实现 | Mean (ms) | P50 (ms) | P95 (ms) | 最大绝对误差 | 相对 paged SDPA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B=4, C=512, KV=8, FP16 | Dense SDPA | 0.07758 | 0.07680 | 0.08197 | 0.000244 | — |
|  | Paged gather + SDPA | 0.23974 | 0.23859 | 0.24576 | 0.000244 | 1.00× |
|  | Triton paged | 0.10178 | 0.10138 | 0.10660 | 0.000397 | 2.35× |
| B=16, C=2048, KV=1, BF16 | Dense SDPA | 0.07730 | 0.07680 | 0.08504 | 0.000488 | — |
|  | Paged gather + SDPA | 0.24234 | 0.23962 | 0.24904 | 0.000488 | 1.00× |
|  | Triton paged | 0.19174 | 0.19046 | 0.19564 | 0.002686 | 1.26× |
| B=16, C=2048, KV=8, FP16 | Dense SDPA | 0.10909 | 0.10854 | 0.11167 | 0.000061 | — |
|  | Paged gather + SDPA | 0.52542 | 0.52531 | 0.52941 | 0.000061 | 1.00× |
|  | Triton paged | 0.21632 | 0.21606 | 0.22021 | 0.000366 | 2.43× |

三组的独立复测方向一致；它们仍是 microbenchmark，不包含真实模型层调度、采样、排队或网络开销。

## Nsight Compute 限制

已通过容器内 `scripts/profile_attention.sh` 尝试采集单次 autotuned Triton kernel。`ncu` 可启动，但驱动返回 `ERR_NVGPUCTRPERM`，未得到 DRAM throughput、寄存器/occupancy 或 warp stall 计数。没有修改宿主驱动设置，也没有将估算带宽冒充硬件测量。启用宿主 GPU counter 权限后，可在容器中重跑：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  bash scripts/profile_attention.sh
```

## 复现

```bash
./scripts/docker_build.sh
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 scripts/run_benchmark_sweep.py \
  --output docs/benchmark_results/a100_cuda13_triton36.jsonl \
  --warmup 10 --iterations 50
```

`estimated_kv_read_gbps` 是由理论 K/V 读取字节数除以时间得到的估算量，不是硬件计数器结果。`kernel_config` 记录每组 autotuner 实际选择的 tile、warps 和 stages。
