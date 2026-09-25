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

## 单层 Synthetic Decoder 端到端结果（2026-09-25）

本节记录固定权重、单层、模型无关 decoder 的真实 runtime 闭环。它不是预训练模型质量或生产 serving 吞吐测试，主要用于验证 scheduler、paged KV cache、KV append、decode attention 和 greedy decode 能否在同一条数据流中协作。

### 环境与配置

- GPU：NVIDIA A100-SXM4-40GB，Compute Capability 8.0
- Driver：580.173.02；PyTorch：2.11.0+cu130；CUDA runtime：13.0；Triton：3.6.0
- 容器：基于 `nano-vllm:optimized` 构建的 `inferkernellab:cuda`
- 配置矩阵：batch `1/4/16`，prompt `128/512`，KV heads `32/8/1`（MHA/GQA/MQA），dtype `float16/bfloat16`
- 固定参数：query heads `32`、head dim `64`、block size `16`、vocab size `256`、生成 `16` tokens、seed `2026`
- 每个 backend 使用新的 runner；warmup `1` 次、独立重复 `2` 次，最终指标取重复运行中位数
- 共 `2 × 3 × 2 × 3 × 3 = 108` 条记录，原始 JSONL：`docs/benchmark_results/a100_synthetic_decode.jsonl`

### 指标口径

- `TTFT`：该 batch 所有 prompt prefill step 加第一个 decode step；表格中的 TTFT 使用 wall-clock，JSONL 同时保留 CUDA Event 设备时间版本。
- `TPOT`：JSONL 显式记录每个生成 token 的平均 decode step 时间，P50/P95 另外描述 step 分布。
- `decode step P50/P95`：每个 runtime decode step 的 CUDA Event 设备时间分位数；prefill 和 decode 的投影、KV append、attention、logits 都在 step 范围内。
- `tokens/s`：batch 生成 token 总数除以 decode 阶段耗时；表格使用 device tokens/s，`end_to_end_tokens_per_sec` 使用包含 prefill 的 wall-clock 吞吐。
- `peak_memory_allocated_bytes`：单次 runner 的 PyTorch allocated memory 峰值；不等于显存 reserved 或线上进程总显存。
- `matches_reference`：逐步 logits 与 `paged_reference` 按 dtype 容差执行 `torch.allclose`。
- `token_match_rate`：最终生成 token 序列与 `paged_reference` 的位置一致率；greedy argmax 对很小的 logit margin 不连续，必须和 logits 误差分开解释。

### 按 dtype 的形状矩阵中位数

下表是在 18 个 shape 上取中位数，不是某一个 workload 的承诺值。延迟单位为 ms，吞吐单位为 device tokens/s。

| dtype | attention backend | TTFT | TPOT | decode P50 | decode P95 | device tokens/s | end-to-end tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | paged_reference | 5.207 | 2.826 | 2.799 | 2.972 | 1415.2 | 1337.8 |
| FP16 | paged_sdpa | 3.747 | 1.451 | 1.426 | 1.608 | 2757.1 | 2487.3 |
| FP16 | triton_paged | 4.126 | 1.719 | 1.674 | 1.910 | 2327.0 | 2127.1 |
| BF16 | paged_reference | 5.562 | 2.824 | 2.779 | 3.011 | 1416.5 | 1338.1 |
| BF16 | paged_sdpa | 3.864 | 1.444 | 1.411 | 1.646 | 2769.8 | 2492.4 |
| BF16 | triton_paged | 4.360 | 1.713 | 1.665 | 1.924 | 2335.5 | 2128.4 |

### 结果解释

- 108/108 条记录的 logits 都通过 dtype-aware correctness check；FP16 最大绝对误差为 `5.97e-4`，BF16 最大绝对误差为 `1.57e-3`。
- greedy token 一致率最低为 `91.4%`，不是把它当作 logits correctness 失败：差异集中在 logits margin 很小的 argmax 位置。选定的 `B=4, context=512, KV heads=8` 配置在 FP16/BF16 的三个 attention backend 中 token match rate 均为 `100%`。
- Triton 相对 paged SDPA 的 decode step P50 比值 `paged_sdpa / triton` 中位数为 `0.841x`，范围 `0.785–0.913`，36 个 shape 中 `0/36` 胜出；即当前 synthetic decoder 中 PyTorch SDPA 仍更快。
- 这个结论与 attention-only benchmark 不矛盾：attention-only 的 Triton direct-paged 路径避免了显式 gather，但端到端路径还包含投影、KV append、kernel launch 和 runtime 调度，PyTorch SDPA backend 可能更成熟。
- 选定 `B=4, context=512, KV heads=8, FP16` 形状的 append 组合数据见 `docs/benchmark_results/a100_synthetic_decode_triton_append.jsonl`：Torch append + Triton attention 的 decode device P50 为 `1.670 ms`，Triton append + Triton attention 为 `1.502 ms`。该结果只有两次独立重复，不外推为普遍收益。

### 复现命令

```bash
./scripts/docker_build.sh
./scripts/docker_test.sh

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 scripts/run_decode_sweep.py \
  --device cuda --dtypes float16,bfloat16 \
  --batch-sizes 1,4,16 --context-lens 128,512 \
  --num-kv-heads-list 32,8,1 --max-new-tokens 16 \
  --num-heads 32 --head-dim 64 --block-size 16 \
  --vocab-size 256 --seed 2026 --max-num-batched-tokens 2048 \
  --warmup 1 --repeats 2 \
  --backends paged_reference,paged_sdpa,triton_paged \
  --output docs/benchmark_results/a100_synthetic_decode.jsonl
```

`ncu` 的 `ERR_NVGPUCTRPERM` 限制仍然存在。本节没有把理论 KV 读取量写成硬件 DRAM throughput，也没有声称已经测得 occupancy 或寄存器瓶颈。
