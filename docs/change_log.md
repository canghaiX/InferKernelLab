# 重要改动记录

本文件用于面试复盘，记录真实实现、取舍、验证和局限。性能数字以 `benchmark_report.md` 与对应 JSONL 为准，不手填预测结果。

## 2026-09-23：基准、边界与文档升级

### 改动点

- **allocator 释放原子化**：此前逐个 block 释放，遇到非法 id 或重复 id 时可能造成 allocator 状态部分改变。现在先校验整组 id、分配状态与 owner，再统一更新；runtime 释放时传入 request id。
- **补齐 SDPA 对照**：新增连续 KV 的 PyTorch SDPA 基线，以及包含 KV gather 的 paged SDPA 基线。保留原 `dense` / `paged_reference` 路径作为 correctness oracle 和兼容输出。
- **保持 batch 基线公平**：等长 dense KV 在计时前堆叠，计时期间执行单次 batched SDPA；paged SDPA 使用 batched gather 后执行 batched SDPA，不再把逐请求 attention loop 当作等长 batch 的 SDPA 基线。
- **覆盖物理页离散访问**：benchmark 在不同请求的 logical block 之间插入保留物理 block，避免测试布局退化为连续物理页；配置和 JSONL 记录该布局。
- **减少 paged gather 不必要同步**：slot table 容量可由 `start`、`length`、`block_size` 在 CPU 侧推导，移除对 GPU 张量 `.item()` 的范围检查，避免基准路径额外 GPU→CPU 同步。
- **加强 Triton wrapper 校验**：默认检查维度、head 数整除关系、dtype/device、metadata 形状和索引范围；benchmark 先严格校验一次，再用预校验 metadata 计时，避免重复同步污染 kernel 测量。
- **调整 autotune 维度**：将 table capacity 加入 autotune key，并在 JSON 记录最终 `block_n`、`num_warps`、`num_stages`，方便复盘配置选择。
- **容器环境统一**：项目 CUDA 镜像默认基于 `nano-vllm:optimized`，不在宿主 Python 环境安装项目依赖。

### 验证与结果

- `tests/test_cache.py`、`tests/test_attention.py`、`tests/test_benchmark.py` focused suite：30 项通过。
- `./scripts/docker_test.sh` 在 `nano-vllm:optimized` 派生镜像中全量通过：33 项测试通过，CUDA/Triton 用例实际执行。
- 最终 sweep：54/54 组通过 dtype-aware correctness check；原始数据与 9 次独立复测保存在 `docs/benchmark_results/`。
- Triton 相对 paged gather + SDPA 的 mean-latency ratio 中位数为 2.179×（54/54 组大于 1）；相对 dense SDPA 为 0.671×（0/54 组大于 1）。不把后者包装成 Triton speedup。
- 初始 smoke 配置：A100、FP16、batch=2、context=64、heads=4、KV heads=2、head dim=32、block size=16，warmup=2、iterations=5。该结果只验证链路，不作为稳定性能结论。
- batched 基线 smoke（同配置，仅 5 次采样）：dense SDPA 约 0.068 ms，paged gather + SDPA 约 0.227 ms，Triton 约 0.105 ms；Triton 约为 paged SDPA 的 2.16x，但只有 dense SDPA 的 0.65x。样本太少，只用于确认“比较对象会改变结论”，不引用作简历性能数字。
- Nsight Compute 已在容器中执行，但宿主 NVIDIA 驱动拒绝 GPU performance counter 访问，报错 `ERR_NVGPUCTRPERM`。未采集或伪造 DRAM/occupancy 指标；需要宿主管理员启用计数器权限后重试。

### 面试时怎么讲

1. 先解释原始基线的问题：只与显式 PyTorch reference 比，会把 Python loop 和索引开销误当成 kernel 优势。
2. 再解释新的比较：dense SDPA 看连续布局下的成熟 attention 路径；paged gather + SDPA 将 gather 成本纳入对照；Triton 直接读 paged cache。
3. 说明测量边界：CUDA Event 报告 GPU 时间线耗时，不等同线上请求 latency；估算带宽不是硬件 DRAM counter。
4. 主动说明限制：当前 kernel 以 `(request, query_head)` 为 program，GQA 下同组 query head 可能重复读取 KV；进一步优化可将 query-head group 融合以提高 KV 复用。

## 后续记录模板

每个重要迭代追加一节，按“问题 → 方案与取舍 → 验证 → 实测 → 局限/下一步”填写。没有实际运行的数据标记为“未测”，不要写预期 speedup。

## 2026-09-25：单层 synthetic decoder 端到端闭环

### 改动点

- **runtime hook**：在不改变原有 `InferenceRuntime.step()` 调用方式的前提下，增加 `prefill_fn(request, start_token, end_token)` 和 `decode_fn(requests)`。prefill 范围取 scheduler 前后的 `cached_tokens` 差值；decode hook 在生成计数更新前执行，便于模型先计算当前 step，再由 runtime 统一推进状态和释放 block。
- **模型无关 decoder**：新增单层固定权重 decoder，包含 embedding、Q/K/V projection、paged KV append、三种 decode attention backend、输出 projection、vocab head 和 greedy argmax。不引入 HuggingFace、权重下载或额外依赖。
- **公平 backend 对照**：默认所有 attention 对照都使用 Torch KV append；Triton append 作为单独组合实验，不把 append 差异混入 attention-only 结论。
- **端到端指标**：新增 TTFT、prefill time、decode step P50/P95、wall/device tokens/s、显存峰值、peak block 和 logits/token correctness 字段。
- **数据可追溯**：新增 `a100_synthetic_decode.jsonl`（108 条记录）和 `a100_synthetic_decode_triton_append.jsonl`（选定形状 2 条记录）。

### 验证与实际结果

- 宿主 CPU 测试：39 项通过；Docker 派生镜像中的 CUDA/Triton 全量测试：39 项通过。
- A100 synthetic sweep：2 个 dtype × 3 个 batch × 2 个 context × 3 个 KV head × 3 个 attention backend = 108 条；108/108 条 logits 通过 dtype-aware `allclose`。
- FP16 最大绝对误差 `5.97e-4`，BF16 最大绝对误差 `1.57e-3`。greedy token 一致率最低为 `91.4%`，原因是近似相等的 logits 在不同数值路径下发生 argmax 翻转；稳定选定的 `B=4,C=512,KV=8` 配置在 FP16/BF16 的三个 attention backend 中均为 100%。
- synthetic decode 中 Triton 相对 paged SDPA 的 decode step P50 中位数比值为 `0.841x`，即当前 Triton 约慢 18.9%；这是真实的“尚未胜过 PyTorch SDPA”结果，不把 attention-only 的局部收益包装成端到端收益。
- 选定 `B=4,C=512,KV=8,FP16` 形状下，Triton append + Triton attention 的 device decode step P50 为约 `1.502 ms`，Torch append + Triton attention 为约 `1.670 ms`；仅作为组合路径示例，不能外推为普遍加速。

### 面试时怎么讲

1. 先说清楚这是模型无关 synthetic decoder，不冒充真实预训练模型；它的价值是把 cache、scheduler、append、attention 和 token 生成串成可测闭环。
2. 说明为什么 attention-only Triton 可能胜过 paged gather + SDPA，但端到端 synthetic decoder 仍输给 paged SDPA：PyTorch SDPA backend 可能更成熟，且端到端包含投影、launch 和 Python/runtime 开销。
3. 区分 logits `allclose` 与 greedy token 一致率：argmax 对小 margin 不连续，token mismatch 不必然表示数值实现错误。
4. 强调下一步不是继续堆 synthetic 层数，而是解决 profiler counter 权限或接入真实小模型做 TTFT/TPOT 复测。
