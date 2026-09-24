# 设计说明

## 项目边界

MiniTrainBench 展示训练与预训练基础设施。本项目聚焦推理侧的 paged KV cache、decode attention kernel 和可复现实验。当前 runtime 不执行真实模型，不管理权重或 logits；scheduler 是便于检查 token-budget 规则的示例实现。

## 逻辑与物理 KV 位置

请求看到的是连续逻辑 token 序列，allocator 管理的是物理 block。block table 是两者之间的间接层：

```text
逻辑位置 p
  -> 逻辑块 p // block_size
  -> 物理块 block_table[逻辑块]
  -> cache slot = 物理块 * block_size + p % block_size
```

请求无需占用连续物理内存。block table 也为后续 prefix sharing 留出表示空间，但本项目尚未实现引用计数或 prefix-cache 语义。

## Reference、SDPA 与 Triton

- PyTorch reference 显式读取逻辑 token，并按请求执行，主要用途是正确性 oracle，不预期有竞争力的延迟。
- dense SDPA 使用预先连续堆叠的 K/V 和单次 batched PyTorch `scaled_dot_product_attention`，避免把逐请求 Python loop 当作 SDPA 基线。
- paged SDPA 在计时过程中用规则索引一次性 gather 出 batched 连续 K/V，再执行 batched SDPA；计时包含 gather，提供与直接 paged 访问更接近的端到端比较。
- Triton kernel 为每个 `(request, query_head)` 分配一个 program，在逻辑 block table 上遍历 K/V，并使用 online softmax 累积输出。

Triton 目前限制 `head_dim <= 128`，并通过 `head_dim`、table capacity 选择 autotune 配置。wrapper 默认检查设备、dtype、形状、上下文长度和物理 block 编号；benchmark 先做一次完整校验，再用预校验张量计时，避免每次重复触发 GPU 同步。

## 在线 softmax

对一个 tile 的分数 `s`，kernel 维护当前最大分数 `m`、归一化分母 `l` 和加权值累积 `o`。当新 tile 的最大值为 `m'` 时，旧累积按 `exp(m-m')` 重缩放，再合并新 tile 的 `exp(s-m')` 权重。这样无需保存整条 context 的 score/probability 矩阵，且可稳定地跨 tile 归约。

## 正确性与边界

- allocator 在释放前先验证整组 block，拒绝重复、未分配、越界或 owner 不匹配的释放，避免部分更新导致 allocator 状态损坏。
- 支持 MHA/GQA/MQA 的条件是 `num_heads % num_kv_heads == 0`；query head 按组映射到对应 KV head。
- Triton block table 需是规则二维张量；ragged request 用有效物理 block 填充，另由 `context_lens` 限定有效 token 数。
- reference 和 SDPA 路径覆盖不同请求长度；Triton wrapper 检查上下文不能超过 table capacity。
- Triton CUDA 测试分别覆盖 FP16、BF16、FP32；各 dtype 使用不同数值容差。

## Benchmark 解释

benchmark 在相同 batch、context、head layout、dtype、warmup 和迭代次数下测量各实现。物理分配在 logical blocks 之间插入保留 block，使逻辑访问不对应连续物理页。`dense` / `paged_reference` 仅是数学参考，不代表生产级优化实现；性能结论优先对比 `triton_paged` 与 `paged_sdpa`，并同时报告 P50/P95 和最大误差。

理论 KV 读取字节数用于计算 `estimated_kv_read_gbps`，不是硬件测量。要报告 DRAM throughput、寄存器或 occupancy，需要 Nsight Compute 硬件计数器。当前容器的 `ncu` 被宿主驱动的 `ERR_NVGPUCTRPERM` 权限策略阻止，因此现阶段不报告硬件计数器结果。
