# InferKernelLab 面试指南

## 一分钟项目介绍

InferKernelLab 聚焦 LLM decode 阶段的 KV cache 与 attention kernel。我实现了逻辑到物理 block 映射、PyTorch correctness reference、连续 KV 的 SDPA 基线、paged gather + SDPA 基线，以及直接访问 paged KV 的 Triton kernel；本轮又用固定权重的单层 synthetic decoder 把 embedding、KV append、attention、greedy decode 和 runtime scheduler 串成端到端闭环。项目重点不是声称某个数字普遍更快，而是用统一输入、正确性误差和可复现测量解释不同实现的成本。它和 MiniTrainBench 形成互补：一个展示训练/预训练系统，一个展示推理内存与 GPU kernel 优化。

## 讲解顺序

1. **问题**：长上下文 decode 每步都要读取历史 K/V；显存容量、带宽和访问布局直接影响吞吐。
2. **内存设计**：物理 cache 以固定 block 管理，请求通过 block table 将逻辑 token 映射到物理 slot。
3. **计算实现**：reference 保证数学正确；SDPA 提供 fused baseline；Triton kernel 在线读取 page，并以 online softmax 累积结果。
4. **实验方法**：固定 shape、dtype、warmup、迭代数与 CUDA Event 协议，比较误差和 P50/P95；profile 权限不可用时明确标注限制。
5. **下一步**：优先减少 GQA 对同一 KV 的重复读取、分析长 context 下的带宽瓶颈，并在获得硬件 counter 权限后验证瓶颈假设。

## 高频知识点与项目关联

### Paged KV cache

- **为什么分页**：避免为每个请求预留最大长度的连续 KV 区域，便于按需分配、回收和处理不同长度请求；代价是 block table 间接寻址和页内碎片。
- **地址计算**：`logical_block = token_pos // block_size`；`physical_block = block_table[logical_block]`；`slot = physical_block * block_size + token_pos % block_size`。
- **block size 的取舍**：小 block 减少尾部浪费但增加 table 与寻址开销；大 block 降低元数据开销但可能增加内部碎片。
- **容量估算**：单请求 KV 字节数约为 `2 * num_layers * tokens * num_kv_heads * head_dim * bytes_per_element`；系数 2 来自 K 和 V。
- **追问**：共享 prefix 需要哪些状态？除了 block table，还需要引用计数/所有权、不可变共享页语义，以及写时复制或尾块隔离。

### MHA、GQA 与 MQA

- MHA 中 `num_kv_heads == num_heads`；GQA 中多个 query heads 共享一个 KV head；MQA 中所有 query heads 共享单个 KV head。
- 本项目按 `query_head * num_kv_heads // num_heads` 映射 KV head，因此要求 `num_heads` 可被 `num_kv_heads` 整除。
- GQA/MQA 可减少 KV cache 存储和理论读取量；但如果 kernel 按 query head 单独发 program，同一 KV head 仍可能被重复加载。
- **可继续优化**：让一个 CTA 同时处理一个 query-head group，复用 K/V tile；需权衡寄存器占用、并行度和 softmax 状态数量。

### Decode attention 与 online softmax

- Decode 每步 query 长度通常为 1，主要成本是扫描历史 K/V；常见长上下文场景更接近 memory-bound，而 prefill 的 Q/K/V 计算规模和并行方式不同。
- online softmax 维护当前最大值 `m`、归一化分母 `l` 和加权和 `o`。新 tile 到来后用 `alpha = exp(m - m_new)` 重缩放旧 `l/o`，再并入新 tile 的指数权重，最终输出 `o/l`。
- 数值稳定性来自减去 running max；accumulator 使用 FP32，输入和输出可为 FP16/BF16。
- **追问**：为什么不能每个 tile 独立 softmax 再拼接？因为不同 tile 的归一化分母不同，必须用 running max 与 denominator 做全序列一致归一化。

### Prefill、Decode 与 Synthetic Decoder

- **Prefill** 通常一次处理多个 prompt token，计算量更偏矩阵乘和高并行；**decode** 每步通常只有一个新 token，但要扫描历史 KV，长上下文时更容易受显存访问和 launch 开销影响。
- 本项目的 runtime 在 prefill 阶段按 token budget 分段，并通过 `prefill_fn(request, start_token, end_token)` 把实际推进的 prompt 范围交给模型；decode 阶段先调用 `decode_fn(requests)`，再更新 `generated_tokens` 和释放已完成 request 的 block。
- synthetic decoder 是固定权重的单层闭环，不是外部预训练模型。它验证的是 embedding、投影、KV append、paged attention、logits 和 scheduler/cache 生命周期的组合关系，不能用来声称模型质量或生产 serving 性能。
- `TTFT` 包含 prefill 和第一个 decode step；`TPOT` 在本项目中显式记录平均 decode step 时间，P50/P95 另外描述 step 分布。两者都要说明是 CUDA Event 设备时间还是包含 Python/runtime 的 wall time。
- **追问**：为什么不直接接 HuggingFace？本轮目标是隔离 runtime/kernel 变量、避免下载权重和额外依赖；下一步若接真实模型，应固定 tokenizer、权重、采样和 batching 协议后重新测量。

### 性能与 profiling

- 区分 reference correctness oracle、dense SDPA、paged gather + SDPA 与 direct-paged Triton。reference 很慢不代表优化 kernel 快于成熟 fused kernel。
- 比较 paged kernel 时，paged SDPA 的计时包含 gather；dense SDPA 表示连续 KV 的上界式对照，二者含义不同。
- 等长 batch 的 dense SDPA KV 在计时前已连续堆叠；paged SDPA 在计时中批量 gather 并调用一次 batched SDPA。两者都避免逐请求串行执行 attention，但 paged 路径额外包含索引和 gather。
- benchmark 在 request 的逻辑 block 之间插入保留物理 block，验证 block table 访问不是简单连续寻址；这是可控 microbenchmark 布局，不代表所有 serving allocator 的真实碎片分布。
- CUDA Event 记录设备时间线，不直接包含网络、请求排队等线上延迟；warmup 用来排除首次 compile/autotune，P50/P95 描述分布但需足够样本。
- `estimated_kv_read_gbps` 是理论字节数除以时间，不是硬件读数。真实 DRAM throughput、寄存器和 occupancy 需 Nsight Compute counter。
- 当前容器 `ncu` 因 `ERR_NVGPUCTRPERM` 无法访问计数器；面试应坦诚说明，没有 profiler 数据时不要推断带宽饱和或 occupancy 瓶颈。
- autotune key 包含 `head_dim` 与 table capacity，以便不同 head 维度和 context bucket 分别选择 `block_n`、warps、stages。讨论时应查看实测输出的 `kernel_config`，不要假设某配置必然更好。

### 内存 allocator 与 runtime

- allocator 释放先整组校验，再统一修改空闲/占用集合，保证非法 id、重复 id、owner 错误时不会部分释放。
- runtime/scheduler 通过可选 callback 支持 synthetic decoder，但仍不负责加载模型、执行完整 Transformer 或提供 serving API。这是项目边界，不能描述成可直接 serving 的框架。
- callback 异常会直接向调用方抛出；benchmark 每次新建 runner，避免异常后复用半完成的 scheduler/cache。生产 runtime 还需要取消请求、回滚分配和更明确的错误状态。
- 默认 attention 对照统一使用 Torch KV append；Triton append 单独记录，避免把 append 差异误认为 attention kernel 差异。
- 可继续验证 out-of-block、OOM 时的事务语义、请求取消、prefix sharing 和真实模型 KV 写入路径。

## 常见追问

- **为什么写 reference？** 给优化路径提供独立、易读的 correctness oracle；reference 不用于性能结论。
- **为什么比较 paged gather + SDPA？** 将 page gather 成本纳入 baseline，避免只比较裸 attention 或只和低效 Python 参考比。
- **为什么 Triton 可能更快/更慢？** 取决于 context、batch、GQA 比例、launch 开销、tile 和 SDPA 后端；结论必须按 shape 报告。
- **为什么 attention-only 中 Triton 可能更好，但端到端 synthetic decoder 中反而输给 paged SDPA？** attention-only 主要观察 gather 与直接 paged 访问的差异；端到端还包含投影、KV append、kernel launch 和 runtime 开销，PyTorch SDPA backend 可能有更成熟的融合和调度。
- **为什么 logits allclose 通过但 token match rate 不是 100%？** logits 是连续值，`allclose` 允许很小误差；greedy argmax 在两个候选 logit 很接近时是不连续的，小舍入差异即可改变后续 token。要同时报告 logits 误差和最终生成序列的一致率。
- **怎么证明优化有效？** 同环境、同输入、正确性通过、足够 warmup/采样、重复运行趋势一致，并能解释硬件计数器；当前最后一项受宿主 counter 权限限制。
- **下一步做什么？** 先让 profiler 可采样，再评估 GQA KV reuse；之后再考虑 prefill kernel 或真实模型 adapter，而不是同时扩成不完整 serving 系统。

## 简历表述模板

> 实现支持 MHA/GQA/MQA 的 paged KV cache 与 Triton decode attention；构建 PyTorch reference、dense SDPA 和 paged gather + SDPA 对照，覆盖（报告中的配置范围），在 A100 上测得（仅填写可复现数据），最大误差（填写对应 dtype 数据）。

实际写简历时只填 `benchmark_report.md` 中已复现的形状和测量结果，并明确这是 microbenchmark，不等同整模型 serving 吞吐。
