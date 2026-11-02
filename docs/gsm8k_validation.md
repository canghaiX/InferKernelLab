# Qwen3/SGLang GSM8K 实验验收（2026-11-03）

**实现与组件 smoke 已通过，完整三 seed BF16 训练闭环尚未通过验收。**
全部 256 × 8 rollout 的 logprob 预检失败，正式实验在 optimizer 更新前停止，
因此没有三个更新模型的 test EM、均值/标准差或训练提升结论。
原始摘要见 [`benchmark_results/qwen3_gsm8k_validation.json`](benchmark_results/qwen3_gsm8k_validation.json)。

## 固定条件

- 本地 Qwen3-0.6B；模型和 tokenizer SHA-256 写入 JSON，未下载权重。
- GSM8K main train 按 seed 2026 固定选 256 题；完整官方 test 1,319 题。
  源文件 SHA-256 校验和 MIT 许可记录在数据 manifest。
- A100-SXM4-40GB，推理独立一张卡，训练使用两张卡。
- SGLang 0.5.18，FA3，`--rl-on-policy-target fsdp`，PyTorch sampler；
  容器 Torch 2.13.0/CUDA 13.0、Transformers 5.12.1。
- HF trainer 为 Torch 2.6.0/CUDA 12.4、Transformers 4.57.6；SDPA、BF16 autocast。
- temperature 0.8，top-p 1、禁用 top-k、每题 8 个回答、最多 256 token，保留 EOS。
  每个请求以 base seed 和序号的哈希派生 seed，避免相邻 seed 间复用随机流。
- 最大绝对 logprob 误差门槛固定为 0.5，未为通过实验而提高门槛。

## 完整 rollout 预检

| Rollout seed | Token 数 | Max error | P99 error | Mean error | 结果 |
|---|---:|---:|---:|---:|---|
| 2026 | 422,994 | 0.652631 | 0.155666 | 0.011141 | 失败，无参数更新 |
| 2027 | 419,699 | 1.254301 | 0.154289 | 0.011157 | 失败，无参数更新 |
| 2028 | 425,292 | 0.811586 | 0.153992 | 0.011123 | 失败，无参数更新 |

三个 rollout 的本地端到端收集吞吐约为 9,946、10,088、10,056 token/s。
这些数包含 HTTP 收集开销，不能替代统一负载的 serving benchmark。
约 0.38%–0.41% 的 token 初始 importance ratio 已在 [0.8, 1.2] 外，
故不能只用低平均误差来忽略尾部差异。最差 token 的题号、位置和双方 logprob
都记录在 JSON 中。

排查中 HF eager 的全量误差明显高于 SDPA；切换到 SDPA 降低了误差，但仍未达标。
FlashInfer + 对齐模式的一个完整 rollout 最大误差为 0.648108；
关闭对齐模式后为 1.108113。同容器 HF 版本也做过诊断，未建立完整数值一致性。
这些是排查记录，不是已证明的根因；当前不能将问题单独归因于某个 attention kernel。

## 已通过的较小闭环组件

另用 **4 题 × 8 个回答、最多 256 token** 做双卡 BF16 smoke：

- 完整响应 token 预检通过，max error 为 0.409802。
- 执行 2 步 DDP 更新，参数实际改变；两个 rank 参数摘要相同。
- 从第 1 步 DCP checkpoint 恢复到第 2 步，与连续训练的最终参数 SHA-256 完全一致。
- 峰值分配显存 25,746,937,344 bytes（约 24 GiB）；短跑数字不作为稳定性能结论。
- 导出的模型重新通过 SGLang 加载，显式使用 BF16；全量 test 仍为 **677/1,319（51.33%）**，
  与初始模型相同，未观察到 EM 提升。初始和导出 tokenizer 对 1,319 个 prompt 的
  token IDs 逐题比较完全一致。

该 smoke 的训练集合与正式 256 题实验不同，且只使用单个 seed，不能用于替代正式验收。

## 完整测试集基线与边界

初始模型本地 greedy EM 为 **677/1,319 = 51.33%**；同一 no-thinking chat 提示、
最多 256 token，最后标记答案优先、无标记则取最后数字。它是本地完整测试集基线，
不是标准 benchmark 成绩，也不是训练后的收益。

完整 pytest：MiniTrainBench 57 passed，InferKernelLab 92 passed。
覆盖旧 Llama artifact、tiny Qwen3、GRPO loss、EOS mask、DDP、DCP resume、
GSM8K 数据校验/reward、SGLang HTTP 批量请求/响应、失败预检和错误汇总拦截。

复现入口与数据/概率约定见 [`sglang_gsm8k_rollout_design.md`](sglang_gsm8k_rollout_design.md)。
当前脚本按预期会在完整预检失败时停止。先解决尾部 logprob 差异，再执行三组
16 步更新及三个更新模型的全量重评；在此之前不发布更新后均值/标准差。
实验发生在功能提交前，JSON 保留当时父 commit、dirty 状态和源码摘要，
未将它伪装为 clean container provenance。
