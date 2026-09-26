# Grouped GQA/MQA Paged Attention

`triton_paged_grouped` 是现有 `triton_paged` 的显式 opt-in 变体，目标是减少 GQA/MQA
中同一 KV head 被多个 query head 重复读取的问题。旧 backend 和 benchmark schema 保持
兼容，不因为新 kernel 在某个 shape 上没有收益而强行替换默认路径。

## Program Mapping

旧 kernel 使用 `(request, query_head)` 作为 program grid。grouped kernel 使用：

```text
(request, kv_head, query_head_group)
```

一个 program 在 query-head group 内维护独立的 online softmax 状态，但复用同一组 K/V
tile。`query_group_size` 默认是 8；当 `num_heads == num_kv_heads` 时 wrapper 退化到旧
kernel，避免 MHA 引入额外寄存器开销。

当前实现支持 group size `1/2/4/8`、head dimension 不超过 128、ragged context、非连续
physical block table、multi-layer cache，以及 FP16/BF16。尾部不足一个完整 group 的 query
heads 通过 mask 处理。

## Correctness

测试把 grouped output 与 PyTorch paged reference 对齐，覆盖：

- MHA、GQA 和 MQA；
- FP16、BF16；
- 跨 block boundary 的 context；
- multi-layer cache 的指定 layer；
- HF Llama adapter 的 eager oracle 和 greedy token match。

`logits_max_error` 和 `token_match_rate` 分开记录。argmax 对很小的 logit margin 不连续，
所以 token 差异不能单独作为浮点 correctness 结论。

## Benchmark

`run_decode_sweep.py` 可同时运行：

```bash
python3 scripts/run_decode_sweep.py \
  --device cuda --batch-sizes 4,16 --context-lens 128,512 \
  --num-kv-heads-list 8,1 --backends paged_sdpa,triton_paged,triton_paged_grouped
```

结果额外记录：

- `kernel_variant=grouped`；
- `query_group_size`；
- `kv_reuse_factor`；
- grouped Triton autotune 的 block/warp/stage 配置。

性能结论只使用相同 shape、dtype、warmup、重复次数和 CUDA Event 口径下的 P50/P95。
理论 KV read reuse 不是 Nsight DRAM counter；如果宿主仍返回 `ERR_NVGPUCTRPERM`，报告
不会伪造 bandwidth 或 occupancy 数据。

截至 2026-09-26 的 A100 smoke 中，grouped kernel 已通过 FP16/BF16 correctness，但在
当前 synthetic shape 上没有稳定优于旧 Triton backend，因此保持显式 opt-in。具体数据见
`docs/benchmark_results/grouped_gqa_smoke.md`。
