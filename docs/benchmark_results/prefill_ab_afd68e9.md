# Batched Prefill A/B

## Configuration

- Commit: `afd68e9` (`perf: add batched llama prefill`), clean worktree.
- GPU: NVIDIA A100-SXM4-40GB; CUDA 12.4; PyTorch 2.6.0+cu124.
- Model: deterministic random 2-layer Llama, hidden size 128, intermediate size 512,
  8 attention heads, 2 KV heads, BF16, block size 16.
- Backend: `paged_sdpa`, Torch KV append, seed 2026, `max_num_batched_tokens=2048`.
- Measurement: 10 warmup runs and 50 measured runs per mode; wall-clock runner result.
- Oracle: Transformers eager full-sequence path; all four records have `git_dirty=false`.

## Results

| Shape | Mode | Prefill ms | Prefill P50 ms | TTFT ms | Decode P50 ms | Max error | Token match |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 × C128 | token | 747.876 | 746.299 | 751.610 | 3.524 | 4.88e-3 | 100% |
| B2 × C128 | batched | 4.694 | 4.508 | 8.274 | 3.479 | 5.86e-3 | 100% |
| B4 × C512 | token | 6009.891 | 6013.398 | 6014.581 | 4.361 | 7.81e-3 | 100% |
| B4 × C512 | batched | 5.623 | 5.509 | 10.229 | 4.369 | 5.86e-3 | 100% |

Relative to the token baseline, batched Prefill improves total Prefill latency by
`159.3x` on B2×C128 and `1068.7x` on B4×C512; the geometric mean is `412.7x`.
TTFT improves by `90.8x` and `588.0x`, respectively, with a `231.1x` geometric
mean. Decode P50 remains effectively unchanged, so the improvement comes from
removing the prompt-side per-token Python/model dispatch rather than changing the
decode kernel.

Both modes produce greedy token match rate `1.0`, release all KV blocks, and pass
the dtype-aware BF16 logits tolerance of `3e-2`. The raw JSON records were kept
outside the repository to avoid committing repeated benchmark payloads; rerun the
same experiment with:

```bash
IMAGE=inferkernellab:cuda scripts/run_prefill_ab.sh
```
