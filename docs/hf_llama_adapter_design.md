# Transformers/Llama Paged Adapter

## Scope

`inferkernellab.hf_benchmark` adds an optional integration path for a real
`LlamaForCausalLM`. The core package and existing attention tests do not import
Transformers. Install the adapter with `pip install 'inferkernellab[hf]'`.

`--model-source random` creates a deterministic tiny `LlamaConfig` locally and
never downloads weights. `--model-source path --model-path PATH` is the only
checkpoint-loading path and passes `local_files_only=True`, so a missing local
file fails instead of silently reaching the network.

## Execution path

The runner reuses `InferenceRuntime`, `TokenBudgetScheduler`, `BlockAllocator`,
`PagedKVCache`, and the existing paged attention backends. For every decoder
layer it performs:

1. embedding and RMSNorm;
2. Q/K/V projection and Transformers-compatible RoPE;
3. per-layer K/V write into logical request positions;
4. paged reference, paged SDPA, Triton attention, or grouped GQA/MQA Triton attention over the layer's cache;
5. output projection, residual, RMSNorm, gated SiLU MLP, and residual.

The cache layout is `[layer, physical_block, token_in_block, kv_head, head_dim]`.
The same request block table is used at every layer, while the layer index selects
the correct K/V plane. Runtime reserves new blocks before decode, then releases
the request table when generation finishes.

## Correctness and metrics

The optimized path is compared with a Transformers eager full-sequence oracle.
The oracle is intentionally used for both the final prompt logits and every
greedy decode decision, avoiding tokenizer or remote checkpoint assumptions.
The record reports logits max error, greedy token match rate, prefill latency,
decode P50/P95, TTFT, TPOT, peak KV memory, backend/kernel configuration, and
cache release status.

This is an auditable model adapter, not a production serving API: quantization,
continuous batching, tensor parallel inference, streaming HTTP, and automatic
remote model download are out of scope.
