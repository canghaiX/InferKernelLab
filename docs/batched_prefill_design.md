# Batched Prefill Design

## Goal

The original model adapter executed one request and one prompt token per
runtime callback. That path is useful as a correctness baseline, but it makes
Python dispatch and repeated small matrix multiplications dominate tiny-model
TTFT. The new path keeps the scheduler and paged cache contract while running
one scheduler Prefill batch at a time.

## Runtime contract

`InferenceRuntime` accepts either the historical callback:

```text
prefill_fn(request, start_token, end_token)
```

or the batched callback:

```text
prefill_batch_fn(requests, starts, ends)
```

The runtime snapshots `cached_tokens` before scheduling, reserves capacity for
every request, and passes the exact logical ranges advanced by the scheduler.
The callbacks are mutually exclusive. Existing callers that only provide
`prefill_fn` keep the original behavior.

## HF Llama path

For a batch of ragged ranges, the adapter pads token IDs to the largest range
in that scheduler batch. Embedding, RMSNorm, Q/K/V projection, RoPE, output
projection, MLP, and final LM head then run over the padded tensor. Only valid
tokens are written to the paged cache through flattened physical slots:

```text
(request_i, logical_position)
  -> block_table_i[logical_position // block_size]
  -> physical_slot
```

After K/V writes, the attention helper gathers each request's complete prefix
from paged storage. A boolean mask combines three conditions:

1. the key position is inside that request's context;
2. the query row is a real, non-padding token;
3. the key position is no later than the query's absolute position.

PyTorch SDPA consumes the resulting `[batch, query_heads, query_tokens,
context_tokens]` mask. GQA/MQA uses SDPA `enable_gqa=True` when available and a
head expansion fallback otherwise. Decode continues to use the selected
`paged_reference`, `paged_sdpa`, Triton, or grouped Triton backend.

## Synthetic path

The synthetic decoder does not need prompt attention logits, so its batched
callback combines token construction, Q/K/V projection, and flattened paged
KV writes. This isolates scheduler and cache-write overhead without changing
the existing decode correctness path.

## Measurement and correctness

`prefill_mode=token` is the historical regression baseline and
`prefill_mode=batched` is the default optimized path. Both modes use the same
deterministic prompts, model weights, token budget, block layout, and decode
backend. HF correctness compares final prompt and decode logits with the
Transformers eager full-sequence oracle and checks greedy token matches.

Results retain schema version `1` and add fields for mode, Prefill batch count,
effective token count, padding count, and Prefill P50/P95. TTFT remains the
sum of all scheduled Prefill steps plus the first decode step. CUDA benchmark
results must report the exact command, commit provenance, dtype, GPU, warmup,
iterations, and whether the 2x target was met; a target is not presented as a
guaranteed speedup.

## Boundaries

This implementation intentionally uses PyTorch SDPA for Prefill rather than a
new Triton causal-Prefill kernel. Quantization, prefix sharing, tensor-parallel
inference, streaming HTTP, and production continuous batching remain outside
the scope of this iteration.
