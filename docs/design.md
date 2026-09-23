# Design Notes

## Why this project is separate

`MiniTrainBench` is a distributed training runtime and evidence suite. This
repository isolates inference memory management and GPU kernels so a benchmark
can answer one question at a time.

## Logical versus physical KV positions

A request sees a logical token sequence. The allocator owns physical blocks.
The block table is the indirection layer:

```text
logical position p
  -> logical block p // block_size
  -> physical block block_table[logical block]
  -> cache slot physical_block * block_size + p % block_size
```

This means a request does not need a contiguous physical allocation. It also
makes prefix sharing possible, although prefix-cache reference semantics are a
follow-up milestone.

## Reference versus optimized implementation

The PyTorch implementation intentionally loops over requests and reads logical
tokens explicitly. It is not expected to be fast. It provides the correctness
oracle for the Triton implementation and exposes the cost of paged addressing.

The Triton kernel assigns one program to each `(request, query_head)` pair and
performs an online softmax over the paged K/V sequence. It currently uses a
bounded reference loop and is intended for correctness and profiler work before
more aggressive tiling and pipelining are added.

## Metrics to add next

- P50/P95 latency instead of only mean latency.
- Dense contiguous KV versus paged KV baseline.
- Context-length sweep and batch-size sweep.
- Nsight Compute SOL, DRAM throughput, register pressure, and warp stalls.
- A replay trace that reports TTFT, TPOT, and active-request count.

## Runtime boundary

`InferenceRuntime` owns resource lifecycle but intentionally does not own model
weights or logits. This keeps scheduler and KV-cache experiments independent of
any particular Hugging Face model. A future model adapter can call `step()` to
obtain a batch, run prefill/decode, and write the newly produced K/V values to
the request's block table.
