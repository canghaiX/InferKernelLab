# SGLang GSM8K rollout

The optional `--engine sglang` path keeps tokenization and artifact creation in
InferKernelLab while sending local token IDs to a fixed SGLang 0.5.18 server.
The native `/generate` request sends a batch of `input_ids` and per-request
sampling settings. Generated IDs and token logprobs are checked for matching
lengths/IDs and finite, nonpositive probabilities; EOS is kept with its logprob.
The original `paged_hf` Llama path and artifact schema version 1 remain usable.

Protocol: [SGLang v0.5.18 sampling parameters](https://github.com/sgl-project/sglang/blob/v0.5.18/docs/docs/basic_usage/sampling_params.mdx).

The reproduction policy is temperature 0.8, top-p 1, top-k disabled, and eight
responses per prompt group. Each flattened request hashes the base seed and index into a 31-bit seed,
avoiding stream overlap across adjacent base seeds.
With top-p 1 the policy logprob is temperature-scaled and untruncated; the
trainer requires this setting for SGLang GRPO artifacts. There is no fallback
to a downloaded model or a different inference engine on failure.

## Data, prompt and reward

The MiniTrainBench preparation script explicitly downloads and verifies the
MIT-licensed official GSM8K files. Its fixed 256-example train subset and full
1,319-example test split are experiment artifacts, never repository data.
GSM8K rollout requires `--data-manifest`. The artifact records its SHA-256,
embedded metadata, prompt-file digest, local model/tokenizer fingerprints,
SGLang version/server settings digest, source hash and rollout throughput.
The container launcher mounts the actual local weight directory; the HTTP
server's self-reported model path is metadata, not a remote weight checksum.

Prompts use the local Qwen chat template with `enable_thinking=False`, request
step-by-step reasoning and a final `####` numeric answer, and use deterministic
local tokenization. Reward prefers the last marked number, otherwise the last
number in the text. Commas are removed and decimal numeric equality is used.
This fallback may also score truncated outputs, so the local EM is not an
unqualified claim of standard GSM8K benchmark performance.

MiniTrainBench verifies the dataset/model/tokenizer boundary independently,
decodes generated IDs to check stored text, recomputes reward and teacher-forces
every response token before any policy update. Tiny successful probes do not
establish full-dataset BF16 equivalence: the complete 256 × 8 experiment still
fails the explicit maximum-logprob-error gate of 0.5 on the tested kernels.
See [`gsm8k_validation.md`](gsm8k_validation.md) for measured results.

## Evaluation

```bash
PYTHONPATH=src python3 scripts/evaluate_gsm8k.py \
  --model-path /data/models/Qwen/Qwen3-0.6B \
  --data /path/to/gsm8k/test.jsonl --base-url http://127.0.0.1:30000 \
  --batch-size 128 --max-new-tokens 256 --output /tmp/gsm8k-eval.json
```

The evaluator uses the same prompt and endpoint with temperature 0, no sampling
logprob request, and full EOS metadata. It reports exact match, counts,
request throughput, batch latency and fingerprints. An updated model must be
served explicitly before evaluating it. Initial-model scores and throughput
are not evidence of GRPO improvement. No three-seed updated-policy mean/std is
published while the full preflight fails.

During full-suite verification a separate paged prefill issue was found:
masked, uninitialized cache slots can contain NaNs which SDPA can propagate
through zero attention weights. The batched prefill path now zeroes invalid
key/value positions; a ragged-prompt test fills unused cache with NaNs to
verify the fix.

## Logprob diagnosis

`scripts/compare_grpo_logprobs.py` replays selected response positions as
`prefix + target`, requests that target's candidate score through the SGLang
0.5.18 sampler endpoint, and compares it with both the stored generation score
and local HF teacher-forced score. `--preflight-json` selects the actual worst
token from a failed MiniTrainBench report and verifies the rollout file digest.
The one-step sampled token is diagnostic only; the requested candidate token is
the scored target. The score retains the configured temperature and untruncated
top-p 1 distribution.

Measured rollout scores matched the same-server teacher-forced sampler exactly
at 67 checked positions (64 sampled positions plus each seed's worst token).
Rare tail-token SGLang/HF forward differences remained as high as 1.767 on the
host HF/Torch stack and 0.724 when HF ran in the SGLang container. Same-container
execution reduced the error but did not make all worst cases pass 0.5. This
isolates the discrepancy to the SGLang/HF numerical forward path rather than
stored rollout token alignment; the specific layer/operator cause remains open.
The training-side gate continues to stop before optimizer updates.

The diagnostic CLI now has opt-in `--capture-final-hidden-states`,
`--prefix-forward-diagnostic`, and `--repeat-sglang-scores N` controls. With
the server started using `--return-hidden-states-mode full`, these compare the
SGLang final request state against HF and compare full-sequence versus prefix
HF scores. On the three recorded worst cases, SGLang repeated scores were
bitwise stable; full/prefix HF scores differed by 0.592 and 0.774 on two cases.
The SGLang HTTP response does not expose per-layer activations, so this is not
a layerwise attribution and does not establish an operator-level root cause.

For layerwise capture, start SGLang with a diagnostic `forward_hooks` factory
and run `scripts/capture_sglang_layer_probe.py` on one fixed token. The probe
compares input/post-attention norms, attention, gate/up/down projections, MLP,
and combined block outputs. On the three full-preflight worst tokens, the
first block output above 0.1 max error occurred at layers 15, 17, and 7; early
layer outputs were close and divergence grew with depth. Because the first
divergent layer differs per sample and the hook compares module outputs across
different residual representations, this is evidence of accumulated numeric
path divergence, not proof of a single defective operator. Results are in
[`benchmark_results/qwen3_layerwise_logprob_diagnostic.json`](benchmark_results/qwen3_layerwise_logprob_diagnostic.json).

## HF Qwen3 correctness reference

`--engine hf_reference` is a local-only Qwen3 rollout engine for establishing a trainable policy reference. It generates one completion at a time with Transformers `DynamicCache`, stores each temperature-scaled untruncated policy logprob, and keeps EOS with its score. GRPO requires `top_p=1`; FP32 weights and the selected autocast dtype are recorded in the manifest.

The manifest's `forward_contract` binds the rollout to exact PyTorch and Transformers versions, attention implementation, weight dtype, autocast dtype, and incremental-cache mode. MiniTrainBench rejects mismatches before loading the actor or creating an optimizer. Existing `sglang` and Llama `paged_hf` artifacts keep their current interpretation.

This serial path is a correctness baseline with intentionally low throughput. SGLang remains a separate serving engine for deployment-style evaluation and performance work; its logprobs are not used as the HF reference behavior policy. The 0.5 maximum absolute logprob gate still controls whether updates may proceed.

The initial A100 smoke with four GSM8K prompts and eight completions passed
preflight for 1,024 tokens (max absolute error 0.0), but its 32-token cap
truncated all responses and produced no reward variance. An expanded smoke with
eight prompts and a 256-token cap generated 11,332 tokens in 337 seconds; the
trainer preflight again had max error 0.0. One dual-GPU BF16 GRPO update changed
the policy with consistent rank digests and peaked at 19.7 GiB allocated memory.
Step-2 DCP resume then passed with matching rank digests and peaked at 23.9 GiB.
The seed-2026 follow-up is complete: its 256 × 8 HF reference rollout passed
all 419,959 tokens with max/mean/P99 logprob error 0.0. MiniTrainBench completed
16 dual-A100 BF16/DDP steps with changed parameters and consistent ranks, then
verified the final model/TrainState DCP digest. SGLang 0.5.18/FA3 greedy evaluation
on all 1,319 test questions gave 677 correct before and after (51.3268% each):
no measured EM improvement. This is one seed, not a stability or standard
benchmark claim. HF rollout/training ran on the host with dirty provenance;
the container fingerprint applies to SGLang evaluation only. See the
[final report](https://github.com/canghaiX/MiniTrainBench/blob/main/docs/gsm8k_single_seed_result.md). Training processes one
completion at a time to release its autograd cache graph; DDP buffer broadcasts
are disabled for this variable-forward-count path, while gradient reduction
continues on the final completion backward.
# Trainer Batch Alignment Diagnostic

`scripts/compare_grpo_logprobs.py --trainer-batch-diagnostic` additionally scores all completions for a prompt in one right-padded HF forward, matching MiniTrainBench's preflight batch shape (including token-id-0 padding, binary attention mask, BF16 autocast, SDPA, and rollout temperature). Each selected token records that result alongside the one-completion HF forward and SGLang teacher-forced score. The summary at `docs/benchmark_results/qwen3_trainer_batch_alignment_diagnostic.json` captures the three fixed worst-token cases and their rollout hashes.

The measured batch shape changes the error materially for some cases, but removing it does not bring all three worst cases below the fixed 0.5 maximum absolute logprob threshold. The diagnostic therefore does not justify a trainer compatibility patch or an optimizer update.
