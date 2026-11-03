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
