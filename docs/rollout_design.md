# Offline Llama Rollout

`python -m inferkernellab.rollout` turns a local Llama checkpoint and JSONL prompts into an
immutable rollout artifact for MiniTrainBench's optional GRPO experiment. It supports a seeded
tiny random model for smoke tests and explicit local model paths; model and tokenizer loading
never reaches the network. Token-ID prompts do not require a tokenizer.

Each file starts with a manifest containing model and tokenizer SHA-256 fingerprints, sampling
configuration, generation limits, reward identifier, runtime backend, latency, cache memory,
environment, and source revision. Each following line is one prompt group with its target and
multiple completions. A completion records the response including EOS, one policy logprob and
one sampling logprob per emitted token, deterministic toy reward, and finish reason. The policy
logprob is computed under `softmax(logits / temperature)` before top-p filtering. The sampling
logprob is under the renormalized top-p distribution actually sampled from. GRPO consumes the
former; the latter makes behavior reproducible and auditable.

The current reward is `token_prefix_accuracy_v1`: matching leading response tokens divided by
the greater of response and target lengths. It is only a deterministic plumbing/evaluation
signal. It is not a learned reward model, preference signal, or evidence of language quality.
Rollouts are not reusable after changing model weights, tokenizer files, temperature, or the
recorded generation policy. The trainer verifies model/tokenizer fingerprints and recomputes all
stored policy logprobs before the first update.

An end-to-end local smoke is available at
[`../MiniTrainBench/scripts/run_grpo_cycle_smoke.sh`](../../MiniTrainBench/scripts/run_grpo_cycle_smoke.sh).
