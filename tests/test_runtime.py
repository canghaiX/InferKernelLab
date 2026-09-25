from inferkernellab.cache import PagedKVCache
from inferkernellab.runtime import InferenceRuntime
from inferkernellab.scheduler import TokenBudgetScheduler


def test_runtime_grows_and_releases_kv_blocks():
    cache = PagedKVCache(8, 4, 1, 2)
    runtime = InferenceRuntime(cache, TokenBudgetScheduler(max_num_batched_tokens=4))
    request = runtime.submit(3, prompt_tokens=5, max_new_tokens=2)
    assert runtime.stats().used_blocks == 2

    assert runtime.step().phase == "prefill"
    assert runtime.step().phase == "prefill"
    assert runtime.step().phase == "decode"
    assert request.generated_tokens == 1
    assert runtime.step().phase == "decode"
    assert request.status == "finished"
    assert runtime.stats().used_blocks == 0


def test_runtime_callbacks_receive_prefill_ranges_and_decode_before_accounting():
    cache = PagedKVCache(8, 4, 1, 2)
    events = []

    def prefill(request, start_token, end_token):
        events.append(("prefill", request.request_id, start_token, end_token, request.cached_tokens))

    def decode(requests):
        events.append(("decode", tuple(request.generated_tokens for request in requests)))

    runtime = InferenceRuntime(
        cache,
        TokenBudgetScheduler(max_num_batched_tokens=4),
        prefill_fn=prefill,
        decode_fn=decode,
    )
    request = runtime.submit(9, prompt_tokens=5, max_new_tokens=2)

    assert runtime.step().phase == "prefill"
    assert runtime.step().phase == "prefill"
    assert runtime.step().phase == "decode"
    assert runtime.step().phase == "decode"
    assert events == [
        ("prefill", 9, 0, 4, 4),
        ("prefill", 9, 4, 5, 5),
        ("decode", (0,)),
        ("decode", (1,)),
    ]
