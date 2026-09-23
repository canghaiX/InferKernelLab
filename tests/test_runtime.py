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

