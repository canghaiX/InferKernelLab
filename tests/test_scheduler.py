from inferkernellab.scheduler import InferenceRequest, TokenBudgetScheduler


def test_scheduler_chunks_prefill_then_decodes():
    scheduler = TokenBudgetScheduler(max_num_seqs=4, max_num_batched_tokens=4)
    request = InferenceRequest(1, prompt_tokens=7, max_new_tokens=2)
    scheduler.add(request)
    batch = scheduler.schedule()
    assert batch.phase == "prefill"
    assert batch.scheduled_tokens == 4
    assert request.cached_tokens == 4
    batch = scheduler.schedule()
    assert batch.phase == "prefill"
    assert request.cached_tokens == 7
    batch = scheduler.schedule()
    assert batch.phase == "decode"
    scheduler.mark_decode_step(batch.requests)
    assert request.generated_tokens == 1


def test_scheduler_marks_finished_requests():
    scheduler = TokenBudgetScheduler()
    request = InferenceRequest(1, prompt_tokens=1, max_new_tokens=1)
    scheduler.add(request)
    scheduler.schedule()
    batch = scheduler.schedule()
    scheduler.mark_decode_step(batch.requests)
    assert request.status == "finished"
    assert scheduler.finished == [request]

