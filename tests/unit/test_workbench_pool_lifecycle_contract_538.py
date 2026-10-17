"""#538 D2-WB-1: WorkbenchPool bounded stop, worker failure and safe release.

Owner target (2026-10-17): while a worker has not really ended, the pool must
not report a normal stop or shut down the batcher it may still use. D2-WB-1
covers bounded stop, worker failure and resource-safe release. The contract,
including the two choices beyond the ruling (fresh restart, late callbacks
dropped and counted), is in failure_degradation_537.md §12/§13.

Same harness as the D2-WB-0 diagnostics: each scenario runs in a fresh
interpreter under an outside timeout, with CPU fakes. A hang therefore fails
the test instead of blocking pytest. CUDA is out of scope (D2-GPU).
"""

# scope: perception
# function: contract
# lifecycle: active

from __future__ import annotations

import pytest

from tests.unit.test_workbench_pool_lifecycle_observed_538 import _run


def test_stuck_worker_stop_is_bounded_and_keeps_the_batcher() -> None:
    out = _run(
        """
async def main():
    MODE["process"] = "block"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    t0 = time.monotonic()
    try:
        pool.stop(timeout=1.0)
    except d.WorkbenchPoolStopTimeout:
        print("@stop1=timeout")
    print(f"@stop1_s={round(time.monotonic() - t0)}")
    print(f"@state={pool.state.value}")
    print(f"@batcher_down={FakeBatcher.last.down}")
    try:
        await pool.start()
    except d.WorkbenchPoolStopTimeout:
        print("@restart_while_stopping=refused")
    GATE.set()                                       # the worker finishes its frame
    wait_until(lambda: not pool._workers[0].is_alive())
    pool.stop(timeout=1.0)                           # retry: now it really stops
    print(f"@state2={pool.state.value}")
    print(f"@alive_at_shutdown={FakeBatcher.last.alive_at_shutdown}")
    print(f"@used_after_shutdown={FakeBatcher.last.used_after_shutdown}")
    stats = pool.get_stats()
    print(f"@late_callbacks={stats['late_callbacks']}")
    print(f"@stop_timeouts={stats['stop_timeouts']}")
asyncio.run(main())
"""
    )
    assert out == {
        "stop1": "timeout",
        "stop1_s": "1",
        "state": "stopping",
        "batcher_down": "False",
        "restart_while_stopping": "refused",
        "state2": "stopped",
        "alive_at_shutdown": "0",
        "used_after_shutdown": "0",
        "late_callbacks": "1",  # the result came in while stopping
        "stop_timeouts": "1",
    }


def test_full_queue_with_stuck_workers_shares_one_deadline() -> None:
    out = _run(
        """
async def main():
    MODE["process"] = "block"
    pool = make_pool(2)
    await pool.start()
    for sid in ("stream_0", "stream_1"):
        pool.put_frame(sid, frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    for i in range(4):
        pool.put_frame("stream_0", frame(), float(1 + i))
    print(f"@queue_depth={pool.get_stats()['queue_depth']}")
    t0 = time.monotonic()
    try:
        pool.stop(timeout=1.0)
    except d.WorkbenchPoolStopTimeout:
        print("@stop=timeout")
    print(f"@stop_s={round(time.monotonic() - t0)}")
    GATE.set()
    wait_until(lambda: not any(w.is_alive() for w in pool._workers))
    pool.stop()
    print(f"@discarded_on_stop={pool.get_stats()['discarded_on_stop']}")
asyncio.run(main())
"""
    )
    assert out == {
        "queue_depth": "4",
        "stop": "timeout",
        "stop_s": "1",
        "discarded_on_stop": "4",
    }


def test_worker_exception_fails_the_pool() -> None:
    out = _run(
        """
async def main():
    MODE["process"] = "raise"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.state is d.PoolState.FAILED)
    stats = pool.get_stats()
    print(f"@state={stats['state']}")
    print(f"@failure={stats['failure']}")
    print(f"@failed_stream={stats['failed_stream']}")
    print(f"@thread_errors={THREAD_ERRORS}")
    try:
        pool.put_frame("stream_0", frame(), 1.0)
    except d.WorkbenchPoolFailed as e:
        print(f"@put_frame=refused:{type(e.__cause__).__name__}")
    try:
        await pool.start()
    except d.WorkbenchPoolFailed:
        print("@start=refused")
    try:
        pool.stop()
    except d.WorkbenchPoolFailed as e:
        print(f"@stop=raised:{e.__cause__ is pool.failure}")
    print(f"@state2={pool.state.value}")
    print(f"@batcher_down={FakeBatcher.last.down}")
    print(f"@alive_at_shutdown={FakeBatcher.last.alive_at_shutdown}")
    pool.stop()                                      # nothing left to report
    print("@stop2=ok")
asyncio.run(main())
"""
    )
    assert out == {
        "state": "failed",
        "failure": "RuntimeError('process_frame failed')",
        "failed_stream": "stream_0",
        "thread_errors": "[]",
        "put_frame": "refused:RuntimeError",
        "start": "refused",
        "stop": "raised:True",
        "state2": "stopped",
        "batcher_down": "True",
        "alive_at_shutdown": "0",
        "stop2": "ok",
    }


def test_partial_start_failure_rolls_back() -> None:
    out = _run(
        """
async def main():
    MODE["construct_fail_at"] = 1
    pool = make_pool(3)
    try:
        await pool.start()
    except RuntimeError as e:
        print(f"@start_error={e}")
    print(f"@state={pool.state.value}")
    print(f"@workers_alive={sum(w.is_alive() for w in pool._workers)}")
    print(f"@batcher_down={FakeBatcher.last.down}")
    MODE["construct_fail_at"] = None
    await pool.start()                               # a clean retry works
    print(f"@retry_state={pool.state.value}")
    print(f"@retry_workers={sum(w.is_alive() for w in pool._workers)}")
    pool.stop()
asyncio.run(main())
"""
    )
    assert out == {
        "start_error": "workbench construction failed",
        "state": "stopped",
        "workers_alive": "0",
        "batcher_down": "True",
        "retry_state": "running",
        "retry_workers": "3",
    }


def test_missing_extension_is_refused_before_the_batcher() -> None:
    out = _run(
        """
async def main():
    d.PerceptionPipelineConfig = None
    pool = make_pool(1)
    try:
        await pool.start()
    except RuntimeError as e:
        print(f"@start_error={type(e).__name__}")
    print(f"@state={pool.state.value}")
    print(f"@batcher_created={FakeBatcher.last is not None}")
asyncio.run(main())
"""
    )
    assert out == {
        "start_error": "RuntimeError",
        "state": "created",
        "batcher_created": "False",
    }


def test_restart_gets_fresh_queues_and_workers() -> None:
    out = _run(
        """
async def main():
    pool = make_pool(1)
    await pool.start()
    pool.stop()
    pool.stop()
    await pool.start()
    print(f"@queues={len(pool._queues)}")
    print(f"@workers_alive={[w.is_alive() for w in pool._workers]}")
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    await asyncio.sleep(0.3)
    print(f"@results={len(RESULTS)}")
    pool.stop()
asyncio.run(main())
"""
    )
    assert out == {"queues": "1", "workers_alive": "[True]", "results": "1"}


def test_late_callback_after_loop_closed_is_dropped() -> None:
    out = _run(
        """
async def main():
    MODE["process"] = "block"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    return pool
pool = asyncio.run(main())                       # the loop is closed now
GATE.set()
wait_until(lambda: pool.get_stats()["late_callbacks"] == 1)
print(f"@late_callbacks={pool.get_stats()['late_callbacks']}")
print(f"@worker_alive={pool._workers[0].is_alive()}")
print(f"@thread_errors={THREAD_ERRORS}")
pool.stop()
print(f"@state={pool.state.value}")
"""
    )
    assert out == {
        "late_callbacks": "1",
        "worker_alive": "True",
        "thread_errors": "[]",
        "state": "stopped",
    }


_QUEUED_THEN = """
async def main():
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    # Block the loop (no await) until the worker has queued its callback.
    wait_until(lambda: FakeWorkbench.processed == 1)
    time.sleep(0.2)
    {action}
    await asyncio.sleep(0.2)                         # now the loop runs the callback
    print(f"@results={{len(RESULTS)}}")
    print(f"@late_callbacks={{pool.get_stats()['late_callbacks']}}")
    print(f"@state={{pool.state.value}}")
    if pool.state is d.PoolState.RUNNING:
        pool.stop()
asyncio.run(main())
"""


def test_callback_queued_before_stop_is_dropped_when_it_runs() -> None:
    out = _run(_QUEUED_THEN.format(action="pool.stop()"))
    assert out == {"results": "0", "late_callbacks": "1", "state": "stopped"}


def test_callback_queued_before_restart_is_dropped_when_it_runs() -> None:
    out = _run(_QUEUED_THEN.format(action="pool.stop(); await pool.start()"))
    assert out == {"results": "0", "late_callbacks": "1", "state": "running"}


def test_callback_queued_while_running_is_delivered() -> None:
    """Positive control for the two cases above."""
    out = _run(_QUEUED_THEN.format(action="pass"))
    assert out == {"results": "1", "late_callbacks": "0", "state": "running"}


@pytest.mark.parametrize("failure", ["False", '"raise"'])
def test_batcher_not_released_keeps_stopping(failure: str) -> None:
    out = _run(
        f"""
async def main():
    MODE["shutdown"] = {failure}
    pool = make_pool(1)
    await pool.start()
    try:
        pool.stop()
    except (d.WorkbenchPoolStopTimeout, OSError) as e:
        print(f"@stop1={{type(e).__name__}}")
    print(f"@state={{pool.state.value}}")
    print(f"@batcher_kept={{pool._batcher is FakeBatcher.last}}")
    try:
        await pool.start()
    except d.WorkbenchPoolStopTimeout:
        print("@restart=refused")
    MODE["shutdown"] = True
    pool.stop()                                      # retry once the batcher releases
    print(f"@state2={{pool.state.value}}")
    print(f"@batcher_cleared={{pool._batcher is None}}")
asyncio.run(main())
"""
    )
    assert out == {
        "stop1": "WorkbenchPoolStopTimeout" if failure == "False" else "OSError",
        "state": "stopping",
        "batcher_kept": "True",
        "restart": "refused",
        "state2": "stopped",
        "batcher_cleared": "True",
    }
