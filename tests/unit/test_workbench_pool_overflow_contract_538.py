"""#538 D3-F02: WorkbenchPool overflow counts the frames it actually drops.

Owner ruling (#538 comment 6393037794, async_ownership_538.md §9): keep
latest-wins / drain-all, but ``drops`` is the number of frames actually
dropped, ``overflow_events`` the number of overflows, ``discarded_on_stop``
stays separate, and every stream keeps frame identity with an observable
gap. Concurrent producers must not drop a frame without recording it.
Contract: async_ownership_538.md §10.

Same harness as the D2-WB tests: each scenario runs in a fresh interpreter
under an outside timeout, with CPU fakes.
"""

# scope: perception
# function: contract
# lifecycle: active

from __future__ import annotations

import json

import pytest

from tests.unit.test_workbench_pool_lifecycle_observed_538 import _run

_TELEMETRY = """
from saccade.perception.online_telemetry import OnlineTelemetry
TEL = OnlineTelemetry()
def make_counted_pool(n=1):
    global POOL
    POOL = d.WorkbenchPool(
        "fake.engine", n_streams=n, on_result=on_result, input_hw=(8, 8), telemetry=TEL
    )
    return POOL
def counters():
    s = TEL.summary()
    return s.get("workbench_drops", 0), s.get("workbench_overflow_events", 0)
"""


def _streams(out: dict[str, str]) -> dict[str, dict[str, int]]:
    return json.loads(out["streams"])


def test_one_overflow_drops_and_counts_every_queued_frame() -> None:
    """Ruling example: drain 4 frames, enqueue the new one -> drops += 4,
    overflow_events += 1 (it used to be drops += 1 for 4 lost frames)."""
    out = _run(
        _TELEMETRY
        + """
async def main():
    MODE["process"] = "block"
    pool = make_counted_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)          # taken by the worker
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    for i in range(1, 5):
        pool.put_frame("stream_0", frame(), float(i))  # fills the queue (4)
    pool.put_frame("stream_0", frame(), 5.0)          # overflow
    st = pool.get_stats()
    print(f"@queue_depth={st['queue_depth']}")
    print(f"@drops={st['drops']}")
    print(f"@overflow_events={st['overflow_events']}")
    print(f"@telemetry={counters()}")
    GATE.set()
    wait_until(lambda: len(RESULTS) == 2)
    await asyncio.sleep(0.05)
    pool.stop()
    print(f"@results={RESULTS}")
    print("@streams=" + json.dumps(pool.get_stats()["streams"]))
import json
asyncio.run(main())
"""
    )
    assert out is not None
    assert out["queue_depth"] == "1"
    assert out["drops"] == "4"
    assert out["overflow_events"] == "1"
    assert out["telemetry"] == "(4, 1)"
    assert out["results"] == "[0.0, 5.0]"  # frames 1..4 were dropped
    s0 = _streams(out)["stream_0"]
    assert s0 == {
        "submitted": 6,
        "dropped": 4,
        "overflow_events": 1,
        "dequeued": 2,
        "discarded_on_stop": 0,
        "last_dequeued_seq": 6,
        "gaps": 1,
        "gap_frames": 4,
        "queued": 0,
    }


def test_repeated_overflows_accumulate_frames_and_events() -> None:
    out = _run(
        _TELEMETRY
        + """
async def main():
    MODE["process"] = "block"
    pool = make_counted_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    for i in range(1, 10):                           # 9 more: two overflows
        pool.put_frame("stream_0", frame(), float(i))
    st = pool.get_stats()
    print(f"@drops={st['drops']}")
    print(f"@overflow_events={st['overflow_events']}")
    print(f"@queue_depth={st['queue_depth']}")
    print(f"@telemetry={counters()}")
    pool.reset_stats()
    st = pool.get_stats()
    print(f"@after_reset={st['drops']},{st['overflow_events']},{st['streams']['stream_0']['dropped']}")
    GATE.set()
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    pool.stop()
asyncio.run(main())
"""
    )
    assert out is not None
    # 1..4 fill, 5 drains 4; 6..8 fill to 4, 9 drains 4.
    assert out["drops"] == "8"
    assert out["overflow_events"] == "2"
    assert out["queue_depth"] == "1"
    assert out["telemetry"] == "(8, 2)"
    # reset_stats() resets the totals; per-stream counts are kept.
    assert out["after_reset"] == "0,0,8"


def test_discarded_on_stop_is_not_an_overflow_drop() -> None:
    out = _run(
        _TELEMETRY
        + """
async def main():
    MODE["process"] = "block"
    pool = make_counted_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    for i in range(1, 4):
        pool.put_frame("stream_0", frame(), float(i))  # 3 queued, no overflow
    try:
        pool.stop(timeout=0.5)
    except d.WorkbenchPoolStopTimeout:
        pass
    GATE.set()
    wait_until(lambda: not any(w.is_alive() for w in pool._workers))
    pool.stop()
    st = pool.get_stats()
    print(f"@drops={st['drops']}")
    print(f"@overflow_events={st['overflow_events']}")
    print(f"@discarded_on_stop={st['discarded_on_stop']}")
    print(f"@telemetry={counters()}")
    print("@streams=" + json.dumps(st["streams"]))
import json
asyncio.run(main())
"""
    )
    assert out is not None
    assert (out["drops"], out["overflow_events"], out["discarded_on_stop"]) == (
        "0",
        "0",
        "3",
    )
    assert out["telemetry"] == "(0, 0)"
    s0 = _streams(out)["stream_0"]
    assert s0["submitted"] == 4 and s0["dequeued"] == 1
    assert s0["dropped"] == 0 and s0["discarded_on_stop"] == 3


def test_concurrent_producers_never_lose_an_unrecorded_frame() -> None:
    """8 producer threads per stream against a slow worker. Before the fix a
    racing producer could refill the queue between the drain and the final
    put_nowait (queue.Full escaped put_frame), and every overflow counted 1.

    After stop() every frame is accounted for exactly once, per stream:
    submitted == dequeued + dropped + discarded_on_stop + queued; the totals
    and the telemetry agree with the per-stream counts.
    """
    out = _run(
        _TELEMETRY
        + """
import json
orig = FakeWorkbench.process_frame
def slow(self, *a, **kw):
    time.sleep(0.001)
    return orig(self, *a, **kw)
FakeWorkbench.process_frame = slow
ERRORS = []
async def main():
    pool = make_counted_pool(2)
    await pool.start()
    def produce(sid, k):
        for i in range(400):
            try:
                pool.put_frame(sid, frame(), float(k * 1000 + i))
            except Exception as e:
                ERRORS.append(type(e).__name__)
    threads = [
        threading.Thread(target=produce, args=(f"stream_{j % 2}", j)) for j in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pool.stop()
    st = pool.get_stats()
    print(f"@errors={sorted(set(ERRORS))}")
    print(f"@drops={st['drops']}")
    print(f"@overflow_events={st['overflow_events']}")
    print(f"@discarded_on_stop={st['discarded_on_stop']}")
    print(f"@telemetry={list(counters())}")
    print("@streams=" + json.dumps(st["streams"]))
    print(f"@thread_errors={THREAD_ERRORS}")
asyncio.run(main())
""",
        timeout=120.0,
    )
    assert out is not None
    assert out["errors"] == "[]"
    assert out["thread_errors"] == "[]"
    streams = _streams(out)
    for sid, c in streams.items():
        assert c["submitted"] == 8 * 400, sid
        assert c["queued"] == 0, sid
        assert (
            c["submitted"] == c["dequeued"] + c["dropped"] + c["discarded_on_stop"]
        ), (sid, c)
        # every frame identity the worker skipped was dropped or discarded
        assert c["gap_frames"] <= c["dropped"] + c["discarded_on_stop"], (sid, c)
        assert c["last_dequeued_seq"] <= c["submitted"], (sid, c)
    total_dropped = sum(c["dropped"] for c in streams.values())
    total_events = sum(c["overflow_events"] for c in streams.values())
    assert total_dropped > 0 and total_events > 0  # the race was exercised
    assert int(out["drops"]) == total_dropped
    assert int(out["overflow_events"]) == total_events
    assert json.loads(out["telemetry"]) == [total_dropped, total_events]
    assert int(out["discarded_on_stop"]) == sum(
        c["discarded_on_stop"] for c in streams.values()
    )
    # one overflow drops at most the queue capacity
    assert total_dropped <= 4 * total_events


def test_put_after_stop_is_refused_not_counted() -> None:
    """A put refused because the pool stopped is not a drop or a discard."""
    out = _run(
        _TELEMETRY
        + """
async def main():
    pool = make_counted_pool(1)
    await pool.start()
    pool.stop()
    try:
        pool.put_frame("stream_0", frame(), 0.0)
    except d.WorkbenchPoolNotRunning:
        print("@refused=yes")
    st = pool.get_stats()
    c = st["streams"]["stream_0"]
    print(f"@counts={c['submitted']},{c['dropped']},{c['discarded_on_stop']},{st['drops']}")
asyncio.run(main())
"""
    )
    assert out == {"refused": "yes", "counts": "0,0,0,0"}


def test_stop_bounds_a_producer_holding_the_stream_lock_and_can_retry() -> None:
    """Admission can be paused even after every worker has ended.

    stop must report a timeout and retain its batcher, rather than wait for
    that producer indefinitely. A later admission is refused, and retrying
    stop accounts for the frame admitted before stopping exactly once.
    """
    out = _run(
        _TELEMETRY
        + """
import json
async def main():
    pool = make_counted_pool(1)
    await pool.start()
    inside = threading.Event()
    resume = threading.Event()
    q = pool._queues[0]
    original_put = q.put_nowait
    def paused_put(item):
        inside.set()
        assert resume.wait(5.0)
        original_put(item)
    q.put_nowait = paused_put
    errors = []
    def produce(ts):
        try:
            pool.put_frame("stream_0", frame(), ts)
        except Exception as e:
            errors.append(type(e).__name__)
    admitted = threading.Thread(target=produce, args=(0.0,))
    admitted.start()
    assert inside.wait(5.0)
    # This producer waits for the same stream lock, but must not be admitted
    # once the pool enters STOPPING.
    checked = threading.Event()
    original_check = pool._check_running
    def observe_check():
        original_check()
        if threading.current_thread().name == "late-producer":
            checked.set()
    pool._check_running = observe_check
    refused = threading.Thread(target=produce, args=(1.0,), name="late-producer")
    refused.start()
    assert checked.wait(5.0)
    release = threading.Timer(1.5, resume.set)
    release.start()  # bound the diagnostic even if stop's timeout is broken
    t0 = time.monotonic()
    outcome = "returned"
    try:
        pool.stop(timeout=0.15)
    except d.WorkbenchPoolStopTimeout:
        outcome = "timeout"
    print(f"@outcome={outcome}")
    print(f"@bounded={time.monotonic() - t0 < 0.8}")
    print(f"@state={pool.state.value}")
    print(f"@batcher_kept={pool._batcher is FakeBatcher.last and not FakeBatcher.last.down}")
    resume.set()
    release.cancel()
    release.join(5.0)
    admitted.join(5.0)
    refused.join(5.0)
    assert not admitted.is_alive() and not refused.is_alive()
    pool.stop()
    first = pool.get_stats()
    pool.stop()
    print(f"@idempotent={first == pool.get_stats()}")
    print(f"@errors={errors}")
    print(f"@discarded={first['discarded_on_stop']}")
    print(f"@timeouts={first['stop_timeouts']}")
    print(f"@telemetry={counters()}")
    print("@streams=" + json.dumps(first["streams"]))
asyncio.run(main())
"""
    )
    assert out is not None
    assert out["outcome"] == "timeout"
    assert out["bounded"] == "True"
    assert out["state"] == "stopping"
    assert out["batcher_kept"] == "True"
    assert out["idempotent"] == "True"
    assert out["errors"] == "['WorkbenchPoolNotRunning']"
    assert out["discarded"] == "1" and out["timeouts"] == "1"
    assert out["telemetry"] == "(0, 0)"
    c = _streams(out)["stream_0"]
    assert c["submitted"] == c["discarded_on_stop"] == 1
    assert c["dequeued"] == c["dropped"] == c["queued"] == 0


def test_restart_keeps_frame_identity_and_exposes_stop_discard_gap() -> None:
    out = _run(
        _TELEMETRY
        + """
import json
async def main():
    MODE["process"] = "block"
    pool = make_counted_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    assert wait_until(lambda: pool.get_stats()["streams"]["stream_0"]["dequeued"] == 1)
    for ts in (1.0, 2.0, 3.0):
        pool.put_frame("stream_0", frame(), ts)
    try:
        pool.stop(timeout=0.05)
    except d.WorkbenchPoolStopTimeout:
        pass
    GATE.set()
    assert wait_until(lambda: not any(w.is_alive() for w in pool._workers))
    pool.stop()
    await pool.start()
    pool.put_frame("stream_0", frame(), 4.0)
    assert wait_until(lambda: pool.get_stats()["streams"]["stream_0"]["dequeued"] == 2)
    pool.stop()
    st = pool.get_stats()
    print("@streams=" + json.dumps(st["streams"]))
    print(f"@discarded={st['discarded_on_stop']}")
    print(f"@telemetry={counters()}")
asyncio.run(main())
"""
    )
    assert out is not None
    c = _streams(out)["stream_0"]
    assert c == {
        "submitted": 5,
        "dropped": 0,
        "overflow_events": 0,
        "dequeued": 2,
        "discarded_on_stop": 3,
        "last_dequeued_seq": 5,
        "gaps": 1,
        "gap_frames": 3,
        "queued": 0,
    }
    assert out["discarded"] == "3"
    assert out["telemetry"] == "(0, 0)"


def test_partial_stop_drain_is_counted_before_another_stream_times_out() -> None:
    out = _run(
        _TELEMETRY
        + """
import json
async def main():
    MODE["process"] = "block"
    pool = make_counted_pool(2)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    assert wait_until(lambda: pool.get_stats()["streams"]["stream_0"]["dequeued"] == 1)
    pool.put_frame("stream_0", frame(), 1.0)
    pool.put_frame("stream_0", frame(), 2.0)
    inside = threading.Event()
    resume = threading.Event()
    q = pool._queues[1]
    original_put = q.put_nowait
    def paused_put(item):
        inside.set()
        assert resume.wait(5.0)
        original_put(item)
    q.put_nowait = paused_put
    producer = threading.Thread(target=pool.put_frame, args=("stream_1", frame(), 3.0))
    producer.start()
    assert inside.wait(5.0)
    def release_worker():
        assert wait_until(lambda: pool.state is d.PoolState.STOPPING)
        GATE.set()
    worker_release = threading.Thread(target=release_worker)
    worker_release.start()
    release = threading.Timer(1.5, resume.set)
    release.start()
    outcome = "returned"
    try:
        pool.stop(timeout=0.2)
    except d.WorkbenchPoolStopTimeout:
        outcome = "timeout"
    print(f"@outcome={outcome}")
    print(f"@partial_discarded={pool._discarded_on_stop}")
    resume.set()
    release.cancel()
    release.join(5.0)
    producer.join(5.0)
    worker_release.join(5.0)
    assert not producer.is_alive() and not worker_release.is_alive()
    pool.stop()
    st = pool.get_stats()
    print(f"@discarded={st['discarded_on_stop']}")
    print("@streams=" + json.dumps(st["streams"]))
asyncio.run(main())
"""
    )
    assert out is not None
    assert out["outcome"] == "timeout"
    assert out["partial_discarded"] == "2"
    assert out["discarded"] == "3"
    streams = _streams(out)
    assert streams["stream_0"]["discarded_on_stop"] == 2
    assert streams["stream_1"]["discarded_on_stop"] == 1
    for c in streams.values():
        assert c["submitted"] == c["dequeued"] + c["discarded_on_stop"]
        assert c["queued"] == c["dropped"] == 0


@pytest.mark.gpu
def test_cuda_overflow_counts_frames_and_delivers_the_latest_frame() -> None:
    """Real CUDA Workbench and batcher; only TRT inference is replaced.

    Hold the first inference while overflowing its stream, then let both
    frames finish. This exercises F02 on the actual CUDA processing path;
    it does not qualify a model, throughput or online deployment.
    """
    from tests.unit.test_workbench_pool_gpu_contract_538 import (
        _need_native,
        _run as run_gpu,
    )

    _need_native()
    out = run_gpu(
        """
import json
from saccade.perception.online_telemetry import OnlineTelemetry
async def main():
    telemetry = OnlineTelemetry()
    pool = make_pool(1, telemetry=telemetry)
    PLAN["gate"] = threading.Event()
    await pool.start()
    pool.put_frame("stream_0", frame(0), 0.0)
    await until(IN_INFER.is_set)
    for i in range(1, 6):
        pool.put_frame("stream_0", frame(i * 10), float(i))
    st = pool.get_stats()
    print(f"@pending={st['drops']},{st['overflow_events']},{st['queue_depth']}")
    PLAN["gate"].set()
    await until(lambda: len(RESULTS) == 2)
    pool.stop()
    st = pool.get_stats()
    print("@streams=" + json.dumps(st["streams"]))
    print(f"@timestamps={[r['ts'] for r in RESULTS]}")
    print(f"@inputs_ok={len(SEEN) == 2 and abs(SEEN[0]) < 1e-5 and abs(SEEN[1] - 50.0/255.0) < 1e-3}")
    print(f"@telemetry={st['workbench_drops']},{st['workbench_overflow_events']}")
    print(f"@state={st['state']}")
    print(f"@thread_errors={THREAD_ERRORS}")
asyncio.run(main())
done()
"""
    )
    assert out["pending"] == "4,1,1"
    assert out["timestamps"] == "[0.0, 5.0]"
    assert out["inputs_ok"] == "True"
    assert out["telemetry"] == "4,1"
    assert out["state"] == "stopped"
    assert out["thread_errors"] == "[]"
    c = _streams(out)["stream_0"]
    assert c["submitted"] == 6 and c["dequeued"] == 2
    assert c["dropped"] == c["gap_frames"] == 4
    assert c["last_dequeued_seq"] == 6 and c["gaps"] == 1
    assert c["queued"] == c["discarded_on_stop"] == 0
