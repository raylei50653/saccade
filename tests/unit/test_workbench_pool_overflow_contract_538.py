"""#538 D3-F02: WorkbenchPool overflow counts the frames it actually drops.

Owner ruling (#538 comment 6393037794, async_ownership_538.md §9): keep
latest-wins / drain-all, but ``drops`` is the number of frames actually
dropped, ``overflow_events`` the number of overflows, ``discarded_on_stop``
stays separate, and every stream keeps frame identity with an observable
gap. Concurrent producers must not drop a frame without recording it.
Contract: failure_degradation_537.md §27.

Same harness as the D2-WB tests: each scenario runs in a fresh interpreter
under an outside timeout, with CPU fakes.
"""

# scope: perception
# function: contract
# lifecycle: active

from __future__ import annotations

import json

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
