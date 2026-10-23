"""#537 C0-09: WorkbenchPool stats say that no resource monitor is wired.

The WorkbenchPool path has no resource manager and no VRAM writer
(failure_degradation_537.md §2 C0-09, §19.3). Its stats used to return a fixed
``current_level="NORMAL"``, which looked healthy without any measurement. The
target is a stats shape that states the gap: ``current_level="UNKNOWN"`` and
``resource_monitor_status="not_wired"`` in every pool state, and the
``--workbench`` text report keeps the monitor status. Wiring a degradation
controller is out of scope.

Same harnesses as the D2-WB and caller lifecycle tests: a fresh interpreter
with CPU fakes, so there is no GPU, NVML or RTSP.
"""

# scope: perception, resource
# function: contract
# lifecycle: active

from __future__ import annotations

from tests.unit import test_caller_lifecycle_538 as caller
from tests.unit.test_workbench_pool_lifecycle_observed_538 import _run


def test_stats_report_no_monitor_in_every_pool_state() -> None:
    out = _run(
        """
def show(label, pool):
    stats = pool.get_stats()
    print(f"@{label}={stats['current_level']}/{stats['resource_monitor_status']}"
          f"/{stats['resource_monitor_error']!r}")

async def main():
    MODE["process"] = "raise"
    pool = make_pool(1)
    show("created", pool)
    await pool.start()
    show("running", pool)
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.state is d.PoolState.FAILED)
    show("failed", pool)
    try:
        pool.stop()
    except d.WorkbenchPoolFailed:
        pass
    show("stopped", pool)
asyncio.run(main())
"""
    )
    unknown = "UNKNOWN/not_wired/''"
    assert out == {
        "created": unknown,
        "running": unknown,
        "failed": unknown,
        "stopped": unknown,
    }


def test_run8_workbench_report_keeps_the_monitor_status() -> None:
    out = caller._run(
        r"""
async def idle_producer(*args):
    await asyncio.Event().wait()
run8.run_stream_producer_workbench = idle_producer
async def report_once(interval, report):
    report()
    raise RuntimeError("report complete")
run8._stats_loop = report_once
report(lambda: asyncio.run(checked(run8.run_multistream_perception(args("--workbench")))))
after_pool()
""",
        prelude=caller._POOL_SCRIPT,
    )
    lines = [
        line for line in out["_stdout"].splitlines() if "WorkbenchPool Stats:" in line
    ]
    assert len(lines) == 1
    assert "level=UNKNOWN monitor=not_wired " in lines[0]
    assert "level=NORMAL" not in out["_stdout"]
    assert caller._pick(
        out, "outcome", "leftover_tasks", "pool_state", "batcher_down"
    ) == {
        "outcome": "RuntimeError",
        "leftover_tasks": "0",
        "pool_state": "stopped",
        "batcher_down": "True",
    }
