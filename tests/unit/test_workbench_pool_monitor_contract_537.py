"""#537 C0-09: WorkbenchPool stats say that no resource monitor is wired.

The WorkbenchPool path has no resource manager and no VRAM writer
(failure_degradation_537.md §2 C0-09, §19.3). Its stats used to return a fixed
``current_level="NORMAL"``, which looked healthy without any measurement. The
target is a stats shape that states the gap: ``current_level="UNKNOWN"`` and
``resource_monitor_status="not_wired"`` in every pool state, and the
``--workbench`` text report keeps the monitor status. Wiring a degradation
controller is out of scope.

The host cases reuse the D2-WB and caller lifecycle harnesses: a fresh
interpreter with CPU fakes, so there is no GPU, NVML or RTSP. The ``gpu`` case
reuses the D2-GPU-1 harness: the real WorkbenchPool, Workbench and
BatchingTRTDetector on CUDA, with only the TRT engine faked.
"""

# scope: perception, resource
# function: contract
# lifecycle: active

from __future__ import annotations

import pytest

from tests.unit import test_caller_lifecycle_538 as caller
from tests.unit import test_workbench_pool_gpu_contract_538 as gpu
from tests.unit.test_workbench_pool_lifecycle_observed_538 import _run


def test_stats_report_no_monitor_in_every_pool_state() -> None:
    out = _run(
        """
def show(label, pool):
    stats = pool.get_stats()
    print(f"@{label}={stats['state']}:{stats['current_level']}"
          f"/{stats['resource_monitor_status']}/{stats['resource_monitor_error']!r}")

async def main():
    pool = make_pool(1)
    show("created", pool)
    await pool.start()
    show("running", pool)
    MODE["shutdown"] = False                         # the batcher is not released
    try:
        pool.stop()
    except d.WorkbenchPoolStopTimeout:
        pass
    show("stopping", pool)
    MODE["shutdown"] = True
    pool.stop()                                      # retry once the batcher releases
    show("stopped", pool)

    MODE["process"] = "raise"
    failed = make_pool(1)
    await failed.start()
    failed.put_frame("stream_0", frame(), 0.0)
    print(f"@failed_wait={wait_until(lambda: failed.state is d.PoolState.FAILED)}")
    show("failed", failed)
    try:
        failed.stop()
    except d.WorkbenchPoolFailed:
        pass
    show("failed_stopped", failed)
asyncio.run(main())
"""
    )
    unknown = "UNKNOWN/not_wired/''"
    assert out == {
        "created": f"created:{unknown}",
        "running": f"running:{unknown}",
        "stopping": f"stopping:{unknown}",
        "stopped": f"stopped:{unknown}",
        "failed_wait": "True",
        "failed": f"failed:{unknown}",
        "failed_stopped": f"stopped:{unknown}",
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


@pytest.mark.gpu
def test_real_pool_on_cuda_reports_no_monitor_while_delivering() -> None:
    gpu._need_native()
    out = gpu._run(
        """
def show(label, pool):
    stats = pool.get_stats()
    print(f"@{label}={stats['current_level']}/{stats['resource_monitor_status']}")

async def main():
    pool = make_pool()
    await pool.start()
    PLAN["boxes"] = [track_boxes(k) for k in range(3)]
    await feed(pool, [frame(10)] * 3)
    print(f"@delivered={len(RESULTS)}")
    show("running", pool)
    pool.stop()
    show("stopped", pool)
asyncio.run(main())
done()
"""
    )
    assert out["delivered"] == "3", out
    assert (out["running"], out["stopped"]) == (
        "UNKNOWN/not_wired",
        "UNKNOWN/not_wired",
    ), out
