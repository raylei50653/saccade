"""#538 D1-F04: reconnects yield the loop and stay owned through cleanup.

Each case uses a fresh interpreter and an outside timeout. The client cases
exercise the real MediaMTX async control methods with gated synchronous
operations; caller cases run the real CLI lifecycle with controlled media.
These are host contracts, not RTSP/GPU qualification evidence.
"""

# scope: media, pipeline
# function: contract
# lifecycle: active

from __future__ import annotations

import pytest

from tests.unit.test_caller_lifecycle_538 import _MAIN, _pick, _run


_CLIENT = r"""
import asyncio, threading, time
from saccade.media.mediamtx_client import MediaMTXClient
client = MediaMTXClient()
client.use_cpp = False

async def until(predicate):
    end = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > end:
            raise TimeoutError("condition was not reached")
        await asyncio.sleep(0.001)

async def cancelled(task):
    task.cancel()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError), result
"""


def test_initial_connect_keeps_the_event_loop_responsive() -> None:
    out = _run(
        r"""
entered = threading.Event()
gate = threading.Event()
def connect():
    entered.set()
    assert gate.wait(2), "connect was not released"
    return True
client.connect = connect

async def case():
    ticks = 0
    async def heartbeat():
        nonlocal ticks
        while not gate.is_set():
            ticks += 1
            await asyncio.sleep(0.002)
    timer = threading.Timer(0.2, gate.set)
    timer.start()
    beat = asyncio.create_task(heartbeat())
    try:
        assert await client.connect_async()
        await beat
        print(f"@responsive={ticks >= 10}")
        print(f"@entered={entered.is_set()}")
    finally:
        gate.set()
        timer.join()
        await asyncio.gather(beat, return_exceptions=True)
asyncio.run(case())
""",
        prelude=_CLIENT,
    )
    assert _pick(out, "responsive", "entered") == {
        "responsive": "True",
        "entered": "True",
    }


def test_watchdog_reconnect_keeps_the_event_loop_responsive() -> None:
    out = _run(
        r"""
entered = threading.Event()
gate = threading.Event()
def restart():
    entered.set()
    assert gate.wait(2), "reconnect was not released"
    return True
client._running = True
client._is_alive = lambda: False
client._restart_pipeline = restart

async def case():
    ticks = 0
    async def heartbeat():
        nonlocal ticks
        while not gate.is_set():
            ticks += 1
            await asyncio.sleep(0.002)
    timer = threading.Timer(0.2, gate.set)
    timer.start()
    beat = asyncio.create_task(heartbeat())
    watch = asyncio.create_task(client.watchdog_loop())
    try:
        await until(gate.is_set)
        print(f"@responsive={ticks >= 10}")
        print(f"@entered={entered.is_set()}")
    finally:
        gate.set()
        timer.join()
        await cancelled(watch)
        await beat
asyncio.run(case())
""",
        prelude=_CLIENT,
    )
    assert _pick(out, "responsive", "entered") == {
        "responsive": "True",
        "entered": "True",
    }


def test_failed_reconnect_does_not_end_the_retry_loop() -> None:
    out = _run(
        r"""
attempts = 0
def restart():
    global attempts
    attempts += 1
    client._running = attempts >= 3
    return client._running
client._running = True
client._is_alive = lambda: attempts >= 3
client._restart_pipeline = restart
original_sleep = asyncio.sleep
async def fast_sleep(delay):
    await original_sleep(min(delay, 0.001))
asyncio.sleep = fast_sleep

async def case():
    watch = asyncio.create_task(client.watchdog_loop())
    try:
        await until(lambda: attempts >= 3 or watch.done())
        print(f"@attempts={attempts}")
        print(f"@still_watching={not watch.done()}")
    finally:
        watch.cancel()
        await asyncio.gather(watch, return_exceptions=True)
asyncio.run(case())
""",
        prelude=_CLIENT,
    )
    assert _pick(out, "attempts", "still_watching") == {
        "attempts": "3",
        "still_watching": "True",
    }


def test_cancelled_connect_drains_before_release_even_after_repeated_cancel() -> None:
    out = _run(
        r"""
entered = threading.Event()
gate = threading.Event()
events = []
def connect():
    events.append("connect-enter")
    entered.set()
    assert gate.wait(2), "connect was not released"
    client._running = True
    events.append("connect-return")
    return True
def release():
    events.append("release")
    client._running = False
client.connect = connect
client.release = release

async def case():
    task = asyncio.create_task(client.connect_async())
    try:
        await until(entered.is_set)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        print(f"@draining={not task.done()}")
        print(f"@before_release={events == ['connect-enter']}")
    finally:
        gate.set()
        result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError), result
    await client.release_async()
    await asyncio.sleep(0.02)
    print(f"@events={events}")
    print(f"@running={client._running}")
    print(f"@leftover_tasks={len(asyncio.all_tasks()) - 1}")
asyncio.run(case())
""",
        prelude=_CLIENT,
    )
    assert _pick(
        out, "draining", "before_release", "events", "running", "leftover_tasks"
    ) == {
        "draining": "True",
        "before_release": "True",
        "events": "['connect-enter', 'connect-return', 'release']",
        "running": "False",
        "leftover_tasks": "0",
    }


def test_cancelled_reconnect_finishes_before_owner_release() -> None:
    out = _run(
        r"""
entered = threading.Event()
gate = threading.Event()
events = []
def restart():
    events.append("restart-enter")
    entered.set()
    assert gate.wait(2), "reconnect was not released"
    client._running = True
    events.append("restart-return")
    return True
def release():
    events.append("release")
    client._running = False
client._running = True
client._is_alive = lambda: False
client._restart_pipeline = restart
client.release = release

async def case():
    watch = asyncio.create_task(client.watchdog_loop())
    try:
        await until(entered.is_set)
        watch.cancel()
        await asyncio.sleep(0.02)
        print(f"@draining={not watch.done()}")
    finally:
        gate.set()
        result = await asyncio.gather(watch, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError), result
    await client.release_async()
    await asyncio.sleep(0.02)
    print(f"@events={events}")
    print(f"@running={client._running}")
asyncio.run(case())
""",
        prelude=_CLIENT,
    )
    assert _pick(out, "draining", "events", "running") == {
        "draining": "True",
        "events": "['restart-enter', 'restart-return', 'release']",
        "running": "False",
    }


def test_control_failure_during_cancellation_is_retrieved_and_attached() -> None:
    out = _run(
        r"""
entered = threading.Event()
gate = threading.Event()
def connect():
    entered.set()
    assert gate.wait(2), "connect was not released"
    raise OSError("control failed while cancelling")
client.connect = connect

async def case():
    unhandled = []
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, context: unhandled.append(context))
    task = asyncio.create_task(client.connect_async())
    try:
        await until(entered.is_set)
        task.cancel()
        await asyncio.sleep(0.02)
    finally:
        gate.set()
    error = None
    try:
        await task
    except BaseException as caught:
        error = caught
    print(f"@outcome={type(error).__name__}")
    print(f"@notes={getattr(error, '__notes__', [])}")
    await asyncio.sleep(0.02)
    print(f"@unhandled={unhandled}")
asyncio.run(case())
""",
        prelude=_CLIENT,
    )
    assert _pick(out, "outcome", "unhandled") == {
        "outcome": "CancelledError",
        "unhandled": "[]",
    }
    assert "OSError('control failed while cancelling')" in out["notes"]


_MEDIA = (
    _MAIN
    + r"""
MEDIA = []
class FakeMedia:
    connect_result = True
    fail_watchdog = False
    def __init__(self, *args, **kwargs):
        self.index = len(MEDIA)
        self.releases = 0
        self.watch_stopped = False
        MEDIA.append(self)
    async def connect_async(self):
        return self.connect_result
    def grab_tensor(self):
        return False, None
    async def watchdog_loop(self):
        try:
            if self.fail_watchdog and self.index == 0:
                await asyncio.sleep(0.04)
                raise ValueError("watchdog failed")
            await asyncio.Future()
        finally:
            self.watch_stopped = True
    async def release_async(self):
        self.releases += 1

main.MediaMTXClient = FakeMedia

async def cancel_later(coro):
    task = asyncio.create_task(coro)
    await asyncio.sleep(0.05)
    task.cancel()
    await task

def media_after():
    print(f"@media_count={len(MEDIA)}")
    print(f"@releases={[m.releases for m in MEDIA]}")
    print(f"@watch_stopped={[m.watch_stopped for m in MEDIA]}")
"""
)


def test_watchdog_failure_reaches_caller_and_cancels_sibling_producers() -> None:
    out = _run(
        r"""
FakeMedia.fail_watchdog = True
report(lambda: asyncio.run(checked(main.run_perception(
    detector_factory=FakeDetector, streams=("s1", "s2"), source="rtsp://x/live"))))
after()
media_after()
""",
        prelude=_MEDIA,
    )
    assert _pick(
        out,
        "outcome",
        "leftover_tasks",
        "state",
        "writer_closed",
        "media_count",
        "releases",
        "watch_stopped",
    ) == {
        "outcome": "ValueError",
        "leftover_tasks": "0",
        "state": "stopped",
        "writer_closed": "1",
        "media_count": "2",
        "releases": "[1, 1]",
        "watch_stopped": "[True, True]",
    }


def test_owner_cancellation_stops_watchdog_before_releasing_media() -> None:
    out = _run(
        r"""
async def release(self):
    assert self.watch_stopped, "media was released before watchdog finished"
    self.releases += 1
FakeMedia.release_async = release
report(lambda: asyncio.run(checked(cancel_later(
    main.run_stream_producer("s1", object(), "rtsp://x/live")))))
media_after()
""",
        prelude=_MEDIA,
    )
    assert _pick(out, "outcome", "leftover_tasks", "releases", "watch_stopped") == {
        "outcome": "CancelledError",
        "leftover_tasks": "0",
        "releases": "[1]",
        "watch_stopped": "[True]",
    }


def test_cancelling_initial_connect_retry_releases_media() -> None:
    out = _run(
        r"""
FakeMedia.connect_result = False
report(lambda: asyncio.run(checked(cancel_later(
    main.run_stream_producer("s1", object(), "rtsp://x/live")))))
media_after()
""",
        prelude=_MEDIA,
    )
    assert _pick(out, "outcome", "leftover_tasks", "releases", "watch_stopped") == {
        "outcome": "CancelledError",
        "leftover_tasks": "0",
        "releases": "[1]",
        "watch_stopped": "[False]",
    }


def test_dali_producer_does_not_require_a_watchdog_and_releases_on_failure() -> None:
    out = _run(
        r"""
class FakeDali:
    released = 0
    def __init__(self, **kwargs):
        pass
    def connect(self):
        return True
    def grab_tensor(self):
        raise LookupError("DALI grab failed")
    def release(self):
        FakeDali.released += 1
main.DALIMediaClient = FakeDali
main.os.path.isfile = lambda path: True
report(lambda: asyncio.run(checked(
    main.run_stream_producer("s1", object(), "file.mp4"))))
print(f"@released={FakeDali.released}")
""",
        prelude=_MAIN,
    )
    assert _pick(out, "outcome", "leftover_tasks", "released") == {
        "outcome": "LookupError",
        "leftover_tasks": "0",
        "released": "1",
    }


def test_dali_connect_error_still_releases_media() -> None:
    out = _run(
        r"""
class FakeDali:
    released = 0
    def __init__(self, **kwargs):
        pass
    def connect(self):
        raise LookupError("DALI connect failed")
    def release(self):
        FakeDali.released += 1
main.DALIMediaClient = FakeDali
main.os.path.isfile = lambda path: True
report(lambda: asyncio.run(checked(
    main.run_stream_producer("s1", object(), "file.mp4"))))
print(f"@released={FakeDali.released}")
""",
        prelude=_MAIN,
    )
    assert _pick(out, "outcome", "leftover_tasks", "released") == {
        "outcome": "LookupError",
        "leftover_tasks": "0",
        "released": "1",
    }


@pytest.mark.parametrize(
    ("source", "expected_kwargs"),
    [
        (None, "{}"),
        ("definitely-missing-538-watchdog.mp4", "{}"),
        ("rtsp://x/live", "{'rtsp_url': 'rtsp://x/live'}"),
        ("rtsps://x/live", "{'rtsp_url': 'rtsps://x/live'}"),
    ],
)
def test_rtsp_factory_does_not_mask_watchdog_with_a_dummy_video(
    source: str | None, expected_kwargs: str
) -> None:
    out = _run(
        r"""
class RecordingMedia(FakeMedia):
    def __init__(self, *args, **kwargs):
        print(f"@kwargs={kwargs}")
        super().__init__(*args, **kwargs)
main.MediaMTXClient = RecordingMedia
"""
        + "report(lambda: asyncio.run(checked(cancel_later(\n"
        + f"    main.run_stream_producer('s1', object(), {source!r})))))\n"
        + "media_after()\n",
        prelude=_MEDIA,
    )
    assert _pick(out, "outcome", "leftover_tasks", "kwargs", "watch_stopped") == {
        "outcome": "CancelledError",
        "leftover_tasks": "0",
        "kwargs": expected_kwargs,
        "watch_stopped": "[True]",
    }


def test_unexpected_watchdog_return_fails_and_releases_the_producer() -> None:
    out = _run(
        r"""
async def returned(self):
    self.watch_stopped = True
FakeMedia.watchdog_loop = returned
report(lambda: asyncio.run(checked(
    main.run_stream_producer("s1", object(), "rtsp://x/live"))))
media_after()
""",
        prelude=_MEDIA,
    )
    assert _pick(out, "outcome", "leftover_tasks", "releases", "watch_stopped") == {
        "outcome": "RuntimeError",
        "leftover_tasks": "0",
        "releases": "[1]",
        "watch_stopped": "[True]",
    }


def test_media_cleanup_error_does_not_replace_the_pump_error() -> None:
    out = _run(
        r"""
def grab(self):
    raise LookupError("pump failed")
async def release(self):
    self.releases += 1
    raise OSError("release failed")
FakeMedia.grab_tensor = grab
FakeMedia.release_async = release
report(lambda: asyncio.run(checked(
    main.run_stream_producer("s1", object(), "rtsp://x/live"))))
media_after()
""",
        prelude=_MEDIA,
    )
    assert _pick(out, "outcome", "leftover_tasks", "note", "releases") == {
        "outcome": "LookupError",
        "leftover_tasks": "0",
        "note": "cleanup media.release() also failed",
        "releases": "[1]",
    }
    assert "OSError('release failed')" in out["notes_full"]
