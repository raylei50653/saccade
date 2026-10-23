"""#537 health CLI (REQ-535-07-03): what could not be observed is not healthy.

``saccade.pipeline.health`` (``saccade_ctl.sh health``; also printed once when
the orchestrator starts, SE-48) used to report an NVML failure as OK, an
unreachable Redis as a queue depth of 0 and a missing user unit as
``inactive/failed``, and showed the health process's own loop latency and
allocator as stress metrics without saying so (failure_degradation_537.md
§24). The target: each check is ok / degraded / unknown with its source; the
status is DEGRADED when any check is degraded, else UNKNOWN when any is
unknown, else OK; ``overall_ok`` only for OK. The exit code is unchanged (no
readiness policy is chosen).

Host only: systemctl, NVML, Redis and torch's CUDA queries are replaced; no
service, GPU or Redis is touched.
"""

# scope: pipeline, resource
# function: contract
# lifecycle: active

from __future__ import annotations

import subprocess
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import saccade.pipeline.health as H

OK, DEGRADED, UNKNOWN = H.OK, H.DEGRADED, H.UNKNOWN


def _vram(warn: bool = False) -> H.VramStatus:
    return H.VramStatus(
        used_gb=1.0, total_gb=12.0, pct=90.0 if warn else 8.3, warn=warn
    )


def _report(
    systemd: tuple[str, ...] = (OK, OK, OK),
    redis: str = OK,
    vram: H.VramStatus | None = None,
    vram_missing: bool = False,
) -> H.HealthReport:
    return H.HealthReport(
        timestamp=datetime(2026, 10, 23, 12, 0, 0),
        systemd=[H.ServiceStatus(f"svc{i}", s, "x") for i, s in enumerate(systemd)],
        vram=None if vram_missing else (vram or _vram()),
        redis=H.ServiceStatus("redis", redis, "x"),
        vram_error="NVMLError: init failed" if vram_missing else "",
    )


# ── Overall status ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({}, "OK"),
        ({"vram_missing": True}, "UNKNOWN"),  # the old report said OK
        ({"vram": _vram(warn=True)}, "DEGRADED"),
        ({"systemd": (OK, UNKNOWN, OK)}, "UNKNOWN"),
        ({"redis": DEGRADED}, "DEGRADED"),
        ({"systemd": (UNKNOWN, OK, OK), "redis": DEGRADED}, "DEGRADED"),
        ({"systemd": (DEGRADED, OK, OK), "vram_missing": True}, "DEGRADED"),
        ({"systemd": (UNKNOWN, UNKNOWN, UNKNOWN), "vram_missing": True}, "UNKNOWN"),
    ],
)
def test_status_degraded_over_unknown_over_ok(
    kwargs: dict[str, Any], status: str
) -> None:
    report = _report(**kwargs)
    assert report.status == status
    assert report.overall_ok is (status == "OK")
    assert H.render(report).splitlines()[0].endswith(f"Saccade Status — {status}")


def test_vram_status_follows_the_reading() -> None:
    assert _report().vram_status == OK
    assert _report(vram=_vram(warn=True)).vram_status == DEGRADED
    assert _report(vram_missing=True).vram_status == UNKNOWN


# ── systemd ───────────────────────────────────────────────────────────────────


def _systemctl(stdout: str = "", returncode: int = 0, stderr: str = "") -> MagicMock:
    return MagicMock(
        return_value=SimpleNamespace(
            stdout=stdout, stderr=stderr, returncode=returncode
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "status", "detail"),
    [
        ("LoadState=loaded\nActiveState=active\n", OK, "active"),
        ("ActiveState=active\nLoadState=loaded\n", OK, "active"),  # any order
        ("LoadState=loaded\nActiveState=inactive\n", DEGRADED, "loaded/inactive"),
        ("LoadState=loaded\nActiveState=failed\n", DEGRADED, "loaded/failed"),
        ("LoadState=loaded\nActiveState=activating\n", DEGRADED, "loaded/activating"),
        ("LoadState=masked\nActiveState=inactive\n", DEGRADED, "masked/inactive"),
        ("LoadState=not-found\nActiveState=inactive\n", UNKNOWN, "no such user unit"),
        ("", UNKNOWN, "no unit state"),
        ("LoadState=loaded\n", UNKNOWN, "no unit state"),
    ],
)
async def test_systemd_unit_state(stdout: str, status: str, detail: str) -> None:
    run = _systemctl(stdout)
    with patch("saccade.pipeline.health.subprocess.run", run):
        s = await H.check_systemd("yolo-perception")
    assert (s.name, s.status) == ("yolo-perception", status)
    assert detail in s.detail
    assert s.ok is (status == OK)
    argv = run.call_args.args[0]
    assert argv == [
        "systemctl",
        "--user",
        "show",
        "-p",
        "LoadState,ActiveState",
        "yolo-perception",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "No such file or directory", "systemctl"),
        subprocess.TimeoutExpired(["systemctl"], 3),
        PermissionError(13, "denied"),
    ],
)
async def test_systemctl_that_cannot_answer_is_unknown(error: Exception) -> None:
    with patch("saccade.pipeline.health.subprocess.run", side_effect=error):
        s = await H.check_systemd("mediamtx")
    assert s.status == UNKNOWN
    assert "systemctl unavailable" in s.detail and type(error).__name__ in s.detail


@pytest.mark.asyncio
async def test_systemctl_error_exit_is_unknown() -> None:
    run = _systemctl(returncode=1, stderr="Failed to connect to bus: No medium found")
    with patch("saccade.pipeline.health.subprocess.run", run):
        s = await H.check_systemd("mediamtx")
    assert s.status == UNKNOWN
    assert "exit 1" in s.detail and "Failed to connect to bus" in s.detail


# ── VRAM ──────────────────────────────────────────────────────────────────────


def _nvml(**overrides: Any) -> MagicMock:
    nvml = MagicMock()
    nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(
        used=3 * 1024**3, total=12 * 1024**3
    )
    for name, value in overrides.items():
        setattr(nvml, name, value)
    return nvml


def test_nvml_init_failure_is_unknown_with_its_reason() -> None:
    nvml = _nvml(
        nvmlInit=MagicMock(side_effect=RuntimeError("NVML Shared Library Not Found"))
    )
    with patch.object(H, "pynvml", nvml):
        vram, error = H.check_vram()
    assert vram is None
    assert error == "RuntimeError: NVML Shared Library Not Found"
    nvml.nvmlShutdown.assert_not_called()


@pytest.mark.parametrize(
    "failing", ["nvmlDeviceGetHandleByIndex", "nvmlDeviceGetMemoryInfo"]
)
def test_nvml_query_failure_is_unknown_and_shuts_down(failing: str) -> None:
    nvml = _nvml(**{failing: MagicMock(side_effect=RuntimeError("GPU is lost"))})
    with patch.object(H, "pynvml", nvml):
        vram, error = H.check_vram()
    assert vram is None and "GPU is lost" in error
    nvml.nvmlShutdown.assert_called_once()


def test_nvml_reading_of_device_zero() -> None:
    nvml = _nvml()
    with patch.object(H, "pynvml", nvml), patch.object(H, "VRAM_WARN_PCT", 85.0):
        vram, error = H.check_vram()
    assert error == ""
    assert vram == H.VramStatus(used_gb=3.0, total_gb=12.0, pct=25.0, warn=False)
    nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
    nvml.nvmlShutdown.assert_called_once()


@pytest.mark.parametrize(("used", "warn"), [(89, False), (90, True), (91, True)])
def test_vram_warn_threshold_is_inclusive(used: int, warn: bool) -> None:
    # VRAM_WARN_PCT comes from the environment; pin it.
    at = SimpleNamespace(used=used, total=100)
    nvml = _nvml(nvmlDeviceGetMemoryInfo=MagicMock(return_value=at))
    with patch.object(H, "pynvml", nvml), patch.object(H, "VRAM_WARN_PCT", 90.0):
        vram, _ = H.check_vram()
    assert vram is not None and vram.warn is warn


# ── Redis and stress metrics ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unreachable_redis_has_no_queue_depth() -> None:
    client = MagicMock()
    client.ping = AsyncMock(side_effect=ConnectionError("Connection refused"))
    client.aclose = AsyncMock()
    with patch("saccade.pipeline.health.aioredis.from_url", return_value=client):
        status, depth = await H.check_redis()
    assert status.status == DEGRADED and "Connection refused" in status.detail
    assert depth is None


def test_fragmentation_is_not_zero_when_nothing_is_measured() -> None:
    cuda = MagicMock()
    cuda.is_available.return_value = False
    with patch("saccade.pipeline.health.torch.cuda", cuda):
        assert H.get_vram_frag() is None
    cuda.is_available.return_value = True
    cuda.memory_reserved.return_value = 0
    with patch("saccade.pipeline.health.torch.cuda", cuda):
        assert H.get_vram_frag() is None
    cuda.memory_reserved.return_value = 100
    cuda.memory_allocated.return_value = 75
    with patch("saccade.pipeline.health.torch.cuda", cuda):
        assert H.get_vram_frag() == pytest.approx(25.0)


# ── The checker and its report ────────────────────────────────────────────────


async def _active(service: str) -> H.ServiceStatus:
    return H.ServiceStatus(service, OK, "active")


async def _redis_down() -> tuple[H.ServiceStatus, None]:
    return H.ServiceStatus("redis", DEGRADED, "Connection refused"), None


async def _redis_up() -> tuple[H.ServiceStatus, int]:
    return H.ServiceStatus("redis", OK, "connected"), 4


@pytest.mark.asyncio
async def test_nvml_failure_with_everything_else_up_is_not_ok() -> None:
    """The reproduced case: the old checker printed ``Saccade Status — OK``."""
    nvml = _nvml(nvmlInit=MagicMock(side_effect=RuntimeError("Driver Not Loaded")))
    with (
        patch.object(H, "check_systemd", _active),
        patch.object(H, "check_redis", _redis_up),
        patch.object(H, "pynvml", nvml),
        patch.object(H, "get_vram_frag", lambda: None),
    ):
        report = await H.HealthChecker().run()
    assert report.vram is None and report.status == "UNKNOWN" and not report.overall_ok
    text = H.render(report)
    assert "Saccade Status — UNKNOWN" in text
    assert "? unknown: RuntimeError: Driver Not Loaded" in text
    assert "Saccade Status — OK" not in text


@pytest.mark.asyncio
async def test_report_names_sources_and_unknowns() -> None:
    with (
        patch.object(H, "check_systemd", _active),
        patch.object(H, "check_redis", _redis_down),
        patch.object(H, "pynvml", _nvml()),
        patch.object(H, "get_vram_frag", lambda: None),
        patch.object(H, "REDIS_URL", "redis://user:s3cret@cache.local:6380/2"),
    ):
        report = await H.HealthChecker().run()
        text = H.render(report)
    assert report.status == "DEGRADED"
    assert report.stress is not None and report.stress.l3_queue_depth is None
    assert "Systemd (systemctl --user)" in text
    assert "VRAM (NVML GPU 0, all processes)" in text
    assert "Services (REDIS_URL redis://cache.local:6380/2)" in text
    assert "s3cret" not in text and "user:" not in text
    assert "L3 Queue Depth (saccade:stream): unknown (Redis not reachable)" in text
    assert "0 items" not in text
    assert "Loop Latency (this process)" in text
    assert "VRAM Fragmentation (this process): n/a" in text
    assert "Stress Metrics (not part of the status)" in text


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("redis://localhost:6379/0", "redis://localhost:6379/0"),
        ("redis://user:s3cret@cache.local:6380/2", "redis://cache.local:6380/2"),
        ("redis://:s3cret@h/1", "redis://h/1"),
        ("rediss://u:s3cret@h/2", "rediss://h/2"),
        ("redis://user:s3cret@[::1]:6379/0", "redis://[::1]:6379/0"),
        ("redis://[::1]/0", "redis://[::1]/0"),
        ("unix:///tmp/redis.sock?db=0&password=s3cret", "unix:///tmp/redis.sock"),
        ("redis+unix://u:s3cret@/tmp/r.sock", "redis+unix:///tmp/r.sock"),
        ("redis://u:s3cret@h:notaport/0", "redis://h:notaport/0"),
        ("redis://u:s3cret@h:65536/0", "redis://h:65536/0"),
        ("redis://u:s3cret@[::1/0", "<unparseable URL>"),
    ],
)
def test_redacted_url_never_fails_or_leaks(url: str, shown: str) -> None:
    assert H.redacted_url(url) == shown
    assert "s3cret" not in H.redacted_url(url)


@pytest.mark.parametrize(
    "url", ["redis://u:s3cret@cache:bad/0", "redis://cache:65536/0"]
)
def test_report_renders_with_an_invalid_redis_url(url: str) -> None:
    """Codex r1: the report used to raise on an invalid port while redacting."""
    with patch.object(H, "REDIS_URL", url):
        text = H.render(_report(redis=DEGRADED))
    assert "Saccade Status — DEGRADED" in text
    assert "Services (REDIS_URL redis://cache:" in text and "s3cret" not in text


def test_report_marks_each_state() -> None:
    report = _report(systemd=(OK, DEGRADED, UNKNOWN))
    lines = H.render(report).splitlines()
    assert any(line.startswith("  ● svc0") and "ok: x" in line for line in lines)
    assert any(line.startswith("  ✗ svc1") and "degraded: x" in line for line in lines)
    assert any(line.startswith("  ? svc2") and "unknown: x" in line for line in lines)
