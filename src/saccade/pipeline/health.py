"""
src/saccade/pipeline/health.py

Saccade system health monitor with Industrial Stress Metrics.

Each check is ``ok``, ``degraded`` (observed and not healthy) or ``unknown``
(could not be observed), and the report names where each one comes from. A
check that could not be observed never counts as healthy: the overall status
is DEGRADED when any check is degraded, else UNKNOWN when any is unknown,
else OK (#537, docs/architecture/failure_degradation_537.md §24). The report
is informational: the exit code stays 0 and no readiness policy is implied.
"""

from __future__ import annotations

import asyncio
import os
import string
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urlsplit

import pynvml
import redis.asyncio as aioredis
import torch


# ── Config (override via .env) ────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VRAM_WARN_PCT = float(os.getenv("VRAM_WARN_PCT", "85"))
VRAM_DEVICE = 0

SYSTEMD_SERVICES = [
    "yolo-perception",
    "yolo-orchestrator",
    "mediamtx",
]

OK, DEGRADED, UNKNOWN = "ok", "degraded", "unknown"


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class StressMetrics:
    l3_queue_depth: Optional[int]  # None: Redis was not reachable
    loop_latency_ms: float
    vram_fragmentation_pct: Optional[float]  # None: no CUDA allocation here


@dataclass
class ServiceStatus:
    name: str
    status: str  # OK / DEGRADED / UNKNOWN
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass
class VramStatus:
    used_gb: float
    total_gb: float
    pct: float
    warn: bool

    @property
    def bar(self) -> str:
        filled = int(self.pct / 10)
        return "█" * filled + "░" * (10 - filled)


@dataclass
class HealthReport:
    timestamp: datetime
    systemd: list[ServiceStatus]
    vram: Optional[VramStatus]
    redis: ServiceStatus
    stress: Optional[StressMetrics] = None
    vram_error: str = ""  # why vram is None
    status: str = field(init=False)  # "OK" / "DEGRADED" / "UNKNOWN"
    overall_ok: bool = field(init=False)

    @property
    def vram_status(self) -> str:
        if self.vram is None:
            return UNKNOWN
        return DEGRADED if self.vram.warn else OK

    def __post_init__(self) -> None:
        checks = [s.status for s in self.systemd] + [
            self.redis.status,
            self.vram_status,
        ]
        if DEGRADED in checks:
            self.status = "DEGRADED"
        elif UNKNOWN in checks:
            self.status = "UNKNOWN"
        else:
            self.status = "OK"
        self.overall_ok = self.status == "OK"


# ── Checkers ──────────────────────────────────────────────────────────────────


async def check_systemd(service: str) -> ServiceStatus:
    """The user unit's state (``systemctl --user``). Only a loaded, active
    unit is ok; a unit this user scope does not have, or a systemctl that
    cannot answer, is unknown -- the service may run elsewhere (system units)."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", "-p", "LoadState,ActiveState", service],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as e:
        return ServiceStatus(
            service, UNKNOWN, f"systemctl unavailable: {type(e).__name__}: {e}"
        )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        return ServiceStatus(
            service, UNKNOWN, f"systemctl exit {result.returncode}: {msg}"
        )
    props = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    load, active = props.get("LoadState", ""), props.get("ActiveState", "")
    if not load or not active:
        return ServiceStatus(
            service, UNKNOWN, f"no unit state in {result.stdout.strip()!r}"
        )
    if load == "not-found":
        return ServiceStatus(service, UNKNOWN, "no such user unit")
    if load == "loaded" and active == "active":
        return ServiceStatus(service, OK, "active")
    return ServiceStatus(service, DEGRADED, f"{load}/{active}")


def check_vram() -> Tuple[Optional[VramStatus], str]:
    """Device-wide memory of GPU ``VRAM_DEVICE`` (all processes), or None and
    the reason it could not be read."""
    try:
        pynvml.nvmlInit()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(VRAM_DEVICE)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pct = (mem.used / mem.total) * 100
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return VramStatus(
        used_gb=round(mem.used / 1024**3, 1),
        total_gb=round(mem.total / 1024**3, 1),
        pct=round(pct, 1),
        warn=pct >= VRAM_WARN_PCT,
    ), ""


# Plain ASCII scheme / host / port / path characters. Anything else -- '@'
# (userinfo), '?' (query), '#' (fragment), '%' (escapes), non-ASCII look-alikes
# such as a fullwidth '＠' -- may carry or hide a secret.
_PLAIN_URL_CHARS = frozenset(string.ascii_letters + string.digits + "+-._~:/[]")
UNPARSEABLE_URL = "<unparseable URL>"


def url_may_carry_secrets(url: str) -> bool:
    """Whether ``url`` has any character outside the plain set above (an
    allowlist: a secret cannot sit in a URL made only of these)."""
    return not set(url) <= _PLAIN_URL_CHARS


def redacted_url(url: str) -> str:
    """``url`` for the report, without its credentials. Only the netloc's
    userinfo is cut; everything else stays as written (an IPv6 host keeps its
    brackets, an invalid port is shown, not parsed) and the whole rest must be
    plain, or the URL is replaced. Nothing is discarded before that check, so
    a URL with a query or fragment, userinfo the split did not separate (a '?'
    or '#' in a password moves '@host' out of the netloc), escapes or
    look-alike characters is never shown. The netloc is cut from ``url`` as
    written (up to the first '/', '?' or '#'), not taken from urlsplit, which
    drops tabs and newlines. The report never fails on REDIS_URL and never
    prints a password from it."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return UNPARSEABLE_URL
    prefix = f"{parts.scheme}://"
    if parts.scheme and url.startswith(prefix):
        after = url[len(prefix) :]
        end = min((i for i in map(after.find, "/?#") if i >= 0), default=len(after))
        shown = prefix + after[:end].rpartition("@")[2] + after[end:]
    else:
        shown = url
    return UNPARSEABLE_URL if url_may_carry_secrets(shown) else shown


def redis_error_detail(error: Exception) -> str:
    """The Redis error for the report. Parser and connection errors can quote
    REDIS_URL, so when it may carry secrets only the error type is shown."""
    if url_may_carry_secrets(REDIS_URL):
        return (
            f"{type(error).__name__} (message hidden: REDIS_URL may carry credentials)"
        )
    return str(error)


async def check_redis() -> Tuple[ServiceStatus, Optional[int]]:
    """Redis at REDIS_URL and the length of ``saccade:stream``; the length is
    None when it could not be read (not 0)."""
    from typing import cast, Awaitable, Any

    r: Any = None
    try:
        r = aioredis.from_url(REDIS_URL, socket_timeout=3)
        await cast(Awaitable[Any], r.ping())
        depth = await cast(Awaitable[int], r.xlen("saccade:stream"))
        return ServiceStatus(name="redis", status=OK, detail="connected"), int(depth)
    except Exception as e:
        return ServiceStatus(
            name="redis", status=DEGRADED, detail=redis_error_detail(e)
        ), None
    finally:
        if r is not None:
            try:
                await cast(Awaitable[Any], r.aclose())
            except Exception:
                pass


async def measure_loop_latency() -> float:
    """This process's own event loop (not a service's)."""
    start = time.perf_counter()
    await asyncio.sleep(0)
    return (time.perf_counter() - start) * 1000


def get_vram_frag() -> Optional[float]:
    """This process's own torch allocator (not a service's); None without
    CUDA or when this process has reserved nothing."""
    if not torch.cuda.is_available():
        return None
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    if reserved == 0:
        return None
    return float((1.0 - (allocated / reserved)) * 100)


# ── Aggregator ────────────────────────────────────────────────────────────────


class HealthChecker:
    async def run(self) -> HealthReport:
        systemd_results = await asyncio.gather(
            *[check_systemd(s) for s in SYSTEMD_SERVICES]
        )
        redis_status, q_depth = await check_redis()
        loop_latency = await measure_loop_latency()
        frag = get_vram_frag()
        vram, vram_error = check_vram()

        return HealthReport(
            timestamp=datetime.now(),
            systemd=list(systemd_results),
            vram=vram,
            redis=redis_status,
            stress=StressMetrics(
                l3_queue_depth=q_depth,
                loop_latency_ms=loop_latency,
                vram_fragmentation_pct=frag,
            ),
            vram_error=vram_error,
        )


# ── Renderer ──────────────────────────────────────────────────────────────────

_MARK = {OK: "●", DEGRADED: "✗", UNKNOWN: "?"}


def render(report: HealthReport) -> str:
    lines: list[str] = []
    ts = report.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"[{ts}] Saccade Status — {report.status}")
    lines.append("─" * 50)

    lines.append("Systemd (systemctl --user)")
    for s in report.systemd:
        lines.append(f"  {_MARK[s.status]} {s.name:<24} {s.status}: {s.detail}")

    lines.append("")
    lines.append(f"VRAM (NVML GPU {VRAM_DEVICE}, all processes)")
    if report.vram:
        warn = f"  ⚠ WARN (>= {VRAM_WARN_PCT:g}%)" if report.vram.warn else ""
        lines.append(
            f"  {_MARK[report.vram_status]} {report.vram.used_gb}GB / {report.vram.total_gb}GB  "
            f"{report.vram.bar}  {report.vram.pct}%{warn}"
        )
    else:
        lines.append(f"  {_MARK[UNKNOWN]} unknown: {report.vram_error or 'not read'}")

    lines.append("")
    lines.append(f"Services (REDIS_URL {redacted_url(REDIS_URL)})")
    r = report.redis
    lines.append(f"  {_MARK[r.status]} {r.name:<24} {r.status}: {r.detail}")

    if report.stress:
        st = report.stress
        depth = (
            f"{st.l3_queue_depth} items"
            if st.l3_queue_depth is not None
            else "unknown (Redis not reachable)"
        )
        frag = (
            f"{st.vram_fragmentation_pct:.1f}%"
            if st.vram_fragmentation_pct is not None
            else "n/a (no CUDA allocation in this process)"
        )
        lines.append("")
        lines.append("Stress Metrics (not part of the status)")
        lines.append(f"  ⚡ Loop Latency (this process): {st.loop_latency_ms:.2f} ms")
        lines.append(f"  📦 L3 Queue Depth (saccade:stream): {depth}")
        lines.append(f"  🧩 VRAM Fragmentation (this process): {frag}")

    return "\n".join(lines)


# ── Entrypoint ────────────────────────────────────────────────────────────────


async def _main() -> None:
    checker = HealthChecker()
    report = await checker.run()
    print(render(report))


if __name__ == "__main__":
    asyncio.run(_main())
