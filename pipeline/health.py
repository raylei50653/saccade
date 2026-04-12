"""
pipeline/health.py

Saccade system health monitor.
Run directly for a one-shot status snapshot:

    python -m pipeline.health

Or import and call periodically from orchestrator.py:

    from pipeline.health import HealthChecker
    checker = HealthChecker()
    report = await checker.run()
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
import pynvml
import redis.asyncio as aioredis


# ── Config (override via .env) ────────────────────────────────────────────────

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379")
VRAM_WARN_PCT    = float(os.getenv("VRAM_WARN_PCT", "85"))

SYSTEMD_SERVICES = [
    "yolo-perception",
    "yolo-cognition",
    "mediamtx",
]


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ServiceStatus:
    name: str
    ok: bool
    detail: str = ""


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
    llama: ServiceStatus
    redis: ServiceStatus
    overall_ok: bool = field(init=False)

    def __post_init__(self) -> None:
        checks = [
            all(s.ok for s in self.systemd),
            self.llama.ok,
            self.redis.ok,
            (not self.vram.warn if self.vram else True),
        ]
        self.overall_ok = all(checks)


# ── Checkers ──────────────────────────────────────────────────────────────────

async def check_systemd(service: str) -> ServiceStatus:
    """Check a single Systemd service via systemctl is-active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", service],
            timeout=3,
        )
        ok = result.returncode == 0
        return ServiceStatus(name=service, ok=ok, detail="running" if ok else "inactive/failed")
    except Exception as e:
        return ServiceStatus(name=service, ok=False, detail=str(e))


def check_vram() -> Optional[VramStatus]:
    """Query VRAM usage via NVML (first GPU only)."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        used_gb  = mem.used  / 1024 ** 3
        total_gb = mem.total / 1024 ** 3
        pct      = (mem.used / mem.total) * 100

        return VramStatus(
            used_gb=round(used_gb, 1),
            total_gb=round(total_gb, 1),
            pct=round(pct, 1),
            warn=pct >= VRAM_WARN_PCT,
        )
    except Exception:
        return None


async def check_llama_server() -> ServiceStatus:
    """Hit llama-server /health endpoint and measure latency."""
    url = f"{LLAMA_SERVER_URL}/health"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            resp = await client.get(url)
            ok = resp.status_code == 200
            return ServiceStatus(
                name="llama-server",
                ok=ok,
                detail=f"HTTP {resp.status_code}",
            )
    except httpx.ConnectError:
        return ServiceStatus(name="llama-server", ok=False, detail="connection refused")
    except httpx.TimeoutException:
        return ServiceStatus(name="llama-server", ok=False, detail="timeout")
    except Exception as e:
        return ServiceStatus(name="llama-server", ok=False, detail=str(e))


async def check_redis() -> ServiceStatus:
    """Ping Redis and report queue depth."""
    try:
        r = aioredis.from_url(REDIS_URL, socket_timeout=3)
        await r.ping()
        depth = await r.llen("saccade:events")
        await r.aclose()
        return ServiceStatus(name="redis", ok=True, detail=f"queue depth: {depth}")
    except Exception as e:
        return ServiceStatus(name="redis", ok=False, detail=str(e))


# ── Aggregator ────────────────────────────────────────────────────────────────

class HealthChecker:
    async def run(self) -> HealthReport:
        systemd_results = await asyncio.gather(*[check_systemd(s) for s in SYSTEMD_SERVICES])
        llama, redis_status = await asyncio.gather(
            check_llama_server(),
            check_redis(),
        )

        return HealthReport(
            timestamp=datetime.now(),
            systemd=list(systemd_results),
            vram=check_vram(),
            llama=llama,
            redis=redis_status,
        )


# ── Renderer ──────────────────────────────────────────────────────────────────

def render(report: HealthReport) -> str:
    lines: list[str] = []
    ts = report.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    overall = "OK" if report.overall_ok else "DEGRADED"

    lines.append(f"[{ts}] Saccade Status — {overall}")
    lines.append("─" * 50)

    lines.append("Systemd")
    for s in report.systemd:
        dot = "●" if s.ok else "✗"
        lines.append(f"  {dot} {s.name:<24} {s.detail}")

    lines.append("")
    if report.vram:
        warn = "  ⚠ WARN" if report.vram.warn else ""
        lines.append("VRAM")
        lines.append(
            f"  {report.vram.used_gb}GB / {report.vram.total_gb}GB  "
            f"{report.vram.bar}  {report.vram.pct}%{warn}"
        )
    else:
        lines.append("VRAM  — unavailable (no GPU or pynvml not installed)")

    lines.append("")
    lines.append("Services")
    for svc in [report.llama, report.redis]:
        dot = "●" if svc.ok else "✗"
        lines.append(f"  {dot} {svc.name:<24} {svc.detail}")

    return "\n".join(lines)


# ── Entrypoint ────────────────────────────────────────────────────────────────

async def _main() -> None:
    checker = HealthChecker()
    report  = await checker.run()
    print(render(report))

    if not report.overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_main())
