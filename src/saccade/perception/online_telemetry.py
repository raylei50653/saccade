from __future__ import annotations

import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


@dataclass(frozen=True)
class ResourceSample:
    timestamp: float
    cpu_percent: float
    gpu_util_percent: float
    gpu_mem_util_percent: float
    gpu_mem_used_mb: float


class OnlineTelemetry:
    """Rolling latency/resource telemetry for online perception pipelines."""

    def __init__(self, window: int = 512, sample_interval_sec: float = 1.0):
        self.window = int(window)
        self.sample_interval_sec = float(sample_interval_sec)
        self._stages: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._counters: dict[str, int] = defaultdict(int)
        self._resources: deque[ResourceSample] = deque(maxlen=max(8, self.window))
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_cpu_total: int | None = None
        self._last_cpu_idle: int | None = None

    def observe(self, stage: str, elapsed_ms: float) -> None:
        with self._lock:
            self._stages[stage].append(float(elapsed_ms))

    def increment(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[counter] += int(amount)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop, name="online-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            stages = {name: list(values) for name, values in self._stages.items()}
            counters = dict(self._counters)
            resources = list(self._resources)

        out: dict[str, Any] = {}
        for name, values in stages.items():
            out[f"{name}_ms_mean"] = round(float(np.mean(values)), 3) if values else 0.0
            out[f"{name}_ms_p95"] = round(_percentile(values, 95), 3)
            out[f"{name}_ms_p99"] = round(_percentile(values, 99), 3)
        out.update(counters)

        if resources:
            cpu = [s.cpu_percent for s in resources]
            gpu = [s.gpu_util_percent for s in resources]
            gpu_mem = [s.gpu_mem_util_percent for s in resources]
            gpu_used = [s.gpu_mem_used_mb for s in resources]
            out.update(
                {
                    "cpu_util_mean": round(float(np.mean(cpu)), 2),
                    "cpu_util_p95": round(_percentile(cpu, 95), 2),
                    "gpu_util_mean": round(float(np.mean(gpu)), 2),
                    "gpu_util_p95": round(_percentile(gpu, 95), 2),
                    "gpu_mem_util_mean": round(float(np.mean(gpu_mem)), 2),
                    "gpu_mem_used_mb_mean": round(float(np.mean(gpu_used)), 1),
                }
            )
        return out

    def _sample_loop(self) -> None:
        while self._running:
            sample = ResourceSample(
                timestamp=time.time(),
                cpu_percent=self._read_cpu_percent(),
                gpu_util_percent=0.0,
                gpu_mem_util_percent=0.0,
                gpu_mem_used_mb=0.0,
            )
            gpu = self._read_gpu()
            if gpu is not None:
                sample = ResourceSample(sample.timestamp, sample.cpu_percent, *gpu)
            with self._lock:
                self._resources.append(sample)
            time.sleep(self.sample_interval_sec)

    def _read_cpu_percent(self) -> float:
        try:
            parts = [
                int(x)
                for x in open("/proc/stat", encoding="utf-8").readline().split()[1:]
            ]
        except Exception:
            return 0.0
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        if self._last_cpu_total is None or self._last_cpu_idle is None:
            self._last_cpu_total = total
            self._last_cpu_idle = idle
            return 0.0
        total_delta = max(1, total - self._last_cpu_total)
        idle_delta = idle - self._last_cpu_idle
        self._last_cpu_total = total
        self._last_cpu_idle = idle
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))

    def _read_gpu(self) -> tuple[float, float, float] | None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=0.5,
            )
        except Exception:
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        line = proc.stdout.strip().splitlines()[0]
        try:
            gpu_util, mem_util, mem_used = [float(x.strip()) for x in line.split(",")]
        except Exception:
            return None
        return gpu_util, mem_util, mem_used
