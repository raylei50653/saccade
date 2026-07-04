from __future__ import annotations

import subprocess

from saccade.perception.online_telemetry import OnlineTelemetry


def test_online_telemetry_summarizes_stage_windows() -> None:
    telemetry = OnlineTelemetry(window=4)
    telemetry.observe("stage", 1.0)
    telemetry.observe("stage", 3.0)
    telemetry.increment("drops")
    telemetry.increment("drops", 2)

    summary = telemetry.summary()

    assert summary["stage_ms_mean"] == 2.0
    assert summary["stage_ms_p95"] > 0.0
    assert summary["drops"] == 3


def test_online_telemetry_gpu_probe_handles_missing_nvidia_smi(monkeypatch) -> None:
    telemetry = OnlineTelemetry()

    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _raise)

    assert telemetry._read_gpu() is None
