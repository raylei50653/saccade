"""#537 D-537-4A: NVML measurements have explicit acquisition validity.

Successful zero usage is a measurement; failed initialization, failed reads and
invalid raw samples raise ResourceMonitorUnavailable with their cause intact.
All NVML functions are fake, so these contracts require no GPU or driver.
Dispatcher lifecycle and publication are tested in their separate contract file.
"""

# scope: resource
# function: contract
# lifecycle: active

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import saccade.resource.resource_manager as rm
from saccade.resource.resource_manager import DegradationLevel as L
from saccade.resource.resource_manager import (
    ResourceManager,
    ResourceMonitorUnavailable,
)

_MIB = 1024**2


def _sample(
    *, total: float = 100 * _MIB, used: float = 0, free: float = 100 * _MIB
) -> SimpleNamespace:
    return SimpleNamespace(total=total, used=used, free=free)


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock()
    fake.nvmlDeviceGetMemoryInfo.return_value = _sample()
    monkeypatch.setattr(rm, "pynvml", fake)
    return fake


@pytest.mark.parametrize(
    ("used", "free", "expected_percent"),
    [(0, 100, 0.0), (100, 0, 100.0), (25, 75, 25.0), (97, 3, 97.0)],
    ids=["valid-zero", "valid-full", "partial", "emergency"],
)
def test_valid_measurements_preserve_units_and_boundaries(
    nvml: MagicMock, used: float, free: float, expected_percent: float
) -> None:
    nvml.nvmlDeviceGetMemoryInfo.return_value = _sample(
        used=used * _MIB, free=free * _MIB
    )
    manager = ResourceManager(gpu_id=3)
    assert vars(manager.get_stats()) == {
        "free_mb": free,
        "used_mb": used,
        "total_mb": 100.0,
        "usage_percent": expected_percent,
    }
    # A measurement/preflight alone must not advance the hysteresis state.
    assert manager.current_level is L.NORMAL
    nvml.nvmlInit.assert_called_once_with()
    nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(3)
    nvml.nvmlDeviceGetMemoryInfo.assert_called_once_with(
        nvml.nvmlDeviceGetHandleByIndex.return_value
    )
    manager.close()
    nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "free", [0, 50, 150], ids=["free-zero", "not-a-sum", "free-over-total"]
)
def test_free_is_independent_of_total_minus_used(nvml: MagicMock, free: float) -> None:
    """Approved domain requires finite free >= 0, without an accounting equation."""
    nvml.nvmlDeviceGetMemoryInfo.return_value = _sample(
        used=25 * _MIB, free=free * _MIB
    )
    stats = ResourceManager().get_stats()
    assert (stats.total_mb, stats.used_mb, stats.free_mb) == (100.0, 25.0, free)
    assert stats.usage_percent == 25.0


def test_init_failure_is_constructible_but_never_returns_stats_or_retries(
    nvml: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    original = RuntimeError("NVML library missing")
    nvml.nvmlInit.side_effect = original
    manager = ResourceManager(gpu_id=2)
    assert manager.current_level is L.NORMAL
    assert "NVML library missing" in capsys.readouterr().out
    nvml.nvmlInit.side_effect = None  # backend recovery cannot repair this instance
    manager.current_level = L.EMERGENCY
    for _ in range(3):
        for read in (manager.get_stats, manager.decide_degradation_level):
            with pytest.raises(
                ResourceMonitorUnavailable, match="NVML library missing"
            ) as exc:
                read()
            assert isinstance(exc.value, RuntimeError)
            assert exc.value.__cause__ is original
        assert manager.current_level is L.EMERGENCY
    assert capsys.readouterr().out == ""  # no per-read diagnostic flooding
    nvml.nvmlInit.assert_called_once_with()
    nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
    nvml.nvmlDeviceGetMemoryInfo.assert_not_called()
    manager.close()
    nvml.nvmlShutdown.assert_not_called()

    fresh = ResourceManager(gpu_id=2)
    assert fresh.current_level is L.NORMAL  # a fresh manager follows R0
    assert fresh.get_stats().usage_percent == 0.0
    assert fresh.decide_degradation_level() is L.NORMAL
    assert nvml.nvmlInit.call_count == 2
    fresh.close()
    nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "failure_point", ["nvmlDeviceGetHandleByIndex", "nvmlDeviceGetMemoryInfo"]
)
def test_read_failure_keeps_original_cause_and_can_recover_without_reinit(
    nvml: MagicMock, failure_point: str, capsys: pytest.CaptureFixture[str]
) -> None:
    manager = ResourceManager()
    nvml.nvmlDeviceGetMemoryInfo.return_value = _sample(used=97 * _MIB, free=3 * _MIB)
    assert manager.decide_degradation_level() is L.EMERGENCY
    capsys.readouterr()
    original = RuntimeError(f"{failure_point} failed")
    getattr(nvml, failure_point).side_effect = original
    for _ in range(3):
        for read in (manager.get_stats, manager.decide_degradation_level):
            with pytest.raises(ResourceMonitorUnavailable) as exc:
                read()
            assert exc.value.__cause__ is original
            assert str(original) in str(exc.value)
        assert manager.current_level is L.EMERGENCY
    assert capsys.readouterr().out == ""

    getattr(nvml, failure_point).side_effect = None
    nvml.nvmlDeviceGetMemoryInfo.return_value = _sample()
    assert manager.get_stats().usage_percent == 0.0
    assert manager.current_level is L.EMERGENCY
    assert manager.decide_degradation_level() is L.FAST_PATH
    nvml.nvmlInit.assert_called_once_with()
    manager.close()
    nvml.nvmlShutdown.assert_called_once_with()


_INVALID_SAMPLES = [
    ("total", 0),
    ("total", -1),
    ("total", float("nan")),
    ("total", float("inf")),
    ("total", float("-inf")),
    ("used", -1),
    ("used", 101 * _MIB),
    ("used", float("nan")),
    ("used", float("inf")),
    ("used", float("-inf")),
    ("free", -1),
    ("free", float("nan")),
    ("free", float("inf")),
    ("free", float("-inf")),
    ("total", None),
    ("used", "unknown"),
    ("free", object()),
]


@pytest.mark.parametrize(
    ("field", "value"),
    _INVALID_SAMPLES,
    ids=[f"{field}-{index}" for index, (field, _) in enumerate(_INVALID_SAMPLES)],
)
def test_invalid_raw_sample_has_validation_cause_and_does_not_recover_level(
    nvml: MagicMock, field: str, value: object
) -> None:
    manager = ResourceManager()
    manager.current_level = L.EMERGENCY
    invalid = _sample()
    setattr(invalid, field, value)
    nvml.nvmlDeviceGetMemoryInfo.return_value = invalid
    for read in (manager.get_stats, manager.decide_degradation_level):
        with pytest.raises(ResourceMonitorUnavailable) as exc:
            read()
        assert isinstance(exc.value.__cause__, ValueError)
        assert "invalid NVML memory sample" in str(exc.value)
    assert manager.current_level is L.EMERGENCY
    # Invalid samples do not invalidate initialization or require a new manager.
    nvml.nvmlDeviceGetMemoryInfo.return_value = _sample()
    assert manager.decide_degradation_level() is L.FAST_PATH
    nvml.nvmlInit.assert_called_once_with()


def test_missing_raw_sample_field_is_a_validation_failure(nvml: MagicMock) -> None:
    del nvml.nvmlDeviceGetMemoryInfo.return_value.total
    with pytest.raises(ResourceMonitorUnavailable) as exc:
        ResourceManager().get_stats()
    assert isinstance(exc.value.__cause__, ValueError)
    assert "total" in str(exc.value.__cause__)
