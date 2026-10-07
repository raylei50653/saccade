"""``saccade_track`` refuses a torn schedule config at the CLI (#465 PR-10 review).

The PR-10 review found that ``saccade_track --schedule serial`` skipped
``plan_schedule``: ``opt.schedule == "serial" || !plan_schedule(cfg)...``
short-circuits, so a resolved config whose recorded
``steps.schedule.double_buffer`` disagrees with the oracle's own preconditions
(``SACCADE_DETECT_BARRIER``, ``SACCADE_DOUBLE_BUFFER``) ran instead of failing
closed. ``test_shipping_native_config.cpp`` pins ``select_schedule`` itself; this
pins the entrypoint: the real binary, a torn config, with and without the
developer override, must exit 2 with the schedule error -- before any model,
lineage or sequence is touched (none of them exist here), so no GPU is needed.
Skips when ``build/shipping/saccade_track`` has not been built.

#465 PR-C2: ``--schedule serial`` is an option of the developer build
``saccade_track_measurement`` only; the shipping ``saccade_track`` refuses it
as an unknown argument before it reads the config. The override cases run the
developer build (skipped when it is not built).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TRACK = REPO / "build" / "shipping" / "saccade_track"
MEASUREMENT = REPO / "build" / "shipping" / "saccade_track_measurement"
CONFIG = REPO / "configs" / "shipping" / "mamba_whole_graph.resolved.json"

pytestmark = pytest.mark.skipif(
    not TRACK.exists(), reason="build/shipping/saccade_track not built"
)


def _barrier_full(cfg: dict[str, Any]) -> None:
    cfg["host_params"]["env"]["SACCADE_DETECT_BARRIER"] = "full"


def _double_buffer_env_unset(cfg: dict[str, Any]) -> None:
    cfg["host_params"]["env"]["SACCADE_DOUBLE_BUFFER"] = None


def _run(tmp_path: Path, config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    # The serial override exists only in the developer build (PR-C2).
    binary = MEASUREMENT if "--schedule" in extra else TRACK
    if not binary.exists():
        pytest.skip(f"{binary.relative_to(REPO)} not built")
    return subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--lineage",
            str(tmp_path / "no_such.lineage.json"),
            "--out",
            str(tmp_path / "out"),
            *extra,
            str(tmp_path / "no_such_sequence"),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("tear", [_barrier_full, _double_buffer_env_unset])
@pytest.mark.parametrize("override", [(), ("--schedule", "serial")])
def test_torn_schedule_config_fails_closed(
    tmp_path: Path, tear: Any, override: tuple[str, ...]
) -> None:
    cfg = json.loads(CONFIG.read_text())
    tear(cfg)
    torn = tmp_path / "torn.resolved.json"
    torn.write_text(json.dumps(cfg))
    r = _run(tmp_path, torn, *override)
    assert r.returncode == 2, r.stderr
    assert "shipping schedule:" in r.stderr, r.stderr
    assert not (tmp_path / "out").exists()


def test_committed_config_passes_the_schedule_check(tmp_path: Path) -> None:
    # Control: with the committed config the run gets past the schedule plan
    # and fails later, on the missing lineage -- not with the schedule error.
    for override in ((), ("--schedule", "serial")):
        r = _run(tmp_path, CONFIG, *override)
        assert r.returncode == 2
        assert "shipping schedule:" not in r.stderr, r.stderr


def test_shipping_entrypoint_refuses_the_override(tmp_path: Path) -> None:
    # PR-C2: no developer option on the shipping binary; refused before the
    # config is read (this one does not exist).
    for extra in (
        ("--schedule", "serial"),
        ("--measurement-mutation", "none"),
        ("--max-frames", "5"),
    ):
        r = subprocess.run(
            [
                str(TRACK),
                "--config",
                str(tmp_path / "no_such.resolved.json"),
                "--lineage",
                str(tmp_path / "no_such.lineage.json"),
                "--out",
                str(tmp_path / "out"),
                *extra,
                str(tmp_path / "no_such_sequence"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert r.returncode == 2
        assert r.stderr == f"saccade_track: unknown argument {extra[0]}\n", r.stderr
        assert not (tmp_path / "out").exists()
