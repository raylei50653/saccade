"""Auto-marking for the research test quarantine.

Every test below tests/research/ gets ``research`` (excluded from the default
collection via addopts ``-m 'not research'``; run explicitly with
``pytest -m research tests/research/``).

Files listed in ``PACKET_BOUND_FILES`` additionally get ``packet_bound``:
they read a sealed evidence packet or frozen dated artifact and effectively
skip when that artifact is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RESEARCH_DIR = Path(__file__).resolve().parent

PACKET_BOUND_FILES = {
    # safe_region
    "test_safe_region_a1_audit.py",
    "test_safe_region_asset_r1_conversion.py",
    # gap_conditioned_motion (exec runners inside dated evidence dirs)
    "test_gap_conditioned_motion_e0.py",
    "test_gap_conditioned_motion_e1_m0.py",
    "test_gap_conditioned_motion_e2_family.py",
    "test_gap_conditioned_motion_e3_signals.py",
    "test_gap_conditioned_motion_phase_b.py",
    # d_online_stage2 (dated out/signal_study artifacts)
    "test_d_online_stage2_q1q3.py",
    "test_d_online_stage2_q4.py",
    "test_d_online_stage2_q45_atlas.py",
    # misc
    "test_boolean_atom_partial_order.py",
    "test_d0_bridge_estimator_fidelity.py",
    "test_portable_or_tail.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # This hook fires with the FULL collected item list (conftest hooks are
    # global once loaded), so restrict marking to items under this directory.
    for item in items:
        if item.path is None or _RESEARCH_DIR not in item.path.parents:
            continue
        item.add_marker(pytest.mark.research)
        if item.path.name in PACKET_BOUND_FILES:
            item.add_marker(pytest.mark.packet_bound)
