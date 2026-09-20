"""ADR 026: scope a CLOSED packet's pinned currency tests to the attested arm.

The H0/GCTM packets pin their own targeted test files by sha256, and those
files compare the packets' frozen digests against the working tree.  That is a
*currency* assertion ("HEAD still is the frozen coordinate"), not a historical
one.  Once the supersession ledger records that a packet is historical for some
path, the development arm skips that packet's targeted tests with the reason,
and ``SACCADE_ATTESTED_CONSUMER=1`` runs them again (where they fail, correctly,
because the consumer is claiming currency).

Unrecorded drift is never skipped: the pinned tests fail as before and
``test_frozen_source_evolution_policy.py`` names the missing ledger entry.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import frozen_source_status as frozen  # noqa: E402


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if frozen.attested_consumer_requested():
        return
    try:
        report = frozen.evaluate(_REPO, mode="development")
    except Exception:  # noqa: BLE001 - never let the guard hide the real failure
        return
    to_skip = frozen.targeted_tests_to_skip(report, _REPO)
    if not to_skip:
        return
    for item in items:
        try:
            rel = Path(str(item.fspath)).resolve().relative_to(_REPO).as_posix()
        except ValueError:
            continue
        reason = to_skip.get(rel)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
