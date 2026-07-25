"""H0-R5 diagnostic: qualification vs authoritative extension-load vector parity.

Audit finding (2026-07-25): qualification inserts ``extension.parent`` into
``sys.path`` before ``import saccade_tracking_ext``; the controller child vector
does not. Under ``python -I -B`` site still loads ``saccade_build.pth`` →
``<repo>/build``, so the controller can import a non-h0_phase_a artifact and fail
membership for the expected realpath.

These tests are diagnostic only. They document whether each surface's *source*
explicitly bootstraps ``extension.parent``. They do **not** assert full
byte-policy equality of child scripts, argv, flags, environment, or path order.
A later seal-grade gate should compare a shared canonical vector/spec instead.

These tests do not authorize repair, seal, or launch.
"""

# scope: system
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, TOOLS.as_posix())

import qualify_h0_phase_a as qualify  # noqa: E402
import run_h0_phase_a as controller  # noqa: E402
import verify_h0_phase_a as verifier  # noqa: E402


def _source(fn) -> str:
    return inspect.getsource(fn)


def _explicitly_bootstraps_extension_parent(text: str) -> bool:
    """True when source text literally names both insert and extension.parent.

    This is a *source-shape* diagnostic, not a full vector policy comparison.
    A shared builder that hides the insert behind a helper will correctly fail
    this check until this diagnostic is replaced by canonical-vector equality.
    """
    return "sys.path.insert" in text and "extension.parent" in text


def test_membership_predicate_is_shared_module() -> None:
    import h0_runtime_confinement as confinement

    assert callable(confinement.assert_extension_plugin_membership)
    assert "assert_extension_plugin_membership" in _source(
        qualify._qualify_confined_extension_load
    )
    assert "assert_extension_plugin_membership" in _source(
        controller._verify_extension_load
    )


def test_qualification_load_vector_inserts_extension_parent() -> None:
    source = _source(qualify._qualify_confined_extension_load)
    assert _explicitly_bootstraps_extension_parent(source)


def test_maps_helper_inserts_extension_parent() -> None:
    source = _source(controller._runtime_maps_dependencies)
    assert _explicitly_bootstraps_extension_parent(source)


def test_controller_extension_load_child_script_lacks_extension_parent_insert() -> None:
    """Documents the R5 parity defect: controller child script omits the insert."""
    source = _source(controller._verify_extension_load)
    # Parent may still sys.path.insert tools/ for imports; child -c script must not
    # be confused with that. Look for the f-string that builds the child script.
    match = re.search(
        r"script\s*=\s*\((?P<body>.*?)\)\s*\n\s*vector\s*=\s*\[python",
        source,
        flags=re.S,
    )
    assert match is not None, "could not locate controller extension-load script"
    body = match.group("body")
    assert "import saccade_tracking_ext" in body
    assert not _explicitly_bootstraps_extension_parent(body)


def test_verifier_expected_vector_mirrors_controller_without_insert() -> None:
    source = _source(verifier._expected_extension_load_vector)
    assert "import saccade_tracking_ext" in source
    assert not _explicitly_bootstraps_extension_parent(source)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "H0-R5 parity audit: controller/verifier child-script sources do not "
        "explicitly bootstrap extension.parent (qualification and maps helper do). "
        "Remove xfail only when all four surfaces' checked regions explicitly "
        "bootstrap extension.parent in source — not when a shared builder merely "
        "establishes runtime parity without those literals. Full vector/spec "
        "equality belongs to a later seal-grade gate, not this diagnostic."
    ),
)
def test_all_surfaces_explicitly_bootstrap_extension_parent() -> None:
    """Desired interim state: each surface's source explicitly bootstraps parent.

    Not a claim of full child-script / argv / env / policy byte equality.
    """
    q = _source(qualify._qualify_confined_extension_load)
    c = _source(controller._verify_extension_load)
    v = _source(verifier._expected_extension_load_vector)
    m = _source(controller._runtime_maps_dependencies)

    match = re.search(
        r"script\s*=\s*\((?P<body>.*?)\)\s*\n\s*vector\s*=\s*\[python",
        c,
        flags=re.S,
    )
    assert match is not None
    child = match.group("body")
    assert _explicitly_bootstraps_extension_parent(q)
    assert _explicitly_bootstraps_extension_parent(m)
    assert _explicitly_bootstraps_extension_parent(child)
    assert _explicitly_bootstraps_extension_parent(v)
