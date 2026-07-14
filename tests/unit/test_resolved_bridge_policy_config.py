"""Pin the resolved-config fingerprints the H0 declaration freezes.

The producer earns the right to compute new fingerprints by reproducing one that
already existed: the `s` value H0 carried before its policy target was corrected
(`b1b78318…`). If that stops reproducing, the producer has drifted from whatever
computed the declared value, and every fingerprint it emits is suspect — so it is
pinned here rather than left as a one-off check in a commit message.

The declaration itself is pinned too: the fingerprint it prints must be the one
the producer derives from the preset it names. (The repo-wide scanner that holds
*every* declaration to this is a separate contract test; this file guards the
specific values H0 froze.)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "tools"))

from resolved_bridge_policy_config import (  # noqa: E402
    fingerprint,
    preset_sha256,
    resolve,
)

_H0 = (
    _REPO
    / "docs/modules/semantic/research"
    / "headline_bridge_full_decision_capture_declaration_20260713.md"
)

# `s` — superseded as H0's target, kept as the producer's calibration point.
S_PRESET = "mamba_whole_graph"
S_PRESET_SHA = "093b66ed124063f035ae9cf2a76e4f5426743cd819fb66e3e54994c97ea42cd1"
S_FINGERPRINT = "b1b78318ccbb87a701986f71c86147d83058e598ffd3b21e06f42d6116a51ae6"

# `m` — the preset the bridge-fidelity line is sealed on, and H0's target.
M_PRESET = "mamba_whole_graph_m"
M_PRESET_SHA = "496c4ec22b497c70bc8409227513939b4cd86834bf2210475d0ad655be6937af"
M_FINGERPRINT = "c7a6dbb35168cba75249b7f2c67d8455b6f634732493e455a4bb920aab6d7782"


def test_producer_reproduces_the_previously_declared_s_fingerprint() -> None:
    """The calibration point: reproduce a value the producer did not invent."""
    assert preset_sha256(S_PRESET) == S_PRESET_SHA
    assert fingerprint(S_PRESET) == S_FINGERPRINT


def test_m_fingerprint_is_stable() -> None:
    assert preset_sha256(M_PRESET) == M_PRESET_SHA
    assert fingerprint(M_PRESET) == M_FINGERPRINT


def test_h0_declaration_pins_the_m_fingerprint_it_resolves_to() -> None:
    text = _H0.read_text(encoding="utf-8")
    assert fingerprint(M_PRESET) in text
    assert preset_sha256(M_PRESET) in text


def test_s_and_m_differ_only_in_the_four_bridge_knobs() -> None:
    """The whole retarget rests on this: no fifth field moves silently."""
    s_policy, m_policy = resolve(S_PRESET), resolve(M_PRESET)
    moved = {key for key in s_policy if s_policy[key] != m_policy[key]}

    assert moved == {
        "relink_bridge_px",
        "relink_bridge_h_lo",
        "relink_bridge_h_hi",
        "relink_bridge_dir_bonus",
    }
