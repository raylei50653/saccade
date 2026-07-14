"""Pin every frozen bridge policy in a declaration to the preset it claims.

A declaration that freezes a bridge policy prints a knob table and calls it "the
headline runtime path". Both halves rotted at once in 2026-07-13: `headline` is
overloaded in this repo (`status_2026-07-09.md` keeps an `s` primary and an `m`
capacity track), and P0 froze `s` while auditing `m`-sealed evidence — the knob
table was self-consistent, just transcribed from the wrong preset. Nothing
mechanical objected, and the error propagated into H0 before anyone read the two
documents side by side.

So a frozen-policy table is treated as a claim about a named preset, and checked:

  1. it must carry a `<!-- policy-target: headline | non-headline; … -->` marker
     (silence fails — assuming a target is the original error);
  2. `headline` must mean the preset `HEADLINE_PRESET_REL` names, byte-for-byte;
  3. every knob value it prints must equal that preset's *resolved* value;
  4. any `resolved_bridge_policy_config_v1` fingerprint it declares must be
     reproducible from that preset.

A declaration may still target a non-headline preset — it just has to say so.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_RESEARCH = _REPO / "docs" / "modules" / "semantic" / "research"

sys.path.insert(0, str(_REPO / "scripts" / "tools"))

from resolved_bridge_policy_config import (  # noqa: E402
    fingerprint,
    preset_sha256,
    resolve,
)

# The code's own answer to "which preset is the headline bridge path".
_HEADLINE_REL = re.search(
    r'HEADLINE_PRESET_REL:\s*Final\[str\]\s*=\s*"([^"]+)"',
    (
        _REPO
        / "src"
        / "saccade"
        / "perception"
        / "eval"
        / "consumer_a_bridge_fidelity.py"
    ).read_text(encoding="utf-8"),
).group(1)
HEADLINE_PRESET = Path(_HEADLINE_REL).stem

# A frozen-policy row is a two-cell markdown row whose first cell is nothing but
# backticked knob names. It may pair knobs off:
#   | `relink_bridge_px` | `0.4` |
#   | `relink_bridge_h_lo`, `relink_bridge_h_hi` | `0.6`, `1.7` |
_TABLE_ROW = re.compile(r"^\|([^|\n]+)\|([^|\n]+)\|\s*$", re.MULTILINE)
_TICKED = re.compile(r"`([^`]+)`")
_KNOB = re.compile(r"^(relink_bridge_\w+|relink_\w+|reid_mode)$")

_MARKER = re.compile(
    r"<!--\s*policy-target:\s*(headline|non-headline)\b([^>]*)-->", re.IGNORECASE
)
_PRESET_REF = re.compile(r"configs/presets/([\w.]+)\.yaml")
_FINGERPRINT = re.compile(r"`([0-9a-f]{64})`")

# Amendments and corrections are append-only history: they quote superseded
# values and audit comparisons on purpose, so only the declaration body states
# the policy in force.
_HISTORY = re.compile(r"^##\s+(Amendment|Correction)\b", re.MULTILINE)


def _body(text: str) -> str:
    first = _HISTORY.search(text)
    return text[: first.start()] if first else text


def _frozen_knobs(text: str) -> list[tuple[str, str]]:
    """Knob/value pairs from the frozen-policy table(s) of a declaration body."""
    pairs: list[tuple[str, str]] = []
    for lhs, rhs in _TABLE_ROW.findall(_body(text)):
        knobs = _TICKED.findall(lhs)
        values = _TICKED.findall(rhs)
        if not knobs or len(knobs) != len(values):
            continue
        if not all(_KNOB.match(knob) for knob in knobs):
            continue
        pairs.extend(zip(knobs, values))
    return pairs


def _freezes_a_policy(text: str) -> bool:
    return any(knob.startswith("relink_bridge_") for knob, _ in _frozen_knobs(text))


def _declarations() -> list[Path]:
    return sorted(
        path
        for path in _RESEARCH.rglob("*.md")
        if _freezes_a_policy(path.read_text(encoding="utf-8"))
    )


def _scalar(raw: str) -> Any:
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return float(raw)
    except ValueError:
        return raw


@pytest.fixture(params=_declarations(), ids=lambda p: p.stem)
def declaration(request) -> tuple[Path, str]:
    path = request.param
    return path, path.read_text(encoding="utf-8")


def test_a_frozen_policy_names_its_target_explicitly(declaration) -> None:
    path, text = declaration
    marker = _MARKER.search(text)
    assert marker is not None, (
        f"{path.relative_to(_REPO)} freezes a bridge policy but declares no "
        "policy target. Add `<!-- policy-target: headline -->`, or "
        "`<!-- policy-target: non-headline; preset: <stem>.yaml; reason: … -->`. "
        "Leaving it implicit is the P0 error."
    )


def test_headline_target_is_the_preset_the_code_calls_headline(declaration) -> None:
    path, text = declaration
    marker = _MARKER.search(text)
    if marker is None or marker.group(1).lower() != "headline":
        pytest.skip("not a headline-targeted declaration")

    presets = set(_PRESET_REF.findall(text))
    assert HEADLINE_PRESET in presets, (
        f"{path.relative_to(_REPO)} claims policy-target: headline, but never names "
        f"{HEADLINE_PRESET}.yaml — the preset HEADLINE_PRESET_REL points at."
    )
    assert preset_sha256(HEADLINE_PRESET) in text, (
        f"{path.relative_to(_REPO)} claims policy-target: headline but does not pin "
        f"{HEADLINE_PRESET}.yaml's current bytes "
        f"({preset_sha256(HEADLINE_PRESET)[:12]}…). Either the preset moved under the "
        "declaration or the declaration froze a different one."
    )


def test_printed_knobs_match_the_targets_resolved_policy(declaration) -> None:
    path, text = declaration
    marker = _MARKER.search(text)
    assert marker is not None

    if marker.group(1).lower() == "headline":
        preset = HEADLINE_PRESET
    else:
        named = re.search(r"preset:\s*([\w.]+)\.yaml", marker.group(2))
        assert named, (
            f"{path.relative_to(_REPO)}: a non-headline policy target must name its "
            "preset in the marker, e.g. "
            "`<!-- policy-target: non-headline; preset: mamba_whole_graph.yaml; reason: … -->`"
        )
        preset = named.group(1)

    resolved = resolve(preset)
    checked = 0
    for knob, printed in _frozen_knobs(text):
        if knob not in resolved:
            continue
        expected = resolved[knob]
        actual = _scalar(printed)
        checked += 1
        message = (
            f"{path.relative_to(_REPO)}: prints `{knob} = {printed}` but "
            f"{preset}.yaml resolves it to {expected!r}"
        )
        if isinstance(expected, bool) or isinstance(expected, str):
            assert actual == expected, message
        else:
            assert abs(float(actual) - float(expected)) <= 1e-9, message

    assert checked, f"{path.relative_to(_REPO)}: no knob row was actually compared"


def test_declared_config_fingerprint_is_reproducible(declaration) -> None:
    path, text = declaration
    if "resolved_bridge_policy_config_v1" not in text:
        pytest.skip("declaration pins no resolved-config fingerprint")

    marker = _MARKER.search(text)
    assert marker is not None
    if marker.group(1).lower() == "headline":
        preset = HEADLINE_PRESET
    else:
        named = re.search(r"preset:\s*([\w.]+)\.yaml", marker.group(2))
        assert named
        preset = named.group(1)

    expected = fingerprint(preset)
    assert expected in _FINGERPRINT.findall(_body(text)), (
        f"{path.relative_to(_REPO)} pins a resolved_bridge_policy_config_v1 that "
        f"{preset}.yaml no longer produces. Recompute with "
        f"`scripts/tools/resolved_bridge_policy_config.py --preset {preset}` "
        f"(current: {expected})."
    )
