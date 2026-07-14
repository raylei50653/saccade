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

## Scope is itself load-bearing

All four checks are parametrized over the documents the frozen-policy regex finds,
so *scope decides whether anything is checked at all*. An empty parameter set does
not fail in pytest — it skips, and the suite goes green. Scope is therefore pinned
(`MUST_BE_IN_SCOPE`) and its detection kept independent of table shape, because a
guard that silently checks nothing is this file's own thesis turned against it.

## Which text is "in force"

These documents are append-only: a sealed body is never rewritten, and amendments
and corrections accrete below it. So "what does this declaration freeze *now*" has
to be defined, not guessed:

  * **the effective marker is the LAST one in document order.** Corrections may
    retarget a declaration, and when they do, the later marker supersedes the
    earlier. (Taking the first match would let a superseded target win — and
    would also mean a document could be rescued by a marker it has since
    corrected away.)
  * **every value — preset name, byte hash, knob table, fingerprint — is read
    from the BODY only**, i.e. above the first amendment/correction heading. The
    body is the statement of the policy in force; amendments quote superseded
    values and audit comparisons on purpose, and must never be able to satisfy a
    check.

The two rules interlock: a marker anywhere names the target, but the body's values
are then checked against *that* target. So a correction cannot silently retarget a
declaration without the body being brought along (H0's Amendment 5 does exactly
this), and a historical marker cannot vouch for a body it no longer describes.

This also lets a sealed body keep its marker in a correction — P0's does, because
its § 1 is frozen and must not be edited to add one.
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

# A frozen-policy row is a markdown row whose first cell is nothing but backticked
# knob names and whose second cell holds their values. It may pair knobs off:
#   | `relink_bridge_px` | `0.4` |
#   | `relink_bridge_h_lo`, `relink_bridge_h_hi` | `0.6`, `1.7` |
#
# Trailing cells are allowed and ignored: a knob table with a `rationale` or `note`
# column is an ordinary way to write one, and requiring exactly two cells would let
# such a declaration fall out of scope *entirely* — unguarded, and silently, since
# scope is what decides whether any of these checks run at all.
_TABLE_ROW = re.compile(r"^\|([^|\n]+)\|([^|\n]+)\|", re.MULTILINE)
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
    """The declaration proper — everything above the first amendment/correction."""
    first = _HISTORY.search(text)
    return text[: first.start()] if first else text


def _effective_marker(text: str) -> re.Match[str] | None:
    """The last marker in document order: a later correction supersedes an earlier one."""
    markers = list(_MARKER.finditer(text))
    return markers[-1] if markers else None


def _effective_preset(marker: re.Match[str], path: Path) -> str:
    if marker.group(1).lower() == "headline":
        return HEADLINE_PRESET
    named = re.search(r"preset:\s*([\w.]+)\.yaml", marker.group(2))
    assert named, (
        f"{path}: a non-headline policy target must name its preset in the marker, "
        "e.g. `<!-- policy-target: non-headline; preset: mamba_whole_graph.yaml; "
        "reason: … -->`"
    )
    return named.group(1)


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


# --------------------------------------------------------------------------- #
# Guard the guard's *scope*.                                                    #
#                                                                               #
# Every check below is parametrized over `_declarations()`, and `_declarations()`
# is a regex over table shape. So scope decides whether anything is checked at    #
# all — and when a parametrized fixture comes back empty, pytest does not fail:   #
# it collects nothing, reports a skip, and exits 0. A guard that quietly checks   #
# zero documents is the exact failure it was written to prevent, one level up.    #
# --------------------------------------------------------------------------- #

# Declarations known to freeze a bridge policy. Pinned by name so that a regex
# which stops matching them fails loudly here instead of emptying the suite.
MUST_BE_IN_SCOPE = {
    "headline_bridge_full_decision_capture_declaration_20260713",
    "runtime_bridge_decision_path_identifiability_declaration_20260713",
}


def test_the_guard_actually_has_documents_to_check() -> None:
    in_scope = {path.stem for path in _declarations()}
    assert in_scope, (
        "no document is in scope: the frozen-policy regex matches nothing, so every "
        "check in this file would silently pass. The guard is not guarding."
    )
    missing = MUST_BE_IN_SCOPE - in_scope
    assert not missing, (
        f"{sorted(missing)} freeze a bridge policy but are no longer detected as "
        "doing so. Either the declaration changed shape or the regex rotted — "
        "either way these documents are now unguarded."
    )


def test_scope_detection_is_not_coupled_to_a_two_column_table() -> None:
    """A knob table with a notes column is still a frozen policy.

    Scope keyed on exactly-two-cell rows, so this table — an entirely ordinary way
    to write one — fell out of scope and took all four checks with it.
    """
    with_notes = (
        "| knob | value | rationale |\n"
        "|---|---|---|\n"
        "| `relink_bridge_px` | `0.4` | height-gated |\n"
    )
    assert _freezes_a_policy(with_notes)
    assert _frozen_knobs(with_notes) == [("relink_bridge_px", "0.4")]


def test_a_frozen_policy_names_its_target_explicitly(declaration) -> None:
    path, text = declaration
    assert _effective_marker(text) is not None, (
        f"{path.relative_to(_REPO)} freezes a bridge policy but declares no "
        "policy target. Add `<!-- policy-target: headline -->`, or "
        "`<!-- policy-target: non-headline; preset: <stem>.yaml; reason: … -->`. "
        "Leaving it implicit is the P0 error."
    )


def test_headline_target_is_the_preset_the_code_calls_headline(declaration) -> None:
    path, text = declaration
    marker = _effective_marker(text)
    if marker is None or marker.group(1).lower() != "headline":
        pytest.skip("not a headline-targeted declaration")

    body = _body(text)
    assert HEADLINE_PRESET in set(_PRESET_REF.findall(body)), (
        f"{path.relative_to(_REPO)} claims policy-target: headline, but its body never "
        f"names {HEADLINE_PRESET}.yaml — the preset HEADLINE_PRESET_REL points at."
    )
    assert preset_sha256(HEADLINE_PRESET) in body, (
        f"{path.relative_to(_REPO)} claims policy-target: headline but its body does not "
        f"pin {HEADLINE_PRESET}.yaml's current bytes "
        f"({preset_sha256(HEADLINE_PRESET)[:12]}…). Either the preset moved under the "
        "declaration or the declaration froze a different one."
    )


def test_printed_knobs_match_the_targets_resolved_policy(declaration) -> None:
    path, text = declaration
    marker = _effective_marker(text)
    assert marker is not None
    preset = _effective_preset(marker, path.relative_to(_REPO))

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
    if "resolved_bridge_policy_config_v1" not in _body(text):
        pytest.skip("declaration body pins no resolved-config fingerprint")

    marker = _effective_marker(text)
    assert marker is not None
    preset = _effective_preset(marker, path.relative_to(_REPO))

    expected = fingerprint(preset)
    assert expected in _FINGERPRINT.findall(_body(text)), (
        f"{path.relative_to(_REPO)} pins a resolved_bridge_policy_config_v1 that "
        f"{preset}.yaml no longer produces. Recompute with "
        f"`scripts/tools/resolved_bridge_policy_config.py --preset {preset}` "
        f"(current: {expected})."
    )


# --------------------------------------------------------------------------- #
# The guard's own failure modes. A guard that only ever passes proves nothing,  #
# and the append-only model above is subtle enough that it needs pinning.       #
# --------------------------------------------------------------------------- #
_H0 = (
    _REPO
    / "docs/modules/semantic/research"
    / "headline_bridge_full_decision_capture_declaration_20260713.md"
)
_BODY_MARKER = "<!-- policy-target: headline -->\n\n"


def _knobs_ok(text: str) -> bool:
    marker = _effective_marker(text)
    if marker is None:
        return False
    resolved = resolve(_effective_preset(marker, Path("<mutant>")))
    return all(
        abs(float(_scalar(printed)) - float(resolved[knob])) <= 1e-9
        for knob, printed in _frozen_knobs(text)
        if knob in resolved and not isinstance(resolved[knob], str)
    )


def _retarget_body_to_s(text: str) -> str:
    return text.replace(
        "| `relink_bridge_px` | `0.4` |", "| `relink_bridge_px` | `0.25` |", 1
    )


def test_guard_rejects_the_original_bug() -> None:
    """`s` values under a headline claim — exactly what P0 did."""
    assert not _knobs_ok(_retarget_body_to_s(_H0.read_text(encoding="utf-8")))


def test_guard_rejects_a_correction_that_retargets_without_updating_the_body() -> None:
    mutant = _H0.read_text(encoding="utf-8") + (
        "\n\n## Correction 9\n\n<!-- policy-target: non-headline; "
        "preset: mamba_whole_graph.yaml; reason: mutation -->\n"
    )
    # The later marker wins, so the body's `m` values must now fail against `s`.
    assert (
        _effective_preset(_effective_marker(mutant), Path("<mutant>"))
        == "mamba_whole_graph"
    )
    assert not _knobs_ok(mutant)


def test_a_historical_marker_cannot_vouch_for_a_body_that_moved_under_it() -> None:
    """A marker only names the target; the body still has to match it."""
    mutant = (
        _retarget_body_to_s(
            _H0.read_text(encoding="utf-8").replace(_BODY_MARKER, "", 1)
        )
        + "\n\n## Correction 9\n\n<!-- policy-target: headline -->\n"
    )
    assert _effective_marker(mutant) is not None  # the marker survives...
    assert not _knobs_ok(mutant)  # ...and rescues nothing
