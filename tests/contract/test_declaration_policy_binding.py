"""A declaration's policy target is declared in YAML, and checked against the code.

## Why this is not a Markdown check

P0 froze `configs/presets/mamba_whole_graph.yaml` (the `s` preset) and called it
"the current headline runtime path", then audited `m`-sealed evidence against it.
The knob table was internally consistent — just transcribed from the wrong preset.
Nothing mechanical objected, and the error propagated into H0.

The obvious fix — read the declaration's knob table and compare it to the preset —
was attempted, and it failed. Not on one bug, but on a class of them. Each round of
review found a new way for a document to escape a partial Markdown parser:

  * a table shape the parser did not recognise removed the document from scope
    entirely, instead of failing it;
  * a knob name the schema had never heard of was silently skipped;
  * a `## Correction` shown inside a fenced example truncated the body, so every
    row below it went unchecked;
  * a fence of four backticks — ordinary CommonMark — was not a fence to the
    regex, so the same trick worked again;
  * a marker missing its `-->` was invisible to the very check that claimed to
    catch a missing `-->`, leaving a superseded marker in charge;
  * an identity could be "pinned" in prose, or in a decoy row, or twice.

Every one of those was a real fail-open, and every fix revealed the next. The
lesson is not that the parser needed more regex. It is that **a hand-written
Markdown + HTML-comment + table parser was being trusted with fail-closed contract
authority**, and prose is the wrong substrate for that.

So the authority moves out of the prose entirely.

  * **code** is the authority on what a preset resolves to
    (`scripts/tools/resolved_bridge_policy_config.py`, `HEADLINE_PRESET_REL`);
  * **a sidecar YAML** is the authority on which preset a declaration binds;
  * **the Markdown** is human-readable history — tables, corrections, amendments —
    and nothing here parses it.

This file reads YAML, hashes a byte range, and calls the resolver. It contains no
notion of a heading, a fence, a table or a comment, and it must not acquire one.

## Why a prefix hash and not a file hash

A declaration is append-only: corrections and amendments accrete below a body that
must never change. A whole-file hash would break on every legitimate append. So the
binding pins the **immutable prefix** — a byte count and its SHA-256. The sealed
body cannot change by a single byte; history can still be appended below it; and
the check needs to understand nothing about the document's structure to say so.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_RESEARCH = _REPO / "docs" / "modules" / "semantic" / "research"

sys.path.insert(0, str(_REPO / "scripts" / "tools"))

from resolved_bridge_policy_config import (  # noqa: E402
    SCHEMA_ID,
    fingerprint,
    preset_sha256,
)

SCHEMA = "declaration_policy_binding_v1"

# The code's own answer to "which preset is the headline bridge path". Read from
# the source, so that moving it in code and not in the declarations fails here.
HEADLINE_PRESET_REL = next(
    line.split('"')[1]
    for line in (_REPO / "src/saccade/perception/eval/consumer_a_bridge_fidelity.py")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.startswith("HEADLINE_PRESET_REL")
)

# Declarations that bind no bridge policy, and so need no binding file. Listed
# rather than detected: detection would mean parsing Markdown, which is the thing
# this file exists to stop doing. A new declaration is therefore a deliberate
# decision — write a binding, or record here why it needs none.
NO_POLICY_BINDING = {
    "d0_runtime_shadow_fidelity_declaration_20260712": "capture fidelity; freezes no policy table",
    "discrete_m_capability_declaration_20260712": "capability study; freezes no policy table",
    "safe_domain_runtime_transfer_declaration_20260712": "safe-domain transfer; freezes no policy table",
    "frozen_packet_exact_key_recoverability_declaration_20260713": "key recoverability; freezes no policy table",
    "ambiguous_band_ranking_power_probe_declaration_20260712": "read-only probe; explicitly forbids preset change",
    # R1's *evidence* is captured under `m`, and P0's audit checks that against the
    # policy target. The declaration itself freezes no policy — its only mention of
    # a preset is the row declaring preset changes unauthorized.
    "r1_temporal_reduction_capture_declaration_20260712": "capture declaration; freezes no policy table",
}


def _bindings() -> list[Path]:
    return sorted(_RESEARCH.rglob("*.policy.yaml"))


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(params=_bindings(), ids=lambda p: p.name)
def binding(request) -> tuple[Path, dict[str, Any]]:
    path = request.param
    return path, _load(path)


# --------------------------------------------------------------------------- #
# Scope. Not detected from prose — declared, and its absence is a failure.       #
# --------------------------------------------------------------------------- #
def test_every_declaration_either_binds_a_policy_or_says_why_not() -> None:
    """A new declaration cannot quietly arrive unbound.

    The previous guard decided scope by reading Markdown, so a declaration whose
    tables it could not parse simply fell out of scope and was never checked. Scope
    is now a fact on disk: a declaration has a binding file, or it is named here.
    """
    declarations = {
        path.stem
        for path in _RESEARCH.rglob("*_declaration_*.md")
        if not path.name.endswith(".policy.yaml")
    }
    bound = {path.name[: -len(".policy.yaml")] for path in _bindings()}

    unaccounted = declarations - bound - set(NO_POLICY_BINDING)
    assert not unaccounted, (
        f"{sorted(unaccounted)} have no policy binding and are not listed in "
        "NO_POLICY_BINDING. Write a `<declaration>.policy.yaml`, or record there why "
        "the declaration freezes no bridge policy. Silence is what produced P0."
    )
    stale = set(NO_POLICY_BINDING) - declarations
    assert not stale, f"{sorted(stale)} no longer exist — remove the dead exemption."

    assert bound, "no declaration is bound: the guard would be checking nothing"


def test_a_binding_names_a_declaration_that_exists(binding) -> None:
    path, data = binding
    document = _RESEARCH / data["document"]["path"]
    assert document.is_file(), f"{path.name}: binds {data['document']['path']}, absent"
    assert document.name == path.name[: -len(".policy.yaml")] + ".md", (
        f"{path.name}: binds a document it is not named after"
    )


# --------------------------------------------------------------------------- #
# 1. The schema is complete and unambiguous.                                    #
# --------------------------------------------------------------------------- #
def test_the_binding_schema_is_complete(binding) -> None:
    path, data = binding
    assert data["schema"] == SCHEMA, f"{path.name}: unknown schema {data['schema']!r}"

    shape = {
        "document": {"path"},
        "sealed_prefix": {"bytes", "sha256"},
        "policy_target": {"kind", "preset"},
        "preset_identity": {"sha256", "resolved_schema", "resolved_fingerprint"},
    }
    assert set(data) == {"schema", *shape}, (
        f"{path.name}: unexpected or missing top-level keys: {sorted(set(data))}"
    )
    for section, keys in shape.items():
        assert set(data[section]) == keys, (
            f"{path.name}: `{section}` must have exactly {sorted(keys)}, "
            f"has {sorted(data[section])}"
        )
    assert data["policy_target"]["kind"] in ("headline", "non-headline"), (
        f"{path.name}: kind must be `headline` or `non-headline`"
    )
    assert data["preset_identity"]["resolved_schema"] == SCHEMA_ID, (
        f"{path.name}: resolved_schema must be {SCHEMA_ID}"
    )


# --------------------------------------------------------------------------- #
# 2. The declaration's immutable prefix has not moved.                          #
# --------------------------------------------------------------------------- #
def test_the_sealed_prefix_is_unchanged(binding) -> None:
    """One byte of the frozen body cannot change without this failing.

    History may still be appended below the prefix; that is the point of pinning a
    prefix rather than the file.
    """
    path, data = binding
    document = _RESEARCH / data["document"]["path"]
    raw = document.read_bytes()
    size = data["sealed_prefix"]["bytes"]

    assert len(raw) >= size, (
        f"{path.name}: the document is {len(raw)} bytes but the binding pins a "
        f"{size}-byte prefix — the declaration shrank, so its body was rewritten."
    )
    actual = hashlib.sha256(raw[:size]).hexdigest()
    assert actual == data["sealed_prefix"]["sha256"], (
        f"{path.name}: the first {size} bytes of {document.name} no longer hash to "
        f"the pinned prefix ({data['sealed_prefix']['sha256'][:12]}… vs {actual[:12]}…). "
        f"The frozen body was edited. Append a Correction below it instead — and if "
        f"the change is a legitimate pre-seal edit, re-pin this binding deliberately."
    )


# --------------------------------------------------------------------------- #
# 3-4. The preset exists, and its identity is reproducible from the code.        #
# --------------------------------------------------------------------------- #
def test_the_bound_preset_exists(binding) -> None:
    path, data = binding
    preset = _REPO / data["policy_target"]["preset"]
    assert preset.is_file(), f"{path.name}: no such preset: {preset}"


def test_the_preset_identity_is_reproducible(binding) -> None:
    """Recomputed from the preset, never trusted from the page."""
    path, data = binding
    stem = Path(data["policy_target"]["preset"]).stem
    identity = data["preset_identity"]

    assert identity["sha256"] == preset_sha256(stem), (
        f"{path.name}: pins {identity['sha256'][:12]}… as {stem}.yaml's bytes, but it "
        f"hashes to {preset_sha256(stem)[:12]}…. The preset moved under the binding."
    )
    assert identity["resolved_fingerprint"] == fingerprint(stem), (
        f"{path.name}: pins {identity['resolved_fingerprint'][:12]}… as {SCHEMA_ID}, "
        f"but {stem}.yaml resolves to {fingerprint(stem)[:12]}…. Recompute with "
        f"`scripts/tools/resolved_bridge_policy_config.py --preset {stem}`."
    )


# --------------------------------------------------------------------------- #
# 5. `headline` means what the code means by it.                                #
# --------------------------------------------------------------------------- #
def test_headline_means_the_preset_the_code_calls_headline(binding) -> None:
    """The whole incident, in one assertion.

    `headline` is overloaded in this repo — `status_2026-07-09.md` keeps an `s`
    primary and an `m` capacity track. A declaration may bind either, but it may not
    call `s` the headline: `headline` means whatever `HEADLINE_PRESET_REL` names, and
    nothing else gets to redefine it.
    """
    path, data = binding
    target = data["policy_target"]
    if target["kind"] != "headline":
        pytest.skip("binds a non-headline preset, explicitly")

    assert target["preset"] == HEADLINE_PRESET_REL, (
        f"{path.name}: claims `kind: headline` but binds {target['preset']}, while "
        f"HEADLINE_PRESET_REL names {HEADLINE_PRESET_REL}. This is exactly the P0 "
        f"error: two meanings of `headline`, and the declaration chose the wrong one."
    )


def test_a_non_headline_binding_names_its_preset_explicitly(binding) -> None:
    path, data = binding
    target = data["policy_target"]
    if target["kind"] == "headline":
        pytest.skip("headline target; the preset is the code's to name")
    assert target["preset"], f"{path.name}: non-headline binding names no preset"
