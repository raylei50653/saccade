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

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import re
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
    preset_path,
    preset_sha256,
)

SCHEMA = "declaration_policy_binding_v1"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class _StrictLoader(yaml.SafeLoader):
    """A loader that refuses duplicate keys, at every depth.

    `yaml.safe_load` takes the *last* of a repeated key and says nothing. So a
    binding could carry two `preset:` lines, and the exact-key-set check below —
    which only ever sees the merged mapping — would find one key and pass. The
    binding would then declare two policy targets and pin neither, which is the
    ambiguity this whole line exists to remove, moved from Markdown into YAML.
    """


def _no_duplicate_keys(loader: yaml.Loader, node: yaml.MappingNode) -> dict[str, Any]:
    seen: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r} — a binding that says a thing twice says it "
                "once, silently, and you do not get to see which",
                key_node.start_mark,
            )
        seen[key] = loader.construct_object(value_node, deep=True)
    return seen


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)

# The code's own answer to "which preset is the headline bridge path". Read from
# the source, so that moving it in code and not in the declarations fails here.
HEADLINE_PRESET_REL = next(
    line.split('"')[1]
    for line in (_REPO / "src/saccade/perception/eval/consumer_a_bridge_fidelity.py")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.startswith("HEADLINE_PRESET_REL")
)

# Declarations that bind no bridge policy, and so need no binding file.
#
# Listed rather than detected: detection would mean parsing Markdown, which is the
# thing this file exists to stop doing. A new declaration is therefore a deliberate
# decision — write a binding beside it, or record here why it needs none.
#
# Keyed by path relative to research/, exactly as scope is booked. A stem is not an
# identity: an exemption keyed by stem would excuse a same-named declaration in any
# directory, which is the decoy substitution this contract now forbids.
NO_POLICY_BINDING = {
    "d0_runtime_shadow_fidelity_declaration_20260712.md": "capture fidelity; freezes no policy table",
    "discrete_m_capability_declaration_20260712.md": "capability study; freezes no policy table",
    "safe_domain_runtime_transfer_declaration_20260712.md": "safe-domain transfer; freezes no policy table",
    "frozen_packet_exact_key_recoverability_declaration_20260713.md": "key recoverability; freezes no policy table",
    "ambiguous_band_ranking_power_probe_declaration_20260712.md": "read-only probe; explicitly forbids preset change",
    # R1's *evidence* is captured under `m`, and P0's audit checks that against the
    # policy target. The declaration itself freezes no policy — its only mention of
    # a preset is the row declaring preset changes unauthorized.
    "r1_temporal_reduction_capture_declaration_20260712.md": "capture declaration; freezes no policy table",
    # GCTM D1 is substrate-agnostic diagnostic-only; synthetic fixtures and a
    # future-runtime consumer interface. It freezes no bridge preset / production
    # policy table and cannot satisfy runtime activation authority.
    "gctm_d1_ranking_diagnostic_declaration_20260723.md": (
        "GCTM D1 diagnostic declaration; freezes no bridge policy table"
    ),
}


def _bindings() -> list[Path]:
    return sorted(_RESEARCH.rglob("*.policy.yaml"))


def load_binding(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)


@pytest.fixture(params=_bindings(), ids=lambda p: p.name)
def binding(request) -> tuple[Path, dict[str, Any]]:
    path = request.param
    return path, load_binding(path)


# --------------------------------------------------------------------------- #
# Scope. Not detected from prose — declared, and its absence is a failure.       #
# --------------------------------------------------------------------------- #
def test_every_declaration_either_binds_a_policy_or_says_why_not() -> None:
    """A new declaration cannot quietly arrive unbound.

    The previous guard decided scope by reading Markdown, so a declaration whose
    tables it could not parse simply fell out of scope and was never checked. Scope
    is now a fact on disk: a declaration has a binding file, or it is named here.
    """
    # Booked by path relative to research/, never by stem. Two declarations of the
    # same name in different directories are two declarations; keyed by stem they
    # would cancel each other out, and one of them would be silently credited with
    # the other's binding — or the other's exemption.
    declarations = {
        str(path.relative_to(_RESEARCH))
        for path in _RESEARCH.rglob("*_declaration_*.md")
    }
    bound = {str(_bound_document(path).relative_to(_RESEARCH)) for path in _bindings()}

    unaccounted = declarations - bound - set(NO_POLICY_BINDING)
    assert not unaccounted, (
        f"{sorted(unaccounted)} have no policy binding and are not listed in "
        "NO_POLICY_BINDING. Write a `<declaration>.policy.yaml` beside it, or record "
        "there why it freezes no bridge policy. Silence is what produced P0."
    )
    stale = set(NO_POLICY_BINDING) - declarations
    assert not stale, f"{sorted(stale)} no longer exist — remove the dead exemption."

    assert bound, "no declaration is bound: the guard would be checking nothing"


def _bound_document(binding_path: Path) -> Path:
    """The one document a binding may bind: its own name, in its own directory.

    Identity by *basename* is not identity. A binding could name
    `closed/<same-name>.md`, and if a copy lived there the basename check passed,
    scope still believed the real declaration was bound, and the prefix hash pinned
    the copy — leaving the real sealed body free to change with nothing objecting.
    Verified: with a decoy in place, H0's frozen `relink_bridge_px` could be edited
    from `0.4` to `0.25` — the original P0 bug — and every test still passed.

    This is the same substitution as the preset-path one, on the document side. A
    binding's location decides what it binds; the `path` field must agree, not
    choose.
    """
    return binding_path.with_name(
        binding_path.name[: -len(".policy.yaml")] + ".md"
    ).resolve()


def test_a_binding_binds_exactly_the_declaration_it_sits_beside(binding) -> None:
    path, data = binding
    declared = (_RESEARCH / data["document"]["path"]).resolve()
    expected = _bound_document(path)

    assert declared.is_file(), f"{path.name}: binds {data['document']['path']}, absent"
    assert declared == expected, (
        f"{path.name}: binds {declared}, but a binding may only bind the declaration "
        f"it sits beside ({expected}). A same-named document in another directory is "
        f"a different document — pinning it leaves the real one unguarded."
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

    # `bytes` is fed straight to a slice, and a slice forgives everything. `0` pins
    # nothing at all — and `sha256(b"")` is a real, checkable hash, so the binding
    # would still pass while pinning an empty prefix. A negative value pins all but
    # the tail. `true` is an `int` in Python and slices to `[:1]`. None of that is a
    # pin; all of it type-checks.
    size = data["sealed_prefix"]["bytes"]
    assert type(size) is int, (
        f"{path.name}: sealed_prefix.bytes must be an integer, not {type(size).__name__} "
        f"({size!r}). `true` is an int in Python and would slice to one byte."
    )
    assert size > 0, (
        f"{path.name}: sealed_prefix.bytes is {size}. A prefix of zero pins nothing — "
        f"and sha256(b'') is a perfectly valid hash, so the check would pass while "
        f"leaving the entire sealed body free to change."
    )
    assert _HEX64.fullmatch(data["sealed_prefix"]["sha256"]), (
        f"{path.name}: sealed_prefix.sha256 is not a 64-character hex digest"
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
def test_the_declared_preset_is_the_one_the_resolver_reads(binding) -> None:
    """The path the binding names must be the file the identity is computed from.

    The existence check took the binding's full path, but the identity was computed
    from its `.stem` — and the resolver then reads `configs/presets/<stem>.yaml`,
    whatever path the binding actually named. So a binding could point at some other
    file that happens to share a stem, pass the existence check, and have its
    identity silently computed from the canonical preset instead. Two files, one
    identity: the same substitution P0 made, in a different currency.
    """
    path, data = binding
    declared = (_REPO / data["policy_target"]["preset"]).resolve()
    assert declared.is_file(), f"{path.name}: no such preset: {declared}"

    stem = Path(data["policy_target"]["preset"]).stem
    canonical = preset_path(stem).resolve()
    assert declared == canonical, (
        f"{path.name}: binds {declared}, but the resolver computes identity from "
        f"{canonical}. The binding must name the file its identity comes from."
    )


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


# --------------------------------------------------------------------------- #
# The validator's own failure modes. Each of these passed before.               #
#                                                                               #
# The schema moved the authority out of Markdown, but a schema is only as strict #
# as its validator: a field fed straight to a slice, or a loader that quietly    #
# merges a repeated key, reintroduces exactly the ambiguity the move was for.    #
# --------------------------------------------------------------------------- #
_H0_BINDING = (
    _RESEARCH / "headline_bridge_full_decision_capture_declaration_20260713.policy.yaml"
)


def _rebound(tmp_path: Path, text: str) -> Path:
    path = tmp_path / _H0_BINDING.name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_real_bindings_load_and_validate(binding) -> None:
    """Calibration: the mutations below must be rejected *from* a passing base."""
    path, data = binding
    assert data["schema"] == SCHEMA and data["sealed_prefix"]["bytes"] > 0


@pytest.mark.parametrize(
    "literal,why",
    [
        ("0", "a zero-length prefix pins nothing — and sha256(b'') is a valid hash"),
        ("-1", "a negative prefix pins everything but the tail"),
        ("true", "`true` is an int in Python and slices to one byte"),
        ('"3531"', "a quoted number is a string, and a string is not a byte count"),
    ],
)
def test_a_sealed_prefix_that_pins_nothing_is_rejected(
    tmp_path: Path, literal: str, why: str
) -> None:
    original = load_binding(_H0_BINDING)["sealed_prefix"]["bytes"]
    text = _H0_BINDING.read_text(encoding="utf-8").replace(
        f"bytes: {original}", f"bytes: {literal}", 1
    )
    size = load_binding(_rebound(tmp_path, text))["sealed_prefix"]["bytes"]
    assert not (type(size) is int and size > 0), why


def test_an_empty_prefix_with_the_empty_hash_would_otherwise_pass() -> None:
    """The mutation is not hypothetical: `bytes: 0` has a genuine, matching hash."""
    empty = hashlib.sha256(b"").hexdigest()
    document = (_RESEARCH / load_binding(_H0_BINDING)["document"]["path"]).read_bytes()
    assert hashlib.sha256(document[:0]).hexdigest() == empty
    assert len(document) >= 0  # the old range check was satisfied by any document


def test_a_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    """`yaml.safe_load` takes the last of a repeated key and says nothing.

    So a binding could name two presets, and the exact-key-set check — which only
    sees the merged mapping — would find one `preset` and pass.
    """
    text = _H0_BINDING.read_text(encoding="utf-8").replace(
        "  preset: configs/presets/mamba_whole_graph_m.yaml",
        "  preset: configs/presets/mamba_whole_graph_m.yaml\n"
        "  preset: configs/presets/mamba_whole_graph.yaml",
        1,
    )
    mutant = _rebound(tmp_path, text)

    # The permissive loader accepts it, and silently keeps the *second* preset.
    lax = yaml.safe_load(mutant.read_text(encoding="utf-8"))
    assert lax["policy_target"]["preset"].endswith("mamba_whole_graph.yaml")
    assert set(lax["policy_target"]) == {"kind", "preset"}  # the key check sees one

    with pytest.raises(yaml.constructor.ConstructorError):
        load_binding(mutant)


def test_a_non_canonical_preset_path_is_rejected(tmp_path: Path) -> None:
    """A file that shares a stem is not the file the resolver reads."""
    decoy_dir = tmp_path / "elsewhere"
    decoy_dir.mkdir()
    stem = Path(load_binding(_H0_BINDING)["policy_target"]["preset"]).stem
    decoy = decoy_dir / f"{stem}.yaml"
    decoy.write_text("relink_bridge_px: 0.25\n", encoding="utf-8")

    declared = decoy.resolve()
    canonical = preset_path(stem).resolve()
    assert declared.is_file()  # the existence check passes...
    assert declared != canonical  # ...but it is not what the identity comes from


def test_a_same_named_decoy_in_another_directory_cannot_be_bound(
    tmp_path: Path,
) -> None:
    """Identity by basename is not identity — the document-side substitution.

    A binding could name `closed/<same-name>.md`. If a copy lived there, the
    basename check passed, scope still believed the real declaration was bound, and
    the prefix hash pinned the *copy*. Verified before the fix: with a decoy in
    place, H0's frozen `relink_bridge_px` could be edited from `0.4` to `0.25` — the
    original P0 bug — and every test still passed.

    Same shape as the preset-path substitution, on the other side of the binding.
    """
    real = _bound_document(_H0_BINDING)
    decoy_dir = tmp_path / "closed"
    decoy_dir.mkdir()
    decoy = decoy_dir / real.name
    decoy.write_bytes(real.read_bytes())  # byte-identical: the prefix hash matches

    data = load_binding(_H0_BINDING)
    size, pinned = data["sealed_prefix"]["bytes"], data["sealed_prefix"]["sha256"]
    assert hashlib.sha256(decoy.read_bytes()[:size]).hexdigest() == pinned, (
        "the decoy satisfies the prefix hash — which is exactly why the *identity* "
        "of the document, not just its contents, has to be pinned"
    )
    assert decoy.name == real.name  # ...and the basename check would have passed

    # But a binding may only bind the declaration it sits beside.
    assert decoy.resolve() != _bound_document(_H0_BINDING)


def test_a_decoy_declaration_cannot_inherit_the_real_ones_binding() -> None:
    """Scope booked by stem would credit a `closed/` copy with the real one's binding.

    Placed in the real tree, so this exercises the production bookkeeping rather
    than a restatement of it.
    """
    real = _bound_document(_H0_BINDING)
    decoy = _RESEARCH / "closed" / real.name
    assert not decoy.exists(), "the decoy name is already taken; pick another"

    decoy.write_bytes(real.read_bytes())
    try:
        declarations = {
            str(path.relative_to(_RESEARCH))
            for path in _RESEARCH.rglob("*_declaration_*.md")
        }
        bound = {
            str(_bound_document(path).relative_to(_RESEARCH)) for path in _bindings()
        }
        key = str(decoy.relative_to(_RESEARCH))

        assert key in declarations, "the decoy is a declaration in its own right..."
        assert key not in bound, "...and it does not inherit the real one's binding"
        assert declarations - bound - set(NO_POLICY_BINDING) == {key}, (
            "the decoy must show up as unaccounted, forcing a decision — keyed by "
            "stem it would have been silently absorbed by the real declaration"
        )
    finally:
        decoy.unlink()
