"""#549 S1 model-bundle contract: schemas, examples and the control matrix agree.

The schemas and the R-01..R-09 rules below are an accepted design target
(ADR 028), not implemented; no runtime reads them. This test keeps the
design machine-checkable: the example
manifest is the S0 inventory's N01-N06 bytes, the example allowlist can never
yield a trusted identity, and every schema/semantic row of the verification
matrix is executed against the examples.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
from typing import Any

import pytest
from jsonschema import Draft202012Validator

REPO = pathlib.Path(__file__).resolve().parents[2]
DESIGN = REPO / "docs" / "architecture" / "model_bundle_549"
MANIFEST_SCHEMA = DESIGN / "saccade.model_bundle.v1.schema.json"
ALLOWLIST_SCHEMA = DESIGN / "saccade.trusted_model_bundles.v1.schema.json"
MATRIX = DESIGN / "verification_matrix.json"
INVENTORY = REPO / "docs" / "reference" / "model_runtime_inventory_549.json"

GAPS = {f"G0{i}" for i in range(1, 8)}
ROLE_OF_PAIRING = {
    "backbone_engine": "backbone_engine",
    "head": "head_torchscript",
    "operator": "scan_operator",
    "config": "resolved_config",
    "lineage": "head_lineage",
    "attestation": "realization_attestation",
}
JSON_ROLES = {
    "resolved_config": "saccade.resolved_shipping_config/v1",
    "head_lineage": "saccade.head_artifact_lineage_torchscript/v1",
    "realization_attestation": "saccade.head_realization_attestation/v1",
}
# R-07 depends on owner decision D3 (ADR 028): weights and the binding metadata
# travel in the bundle, the operator only in the runtime package.
CARRIER_OF_ROLE = {
    "backbone_engine": "model_bundle",
    "head_torchscript": "model_bundle",
    "resolved_config": "model_bundle",
    "head_lineage": "model_bundle",
    "realization_attestation": "model_bundle",
    "scan_operator": "runtime_package",
}


def _load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def _validator(path: pathlib.Path) -> Draft202012Validator:
    schema = _load(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


MATRIX_DOC = _load(MATRIX)
TARGETS = {
    "manifest": (REPO / MATRIX_DOC["examples"]["manifest"], MANIFEST_SCHEMA),
    "allowlist": (REPO / MATRIX_DOC["examples"]["allowlist"], ALLOWLIST_SCHEMA),
}


def manifest_rule_errors(m: dict[str, Any]) -> list[str]:
    """R-01..R-08 of the contract (section 3.2): cross-field rules a schema cannot state."""
    errors: list[str] = []
    members = m["members"]
    by_id = {x["id"]: x for x in members}
    if len(by_id) != len(members):
        errors.append("R-01 duplicate member id")
    for carrier in ("model_bundle", "runtime_package"):
        folded = [x["path"].casefold() for x in members if x["carried_by"] == carrier]
        if len(set(folded)) != len(folded):
            errors.append(f"R-02 member paths collide under {carrier} (case-folded)")
    for slot, role in ROLE_OF_PAIRING.items():
        member = by_id.get(m["pairing"][slot])
        if member is None or member["role"] != role:
            errors.append(f"R-03 pairing.{slot} does not name a {role} member")
    roles = [x["role"] for x in members]
    if sorted(roles) != sorted(ROLE_OF_PAIRING.values()):
        errors.append("R-04 v1 needs each role exactly once")
    if m["runtime_contract"]["requires_operator"]["member"] != m["pairing"]["operator"]:
        errors.append("R-05 requires_operator.member is not pairing.operator")
    for x in members:
        if JSON_ROLES.get(x["role"]) != x.get("json_schema"):
            errors.append(f"R-06 member {x['id']} json_schema does not match its role")
        want = CARRIER_OF_ROLE[x["role"]]
        if x["carried_by"] != want:
            errors.append(f"R-07 {x['role']} must be carried by {want}")
    bb, head = m["io"]["backbone_engine"], m["io"]["head"]
    for name, tensors in (
        ("backbone inputs", bb["inputs"]),
        ("backbone outputs", bb["outputs"]),
        ("head inputs", head["inputs"]),
        ("head outputs", head["outputs"]),
    ):
        if [t["ordinal"] for t in tensors] != list(range(len(tensors))):
            errors.append(f"R-08 {name} ordinals are not 0..n-1")
    pairs = [(t["shape"], t["dtype"]) for t in head["inputs"]]
    if pairs != [(t["shape"], t["dtype"]) for t in bb["outputs"]]:
        errors.append("R-08 head inputs do not match backbone outputs (shape, dtype)")
    levels = len(head["inputs"])
    if len(head["outputs"]) != 2 * levels:
        errors.append("R-08 head must emit one cls and one reg tensor per level")
    else:
        post = m["postprocessing"]
        for k, t in enumerate(head["outputs"]):
            channels = post["num_classes"] if k < levels else post["box_channels"]
            level = head["inputs"][k % levels]["shape"]
            shape = [level[0], channels, *level[2:]]
            if t["shape"] != shape:
                errors.append(f"R-08 head output {t['name']} shape is not {shape}")
    resize = m["preprocessing"]["resize"]
    if bb["inputs"][0]["shape"][2:] != [resize["height"], resize["width"]]:
        errors.append("R-08 resize does not produce the backbone input H/W")
    return errors


def allowlist_rule_errors(a: dict[str, Any]) -> list[str]:
    """R-09: an allowlist names each manifest once."""
    shas = [e["manifest_sha256"] for e in a["entries"]]
    return ["R-09 duplicate manifest_sha256"] if len(set(shas)) != len(shas) else []


RULES = {"manifest": manifest_rule_errors, "allowlist": allowlist_rule_errors}


def resolve_level(manifest_sha256: str, allowlist: dict[str, Any]) -> str:
    """Reference identity resolution (contract section 4.1) after bytes already matched."""
    for e in allowlist["entries"]:
        if e["manifest_sha256"] == manifest_sha256 and e["state"] == "approved":
            return "expected_source_verified"
    return "checksum_matched"


def _apply(doc: Any, ops: list[dict[str, Any]]) -> Any:
    doc = copy.deepcopy(doc)

    def walk(pointer: str) -> tuple[Any, str]:
        parts = [
            p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]
        ]
        node = doc
        for p in parts[:-1]:
            node = node[int(p)] if isinstance(node, list) else node[p]
        return node, parts[-1]

    for op in ops:
        parent, key = walk(op["path"])
        if op["op"] == "copy":
            src_parent, src_key = walk(op["from"])
            src = (
                src_parent[int(src_key)]
                if isinstance(src_parent, list)
                else src_parent[src_key]
            )
            value = copy.deepcopy(src)
        else:
            value = op.get("value")
        if op["op"] == "remove":
            del parent[int(key) if isinstance(parent, list) else key]
        elif isinstance(parent, list):
            if key == "-":
                parent.append(value)
            elif op["op"] == "replace":
                parent[int(key)] = value
            else:
                parent.insert(int(key), value)
        else:
            parent[key] = value
    return doc


def test_schemas_and_examples_are_valid() -> None:
    for target, (example, schema) in TARGETS.items():
        doc = _load(example)
        assert not list(_validator(schema).iter_errors(doc)), target
        assert RULES[target](doc) == [], target


def test_example_manifest_is_the_s0_native_baseline() -> None:
    """The example repeats N01-N06 only as a manifest would; S0 stays the fact authority."""
    manifest = _load(TARGETS["manifest"][0])
    inventory = _load(INVENTORY)
    by_id = {a["id"]: a for a in inventory["artifacts"]}
    assert sorted(inventory["scope"]["native_required_ids"]) == sorted(
        m["inventory_id"] for m in manifest["members"]
    )
    for m in manifest["members"]:
        a = by_id[m["inventory_id"]]
        assert (m["path"], m["bytes"], m["sha256"]) == (
            a["path"],
            a["bytes"],
            a["sha256"],
        )
        assert m["rights_audit_item"] == a["licence"]["audit_item"]
    ckpts = {
        s["value"]
        for s in manifest["provenance"]["sources"]
        if s["kind"] == "checkpoint_sha256"
    }
    assert ckpts <= {a.get("sha256") for a in inventory["artifacts"]}


def test_example_allowlist_cannot_yield_a_trusted_identity() -> None:
    manifest_path, _ = TARGETS["manifest"]
    allowlist = _load(TARGETS["allowlist"][0])
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert [e["manifest_sha256"] for e in allowlist["entries"]] == [sha]
    assert all(e["state"] == "example" for e in allowlist["entries"])
    assert resolve_level(sha, allowlist) == "checksum_matched"
    approved = _apply(
        allowlist,
        [
            {"op": "replace", "path": "/entries/0/state", "value": "approved"},
            {
                "op": "replace",
                "path": "/entries/0/approval",
                "value": {
                    "decision_ref": "https://github.com/raylei50653/saccade/issues/549#issuecomment-1",
                    "date": "2026-10-10",
                },
            },
        ],
    )
    assert not list(_validator(ALLOWLIST_SCHEMA).iter_errors(approved))
    assert resolve_level(sha, approved) == "expected_source_verified"
    assert resolve_level("0" * 64, approved) == "checksum_matched"


def test_matrix_shape() -> None:
    rows = MATRIX_DOC["rows"]
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    for r in rows:
        assert set(r["gaps"]) <= GAPS, r["id"]
        executable = r["layer"] in ("schema", "semantic")
        assert (r["current"] == "design_check") == executable, r["id"]
        assert (r["target"] in TARGETS) == executable, r["id"]
        assert (r["expect_error"] is not None) == (
            executable and r["mutation"] is not None
        ), r["id"]
        if not executable:
            assert r["mutation"] is None and r["current"] in {
                "enforced",
                "partial",
                "missing",
            }, r["id"]
    for gap in sorted(GAPS):
        polarities = {r["polarity"] for r in rows if gap in r["gaps"]}
        assert polarities == {"positive", "negative"}, gap


def test_matrix_evidence_references_exist() -> None:
    for r in MATRIX_DOC["rows"]:
        for ref in r["existing_enforcement"]:
            m = re.fullmatch(r"(.+?)(?::(\d+)(?:-(\d+))?)?", ref)
            assert m is not None
            path = REPO / m[1]
            assert path.is_file(), f"{r['id']}: {ref}"
            if m[2]:
                lines = len(path.read_bytes().splitlines())
                assert 1 <= int(m[2]) <= int(m[3] or m[2]) <= lines, f"{r['id']}: {ref}"


@pytest.mark.parametrize(
    "row",
    [
        r
        for r in MATRIX_DOC["rows"]
        if r["layer"] in ("schema", "semantic") and r["mutation"]
    ],
    ids=lambda r: r["id"],
)
def test_matrix_negative_controls(row: dict[str, Any]) -> None:
    example, schema = TARGETS[row["target"]]
    assert row["polarity"] == "negative"
    doc = _apply(_load(example), row["mutation"])
    schema_errors = list(_validator(schema).iter_errors(doc))
    want = row["expect_error"]
    if row["layer"] == "schema":
        found = {
            ("/" + "/".join(str(k) for k in e.absolute_path), e.validator)
            for e in schema_errors
        }
        assert (want["instance_path"], want["keyword"]) in found, (
            f"{row['id']}: {found}"
        )
    else:
        assert not schema_errors, f"{row['id']} should pass the schema and fail a rule"
        rules = {e.split()[0] for e in RULES[row["target"]](doc)}
        assert rules == {want["rule"]}, f"{row['id']}: {rules}"
