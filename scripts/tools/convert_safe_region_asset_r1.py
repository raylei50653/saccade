#!/usr/bin/env python3
"""R1: Deterministic RegionAsset conversion from sealed Q4.5 + T0 evidence.

Research-only packaging. Never mutates evaluator inputs, never reruns the
evaluator, never searches thresholds, and never self-promotes A0→A1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_VERSION = "region_asset_contract_v0"
ID_SCHEME = "region_asset_id_v2"
PRODUCER_KIND = "grammar_atlas"
PRODUCER_CONTRACT_VERSION = "region_asset_v0"
SCHEMA_VERSION = "region_asset_tables_v0"
MEMBERSHIP_DIGEST_ALGORITHM = "universe_membership_digest_v1"
PACK_STUDY_ID = "m_b1_5_safe_region_asset_r1_20260710"

EXPECTED = {
    "n_ps_total": 154,
    "n_ps_g1": 1,
    "n_ps_and": 153,
    "n_ps_or": 0,
    "n_components": 26,
    "n_g1_regions": 1,
    "n_g2_isolated_l0": 6,
    "n_g2_multi_l1": 19,
    "n_g3_null": 1,
    "n_mask_units": 34,
    "n_gt_exposed": 64,
    "n_fp_exposed": 23,
    "n_atom_rows": 870,
    "n_and_rows": 17640,
    "n_or_rows": 17640,
    "n_single_atoms": 870,
    "n_pairwise_atoms": 210,
}

SIGNALS_PRIMARY = [
    "score_m_bridge",
    "abs_log_h",
    "dist_h",
    "abs_ratio_m1",
    "resid_mean",
]
DIRECTIONS = ["high_tail", "low_tail"]
COMBINATORS = ["AND", "OR"]
G1_THR = list(range(87))
G2_THR = list(range(21))

ROLE = "untyped_observation"


def fail(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_id(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def thr_value_repr(value: float) -> str:
    """Shortest-ish round-trip decimal representation."""
    x = float(value)
    # Prefer JSON number form; fall back to .17g if needed for recovery.
    s = json.dumps(x)
    if float(s) == x or abs(float(s) - x) == 0.0:
        return s
    s2 = format(x, ".17g")
    if float(s2) != x:
        fail(f"thr_value_repr failed round-trip for {x!r}")
    return s2


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, (dict, list)):
                    out[k] = canonical_json(v)
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def parse_seq_json(raw: Any) -> dict[str, int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        obj = json.loads(raw)
        return {str(k): int(v) for k, v in obj.items()}
    fail(f"unsupported per_sequence json type: {type(raw)}")


def g1_grid_id(feature: str, direction: str) -> str:
    return f"S::{feature}::{direction}"


def g2_grid_id(
    feature_a: str, direction_a: str, feature_b: str, direction_b: str
) -> str:
    return f"P::{feature_a}::{direction_a}__{feature_b}::{direction_b}"


def pairwise_axis_pairs() -> list[tuple[str, str, str, str]]:
    """Return the sealed 40 pairwise grids: C(5,2) feature pairs × 4 direction combos.

    Leaves are stored in lexicographic (feature, direction) order after RB4.
    """
    pairs: list[tuple[str, str, str, str]] = []
    feats = list(SIGNALS_PRIMARY)
    for i, f0 in enumerate(feats):
        for f1 in feats[i + 1 :]:
            for d0 in DIRECTIONS:
                for d1 in DIRECTIONS:
                    a, b = (f0, d0), (f1, d1)
                    if a > b:
                        a, b = b, a
                    pairs.append((a[0], a[1], b[0], b[1]))
    out = sorted(set(pairs))
    if len(out) != 40:
        fail(f"pairwise axis pair count {len(out)} != 40")
    return out


def canonicalize_pairwise_leaf(
    feature_a: str,
    direction_a: str,
    thr_index_a: int,
    atom_a_id: str,
    feature_b: str,
    direction_b: str,
    thr_index_b: int,
    atom_b_id: str,
    role_a: str = ROLE,
    role_b: str = ROLE,
) -> dict[str, Any]:
    leaf_a = {
        "atom_id": atom_a_id,
        "feature": feature_a,
        "direction": direction_a,
        "thr_index": int(thr_index_a),
        "role": role_a,
    }
    leaf_b = {
        "atom_id": atom_b_id,
        "feature": feature_b,
        "direction": direction_b,
        "thr_index": int(thr_index_b),
        "role": role_b,
    }
    key_a = (
        leaf_a["feature"],
        leaf_a["direction"],
        leaf_a["atom_id"],
        leaf_a["thr_index"],
        leaf_a["role"],
    )
    key_b = (
        leaf_b["feature"],
        leaf_b["direction"],
        leaf_b["atom_id"],
        leaf_b["thr_index"],
        leaf_b["role"],
    )
    if (leaf_a["feature"], leaf_a["direction"]) > (
        leaf_b["feature"],
        leaf_b["direction"],
    ):
        leaf_a, leaf_b = leaf_b, leaf_a
    elif (leaf_a["feature"], leaf_a["direction"]) == (
        leaf_b["feature"],
        leaf_b["direction"],
    ) and key_a > key_b:
        leaf_a, leaf_b = leaf_b, leaf_a
    return {
        "atom_0_id": leaf_a["atom_id"],
        "feature_0": leaf_a["feature"],
        "direction_0": leaf_a["direction"],
        "thr_index_0": leaf_a["thr_index"],
        "role_0": leaf_a["role"],
        "atom_1_id": leaf_b["atom_id"],
        "feature_1": leaf_b["feature"],
        "direction_1": leaf_b["direction"],
        "thr_index_1": leaf_b["thr_index"],
        "role_1": leaf_b["role"],
        "grid_id": g2_grid_id(
            leaf_a["feature"],
            leaf_a["direction"],
            leaf_b["feature"],
            leaf_b["direction"],
        ),
    }


def canonical_cell_key_g1(feature: str, direction: str, thr_index: int) -> str:
    return f"S::{feature}::{direction}::u{int(thr_index)}"


def canonical_cell_key_pairwise(combinator: str, atom_0_id: str, atom_1_id: str) -> str:
    return f"{combinator}::{atom_0_id}::{atom_1_id}"


def as_int_flag(v: Any) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)) and not pd.isna(v):
        return int(v)
    if isinstance(v, str) and v.strip() != "":
        return int(float(v)) if "." in v else int(v)
    return 0


@dataclass(frozen=True)
class Paths:
    q45: Path
    t0_evidence: Path
    q45_evidence: Path
    candidate_events: Path
    contract: Path
    boolean_contract: Path


def default_paths() -> Paths:
    return Paths(
        q45=REPO_ROOT / "out/signal_study/m_b1_5_stage2_q45_20260710",
        t0_evidence=REPO_ROOT
        / "docs/modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710",
        q45_evidence=REPO_ROOT
        / "docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710",
        candidate_events=REPO_ROOT
        / "out/signal_study/m_b1_5_stage2_q1q3_20260710/d_online_events.parquet",
        contract=REPO_ROOT / "docs/research/eval/safe_region_asset_contract.md",
        boolean_contract=REPO_ROOT
        / "docs/research/eval/boolean_composition_semantics_contract.md",
    )


def preflight(paths: Paths) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "OK",
        "blocking": [],
        "warnings": [],
        "verified_seals": {},
        "inputs": {},
    }

    # Keys map runtime basename → manifest.artifact_hashes key.
    # manifest.json is self-describing and is sealed via committed SHA256SUMS, not artifact_hashes.
    required_runtime = {
        "atom_atlas.parquet": "atom_atlas_parquet",
        "pairwise_and_atlas.parquet": "pairwise_and_atlas_parquet",
        "pairwise_or_atlas.parquet": "pairwise_or_atlas_parquet",
        "per_sequence.csv": "per_sequence",
        "threshold_registry.json": "threshold_registry",
        "summary.json": "summary",
        "region_stability.csv": "region_stability",
    }
    required_present = [
        "manifest.json",
        "atom_atlas.parquet",
        "pairwise_and_atlas.parquet",
        "pairwise_or_atlas.parquet",
        "per_sequence.csv",
        "threshold_registry.json",
        "summary.json",
        "region_stability.csv",
    ]
    if not paths.q45.is_dir():
        report["blocking"].append(f"missing q45 study root: {paths.q45}")
    man_path = paths.q45 / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        hashes = man.get("artifact_hashes") or {}
        for fname in required_present:
            p = paths.q45 / fname
            report["inputs"][fname] = str(p)
            if not p.is_file():
                alt = None
                if fname.endswith(".parquet"):
                    alt = paths.q45 / fname.replace(".parquet", ".csv")
                if alt is None or not alt.is_file():
                    report["blocking"].append(f"missing required runtime artifact: {p}")
                    continue
                p = alt
                report["inputs"][fname] = str(p)
                report["warnings"].append(f"using csv fallback for {fname}: {alt}")
            got = sha256_file(p)
            mkey = required_runtime.get(fname)
            if mkey is None:
                # presence-only for manifest; seal checked via committed evidence copy
                report["verified_seals"][p.name] = {
                    "sha256": got,
                    "expected": None,
                    "ok": True,
                    "note": "presence_only_self_manifest",
                }
                continue
            # csv fallback uses non-parquet hash key when reading csv
            if p.suffix == ".csv" and mkey.endswith("_parquet"):
                mkey = mkey[: -len("_parquet")]
            exp = hashes.get(mkey)
            ok = exp is not None and got == exp
            report["verified_seals"][p.name] = {
                "sha256": got,
                "expected": exp,
                "ok": ok,
            }
            if not ok:
                report["blocking"].append(
                    f"seal mismatch {p.name}: got={got} expected={exp}"
                )
        exp_ev = man.get("source_event_table_sha256")
        report["inputs"]["source_event_table_sha256_declared"] = exp_ev
    else:
        report["blocking"].append(f"missing q45 manifest: {man_path}")
        man = {}

    # committed evidence seals
    sums_path = paths.q45_evidence / "SHA256SUMS.json"
    if sums_path.is_file():
        sums = json.loads(sums_path.read_text(encoding="utf-8"))
        for ent in sums.get("files", []):
            p = paths.q45_evidence / ent["file"]
            if not p.is_file():
                report["blocking"].append(f"missing committed evidence file: {p}")
                continue
            got = sha256_file(p)
            ok = got == ent["sha256"]
            report["verified_seals"][f"evidence/{ent['file']}"] = {
                "sha256": got,
                "expected": ent["sha256"],
                "ok": ok,
            }
            if not ok:
                report["blocking"].append(f"committed seal mismatch {ent['file']}")
    else:
        report["blocking"].append(f"missing {sums_path}")

    t0_sums = paths.t0_evidence / "SHA256SUMS.json"
    if t0_sums.is_file():
        sums = json.loads(t0_sums.read_text(encoding="utf-8"))
        for ent in sums.get("files", []):
            p = paths.t0_evidence / ent["file"]
            if not p.is_file():
                report["blocking"].append(f"missing T0 evidence file: {p}")
                continue
            got = sha256_file(p)
            ok = got == ent["sha256"]
            report["verified_seals"][f"t0/{ent['file']}"] = {
                "sha256": got,
                "expected": ent["sha256"],
                "ok": ok,
            }
            if not ok:
                report["blocking"].append(f"T0 seal mismatch {ent['file']}")
    else:
        report["blocking"].append(f"missing {t0_sums}")

    # candidate rows
    cand = paths.candidate_events
    report["inputs"]["candidate_events"] = str(cand)
    if not cand.is_file():
        # csv fallback
        csv_alt = cand.with_suffix(".csv")
        if csv_alt.is_file():
            cand = csv_alt
            report["inputs"]["candidate_events"] = str(cand)
            report["warnings"].append(f"using csv candidate rows: {csv_alt}")
        else:
            report["blocking"].append(
                "BLOCKED_BY_ARTIFACT: missing candidate rows for membership digest"
            )
    if cand.is_file() and man:
        got = sha256_file(cand)
        exp = man.get("source_event_table_sha256")
        # only parquet matches source_event_table_sha256
        if cand.suffix == ".parquet":
            ok = got == exp
            report["verified_seals"]["d_online_events.parquet"] = {
                "sha256": got,
                "expected": exp,
                "ok": ok,
            }
            if not ok:
                report["blocking"].append(
                    f"candidate event seal mismatch: {got} != {exp}"
                )
        else:
            report["warnings"].append(
                "candidate rows loaded from csv; source_event_table_sha256 is parquet seal only"
            )

    for p in [paths.contract, paths.boolean_contract]:
        if not p.is_file():
            report["blocking"].append(f"missing contract: {p}")

    # E1 check: contract must not retain inverted positive implications in §2.4 body
    if paths.contract.is_file():
        text = paths.contract.read_text(encoding="utf-8")
        if (
            "Status:** **ACCEPTED**" not in text
            and "**Status:** **ACCEPTED**" not in text
        ):
            report["warnings"].append("contract status not marked ACCEPTED")
        if "generator-contract equality ⇏" not in text:
            report["blocking"].append(
                "E1 not applied: missing non-implication for generator-contract"
            )
        if "source_event_table_sha256 ⇏" not in text:
            report["blocking"].append(
                "E1 not applied: missing non-implication for source_event_table_sha256"
            )

    if report["blocking"]:
        report["status"] = (
            "BLOCKED_BY_ARTIFACT"
            if any(
                "missing" in b.lower() or "BLOCKED_BY_ARTIFACT" in b
                for b in report["blocking"]
            )
            else "BLOCKED_BY_PROVENANCE"
        )
    return report


def load_atlases(q45: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def load(name: str) -> pd.DataFrame:
        pq = q45 / f"{name}.parquet"
        csvp = q45 / f"{name}.csv"
        if pq.is_file():
            return pd.read_parquet(pq)
        if csvp.is_file():
            return pd.read_csv(csvp)
        fail(f"missing atlas {name}")

    return load("atom_atlas"), load("pairwise_and_atlas"), load("pairwise_or_atlas")


def load_candidates(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_universe_authorities(
    candidates: pd.DataFrame, substrate_id: str
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    pk_cols = ["event_id"]
    label_cols = ["pair_label", "label_status", "baseline_selected"]
    project_cols = pk_cols + label_cols
    for c in project_cols:
        if c not in candidates.columns:
            fail(f"candidate rows missing required column {c}")

    contract_body = {
        "kind": "candidate_universe_contract",
        "substrate_id": substrate_id,
        "hook_id": "stage1_baudit_d_online_hook",
        "candidate_builder_id": "d_online_events",
        "candidate_builder_version": "stage2_q1q3_v1",
        "prefilter_contract_id": "online_hook_eligible",
        "eligibility_contract": "online_hook_eligible",
        "candidate_key_schema": {
            "primary_key_columns": pk_cols,
            "column_types": {"event_id": "string"},
        },
        "label_exposure_contract_id": "q45_primary_resolved_selected_cohort",
        "label_exposure_columns": label_cols,
        "observation_time_or_frame_range": "stage1_baudit_invocation_counter_frames",
        "predecision_state_snapshot_contract": "pre_decision_online_hook_eligible_rows",
    }
    contract_id = content_id(contract_body)

    row_digests: list[str] = []
    seen_pk: dict[str, str] = {}
    for _, row in candidates.iterrows():
        projected: dict[str, Any] = {}
        for c in project_cols:
            v = row[c]
            if pd.isna(v):
                fail(
                    f"missing required membership column {c} for event {row.get('event_id')}"
                )
            if c == "baseline_selected":
                projected[c] = as_int_flag(v)
            else:
                projected[c] = str(v)
        dig = content_id(projected)
        pk = str(row["event_id"])
        if pk in seen_pk and seen_pk[pk] != dig:
            fail(f"duplicate PK with conflicting payload: {pk}")
        if pk not in seen_pk:
            seen_pk[pk] = dig
            row_digests.append(dig)
    row_digests_sorted = sorted(set(row_digests))
    membership_digest = hashlib.sha256(
        ("\n".join(row_digests_sorted) + "\n").encode("utf-8")
    ).hexdigest()

    instance_body = {
        "kind": "candidate_universe_instance",
        "candidate_universe_contract_id": contract_id,
        "universe_membership_digest": membership_digest,
    }
    instance_id = content_id(instance_body)

    contract_row = {
        "candidate_universe_contract_id": contract_id,
        **{k: v for k, v in contract_body.items() if k != "kind"},
    }
    instance_row = {
        "candidate_universe_instance_id": instance_id,
        "candidate_universe_contract_id": contract_id,
        "universe_membership_digest": membership_digest,
        "universe_hash": membership_digest,
        "membership_digest_algorithm_version": MEMBERSHIP_DIGEST_ALGORITHM,
        "n_candidates": len(row_digests_sorted),
        "membership_status": "SEALED",
        "transport_id": None,
    }
    return contract_row, instance_row, contract_id, instance_id


def build_threshold_registry(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    entry_digests: list[str] = []

    for atom in raw["single_atoms"]:
        thr_index = int(atom["thr_index"])
        thr_value = float(atom["thr_value"])
        entry = {
            "lattice_kind": raw["single_lattice_kind"],
            "feature": atom["feature"],
            "direction": atom["direction"],
            "thr_index": thr_index,
            "atom_id": atom["atom_id"],
            "threshold_value_repr": thr_value_repr(thr_value),
            "thr_value": thr_value,
            "scope": "single_atom",
        }
        # optional quantile only when present
        if (
            "quantile_lattice_point" in atom
            and atom["quantile_lattice_point"] is not None
        ):
            entry["quantile_lattice_point"] = atom["quantile_lattice_point"]
        dig_body = {
            k: entry[k]
            for k in (
                "atom_id",
                "direction",
                "feature",
                "lattice_kind",
                "scope",
                "thr_index",
                "threshold_value_repr",
                "thr_value",
            )
            if k in entry
        }
        if "quantile_lattice_point" in entry:
            dig_body["quantile_lattice_point"] = entry["quantile_lattice_point"]
        entry_digests.append(content_id(dig_body))
        entries.append(entry)

    for atom in raw["pairwise_atoms"]:
        thr_index = int(atom["thr_index"])
        thr_value = float(atom["thr_value"])
        entry = {
            "lattice_kind": raw["pairwise_lattice_kind"],
            "feature": atom["feature"],
            "direction": atom["direction"],
            "thr_index": thr_index,
            "atom_id": atom["atom_id"],
            "threshold_value_repr": thr_value_repr(thr_value),
            "thr_value": thr_value,
            "scope": "pairwise_atom",
        }
        if (
            "quantile_lattice_point" in atom
            and atom["quantile_lattice_point"] is not None
        ):
            entry["quantile_lattice_point"] = atom["quantile_lattice_point"]
        elif "quantile" in atom and atom["quantile"] is not None:
            entry["quantile_lattice_point"] = atom["quantile"]
        dig_body = {
            k: entry[k]
            for k in (
                "atom_id",
                "direction",
                "feature",
                "lattice_kind",
                "scope",
                "thr_index",
                "threshold_value_repr",
                "thr_value",
            )
            if k in entry
        }
        if "quantile_lattice_point" in entry:
            dig_body["quantile_lattice_point"] = entry["quantile_lattice_point"]
        entry_digests.append(content_id(dig_body))
        entries.append(entry)

    entries_digest = hashlib.sha256(
        ("\n".join(sorted(entry_digests)) + "\n").encode("utf-8")
    ).hexdigest()

    reg_body = {
        "kind": "threshold_registry",
        "taxonomy_version": raw["taxonomy_version"],
        "single_lattice_kind": raw["single_lattice_kind"],
        "pairwise_lattice_kind": raw["pairwise_lattice_kind"],
        "signals_primary": list(raw["signals_primary"]),
        "directions": list(raw["directions"]),
        "combinators": list(raw["combinators"]),
        "entries_digest": entries_digest,
    }
    registry_id = content_id(reg_body)

    # assign entry ids
    entry_rows: list[dict[str, Any]] = []
    for e in entries:
        eid_body = {
            "kind": "threshold_registry_entry",
            "threshold_registry_id": registry_id,
            "lattice_kind": e["lattice_kind"],
            "feature": e["feature"],
            "direction": e["direction"],
            "thr_index": e["thr_index"],
        }
        eid = content_id(eid_body)
        entry_rows.append(
            {
                "threshold_registry_entry_id": eid,
                "threshold_registry_id": registry_id,
                **e,
            }
        )

    registry_row = {
        "threshold_registry_id": registry_id,
        "taxonomy_version": raw["taxonomy_version"],
        "single_lattice_kind": raw["single_lattice_kind"],
        "pairwise_lattice_kind": raw["pairwise_lattice_kind"],
        "signals_primary": list(raw["signals_primary"]),
        "directions": list(raw["directions"]),
        "combinators": list(raw["combinators"]),
        "entries_digest": entries_digest,
        "n_single_atoms": int(raw["n_single_atoms"]),
        "n_pairwise_atoms": int(raw["n_pairwise_atoms"]),
        "assignment_group_key_status": raw.get("assignment_group_key_status"),
        "source_file_sha256": "d3e3197fa7812a9ec5f9b06cc2286dcce52d49cf805eba6527c3b24b62a585f4",
    }
    return registry_row, entry_rows, registry_id


def build_predicates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in SIGNALS_PRIMARY:
        for direction in DIRECTIONS:
            body = {
                "kind": "predicate",
                "signal_identity": feature,
                "signal_unit": "dimensionless_signal",
                "predicate_domain": "real_or_missing",
                "predicate_codomain": "{T,F,U}",
                "unknown_value_policy": "three_valued_unknown",
                "final_unknown_action": "no_reject",
                "comparator": ">=" if direction == "high_tail" else "<=",
                "endpoint_policy": "inclusive_threshold",
                "tie_policy": "include_on_threshold",
                "nan_policy": "unknown",
                "posinf_policy": "unknown",
                "neginf_policy": "unknown",
                "missing_value_policy": "unknown",
                "quantile_method": "primary_unique_or_q05_lattice",
                "floating_point_tolerance": "exact_float64_registry_value",
                "clipping_domain": "none",
                "direction": direction,
            }
            pid = content_id(body)
            rows.append(
                {"predicate_id": pid, **{k: v for k, v in body.items() if k != "kind"}}
            )
    return rows


def predicate_id_for(preds: list[dict[str, Any]], feature: str, direction: str) -> str:
    for p in preds:
        if p["signal_identity"] == feature and p["direction"] == direction:
            return p["predicate_id"]
    fail(f"missing predicate for {feature}/{direction}")


def policy_family_for_grid(
    grammar: str,
    grid_id: str,
    universe_instance_id: str,
    preds: list[dict[str, Any]],
) -> dict[str, Any]:
    if grammar == "G1_atom":
        # S::feature::direction
        _, feature, direction = grid_id.split("::", 2)
        pred = predicate_id_for(preds, feature, direction)
        ast = {
            "op": "ATOM",
            "role": ROLE,
            "predicate_id": pred,
            "signal": feature,
            "direction": direction,
            "parameter_axis": "thr_index",
            "lattice_kind": "primary_unique_boundaries",
        }
        max_ops = 1
        combinator = None
    else:
        body = grid_id[3:]  # strip P::
        a, b = body.split("__")
        f0, d0 = a.split("::")
        f1, d1 = b.split("::")
        pred0 = predicate_id_for(preds, f0, d0)
        pred1 = predicate_id_for(preds, f1, d1)
        combinator = "AND" if grammar == "G2_and" else "OR"
        ast = {
            "op": combinator,
            "role": ROLE,
            "children": [
                {
                    "op": "ATOM",
                    "role": ROLE,
                    "predicate_id": pred0,
                    "signal": f0,
                    "direction": d0,
                    "parameter_axis": "thr_index",
                    "axis": 0,
                },
                {
                    "op": "ATOM",
                    "role": ROLE,
                    "predicate_id": pred1,
                    "signal": f1,
                    "direction": d1,
                    "parameter_axis": "thr_index",
                    "axis": 1,
                },
            ],
            "lattice_kind": "primary_quantile_lattice_q05",
        }
        max_ops = 2

    ast_hash = content_id(ast)
    body = {
        "kind": "policy_family_definition",
        "grammar_version": "composition_grammar_g1_g2_g3_v0",
        "truth_semantics_version": "boolean_composition_semantics_contract_v0",
        "composition_level": "observational",
        "candidate_universe_instance_id": universe_instance_id,
        "operator_precedence": "NOT>AND>OR",
        "maximum_nesting_depth": 1 if grammar == "G1_atom" else 1,
        "maximum_operands_per_node": max_ops,
        "not_scope": "none",
        "mixed_role_policy": "forbidden_g7_roles",
        "canonical_policy_ast": ast,
        "canonical_policy_ast_hash": ast_hash,
        "grammar": grammar,
        "grid_id": grid_id,
        "parameter_system": "registered_thr_index_lattice",
    }
    fid = content_id(body)
    return {
        "policy_family_definition_id": fid,
        "grammar": grammar,
        "grid_id": grid_id,
        "candidate_universe_instance_id": universe_instance_id,
        "grammar_version": body["grammar_version"],
        "truth_semantics_version": body["truth_semantics_version"],
        "composition_level": "observational",
        "operator_precedence": body["operator_precedence"],
        "maximum_nesting_depth": body["maximum_nesting_depth"],
        "maximum_operands_per_node": body["maximum_operands_per_node"],
        "not_scope": "none",
        "mixed_role_policy": body["mixed_role_policy"],
        "canonical_policy_ast": ast,
        "canonical_policy_ast_hash": ast_hash,
        "parameter_system": body["parameter_system"],
        "combinator": combinator or "",
    }


def build_truth_digest(
    atom: pd.DataFrame,
    and_df: pd.DataFrame,
    or_df: pd.DataFrame,
    thr_raw: dict[str, Any],
    cohort: dict[str, Any],
    sequence_set: list[str],
    component_rows: list[dict[str, Any]],
) -> str:
    def row_digest_map(rows: list[dict[str, Any]], key_fn) -> str:
        digs = []
        seen: dict[str, str] = {}
        for r in rows:
            k = key_fn(r)
            d = content_id(r)
            if k in seen and seen[k] != d:
                fail(f"duplicate truth key with conflict: {k}")
            if k not in seen:
                seen[k] = d
                digs.append((k, d))
        digs.sort(key=lambda x: x[0])
        return hashlib.sha256(
            ("".join(d for _, d in digs)).encode("utf-8")
            if False
            else ("\n".join(d for _, d in digs) + "\n").encode("utf-8")
        ).hexdigest()

    atom_rows = []
    for _, r in atom.iterrows():
        atom_rows.append(
            {
                "atom_id": str(r["atom_id"]),
                "feature": str(r["feature"]),
                "direction": str(r["direction"]),
                "thr_index": int(r["thr_index"]),
                "lattice_kind": str(r["lattice_kind"]),
                "observed_safe_point": as_int_flag(r["observed_safe_point"]),
                "productive_safe_point": as_int_flag(r["productive_safe_point"]),
                "gt_hurt": int(r["gt_hurt"]),
                "n_neg_captured": int(r["n_neg_captured"]),
                "n_gt_captured": int(r["n_gt_captured"]),
                "n_unresolved_selected": int(r["n_unresolved_selected"]),
                "safety_status": str(r["safety_status"]),
                "mask_sha256": str(r["mask_sha256"]),
                "per_sequence_neg": parse_seq_json(r.get("per_sequence_neg_json")),
                "per_sequence_gt": parse_seq_json(r.get("per_sequence_gt_json")),
            }
        )

    def pairwise_rows(df: pd.DataFrame, combinator: str) -> list[dict[str, Any]]:
        out = []
        for _, r in df.iterrows():
            can = canonicalize_pairwise_leaf(
                str(r["feature_a"]),
                str(r["direction_a"]),
                int(r["thr_index_a"]),
                str(r["atom_a_id"]),
                str(r["feature_b"]),
                str(r["direction_b"]),
                int(r["thr_index_b"]),
                str(r["atom_b_id"]),
            )
            ckey = canonical_cell_key_pairwise(
                combinator, can["atom_0_id"], can["atom_1_id"]
            )
            out.append(
                {
                    "canonical_cell_key": ckey,
                    "combinator": combinator,
                    "atom_0_id": can["atom_0_id"],
                    "atom_1_id": can["atom_1_id"],
                    "feature_0": can["feature_0"],
                    "direction_0": can["direction_0"],
                    "thr_index_0": can["thr_index_0"],
                    "feature_1": can["feature_1"],
                    "direction_1": can["direction_1"],
                    "thr_index_1": can["thr_index_1"],
                    "lattice_kind": str(r["lattice_kind"]),
                    "observed_safe_point": as_int_flag(r["observed_safe_point"]),
                    "productive_safe_point": as_int_flag(r["productive_safe_point"]),
                    "gt_hurt": int(r["gt_hurt"]),
                    "n_neg_captured": int(r["n_neg_captured"]),
                    "n_gt_captured": int(r["n_gt_captured"]),
                    "n_unresolved_selected": int(r["n_unresolved_selected"]),
                    "safety_status": str(r["safety_status"]),
                    "mask_sha256": str(r["mask_sha256"]),
                    "semantic_duplicate_mask": as_int_flag(
                        r.get("semantic_duplicate_mask", 0)
                    ),
                    "empty_region": as_int_flag(r.get("empty_region", 0)),
                    "per_sequence_neg": parse_seq_json(r.get("per_sequence_neg_json")),
                    "per_sequence_gt": parse_seq_json(r.get("per_sequence_gt_json")),
                }
            )
        return out

    and_rows = pairwise_rows(and_df, "AND")
    or_rows = pairwise_rows(or_df, "OR")

    thr_meta = {
        "taxonomy_version": thr_raw["taxonomy_version"],
        "signals_primary": sorted(thr_raw["signals_primary"]),
        "directions": sorted(thr_raw["directions"]),
        "combinators": sorted(thr_raw["combinators"]),
        "single_lattice_kind": thr_raw["single_lattice_kind"],
        "pairwise_lattice_kind": thr_raw["pairwise_lattice_kind"],
        "n_single_atoms": int(thr_raw["n_single_atoms"]),
        "n_pairwise_atoms": int(thr_raw["n_pairwise_atoms"]),
        "assignment_group_key_status": thr_raw.get("assignment_group_key_status"),
    }
    cohort_row = {
        "cohort_definition": cohort,
        "sequence_set": sorted(sequence_set),
        "n_primary_negative": EXPECTED["n_fp_exposed"],
        "n_primary_positive_protect": EXPECTED["n_gt_exposed"],
    }

    # t0 component membership without ::compN ordinals
    t0_rows = []
    for c in component_rows:
        coords = (
            json.loads(c["coords_json"])
            if isinstance(c["coords_json"], str)
            else c["coords_json"]
        )
        t0_rows.append(
            {
                "grammar": c["grammar"],
                "grid_id": c["grid_id"],
                "adjacency": "G1_bilateral"
                if c["grammar"] == "G1_atom"
                else "G2_4neighbor",
                "coordinate_keys_sorted": sorted(
                    [str(x) for x in coords],
                    key=lambda s: s,
                )
                if c["grammar"] == "G1_atom"
                else sorted([canonical_json(x) for x in coords]),
            }
        )

    table_digests = {
        "atom_atlas": row_digest_map(atom_rows, lambda r: r["atom_id"]),
        "pairwise_and_atlas": row_digest_map(
            and_rows, lambda r: r["canonical_cell_key"]
        ),
        "pairwise_or_atlas": row_digest_map(or_rows, lambda r: r["canonical_cell_key"]),
        "threshold_registry_meta": content_id(thr_meta),
        "cohort_contract": content_id(cohort_row),
        "t0_component_membership": row_digest_map(
            t0_rows,
            lambda r: (
                r["grammar"]
                + "|"
                + r["grid_id"]
                + "|"
                + content_id(r["coordinate_keys_sorted"])
            ),
        ),
    }
    return content_id(table_digests)


def derive_claim_level(n_coords: int, shape_class: str) -> str:
    if n_coords == 1 or shape_class == "isolated_point":
        return "L0"
    if n_coords >= 2:
        return "L1"
    fail(f"cannot derive claim_level for n_coords={n_coords} shape={shape_class}")


def convert(paths: Paths, out_dir: Path) -> dict[str, Any]:
    report = preflight(paths)
    if report["status"] != "OK":
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "preflight_block_report.json", report)
        return {"status": report["status"], "report": report, "out_dir": str(out_dir)}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "preflight_report.json", report)

    man = json.loads((paths.q45 / "manifest.json").read_text(encoding="utf-8"))
    thr_raw = json.loads(
        (paths.q45_evidence / "threshold_registry.json").read_text(encoding="utf-8")
    )
    atom, and_df, or_df = load_atlases(paths.q45)
    candidates = load_candidates(
        paths.candidate_events
        if paths.candidate_events.is_file()
        else paths.candidate_events.with_suffix(".csv")
    )
    components = list(
        csv.DictReader((paths.t0_evidence / "component_geometry.csv").open())
    )
    margins = list(csv.DictReader((paths.t0_evidence / "boundary_margin.csv").open()))
    mask_caps = list(
        csv.DictReader(
            (paths.t0_evidence / "productive_capacity_by_per_grid_mask.csv").open()
        )
    )

    # basic count checks
    n_ps_g1 = (
        int(atom["productive_safe_point"].astype(int).sum())
        if atom["productive_safe_point"].dtype != object
        else int(atom["productive_safe_point"].map(as_int_flag).sum())
    )
    n_ps_and = int(and_df["productive_safe_point"].map(as_int_flag).sum())
    n_ps_or = int(or_df["productive_safe_point"].map(as_int_flag).sum())
    if (
        len(atom) != EXPECTED["n_atom_rows"]
        or len(and_df) != EXPECTED["n_and_rows"]
        or len(or_df) != EXPECTED["n_or_rows"]
    ):
        fail("atlas row counts mismatch sealed expectations")
    if (
        n_ps_g1 != 1
        or n_ps_and != 153
        or n_ps_or != 0
        or n_ps_g1 + n_ps_and + n_ps_or != 154
    ):
        fail(f"PS counts mismatch: g1={n_ps_g1} and={n_ps_and} or={n_ps_or}")
    if len(components) != EXPECTED["n_components"]:
        fail(f"component count {len(components)} != 26")

    uni_contract, uni_instance, uni_contract_id, uni_instance_id = (
        build_universe_authorities(candidates, man["substrate_id"])
    )
    thr_reg, thr_entries, thr_registry_id = build_threshold_registry(thr_raw)
    if (
        thr_reg["n_single_atoms"] != EXPECTED["n_single_atoms"]
        or thr_reg["n_pairwise_atoms"] != EXPECTED["n_pairwise_atoms"]
    ):
        fail("threshold registry atom counts mismatch")

    # entry lookup
    entry_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for e in thr_entries:
        entry_by_key[
            (e["lattice_kind"], e["feature"], e["direction"], int(e["thr_index"]))
        ] = e

    preds = build_predicates()

    # families for all G1 grids + all pairwise axis pairs for AND and OR
    families: dict[tuple[str, str], dict[str, Any]] = {}
    for feature in SIGNALS_PRIMARY:
        for direction in DIRECTIONS:
            gid = g1_grid_id(feature, direction)
            families[("G1_atom", gid)] = policy_family_for_grid(
                "G1_atom", gid, uni_instance_id, preds
            )
    for fa, da, fb, db in pairwise_axis_pairs():
        gid = g2_grid_id(fa, da, fb, db)
        families[("G2_and", gid)] = policy_family_for_grid(
            "G2_and", gid, uni_instance_id, preds
        )
        families[("G3_or", gid)] = policy_family_for_grid(
            "G3_or", gid, uni_instance_id, preds
        )

    truth_digest = build_truth_digest(
        atom,
        and_df,
        or_df,
        thr_raw,
        man["cohort_definition"],
        man["sequence_set"],
        components,
    )
    truth_body = {
        "id_scheme": ID_SCHEME,
        "kind": "truth_contract",
        "taxonomy_version": man["taxonomy_version"],
        "substrate_id": man["substrate_id"],
        "candidate_universe_instance_id": uni_instance_id,
        "threshold_registry_id": thr_registry_id,
        "signal_family": sorted(SIGNALS_PRIMARY),
        "sequence_set": sorted(man["sequence_set"]),
        "label_contract": man["cohort_definition"],
        "unresolved_policy": {"unresolved_contaminated_blocks_candidate": True},
        "lattice_contract": {
            "single_lattice_kind": thr_raw["single_lattice_kind"],
            "pairwise_lattice_kind": thr_raw["pairwise_lattice_kind"],
            "combinators": COMBINATORS,
        },
        "normalized_data_content_digest": truth_digest,
    }
    truth_contract_id = content_id(truth_body)
    truth_contract = {
        "truth_contract_id": truth_contract_id,
        **{k: v for k, v in truth_body.items() if k != "kind"},
        "contract_version": CONTRACT_VERSION,
    }

    # evidence bundle
    t0_sums = {
        ent["file"]: ent["sha256"]
        for ent in json.loads((paths.t0_evidence / "SHA256SUMS.json").read_text())[
            "files"
        ]
    }
    evidence_body = {
        "id_scheme": ID_SCHEME,
        "kind": "evidence_bundle",
        "truth_contract_id": truth_contract_id,
        "study_id": man["study_id"],
        "study_git_commit": man["git_commit"],
        "evaluator_source_sha256": man["evaluator_source_sha256"],
        "runner_source_sha256": man["runner_source_sha256"],
        "source_event_table_sha256": man["source_event_table_sha256"],
        "raw_artifact_sha256": {
            **man["artifact_hashes"],
            "t0_component_geometry": t0_sums["component_geometry.csv"],
            "t0_boundary_margin": t0_sums["boundary_margin.csv"],
            "t0_summary": t0_sums["summary.json"],
            "t0_manifest": t0_sums["manifest.json"],
        },
        "terminal_letter": man["terminal_letter"],
    }
    evidence_bundle_id = content_id(evidence_body)
    evidence_bundle = {
        "evidence_bundle_id": evidence_bundle_id,
        **{k: v for k, v in evidence_body.items() if k != "kind"},
    }

    feasibility_body = {
        "id_scheme": ID_SCHEME,
        "kind": "feasibility_contract",
        "truth_contract_id": truth_contract_id,
        "parameter_or_policy_space": {
            "spaces": [
                {
                    "grammar": "G1_atom",
                    "lattice": thr_raw["single_lattice_kind"],
                    "n_coords": 870,
                },
                {
                    "grammar": "G2_and",
                    "lattice": thr_raw["pairwise_lattice_kind"],
                    "n_coords": 17640,
                },
                {
                    "grammar": "G3_or",
                    "lattice": thr_raw["pairwise_lattice_kind"],
                    "n_coords": 17640,
                },
            ]
        },
        "candidate_universe_instance_id": uni_instance_id,
        "safety_loss_definition": {
            "name": "resolved_gt_hurt_exact_zero",
            "predicate": "resolved GT_hurt == 0",
            "unresolved_firewall": "unresolved_contaminated_blocks_candidate",
        },
        "productivity_definition": {
            "name": "n_neg_captured_ge_1",
            "predicate": "n_neg_captured > 0",
            "kind": "count_surrogate",
        },
        "epsilon": {"kind": "exact_zero_count", "value": 0},
        "g_min": {"kind": "count_ge_1", "value": 1},
        "n_gt_exposed": EXPECTED["n_gt_exposed"],
        "n_fp_exposed": EXPECTED["n_fp_exposed"],
        "denominator_owner": "primary_resolved_baseline_selected_cohort",
        "selection_scope": "in_sample_searched_and_evaluated",
        "finite_sample_statement": "observed_GT0_is_not_population_risk_zero",
        "metric_adjacency_edge_policy": {
            "G1": "bilateral",
            "G2": "4neighbor_manhattan_erosion",
            "unsafe": "non_productive_safe_on_registered_grid",
            "off_lattice_neighbor": "fails_full_neighborhood",
        },
    }
    feasibility_contract_id = content_id(feasibility_body)
    feasibility_contract = {
        "feasibility_contract_id": feasibility_contract_id,
        **{k: v for k, v in feasibility_body.items() if k != "kind"},
    }

    # index atlas PS rows
    ps_g1: dict[tuple[str, int], dict[str, Any]] = {}
    for _, r in atom.iterrows():
        if as_int_flag(r["productive_safe_point"]) != 1:
            continue
        gid = g1_grid_id(str(r["feature"]), str(r["direction"]))
        ps_g1[(gid, int(r["thr_index"]))] = r.to_dict()

    ps_and: dict[tuple[str, int, int], dict[str, Any]] = {}
    for _, r in and_df.iterrows():
        if as_int_flag(r["productive_safe_point"]) != 1:
            continue
        can = canonicalize_pairwise_leaf(
            str(r["feature_a"]),
            str(r["direction_a"]),
            int(r["thr_index_a"]),
            str(r["atom_a_id"]),
            str(r["feature_b"]),
            str(r["direction_b"]),
            int(r["thr_index_b"]),
            str(r["atom_b_id"]),
        )
        ps_and[(can["grid_id"], can["thr_index_0"], can["thr_index_1"])] = {
            **r.to_dict(),
            **can,
        }

    # margins by component + cell
    margin_by_comp_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for m in margins:
        margin_by_comp_cell[(m["component_id"], m["cell_id"])] = m

    # grid domains
    grid_domains: list[dict[str, Any]] = []
    grid_domain_ids: dict[tuple[str, str], str] = {}
    for feature in SIGNALS_PRIMARY:
        for direction in DIRECTIONS:
            gid = g1_grid_id(feature, direction)
            body = {
                "kind": "grid_domain",
                "lattice_kind": thr_raw["single_lattice_kind"],
                "grid_id": gid,
                "grammar_family": "G1",
                "axis_descriptors": [{"feature": feature, "direction": direction}],
                "n_registered_coordinates": 87,
            }
            gdid = content_id(body)
            grid_domain_ids[("G1_atom", gid)] = gdid
            grid_domains.append(
                {
                    "grid_domain_id": gdid,
                    "lattice_kind": body["lattice_kind"],
                    "grid_id": gid,
                    "grammar_family": "G1",
                    "axis_descriptors": body["axis_descriptors"],
                    "n_registered_coordinates": 87,
                    "grammar": "G1_atom",
                }
            )
    for fa, da, fb, db in pairwise_axis_pairs():
        gid = g2_grid_id(fa, da, fb, db)
        for grammar, gfam in (("G2_and", "pairwise"), ("G3_or", "pairwise")):
            body = {
                "kind": "grid_domain",
                "lattice_kind": thr_raw["pairwise_lattice_kind"],
                "grid_id": gid,
                "grammar_family": gfam,
                "grammar": grammar,
                "axis_descriptors": [
                    {"feature": fa, "direction": da, "axis": 0},
                    {"feature": fb, "direction": db, "axis": 1},
                ],
                "n_registered_coordinates": 441,
            }
            gdid = content_id(body)
            grid_domain_ids[(grammar, gid)] = gdid
            grid_domains.append(
                {
                    "grid_domain_id": gdid,
                    "lattice_kind": thr_raw["pairwise_lattice_kind"],
                    "grid_id": gid,
                    "grammar_family": "pairwise",
                    "axis_descriptors": body["axis_descriptors"],
                    "n_registered_coordinates": 441,
                    "grammar": grammar,
                }
            )

    # search domains
    def make_search_domain(
        grammar: str, combinator: str, lattice_kind: str, members: list[dict[str, Any]]
    ):
        member_keys = sorted(
            [
                {
                    "grid_domain_id": m["grid_domain_id"],
                    "policy_family_definition_id": m["policy_family_definition_id"],
                    "grid_id": m["grid_id"],
                    "n_registered_coordinates": m["n_registered_coordinates"],
                }
                for m in members
            ],
            key=lambda x: x["grid_id"],
        )
        membership_digest = content_id(member_keys)
        body = {
            "kind": "search_domain",
            "truth_contract_id": truth_contract_id,
            "grammar": grammar,
            "combinator": combinator,
            "lattice_kind": lattice_kind,
            "n_members": len(members),
            "n_registered_coordinates_sum": sum(
                m["n_registered_coordinates"] for m in members
            ),
            "membership_digest": membership_digest,
        }
        sdid = content_id(body)
        return {
            "search_domain_id": sdid,
            "truth_contract_id": truth_contract_id,
            "grammar": grammar,
            "combinator": combinator,
            "lattice_kind": lattice_kind,
            "n_members": len(members),
            "n_registered_coordinates_sum": body["n_registered_coordinates_sum"],
            "membership_digest": membership_digest,
        }, [
            {
                "search_domain_id": sdid,
                "grid_domain_id": m["grid_domain_id"],
                "policy_family_definition_id": m["policy_family_definition_id"],
                "grid_id": m["grid_id"],
                "n_registered_coordinates": m["n_registered_coordinates"],
            }
            for m in sorted(members, key=lambda x: x["grid_id"])
        ]

    g1_members = [
        {
            "grid_domain_id": grid_domain_ids[("G1_atom", g1_grid_id(f, d))],
            "policy_family_definition_id": families[("G1_atom", g1_grid_id(f, d))][
                "policy_family_definition_id"
            ],
            "grid_id": g1_grid_id(f, d),
            "n_registered_coordinates": 87,
        }
        for f in SIGNALS_PRIMARY
        for d in DIRECTIONS
    ]
    g2_members = [
        {
            "grid_domain_id": grid_domain_ids[("G2_and", g2_grid_id(fa, da, fb, db))],
            "policy_family_definition_id": families[
                ("G2_and", g2_grid_id(fa, da, fb, db))
            ]["policy_family_definition_id"],
            "grid_id": g2_grid_id(fa, da, fb, db),
            "n_registered_coordinates": 441,
        }
        for fa, da, fb, db in pairwise_axis_pairs()
    ]
    g3_members = [
        {
            "grid_domain_id": grid_domain_ids[("G3_or", g2_grid_id(fa, da, fb, db))],
            "policy_family_definition_id": families[
                ("G3_or", g2_grid_id(fa, da, fb, db))
            ]["policy_family_definition_id"],
            "grid_id": g2_grid_id(fa, da, fb, db),
            "n_registered_coordinates": 441,
        }
        for fa, da, fb, db in pairwise_axis_pairs()
    ]
    if len(g2_members) != 40 or len(g3_members) != 40 or len(g1_members) != 10:
        fail(
            f"search member counts bad: g1={len(g1_members)} g2={len(g2_members)} g3={len(g3_members)}"
        )

    sd_g1, sdm_g1 = make_search_domain(
        "G1_atom", "ATOM", thr_raw["single_lattice_kind"], g1_members
    )
    sd_g2, sdm_g2 = make_search_domain(
        "G2_and", "AND", thr_raw["pairwise_lattice_kind"], g2_members
    )
    sd_g3, sdm_g3 = make_search_domain(
        "G3_or", "OR", thr_raw["pairwise_lattice_kind"], g3_members
    )
    search_domains = [sd_g1, sd_g2, sd_g3]
    search_domain_members = sdm_g1 + sdm_g2 + sdm_g3

    # coordinates, masks, regions, membership, policy instances
    coordinates: dict[str, dict[str, Any]] = {}
    mask_units: dict[str, dict[str, Any]] = {}
    policy_instances: dict[str, dict[str, Any]] = {}
    region_assets: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []

    def ensure_mask(grammar: str, grid_id: str, mask_sha: str) -> str:
        gdid = grid_domain_ids[(grammar, grid_id)]
        body = {
            "kind": "mask_unit",
            "truth_contract_id": truth_contract_id,
            "grid_id": grid_id,
            "mask_sha256": mask_sha,
        }
        mid = content_id(body)
        if mid not in mask_units:
            mask_units[mid] = {
                "mask_unit_id": mid,
                "truth_contract_id": truth_contract_id,
                "grid_domain_id": gdid,
                "grid_id": grid_id,
                "mask_sha256": mask_sha,
                "grammar": grammar,
            }
        return mid

    def ensure_coord_g1(
        feature: str, direction: str, thr_index: int, raw_alias: str | None = None
    ) -> str:
        gid = g1_grid_id(feature, direction)
        gdid = grid_domain_ids[("G1_atom", gid)]
        lattice = thr_raw["single_lattice_kind"]
        entry = entry_by_key[(lattice, feature, direction, int(thr_index))]
        ckey = canonical_cell_key_g1(feature, direction, thr_index)
        body = {
            "kind": "coordinate",
            "truth_contract_id": truth_contract_id,
            "threshold_registry_id": thr_registry_id,
            "canonical_cell_key": ckey,
            "axis_entry_ids": [entry["threshold_registry_entry_id"]],
        }
        cid = content_id(body)
        if cid not in coordinates:
            coordinates[cid] = {
                "coordinate_id": cid,
                "truth_contract_id": truth_contract_id,
                "grid_domain_id": gdid,
                "threshold_registry_id": thr_registry_id,
                "canonical_cell_key": ckey,
                "grammar": "G1_atom",
                "grid_id": gid,
                "thr_index_0": int(thr_index),
                "thr_index_1": "",
                "threshold_registry_entry_id_0": entry["threshold_registry_entry_id"],
                "threshold_registry_entry_id_1": "",
                "feature_0": feature,
                "direction_0": direction,
                "feature_1": "",
                "direction_1": "",
                "raw_cell_id_alias": raw_alias or ckey,
            }
        return cid

    def ensure_coord_pair(
        grammar: str,
        combinator: str,
        can: dict[str, Any],
        raw_alias: str | None = None,
    ) -> str:
        gid = can["grid_id"]
        gdid = grid_domain_ids[(grammar, gid)]
        lattice = thr_raw["pairwise_lattice_kind"]
        e0 = entry_by_key[
            (lattice, can["feature_0"], can["direction_0"], int(can["thr_index_0"]))
        ]
        e1 = entry_by_key[
            (lattice, can["feature_1"], can["direction_1"], int(can["thr_index_1"]))
        ]
        ckey = canonical_cell_key_pairwise(
            combinator, can["atom_0_id"], can["atom_1_id"]
        )
        body = {
            "kind": "coordinate",
            "truth_contract_id": truth_contract_id,
            "threshold_registry_id": thr_registry_id,
            "canonical_cell_key": ckey,
            "axis_entry_ids": [
                e0["threshold_registry_entry_id"],
                e1["threshold_registry_entry_id"],
            ],
        }
        cid = content_id(body)
        if cid not in coordinates:
            coordinates[cid] = {
                "coordinate_id": cid,
                "truth_contract_id": truth_contract_id,
                "grid_domain_id": gdid,
                "threshold_registry_id": thr_registry_id,
                "canonical_cell_key": ckey,
                "grammar": grammar,
                "grid_id": gid,
                "thr_index_0": int(can["thr_index_0"]),
                "thr_index_1": int(can["thr_index_1"]),
                "threshold_registry_entry_id_0": e0["threshold_registry_entry_id"],
                "threshold_registry_entry_id_1": e1["threshold_registry_entry_id"],
                "feature_0": can["feature_0"],
                "direction_0": can["direction_0"],
                "feature_1": can["feature_1"],
                "direction_1": can["direction_1"],
                "raw_cell_id_alias": raw_alias or ckey,
            }
        return cid

    def ensure_policy_instance(family_id: str, coord_row: dict[str, Any]) -> str:
        bindings = []
        bindings.append(
            {
                "axis": 0,
                "feature": coord_row["feature_0"],
                "direction": coord_row["direction_0"],
                "thr_index": int(coord_row["thr_index_0"]),
                "threshold_registry_entry_id": coord_row[
                    "threshold_registry_entry_id_0"
                ],
            }
        )
        if coord_row.get("threshold_registry_entry_id_1"):
            bindings.append(
                {
                    "axis": 1,
                    "feature": coord_row["feature_1"],
                    "direction": coord_row["direction_1"],
                    "thr_index": int(coord_row["thr_index_1"]),
                    "threshold_registry_entry_id": coord_row[
                        "threshold_registry_entry_id_1"
                    ],
                }
            )
        body = {
            "kind": "policy_instance",
            "policy_family_definition_id": family_id,
            "threshold_registry_id": thr_registry_id,
            "threshold_bindings": bindings,
        }
        pid = content_id(body)
        if pid not in policy_instances:
            policy_instances[pid] = {
                "policy_instance_id": pid,
                "policy_family_definition_id": family_id,
                "threshold_registry_id": thr_registry_id,
                "threshold_bindings_json": bindings,
                "coordinate_id": coord_row["coordinate_id"],
            }
            coordinates[coord_row["coordinate_id"]]["policy_instance_id"] = pid
        return pid

    # Build regions from components
    claim_counter = {"L0": 0, "L1": 0, "g1_L0": 0, "g2_L0": 0, "g2_L1": 0}
    for comp in components:
        grammar = comp["grammar"]
        grid_id = comp["grid_id"]
        n_coords = int(comp["n_coords"])
        shape = comp["shape_class"]
        claim = derive_claim_level(n_coords, shape)
        coords_raw = json.loads(comp["coords_json"])
        family = families[(grammar, grid_id)]
        fam_id = family["policy_family_definition_id"]
        gdid = grid_domain_ids[(grammar, grid_id)]

        # materialize membership coords and collect coordinate digests for region id
        coord_ids_for_region: list[str] = []
        member_specs: list[
            tuple[str, str, dict[str, Any], dict[str, Any]]
        ] = []  # cid, mask_id, atlas, margin

        if grammar == "G1_atom":
            for thr in coords_raw:
                thr_i = int(thr)
                atlas = ps_g1.get((grid_id, thr_i))
                if atlas is None:
                    fail(f"missing G1 PS atlas row {grid_id} thr={thr_i}")
                feature = str(atlas["feature"])
                direction = str(atlas["direction"])
                cid = ensure_coord_g1(
                    feature, direction, thr_i, raw_alias=str(atlas["atom_id"])
                )
                mid = ensure_mask(grammar, grid_id, str(atlas["mask_sha256"]))
                cell_id = str(atlas["atom_id"])
                margin = margin_by_comp_cell.get((comp["component_id"], cell_id), {})
                coord_ids_for_region.append(cid)
                member_specs.append((cid, mid, atlas, margin))
                ensure_policy_instance(fam_id, coordinates[cid])
        else:
            # G2: coords as [i,j] in grid axis order (already canonical)
            body = grid_id[3:]
            a, b = body.split("__")
            f0, d0 = a.split("::")
            f1, d1 = b.split("::")
            for pair in coords_raw:
                i, j = int(pair[0]), int(pair[1])
                atlas = ps_and.get((grid_id, i, j))
                if atlas is None:
                    # try reverse thr if raw atlas order differed but grid is canonic
                    fail(f"missing AND PS atlas row {grid_id} thr=({i},{j})")
                can = {
                    "atom_0_id": atlas["atom_0_id"],
                    "atom_1_id": atlas["atom_1_id"],
                    "feature_0": atlas["feature_0"],
                    "direction_0": atlas["direction_0"],
                    "thr_index_0": atlas["thr_index_0"],
                    "feature_1": atlas["feature_1"],
                    "direction_1": atlas["direction_1"],
                    "thr_index_1": atlas["thr_index_1"],
                    "grid_id": grid_id,
                }
                cid = ensure_coord_pair(
                    "G2_and", "AND", can, raw_alias=str(atlas.get("combo_id", ""))
                )
                mid = ensure_mask("G2_and", grid_id, str(atlas["mask_sha256"]))
                # margin cell_id from T0
                # reconstruct possible cell_id forms
                cell_candidates = [
                    str(atlas.get("combo_id", "")),
                    f"AND::{can['atom_0_id']}::{can['atom_1_id']}",
                    f"AND::{can['atom_1_id']}::{can['atom_0_id']}",
                ]
                margin = {}
                for cell in cell_candidates:
                    if (comp["component_id"], cell) in margin_by_comp_cell:
                        margin = margin_by_comp_cell[(comp["component_id"], cell)]
                        break
                if not margin:
                    # match by thr indices in margin file
                    for (comp_id, cell_id), m in margin_by_comp_cell.items():
                        if comp_id != comp["component_id"]:
                            continue
                        if int(m["thr_index_a"]) == i and int(m["thr_index_b"]) == j:
                            margin = m
                            break
                if not margin:
                    fail(f"missing margin for {comp['component_id']} thr=({i},{j})")
                coord_ids_for_region.append(cid)
                member_specs.append((cid, mid, atlas, margin))
                ensure_policy_instance(fam_id, coordinates[cid])

        coord_digest = hashlib.sha256(
            ("\n".join(sorted(coord_ids_for_region)) + "\n").encode("utf-8")
        ).hexdigest()
        region_body = {
            "id_scheme": ID_SCHEME,
            "kind": "region_asset",
            "truth_contract_id": truth_contract_id,
            "feasibility_contract_id": feasibility_contract_id,
            "policy_family_definition_id": fam_id,
            "grid_domain_id": gdid,
            "adjacency": "G1_bilateral" if grammar == "G1_atom" else "G2_4neighbor",
            "membership": "productive_safe",
            "coordinate_digest": coord_digest,
        }
        region_id = content_id(region_body)
        unique_masks = sorted({mid for _, mid, _, _ in member_specs})
        region_assets.append(
            {
                "region_asset_id": region_id,
                "truth_contract_id": truth_contract_id,
                "feasibility_contract_id": feasibility_contract_id,
                "policy_family_definition_id": fam_id,
                "grid_domain_id": gdid,
                "grammar": grammar,
                "grid_id": grid_id,
                "bounded_status": "HAS_REGION",
                "n_coords": n_coords,
                "n_mask_units": len(unique_masks),
                "shape_class": shape,
                "is_single_cell_width_strip": comp["is_single_cell_width_strip"],
                "is_genuine_2d_thick": comp["is_genuine_2d_thick"],
                "max_full_neighborhood_safe_radius": int(
                    comp["max_full_neighborhood_safe_radius"]
                ),
                "claim_level": claim,
                "action_state": "observation_only",
                "production_forbidden": True,
                "t0_component_id_alias": comp["component_id"],
            }
        )
        claim_counter[claim] += 1
        if grammar == "G1_atom":
            claim_counter["g1_L0"] += 1
        elif claim == "L0":
            claim_counter["g2_L0"] += 1
        else:
            claim_counter["g2_L1"] += 1

        for cid, mid, atlas, margin in member_specs:
            membership_rows.append(
                {
                    "region_asset_id": region_id,
                    "coordinate_id": cid,
                    "mask_unit_id": mid,
                    "productive_safe_point": 1,
                    "n_neg_captured": int(atlas["n_neg_captured"]),
                    "n_gt_captured": int(atlas.get("n_gt_captured", 0)),
                    "gt_hurt": int(atlas["gt_hurt"]),
                    "n_unresolved_selected": int(atlas.get("n_unresolved_selected", 0)),
                    "safety_status": str(atlas.get("safety_status", "")),
                    "nearest_unsafe_distance": margin.get(
                        "nearest_unsafe_distance", ""
                    ),
                    "nearest_unsafe_edge_censored": margin.get(
                        "nearest_unsafe_edge_censored", ""
                    ),
                    "distance_to_lattice_edge": margin.get(
                        "distance_to_lattice_edge", ""
                    ),
                    "full_neighborhood_safe_radius": margin.get(
                        "full_neighborhood_safe_radius", ""
                    ),
                    "edge_touches_lattice": margin.get("edge_touches_lattice", ""),
                    "per_sequence_neg_json": parse_seq_json(
                        atlas.get("per_sequence_neg_json")
                    ),
                    "per_sequence_gt_json": parse_seq_json(
                        atlas.get("per_sequence_gt_json")
                    ),
                    "policy_instance_id": coordinates[cid].get(
                        "policy_instance_id", ""
                    ),
                }
            )

    if (
        claim_counter["g1_L0"] != 1
        or claim_counter["g2_L0"] != 6
        or claim_counter["g2_L1"] != 19
    ):
        fail(f"claim distribution mismatch: {claim_counter}")
    if len(region_assets) != 26:
        fail(f"region count {len(region_assets)} != 26")
    if len(membership_rows) != 154:
        fail(f"membership count {len(membership_rows)} != 154")
    if len(mask_units) != EXPECTED["n_mask_units"]:
        # T0 says 34 productive per-grid mask units; verify against mask_caps
        if len(mask_units) != len(mask_caps):
            fail(
                f"mask unit count {len(mask_units)} != expected 34 (mask_caps={len(mask_caps)})"
            )

    # G3 null record
    null_body = {
        "id_scheme": ID_SCHEME,
        "kind": "null_record",
        "truth_contract_id": truth_contract_id,
        "feasibility_contract_id": feasibility_contract_id,
        "search_domain_id": sd_g3["search_domain_id"],
        "null_reason_class": "no_productive_safe_on_registered_domain",
    }
    null_id = content_id(null_body)
    null_records = [
        {
            "null_record_id": null_id,
            "truth_contract_id": truth_contract_id,
            "feasibility_contract_id": feasibility_contract_id,
            "search_domain_id": sd_g3["search_domain_id"],
            "grammar": "G3_or",
            "policy_family_definition_id": None,
            "null_reason": "no_observed_or_productive_safe_on_registered_or_lattice",
            "n_members": 40,
            "n_registered_coordinates_sum": 40 * 441,
            "n_productive_safe": 0,
            "claim_level": "L0",
            "action_state": "observation_only",
            "production_forbidden": True,
        }
    ]

    # evidence claims
    evidence_claims: list[dict[str, Any]] = []
    for ra in region_assets:
        body = {
            "kind": "evidence_claim",
            "feasibility_contract_id": feasibility_contract_id,
            "evidence_bundle_id": evidence_bundle_id,
            "content_kind": "region_asset",
            "content_id": ra["region_asset_id"],
            "claim_level": ra["claim_level"],
            "claim_scope": "object",
        }
        eid = content_id(body)
        evidence_claims.append(
            {
                "evidence_claim_id": eid,
                "feasibility_contract_id": feasibility_contract_id,
                "evidence_bundle_id": evidence_bundle_id,
                "content_kind": "region_asset",
                "content_id": ra["region_asset_id"],
                "claim_level": ra["claim_level"],
                "claim_scope": "object",
                "selection_scope_note": "in_sample_searched_and_evaluated",
                "finite_sample_statement": "observed_GT0_is_not_population_risk_zero",
                "composition_level": "observational",
                "grammar": ra["grammar"],
                "n_coords": ra["n_coords"],
            }
        )
    for nr in null_records:
        body = {
            "kind": "evidence_claim",
            "feasibility_contract_id": feasibility_contract_id,
            "evidence_bundle_id": evidence_bundle_id,
            "content_kind": "null_record",
            "content_id": nr["null_record_id"],
            "claim_level": "L0",
            "claim_scope": "object",
        }
        eid = content_id(body)
        evidence_claims.append(
            {
                "evidence_claim_id": eid,
                "feasibility_contract_id": feasibility_contract_id,
                "evidence_bundle_id": evidence_bundle_id,
                "content_kind": "null_record",
                "content_id": nr["null_record_id"],
                "claim_level": "L0",
                "claim_scope": "object",
                "selection_scope_note": "in_sample_searched_and_evaluated",
                "finite_sample_statement": "observed_GT0_is_not_population_risk_zero",
                "composition_level": "observational",
                "grammar": "G3_or",
                "n_coords": 0,
            }
        )
    pack_claim_body = {
        "kind": "evidence_claim",
        "feasibility_contract_id": feasibility_contract_id,
        "evidence_bundle_id": evidence_bundle_id,
        "content_kind": "pack",
        "content_id": "pack_ceiling",
        "claim_level": "L1",
        "claim_scope": "pack",
    }
    pack_claim_id = content_id(pack_claim_body)
    evidence_claims.append(
        {
            "evidence_claim_id": pack_claim_id,
            "feasibility_contract_id": feasibility_contract_id,
            "evidence_bundle_id": evidence_bundle_id,
            "content_kind": "pack",
            "content_id": "pack_ceiling",
            "claim_level": "L1",
            "claim_scope": "pack",
            "selection_scope_note": "in_sample_searched_and_evaluated",
            "finite_sample_statement": "observed_GT0_is_not_population_risk_zero",
            "composition_level": "observational",
            "grammar": "G1_G2_G3",
            "n_coords": 154,
        }
    )

    # pack manifest
    pack_body = {
        "kind": "pack",
        "truth_contract_id": truth_contract_id,
        "feasibility_contract_id": feasibility_contract_id,
        "evidence_bundle_id": evidence_bundle_id,
        "candidate_universe_instance_id": uni_instance_id,
        "threshold_registry_id": thr_registry_id,
        "producer_kind": PRODUCER_KIND,
        "producer_contract_version": PRODUCER_CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "grammar_scope": "G1_G2_G3",
        "maturity_declared": "A0",
        "pack_claim_ceiling": "L1",
        "action_state_default": "observation_only",
        "production_forbidden": True,
        "terminal_letter": "B",
        "composition_level": "observational",
        "n_non_null_region_assets": len(region_assets),
        "n_null_records": len(null_records),
        "n_coordinates": len(coordinates),
        "n_mask_units": len(mask_units),
        "n_membership_rows": len(membership_rows),
        "n_policy_instances": len(policy_instances),
        "study_pack_id": PACK_STUDY_ID,
    }
    pack_id = content_id(pack_body)
    manifest = {
        "pack_id": pack_id,
        **{k: v for k, v in pack_body.items() if k != "kind"},
        "contract_version": CONTRACT_VERSION,
        "id_scheme": ID_SCHEME,
        "not_scope": "none",
        "g7_roles": "not_inferred",
        "final_unknown_action": "no_reject",
        "review_status": "A0_PACK_CANDIDATE_AWAITING_CHAT_REVIEW",
        "converter": "scripts/tools/convert_safe_region_asset_r1.py",
        "accepted_counts": {
            "n_productive_safe": 154,
            "n_regions": 26,
            "g1_L0": 1,
            "g2_L0_isolated": 6,
            "g2_L1_multi": 19,
            "g3_L0_null": 1,
            "mask_units": len(mask_units),
        },
    }

    claim_contract = {
        "pack_id": pack_id,
        "pack_claim_ceiling": "L1",
        "maturity_declared": "A0",
        "action_states_allowed": ["observation_only"],
        "action_states_forbidden": [
            "shadow_decision",
            "condition_model",
            "offline_filter",
            "default_off_intervention",
            "production",
        ],
        "production_forbidden": True,
        "forbidden_promotions": [
            "A0_to_A1_self_accept",
            "L1_pack_ceiling_to_every_object_L1",
            "G2_grammar_to_every_G2_L1",
            "safe_region_to_production_policy",
            "observed_GT0_to_population_risk_zero",
            "generator_contract_equality_to_same_universe_instance",
            "source_event_table_sha_to_membership_digest",
            "policy_family_to_concrete_threshold_policy",
            "thr_index_without_registry_to_thr_value",
        ],
        "terminal_b": True,
        "g7_status": "not_inferred",
        "identity_layer_policy": "region_asset_id_v2",
        "capacity_policy": "non_additive_dual_distributions",
        "sequence_policy": "union_and_intersection_when_multi_member",
        "claim_ownership_policy": "feasibility_excludes_claim_level",
        "claim_level_derivation_policy": "geometry_not_grammar_wide",
        "composition_level_policy": "observational_only",
        "policy_equivalence_policy": "family_ast_ne_instance_ne_mask",
        "realization_vs_content_policy": "RB5",
        "universe_instance_policy": "RB8_model_B",
        "threshold_registry_policy": "RB9",
    }

    # pack membership
    pack_membership: list[dict[str, Any]] = []

    def add_pm(kind: str, cid: str) -> None:
        pack_membership.append(
            {"pack_id": pack_id, "content_kind": kind, "content_id": cid}
        )

    add_pm("truth_contract", truth_contract_id)
    add_pm("feasibility_contract", feasibility_contract_id)
    add_pm("evidence_bundle", evidence_bundle_id)
    add_pm("candidate_universe_contract", uni_contract_id)
    add_pm("candidate_universe_instance", uni_instance_id)
    add_pm("threshold_registry", thr_registry_id)
    for p in preds:
        add_pm("predicate_definition", p["predicate_id"])
    for fam in families.values():
        add_pm("policy_family", fam["policy_family_definition_id"])
    for pi in policy_instances.values():
        add_pm("policy_instance", pi["policy_instance_id"])
    for ra in region_assets:
        add_pm("region_asset", ra["region_asset_id"])
    for nr in null_records:
        add_pm("null_record", nr["null_record_id"])
    for cid in coordinates:
        add_pm("coordinate", cid)
    for mid in mask_units:
        add_pm("mask_unit", mid)
    for m in membership_rows:
        add_pm(
            "membership",
            content_id(
                {
                    "region_asset_id": m["region_asset_id"],
                    "coordinate_id": m["coordinate_id"],
                }
            ),
        )

    # sort stable outputs
    family_rows = sorted(families.values(), key=lambda r: (r["grammar"], r["grid_id"]))
    coord_rows = sorted(coordinates.values(), key=lambda r: r["canonical_cell_key"])
    mask_rows = sorted(
        mask_units.values(),
        key=lambda r: (r["grammar"], r["grid_id"], r["mask_sha256"]),
    )
    pi_rows = sorted(policy_instances.values(), key=lambda r: r["policy_instance_id"])
    region_assets = sorted(
        region_assets, key=lambda r: (r["grammar"], r["grid_id"], r["region_asset_id"])
    )
    membership_rows = sorted(
        membership_rows, key=lambda r: (r["region_asset_id"], r["coordinate_id"])
    )
    thr_entries_sorted = sorted(
        thr_entries,
        key=lambda e: (e["lattice_kind"], e["feature"], e["direction"], e["thr_index"]),
    )
    grid_domains = sorted(grid_domains, key=lambda r: (r["grammar"], r["grid_id"]))
    pack_membership = sorted(
        pack_membership, key=lambda r: (r["content_kind"], r["content_id"])
    )
    evidence_claims = sorted(evidence_claims, key=lambda r: r["evidence_claim_id"])
    preds = sorted(preds, key=lambda r: (r["signal_identity"], r["direction"]))

    # write authorities
    write_json(out_dir / "truth_contract.json", truth_contract)
    write_json(out_dir / "candidate_universe_contracts.json", [uni_contract])
    write_json(out_dir / "candidate_universe_instances.json", [uni_instance])
    write_json(out_dir / "threshold_registry.json", thr_reg)
    write_csv(
        out_dir / "threshold_registry_entries.csv",
        thr_entries_sorted,
        [
            "threshold_registry_entry_id",
            "threshold_registry_id",
            "lattice_kind",
            "feature",
            "direction",
            "thr_index",
            "atom_id",
            "threshold_value_repr",
            "thr_value",
            "quantile_lattice_point",
            "scope",
        ],
    )
    write_jsonl(out_dir / "predicate_definitions.jsonl", preds)
    write_jsonl(out_dir / "policy_family_definitions.jsonl", family_rows)
    write_jsonl(out_dir / "policy_instances.jsonl", pi_rows)
    write_json(out_dir / "evidence_bundle.json", evidence_bundle)
    write_json(out_dir / "feasibility_contract.json", feasibility_contract)
    write_jsonl(out_dir / "evidence_claims.jsonl", evidence_claims)
    write_csv(
        out_dir / "grid_domains.csv",
        grid_domains,
        [
            "grid_domain_id",
            "lattice_kind",
            "grid_id",
            "grammar_family",
            "grammar",
            "axis_descriptors",
            "n_registered_coordinates",
        ],
    )
    write_csv(
        out_dir / "search_domains.csv",
        search_domains,
        [
            "search_domain_id",
            "truth_contract_id",
            "grammar",
            "combinator",
            "lattice_kind",
            "n_members",
            "n_registered_coordinates_sum",
            "membership_digest",
        ],
    )
    write_csv(
        out_dir / "search_domain_members.csv",
        search_domain_members,
        [
            "search_domain_id",
            "grid_domain_id",
            "policy_family_definition_id",
            "grid_id",
            "n_registered_coordinates",
        ],
    )
    write_csv(
        out_dir / "region_assets.csv",
        region_assets,
        [
            "region_asset_id",
            "truth_contract_id",
            "feasibility_contract_id",
            "policy_family_definition_id",
            "grid_domain_id",
            "grammar",
            "grid_id",
            "bounded_status",
            "n_coords",
            "n_mask_units",
            "shape_class",
            "is_single_cell_width_strip",
            "is_genuine_2d_thick",
            "max_full_neighborhood_safe_radius",
            "claim_level",
            "action_state",
            "production_forbidden",
            "t0_component_id_alias",
        ],
    )
    write_csv(
        out_dir / "null_records.csv",
        null_records,
        [
            "null_record_id",
            "truth_contract_id",
            "feasibility_contract_id",
            "search_domain_id",
            "grammar",
            "policy_family_definition_id",
            "null_reason",
            "n_members",
            "n_registered_coordinates_sum",
            "n_productive_safe",
            "claim_level",
            "action_state",
            "production_forbidden",
        ],
    )
    write_csv(
        out_dir / "coordinates.csv",
        coord_rows,
        [
            "coordinate_id",
            "truth_contract_id",
            "grid_domain_id",
            "threshold_registry_id",
            "canonical_cell_key",
            "grammar",
            "grid_id",
            "thr_index_0",
            "thr_index_1",
            "threshold_registry_entry_id_0",
            "threshold_registry_entry_id_1",
            "feature_0",
            "direction_0",
            "feature_1",
            "direction_1",
            "policy_instance_id",
            "raw_cell_id_alias",
        ],
    )
    write_csv(
        out_dir / "mask_units.csv",
        mask_rows,
        [
            "mask_unit_id",
            "truth_contract_id",
            "grid_domain_id",
            "grid_id",
            "mask_sha256",
            "grammar",
        ],
    )
    write_csv(
        out_dir / "region_coordinate_membership.csv",
        membership_rows,
        [
            "region_asset_id",
            "coordinate_id",
            "mask_unit_id",
            "productive_safe_point",
            "n_neg_captured",
            "n_gt_captured",
            "gt_hurt",
            "n_unresolved_selected",
            "safety_status",
            "nearest_unsafe_distance",
            "nearest_unsafe_edge_censored",
            "distance_to_lattice_edge",
            "full_neighborhood_safe_radius",
            "edge_touches_lattice",
            "per_sequence_neg_json",
            "per_sequence_gt_json",
            "policy_instance_id",
        ],
    )
    write_json(out_dir / "region_asset_manifest.json", manifest)
    write_csv(
        out_dir / "pack_membership.csv",
        pack_membership,
        ["pack_id", "content_kind", "content_id"],
    )
    write_json(out_dir / "region_claim_contract.json", claim_contract)

    # derived region_mask_link
    link_keys = sorted(
        {(m["region_asset_id"], m["mask_unit_id"]) for m in membership_rows}
    )
    write_csv(
        out_dir / "region_mask_link.csv",
        [{"region_asset_id": a, "mask_unit_id": b} for a, b in link_keys],
        ["region_asset_id", "mask_unit_id"],
    )

    validation = validate_pack(out_dir)
    write_json(out_dir / "validation_report.json", validation)
    if not validation["ok"]:
        fail("pack validation failed: " + "; ".join(validation["errors"][:20]))

    result = {
        "status": "A0_PACK_CANDIDATE_AWAITING_CHAT_REVIEW",
        "out_dir": str(out_dir),
        "pack_id": pack_id,
        "truth_contract_id": truth_contract_id,
        "feasibility_contract_id": feasibility_contract_id,
        "evidence_bundle_id": evidence_bundle_id,
        "candidate_universe_contract_id": uni_contract_id,
        "candidate_universe_instance_id": uni_instance_id,
        "universe_membership_digest": uni_instance["universe_membership_digest"],
        "threshold_registry_id": thr_registry_id,
        "counts": manifest["accepted_counts"],
        "validation": validation,
        "preflight": {
            "status": report["status"],
            "n_seals": len(report["verified_seals"]),
        },
    }
    write_json(out_dir / "conversion_summary.json", result)
    return result


def validate_pack(out_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    def loadj(name: str) -> Any:
        return json.loads((out_dir / name).read_text(encoding="utf-8"))

    def load_csv(name: str) -> list[dict[str, str]]:
        with (out_dir / name).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def load_jsonl(name: str) -> list[dict[str, Any]]:
        rows = []
        with (out_dir / name).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    required = [
        "truth_contract.json",
        "candidate_universe_contracts.json",
        "candidate_universe_instances.json",
        "threshold_registry.json",
        "threshold_registry_entries.csv",
        "predicate_definitions.jsonl",
        "policy_family_definitions.jsonl",
        "policy_instances.jsonl",
        "evidence_bundle.json",
        "feasibility_contract.json",
        "evidence_claims.jsonl",
        "grid_domains.csv",
        "search_domains.csv",
        "search_domain_members.csv",
        "region_assets.csv",
        "null_records.csv",
        "coordinates.csv",
        "mask_units.csv",
        "region_coordinate_membership.csv",
        "region_asset_manifest.json",
        "pack_membership.csv",
        "region_claim_contract.json",
    ]
    for name in required:
        if not (out_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    man = loadj("region_asset_manifest.json")
    truth = loadj("truth_contract.json")
    feas = loadj("feasibility_contract.json")
    evid = loadj("evidence_bundle.json")
    uni_c = loadj("candidate_universe_contracts.json")[0]
    uni_i = loadj("candidate_universe_instances.json")[0]
    thr = loadj("threshold_registry.json")
    thr_e = load_csv("threshold_registry_entries.csv")
    preds = load_jsonl("predicate_definitions.jsonl")
    fams = load_jsonl("policy_family_definitions.jsonl")
    pis = load_jsonl("policy_instances.jsonl")
    regions = load_csv("region_assets.csv")
    nulls = load_csv("null_records.csv")
    coords = load_csv("coordinates.csv")
    masks = load_csv("mask_units.csv")
    memb = load_csv("region_coordinate_membership.csv")
    claims = load_jsonl("evidence_claims.jsonl")
    sdm = load_csv("search_domain_members.csv")
    claim_c = loadj("region_claim_contract.json")

    # PK uniqueness
    def uniq(rows, key, label):
        vals = [r[key] for r in rows]
        if len(vals) != len(set(vals)):
            errors.append(f"duplicate PK in {label}")
        return set(vals)

    region_ids = uniq(regions, "region_asset_id", "region_assets")
    coord_ids = uniq(coords, "coordinate_id", "coordinates")
    mask_ids = uniq(masks, "mask_unit_id", "mask_units")
    fam_ids = uniq(fams, "policy_family_definition_id", "policy_families")
    pi_ids = uniq(pis, "policy_instance_id", "policy_instances")
    uniq(preds, "predicate_id", "predicates")
    entry_ids = uniq(thr_e, "threshold_registry_entry_id", "threshold_entries")
    uniq(claims, "evidence_claim_id", "evidence_claims")

    memb_pks = [(r["region_asset_id"], r["coordinate_id"]) for r in memb]
    if len(memb_pks) != len(set(memb_pks)):
        errors.append("duplicate membership PK")

    # FK resolution
    for r in regions:
        if r["truth_contract_id"] != truth["truth_contract_id"]:
            errors.append("region truth FK mismatch")
        if r["feasibility_contract_id"] != feas["feasibility_contract_id"]:
            errors.append("region feasibility FK mismatch")
        if r["policy_family_definition_id"] not in fam_ids:
            errors.append(
                f"region family FK missing {r['policy_family_definition_id'][:12]}"
            )
    for c in coords:
        if c["threshold_registry_id"] != thr["threshold_registry_id"]:
            errors.append("coord registry FK mismatch")
        if c["threshold_registry_entry_id_0"] not in entry_ids:
            errors.append("coord entry0 FK missing")
        if c.get("threshold_registry_entry_id_1"):
            if c["threshold_registry_entry_id_1"] not in entry_ids:
                errors.append("coord entry1 FK missing")
        # forbidden fields on coordinates
        for banned in (
            "productive_safe_point",
            "region_asset_id",
            "feasibility_contract_id",
        ):
            if banned in c and c[banned] not in ("", None):
                errors.append(f"forbidden field {banned} on coordinates")
    for m in memb:
        if m["region_asset_id"] not in region_ids:
            errors.append("membership region FK missing")
        if m["coordinate_id"] not in coord_ids:
            errors.append("membership coord FK missing")
        if m["mask_unit_id"] not in mask_ids:
            errors.append("membership mask FK missing")
    for n in nulls:
        if n.get("policy_family_definition_id") not in ("", None, "null", "None"):
            errors.append("G3 null must have null policy_family_definition_id")
        if n["claim_level"] != "L0":
            errors.append("G3 null claim_level must be L0")
    for pi in pis:
        if pi["policy_family_definition_id"] not in fam_ids:
            errors.append("policy_instance family FK missing")
        if pi["threshold_registry_id"] != thr["threshold_registry_id"]:
            errors.append("policy_instance registry FK missing")
    for cl in claims:
        if cl["feasibility_contract_id"] != feas["feasibility_contract_id"]:
            errors.append("claim feasibility FK mismatch")
        if cl["evidence_bundle_id"] != evid["evidence_bundle_id"]:
            errors.append("claim evidence FK mismatch")

    # counts / claims
    if len(regions) != 26:
        errors.append(f"n_regions={len(regions)} expected 26")
    if len(memb) != 154:
        errors.append(f"n_membership={len(memb)} expected 154")
    if len(nulls) != 1:
        errors.append(f"n_nulls={len(nulls)} expected 1")
    g1 = [r for r in regions if r["grammar"] == "G1_atom"]
    g2 = [r for r in regions if r["grammar"] == "G2_and"]
    g2_l0 = [r for r in g2 if r["claim_level"] == "L0"]
    g2_l1 = [r for r in g2 if r["claim_level"] == "L1"]
    if len(g1) != 1 or g1[0]["claim_level"] != "L0":
        errors.append("G1 claim/count mismatch")
    if len(g2_l0) != 6 or len(g2_l1) != 19:
        errors.append(f"G2 claims L0={len(g2_l0)} L1={len(g2_l1)}")
    if any(r["claim_level"] not in ("L0", "L1") for r in regions):
        errors.append("unexpected claim_level")

    # RB8
    if (
        uni_i["candidate_universe_contract_id"]
        != uni_c["candidate_universe_contract_id"]
    ):
        errors.append("universe instance/contract FK mismatch")
    if uni_i["universe_membership_digest"] != uni_i.get("universe_hash"):
        errors.append("universe_hash alias mismatch")
    if uni_i["universe_membership_digest"] == evid["source_event_table_sha256"]:
        errors.append("membership digest must not equal source_event_table_sha256")
    if uni_i["membership_status"] != "SEALED":
        errors.append("membership_status not SEALED")
    if "claim_level" in feas:
        errors.append("feasibility must not contain claim_level")

    # RB9
    if thr["n_single_atoms"] != "870" and thr["n_single_atoms"] != 870:
        if int(thr["n_single_atoms"]) != 870:
            errors.append("n_single_atoms != 870")
    if int(thr["n_pairwise_atoms"]) != 210:
        errors.append("n_pairwise_atoms != 210")
    if len(thr_e) != 870 + 210:
        errors.append(f"threshold entries {len(thr_e)} != 1080")

    # G3 search members
    g3_sd = [r for r in load_csv("search_domains.csv") if r["grammar"] == "G3_or"]
    if len(g3_sd) != 1 or int(g3_sd[0]["n_members"]) != 40:
        errors.append("G3 search domain members != 40")
    g3_members = [
        r for r in sdm if r["search_domain_id"] == g3_sd[0]["search_domain_id"]
    ]
    if len(g3_members) != 40:
        errors.append("G3 search_domain_members rows != 40")

    # manifest firewalls
    if man["maturity_declared"] != "A0":
        errors.append("maturity_declared != A0")
    if man["composition_level"] != "observational":
        errors.append("composition_level != observational")
    if man["production_forbidden"] not in (True, "True", "true", 1, "1"):
        errors.append("production_forbidden not true")
    if man["terminal_letter"] != "B":
        errors.append("terminal_letter != B")
    if man["pack_claim_ceiling"] != "L1":
        errors.append("pack_claim_ceiling != L1")
    if claim_c["pack_id"] != man["pack_id"]:
        errors.append("claim_contract pack_id mismatch")

    # region_mask_link derived check
    link = load_csv("region_mask_link.csv")
    derived = sorted({(m["region_asset_id"], m["mask_unit_id"]) for m in memb})
    got = sorted((r["region_asset_id"], r["mask_unit_id"]) for r in link)
    if derived != got:
        errors.append("region_mask_link != DISTINCT membership")

    # pairwise swap invariance smoke: all G2 coords have feature_0 <= feature_1/dir order
    for c in coords:
        if c["grammar"] != "G2_and":
            continue
        a = (c["feature_0"], c["direction_0"])
        b = (c["feature_1"], c["direction_1"])
        if a > b:
            errors.append(
                f"non-canonical pairwise coord axis order {c['canonical_cell_key']}"
            )

    # policy family != policy instance
    if fam_ids & pi_ids:
        errors.append("family ids collided with instance ids")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "regions": len(regions),
            "membership": len(memb),
            "coordinates": len(coords),
            "masks": len(masks),
            "policy_instances": len(pis),
            "policy_families": len(fams),
            "threshold_entries": len(thr_e),
            "claims": len(claims),
            "g1_L0": len(g1),
            "g2_L0": len(g2_l0),
            "g2_L1": len(g2_l1),
            "nulls": len(nulls),
        },
        "pack_id": man["pack_id"],
        "universe_membership_digest": uni_i["universe_membership_digest"],
        "source_event_table_sha256": evid["source_event_table_sha256"],
    }


def authority_fingerprint(out_dir: Path) -> str:
    """Content fingerprint over authority files for two-run determinism."""
    names = [
        "truth_contract.json",
        "candidate_universe_contracts.json",
        "candidate_universe_instances.json",
        "threshold_registry.json",
        "threshold_registry_entries.csv",
        "predicate_definitions.jsonl",
        "policy_family_definitions.jsonl",
        "policy_instances.jsonl",
        "evidence_bundle.json",
        "feasibility_contract.json",
        "evidence_claims.jsonl",
        "grid_domains.csv",
        "search_domains.csv",
        "search_domain_members.csv",
        "region_assets.csv",
        "null_records.csv",
        "coordinates.csv",
        "mask_units.csv",
        "region_coordinate_membership.csv",
        "region_asset_manifest.json",
        "pack_membership.csv",
        "region_claim_contract.json",
    ]
    h = hashlib.sha256()
    for name in names:
        p = out_dir / name
        h.update(name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "out/signal_study/m_b1_5_safe_region_asset_r1_20260710",
    )
    ap.add_argument("--q45", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)

    paths = default_paths()
    if args.q45:
        paths = Paths(
            q45=args.q45,
            t0_evidence=paths.t0_evidence,
            q45_evidence=paths.q45_evidence,
            candidate_events=args.candidates or paths.candidate_events,
            contract=paths.contract,
            boolean_contract=paths.boolean_contract,
        )
    if args.candidates:
        paths = Paths(
            q45=paths.q45,
            t0_evidence=paths.t0_evidence,
            q45_evidence=paths.q45_evidence,
            candidate_events=args.candidates,
            contract=paths.contract,
            boolean_contract=paths.boolean_contract,
        )

    if args.validate_only:
        rep = validate_pack(args.out)
        print(canonical_json(rep))
        return 0 if rep["ok"] else 2

    if args.preflight_only:
        rep = preflight(paths)
        print(canonical_json(rep))
        return 0 if rep["status"] == "OK" else 2

    result = convert(paths, args.out)
    print(canonical_json({k: result[k] for k in result if k != "validation"}))
    if result.get("status") not in ("A0_PACK_CANDIDATE_AWAITING_CHAT_REVIEW",):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
