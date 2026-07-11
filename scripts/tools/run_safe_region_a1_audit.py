#!/usr/bin/env python
"""A1 read-only acceptance audit for the safe-region A0 conversion pack.

Implements the chat-review minimal takeover plan (2026-07-10/11):

  S0  acceptance-unit lock   — one pack_id; pack declares A0; awaiting review
  S1  semantic crosswalk     — pack tables vs sealed Q4.5 / T0 oracle artifacts
                               (one-time raw readback; oracle never used in Q1)
  Q1  pack-only battery      — predeclared golden answers from the accepted
                               A0 baseline; answered from pack tables only
  N1  negative controls      — pack encodes the forbidden promotions

Read-only: never writes into the pack, evidence dirs, or probe studies.
Emits an audit report JSON + per-check table under --out.

Usage:
  .venv/bin/python scripts/tools/run_safe_region_a1_audit.py \
    --pack out/signal_study/m_b1_5_safe_region_asset_r1_20260710 \
    --t0-evidence docs/modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710 \
    --q45-evidence docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710 \
    --out out/signal_study/safe_region_a1_audit_20260711
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

# Predeclared golden answers: the accepted A0 baseline recorded in
# docs/research/threads/closed/safe_region_assetization_20260710.md BEFORE this audit.
GOLDEN = {
    "n_productive_safe": 154,
    "split_g1_g2_g3": (1, 153, 0),
    "n_regions": 26,
    "regions_g1_g2": (1, 25),
    "g2_isolated_L0": 6,
    "g2_multi_L1": 19,
    "n_grid_local_mask_assets": 34,
    "n_unique_prediction_masks": 15,
    "n_null_records": 1,
    "pack_claim_ceiling": "L1",
    "lattice_n": {"G1_atom": 870, "G2_and": 17640, "G3_or": 17640},
    "n_fp_exposed": 23,
    "n_gt_exposed": 64,
    "sequence_set": {
        "MOT17-02-SDP",
        "MOT17-04-SDP",
        "MOT17-05-SDP",
        "MOT17-09-SDP",
        "MOT17-10-SDP",
        "MOT17-11-SDP",
        "MOT17-13-SDP",
    },
}

REQUIRED_FORBIDDEN_PROMOTIONS = {
    "A0_to_A1_self_accept",
    "L1_pack_ceiling_to_every_object_L1",
    "safe_region_to_production_policy",
    "observed_GT0_to_population_risk_zero",
    "policy_family_to_concrete_threshold_policy",
    "thr_index_without_registry_to_thr_value",
}


def _check(results: list, check_id: str, section: str, ok: bool, detail: str) -> None:
    results.append(
        {
            "check_id": check_id,
            "section": section,
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }
    )


def _load_pack(pack: Path) -> dict:
    return {
        "pack_root": str(pack),
        "manifest": json.loads((pack / "region_asset_manifest.json").read_text()),
        "claim": json.loads((pack / "region_claim_contract.json").read_text()),
        "truth": json.loads((pack / "truth_contract.json").read_text()),
        "feas": json.loads((pack / "feasibility_contract.json").read_text()),
        "thr": json.loads((pack / "threshold_registry.json").read_text()),
        "regions": pd.read_csv(pack / "region_assets.csv"),
        "members": pd.read_csv(pack / "region_coordinate_membership.csv"),
        "masks": pd.read_csv(pack / "mask_units.csv"),
        "nulls": pd.read_csv(pack / "null_records.csv"),
        "coords": pd.read_csv(pack / "coordinates.csv"),
        "pack_rows": pd.read_csv(pack / "pack_membership.csv"),
    }


def audit_s0(p: dict, results: list) -> None:
    man, claim = p["manifest"], p["claim"]
    pid = man.get("pack_id")
    ok = bool(pid) and claim.get("pack_id") == pid
    _check(results, "S0.1", "S0", ok, f"single pack_id declared: {pid}")
    ids = set(p["pack_rows"]["pack_id"].unique())
    _check(
        results,
        "S0.2",
        "S0",
        ids == {pid},
        f"pack_membership rows all under pack_id ({sorted(ids)[:2]})",
    )
    _check(
        results,
        "S0.3",
        "S0",
        man.get("maturity_declared") == "A0" and claim.get("maturity_declared") == "A0",
        "pack self-declares maturity A0 (no self-promotion)",
    )
    _check(
        results,
        "S0.4",
        "S0",
        man.get("review_status") == "A0_PACK_CANDIDATE_AWAITING_CHAT_REVIEW",
        f"review_status={man.get('review_status')}",
    )


def audit_s1(p: dict, t0: Path, q45: Path, results: list) -> None:
    reg, mem, masks, nulls, coords = (
        p["regions"],
        p["members"],
        p["masks"],
        p["nulls"],
        p["coords"],
    )

    # --- internal counts vs golden ---
    _check(
        results,
        "S1.1",
        "S1",
        len(mem) == GOLDEN["n_productive_safe"]
        and int(mem["productive_safe_point"].sum()) == GOLDEN["n_productive_safe"]
        and int(mem["gt_hurt"].sum()) == 0,
        f"membership rows={len(mem)}, all productive-safe, gt_hurt sum=0",
    )
    gsplit = reg.groupby("grammar").size().to_dict()
    _check(
        results,
        "S1.2",
        "S1",
        len(reg) == GOLDEN["n_regions"]
        and gsplit.get("G1_atom", 0) == 1
        and gsplit.get("G2_and", 0) == 25,
        f"regions={len(reg)} split={gsplit}",
    )
    g2 = reg[reg["grammar"] == "G2_and"]
    lv = g2.groupby("claim_level").size().to_dict()
    iso_ok = (
        lv.get("L0", 0) == GOLDEN["g2_isolated_L0"]
        and lv.get("L1", 0) == GOLDEN["g2_multi_L1"]
        and (g2.loc[g2["claim_level"] == "L1", "n_coords"] > 1).all()
        and (g2.loc[g2["claim_level"] == "L0", "n_coords"] == 1).all()
    )
    _check(results, "S1.3", "S1", iso_ok, f"G2 claim levels {lv}; L1⇔multi-coord")
    uniq = masks["mask_sha256"].nunique()
    g1_sha = set(masks.loc[masks["grammar"] == "G1_atom", "mask_sha256"])
    g2_sha = set(masks.loc[masks["grammar"] == "G2_and", "mask_sha256"])
    _check(
        results,
        "S1.4",
        "S1",
        len(masks) == GOLDEN["n_grid_local_mask_assets"]
        and uniq == GOLDEN["n_unique_prediction_masks"]
        and g1_sha <= g2_sha,
        f"mask units={len(masks)}, unique sha={uniq}, G1⊂G2={g1_sha <= g2_sha}",
    )
    _check(
        results,
        "S1.5",
        "S1",
        len(nulls) == 1
        and nulls.iloc[0]["grammar"] == "G3_or"
        and int(nulls.iloc[0]["n_productive_safe"]) == 0
        and int(nulls.iloc[0]["n_registered_coordinates_sum"])
        == GOLDEN["lattice_n"]["G3_or"],
        "one G3 domain-null over 17640 registered coords, 0 productive",
    )
    _check(
        results,
        "S1.6",
        "S1",
        (reg["max_full_neighborhood_safe_radius"] == 0).all()
        and (mem["full_neighborhood_safe_radius"] == 0).all()
        and (mem["nearest_unsafe_distance"] > 0).all(),
        "thin-strip geometry: radius 0 everywhere, nearest-unsafe > 0",
    )

    # --- oracle crosswalk: T0 component geometry ---
    geo = pd.read_csv(t0 / "component_geometry.csv")
    j = reg.merge(
        geo,
        left_on="t0_component_id_alias",
        right_on="component_id",
        how="left",
        suffixes=("_pack", "_t0"),
    )
    unmatched = int(j["component_id"].isna().sum())
    mism = j[
        (j["n_coords_pack"] != j["n_coords_t0"])
        | (j["shape_class_pack"] != j["shape_class_t0"])
        | (
            j["max_full_neighborhood_safe_radius_pack"]
            != j["max_full_neighborhood_safe_radius_t0"]
        )
    ]
    _check(
        results,
        "S1.7",
        "S1",
        unmatched == 0 and len(mism) == 0 and len(geo) == GOLDEN["n_regions"],
        f"26 regions ↔ 26 T0 components; n_coords/shape/radius mismatches={len(mism)}",
    )

    # --- oracle crosswalk: per-coordinate capture counts + per-seq support ---
    cap = pd.read_csv(t0 / "productive_capacity.csv")
    mc = mem.merge(coords, on="coordinate_id", how="left")
    mc = mc.merge(
        cap,
        left_on=["grammar", "raw_cell_id_alias"],
        right_on=["grammar", "cell_id"],
        how="left",
        suffixes=("_pack", "_t0"),
    )
    unmatched = int(mc["cell_id"].isna().sum())
    bad_counts = mc[
        (mc["n_neg_captured_pack"] != mc["n_neg_captured_t0"])
        | (mc["n_gt_captured_pack"] != mc["n_gt_captured_t0"])
    ]
    seq_bad = 0
    for _, r in mc.iterrows():
        if pd.isna(r["cell_id"]):
            continue
        pack_seq = json.loads(r["per_sequence_neg_json"])
        t0_seq = json.loads(r["productive_sequences_json"])
        if {k: int(v) for k, v in pack_seq.items()} != {
            k: int(v) for k, v in t0_seq.items()
        }:
            seq_bad += 1
    _check(
        results,
        "S1.8",
        "S1",
        unmatched == 0 and len(bad_counts) == 0 and seq_bad == 0,
        f"154 coords ↔ T0 capacity: unmatched={unmatched}, "
        f"count mismatches={len(bad_counts)}, per-seq json mismatches={seq_bad}",
    )
    pack_sha = set(masks["mask_sha256"])
    t0_sha = set(cap["mask_sha256"])
    _check(
        results,
        "S1.9",
        "S1",
        pack_sha == t0_sha,
        f"unique prediction-mask sha sets equal (|pack|={len(pack_sha)}, |t0|={len(t0_sha)})",
    )

    # --- oracle crosswalk: sealed Q4.5 threshold registry ---
    # The pack registry is a derived envelope (id + digests), not a copy:
    # check the seal, the shared keys, and full entry-table reconstruction.
    raw = (q45 / "threshold_registry.json").read_bytes()
    q45_thr = json.loads(raw)
    seal_ok = p["thr"].get("source_file_sha256") == hashlib.sha256(raw).hexdigest()
    shared_ok = all(
        json.dumps(p["thr"][k], sort_keys=True)
        == json.dumps(q45_thr[k], sort_keys=True)
        for k in set(p["thr"]) & set(q45_thr)
    )
    ent = pd.read_csv(Path(p["pack_root"]) / "threshold_registry_entries.csv")
    atoms = {s["atom_id"]: s for s in q45_thr["single_atoms"]}
    atoms.update({a["atom_id"]: a for a in q45_thr.get("pairwise_atoms", [])})
    bad = unmatched = 0
    for _, r in ent.iterrows():
        src = atoms.get(r["atom_id"])
        if src is None:
            unmatched += 1
            continue
        if (
            abs(float(src["thr_value"]) - float(r["thr_value"])) > 1e-12
            or src["feature"] != r["feature"]
            or src["direction"] != r["direction"]
            or int(src["thr_index"]) != int(r["thr_index"])
        ):
            bad += 1
    _check(
        results,
        "S1.10",
        "S1",
        seal_ok and shared_ok and unmatched == 0 and bad == 0,
        f"registry seal={seal_ok}, shared keys equal={shared_ok}, "
        f"{len(ent)} entries reconstruct sealed thr values (unmatched={unmatched}, bad={bad})",
    )
    seqs = set(p["truth"]["sequence_set"])
    used = set()
    for s in mem["per_sequence_neg_json"]:
        used |= set(json.loads(s))
    _check(
        results,
        "S1.11",
        "S1",
        seqs == GOLDEN["sequence_set"] and used <= seqs,
        f"7-sequence cohort; membership per-seq keys ⊆ cohort ({len(used)} used)",
    )


def audit_q1(p: dict, results: list) -> None:
    """Pack-only battery. No oracle files may be read here."""
    reg, mem, masks, nulls = p["regions"], p["members"], p["masks"], p["nulls"]
    coords = p["coords"]

    per_grammar = coords.groupby("grammar").size().to_dict()
    got = (
        per_grammar.get("G1_atom", 0),
        per_grammar.get("G2_and", 0),
        per_grammar.get("G3_or", 0),
    )
    _check(
        results,
        "Q1.topology",
        "Q1",
        got == GOLDEN["split_g1_g2_g3"]
        and len(reg) == GOLDEN["n_regions"]
        and reg.groupby("grammar").size().get("G2_and", 0) == 25,
        f"PS coords by grammar {got}; components 1+25",
    )
    feas_spaces = {
        s["grammar"]: s["n_coords"]
        for s in p["feas"]["parameter_or_policy_space"]["spaces"]
    }
    _check(
        results,
        "Q1.capacity",
        "Q1",
        feas_spaces == GOLDEN["lattice_n"]
        and p["feas"]["n_fp_exposed"] == GOLDEN["n_fp_exposed"]
        and p["feas"]["n_gt_exposed"] == GOLDEN["n_gt_exposed"],
        f"dual capacity denominators {feas_spaces}, cohort neg/gt {p['feas']['n_fp_exposed']}/{p['feas']['n_gt_exposed']}",
    )
    # sequence union / intersection computable pack-only for every region
    union_fail = 0
    for rid, grp in mem.groupby("region_asset_id"):
        seq_sets = [set(json.loads(s)) for s in grp["per_sequence_neg_json"]]
        u = set().union(*seq_sets)
        i = set.intersection(*seq_sets)
        if not (u >= i and u <= GOLDEN["sequence_set"]):
            union_fail += 1
    _check(
        results,
        "Q1.sequence",
        "Q1",
        union_fail == 0 and mem["region_asset_id"].nunique() == GOLDEN["n_regions"],
        f"per-region sequence union/intersection computable pack-only for all {mem['region_asset_id'].nunique()} regions",
    )
    dup = len(masks) - masks["mask_sha256"].nunique()
    _check(
        results,
        "Q1.grain",
        "Q1",
        len(masks) == 34 and masks["mask_sha256"].nunique() == 15 and dup == 19,
        "duplicate-mask grain answerable pack-only: 34 grid-local units → 15 unique masks",
    )
    _check(
        results,
        "Q1.null",
        "Q1",
        len(nulls) == 1 and nulls.iloc[0]["claim_level"] == "L0",
        "G3 null asset present with L0 claim",
    )


def audit_n1(p: dict, results: list) -> None:
    claim, man, mem, feas = p["claim"], p["manifest"], p["members"], p["feas"]
    fp = set(claim.get("forbidden_promotions", []))
    missing = REQUIRED_FORBIDDEN_PROMOTIONS - fp
    _check(
        results,
        "N1.promotions",
        "N1",
        not missing,
        f"forbidden_promotions covers required set (missing={sorted(missing)})",
    )
    neg_sum = int(mem["n_neg_captured"].sum())
    _check(
        results,
        "N1.capacity",
        "N1",
        claim.get("capacity_policy") == "non_additive_dual_distributions"
        and neg_sum > feas["n_fp_exposed"],
        f"capacity sum→event mass blocked: Σn_neg={neg_sum} ≠ cohort neg={feas['n_fp_exposed']}, policy=non_additive",
    )
    _check(
        results,
        "N1.applicability",
        "N1",
        claim.get("action_states_allowed") == ["observation_only"]
        and "production" in claim.get("action_states_forbidden", [])
        and (p["regions"]["action_state"] == "observation_only").all(),
        "sequence union→applicability blocked: observation_only everywhere",
    )
    _check(
        results,
        "N1.mask_identity",
        "N1",
        claim.get("policy_equivalence_policy") == "family_ast_ne_instance_ne_mask"
        and man["n_policy_instances"] > man["n_mask_units"],
        f"mask equality→policy identity blocked: {man['n_policy_instances']} policy instances > {man['n_mask_units']} mask units",
    )
    _check(
        results,
        "N1.maturity",
        "N1",
        "A0_to_A1_self_accept" in fp and man.get("maturity_declared") == "A0",
        "A0→A1 self-accept forbidden; pack stays A0 until recorded terminal",
    )
    _check(
        results,
        "N1.population",
        "N1",
        feas.get("finite_sample_statement")
        == "observed_GT0_is_not_population_risk_zero",
        "observed GT0→population risk-zero blocked",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--t0-evidence", type=Path, required=True)
    ap.add_argument("--q45-evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    p = _load_pack(args.pack)
    results: list[dict] = []
    audit_s0(p, results)
    audit_s1(p, args.t0_evidence, args.q45_evidence, results)
    audit_q1(p, results)
    audit_n1(p, results)

    df = pd.DataFrame(results)
    n_fail = int((df["status"] == "FAIL").sum())
    verdict = "AUDIT_PASS" if n_fail == 0 else "AUDIT_FAIL"
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "a1_audit_checks.csv", index=False)
    report = {
        "audit": "safe_region_a1_read_only_audit",
        "pack_id": p["manifest"]["pack_id"],
        "pack_root": str(args.pack),
        "read_only": True,
        "n_checks": len(df),
        "n_fail": n_fail,
        "verdict": verdict,
        "scope_note": (
            "S0/S1/Q1/N1 only. No D1 decision-trace, no terminal; "
            "A1 terminal remains a research-owner decision."
        ),
        "golden_source": "docs/research/threads/closed/safe_region_assetization_20260710.md accepted A0 baseline (predeclared)",
    }
    (args.out / "a1_audit_report.json").write_text(json.dumps(report, indent=2))
    with pd.option_context("display.width", 200, "display.max_colwidth", 110):
        print(df.to_string(index=False))
    print(f"\n{verdict}: {len(df)} checks, {n_fail} FAIL")
    print(f"report → {args.out}/a1_audit_report.json")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
