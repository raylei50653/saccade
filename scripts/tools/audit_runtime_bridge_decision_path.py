#!/usr/bin/env python3
"""P0: outcome-blind runtime bridge decision-path identifiability audit.

This source/artifact verifier intentionally reads no CSV row values: the D0
capture header is sufficient to decide whether the frozen packet could replay
the production decision graph.  In particular it never accesses pairs.csv or
any GT/FP field.  It writes a field matrix, an explicitly unobserved funnel,
and exactly one ordered terminal.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
STAMP = "20260713"
TERMINAL = "P0_CAPTURE_SEMANTICS_INVALID"
HEADLINE = {
    "preset": "configs/presets/mamba_whole_graph.yaml",
    "relink_bridge_enabled": True,
    "relink_bridge_px": 0.25,
    "relink_bridge_margin": 0.05,
    "relink_bridge_h_lo": 0.75,
    "relink_bridge_h_hi": 1.33,
    "relink_bridge_spatial_gate": 0.0,
    "relink_bridge_max_speed": 0.0,
    "relink_bridge_dir_bonus": 0.8,
    "reid_mode": "off",
}

SOURCE_PROOFS = {
    "stage_a": "if (!active[cand] || trk_to_det[cand] < 0) return;",
    "stage_b_height": "if (bridge_h_hi > 0.0f)",
    "stage_c_bdist": "float bdist = w * 0.5f * (fwd_r + bwd_r) + (1.0f - w) * dist_h;",
    "stage_d_cutoff": "bool ok = bdist <= bridge_px;",
    "stage_e_margin": "(second_dist - best_dist) < bridge_margin",
    "stage_f_claim": "atomicMax(&bridge_claim[best_lost], key);",
    "stage_g_commit": "track_ids[cand] = track_ids[lost];",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def csv_header_gzip(path: Path) -> list[str]:
    # Do not iterate rows: P0 must remain outcome-blind until its stratum freezes.
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"empty capture: {path}")
    return header


def input_key(path: Path, root: Path) -> str:
    """Use a stable explicit key for a test-injected input outside the repo."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return f"external_injected/{path.name}"


def _eq(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-8
        except (TypeError, ValueError):
            return False
    return actual == expected


def _alignment(name: str, bridge: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "relink_bridge_px": "px",
        "relink_bridge_dir_bonus": "dir_bonus",
        "relink_bridge_h_lo": "h_lo",
        "relink_bridge_h_hi": "h_hi",
        "relink_bridge_spatial_gate": "spatial_gate",
        "relink_bridge_max_speed": "max_speed",
    }
    comparisons = []
    for policy_key, artifact_key in mapping.items():
        present = artifact_key in bridge
        actual = bridge.get(artifact_key)
        comparisons.append(
            {
                "headline_knob": policy_key,
                "artifact_field": artifact_key,
                "expected": HEADLINE[policy_key],
                "actual": actual,
                "status": "match"
                if present and _eq(actual, HEADLINE[policy_key])
                else "mismatch_or_absent",
            }
        )
    return {
        "artifact": name,
        "all_fields_match": all(item["status"] == "match" for item in comparisons),
        "comparisons": comparisons,
    }


def _field_matrix(header: set[str]) -> list[dict[str, Any]]:
    def row(
        stage: str, required: list[str], artifact: str, complete: bool, consequence: str
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "required_fields": required,
            "existing_artifact": artifact,
            "complete": complete,
            "missing_consequence": consequence,
        }

    return [
        row(
            "source_snapshot_alignment",
            ["capture kernel source SHA-256", "headline kernel source SHA-256"],
            "D0/R1 provenance records only a git commit",
            False,
            "a commit label is not an exact source-file identity; capture/source code equivalence is unproven",
        ),
        row(
            "A_pair_eligibility",
            [
                "all candidate/lost state before height gate",
                "frame",
                "candidate slot",
                "lost slot",
            ],
            "D0 v2 capture header",
            False,
            "capture is emitted after eligibility and pre-score continues; raw exclusions are unobservable",
        ),
        row(
            "B_pre_score_gates",
            [
                "all raw pairs",
                "EMA heights",
                "height-gate result",
                "speed/centre state",
            ],
            "D0 has ema_lost/ema_cand only on emitted survivors",
            False,
            "height-gate attrition and disabled-path counterfactuals cannot be observed",
        ),
        row(
            "C_score_construction",
            [
                "fwd_r",
                "bwd_r",
                "dist_h",
                "s_lost",
                "w",
                "bdist",
                "directional provenance",
            ],
            "D0 v2 capture header",
            {"fwd_r", "bwd_r", "dist_h", "s_lost", "w", "bdist"}.issubset(header),
            "scalar formula terms are observable only for the foreign-config survivor population",
        ),
        row(
            "D_pair_cutoff",
            ["bdist", "production threshold", "headline preset stamp"],
            "D0 v2 capture header + manifest",
            False,
            "scalar cutoff is mechanically evaluable, but not for the frozen headline configuration",
        ),
        row(
            "E_candidate_local_ranking",
            ["frame", "candidate slot/index", "complete lost competitor set", "bdist"],
            "D0 v2 export",
            False,
            "exported event key omits frame and slots; best/second-best and margin are unreconstructable",
        ),
        row(
            "F_claim_competition",
            [
                "lost slot",
                "all proposing candidates",
                "detection score",
                "quantized key",
                "candidate index",
            ],
            "D0/R1 canonical packets",
            False,
            "atomicMax winner cannot be replayed",
        ),
        row(
            "G_commit",
            ["winning claim", "post-commit candidate ID", "lost-slot deactivation"],
            "shadow capture",
            False,
            "shadow deliberately suppresses the only bridge writes; final commit is not observed",
        ),
    ]


def audit(root: Path = REPO, *, d0_capture_dir: Path | None = None) -> dict[str, Any]:
    evidence = root / "docs/modules/semantic/research/evidence"
    d0_packet_path = evidence / "d0_runtime_shadow_fidelity_20260712/manifest.json"
    r1_packet_path = evidence / "r1_temporal_reduction_capture_20260712/manifest.json"
    r1_hashes_path = (
        evidence / "r1_temporal_reduction_capture_20260712/frozen_packet_hashes.json"
    )
    r1_export_path = (
        evidence / "r1_temporal_reduction_capture_20260712/export_manifest.json"
    )
    s0_packet_path = evidence / "s0_safe_domain_runtime_transfer_20260713/manifest.json"
    d0_dir = d0_capture_dir or (
        root / "out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z"
    )
    d0_capture = d0_dir / "capture.csv.gz"
    d0_capture_manifest_path = d0_dir / "capture.csv.gz.manifest.json"
    declaration = (
        root
        / "docs/modules/semantic/research/runtime_bridge_decision_path_identifiability_declaration_20260713.md"
    )
    source = root / "src/tracking/tracker_gpu.cu"

    for path in (
        d0_packet_path,
        r1_packet_path,
        r1_hashes_path,
        r1_export_path,
        s0_packet_path,
        d0_capture,
        d0_capture_manifest_path,
        declaration,
        source,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    d0_packet = load_json(d0_packet_path)
    d0_capture_manifest = load_json(d0_capture_manifest_path)
    r1_packet = load_json(r1_packet_path)
    r1_hashes = load_json(r1_hashes_path)
    r1_export = load_json(r1_export_path)
    s0_packet = load_json(s0_packet_path)
    header = csv_header_gzip(d0_capture)
    header_set = set(header)
    source_text = source.read_text(encoding="utf-8")
    source_proofs = {name: text in source_text for name, text in SOURCE_PROOFS.items()}

    expected_d0_hash = str(d0_packet["frozen_inputs"]["capture.csv.gz"])
    actual_d0_hash = sha256(d0_capture)
    d0_hash_ok = expected_d0_hash == actual_d0_hash
    s0_hash_ok = str(s0_packet["input_hashes"]["capture.csv.gz"]) == actual_d0_hash
    d0_alignment = _alignment("D0", dict(d0_capture_manifest["provenance"]["bridge"]))
    r1_alignment = _alignment("R1", dict(r1_export["provenance"]["bridge"]))
    r1_preset = str(r1_hashes["scope"]["preset"])

    configuration_valid = (
        d0_alignment["all_fields_match"]
        and r1_alignment["all_fields_match"]
        and r1_preset == HEADLINE["preset"]
    )
    source_valid = all(source_proofs.values())
    terminal = (
        TERMINAL
        if not (configuration_valid and source_valid and d0_hash_ok and s0_hash_ok)
        else "P0_PAIR_CUTOFF_ONLY"
    )
    matrix = _field_matrix(header_set)
    return {
        "study": "p0_runtime_bridge_decision_path_20260713",
        "terminal": terminal,
        "headline": HEADLINE,
        "label_access": {"gt_or_fp_labels_accessed": False, "p5": "not_entered"},
        "source_proofs": source_proofs,
        "provenance": {
            "d0_capture_sha256": actual_d0_hash,
            "current_kernel_source_sha256": sha256(source),
            "d0_capture_git_commit": d0_capture_manifest["provenance"].get(
                "git_commit"
            ),
            "r1_capture_git_commit": r1_export["provenance"].get("git_commit"),
            "capture_kernel_source_sha256_present": False,
            "d0_packet_hash_match": d0_hash_ok,
            "s0_inherits_same_capture_hash": s0_hash_ok,
            "d0_alignment": d0_alignment,
            "r1_alignment": r1_alignment,
            "r1_frozen_preset": r1_preset,
            "r1_packet_terminal": r1_packet.get("terminal"),
        },
        "d0_header": header,
        "field_sufficiency": matrix,
        "replay": {
            "observed_level": "not_assignable_due_to_capture_semantics_invalid",
            "counterfactual_ceiling_if_headline_alignment_existed": "L1_pair_cutoff_replay",
            "l2_blockers": [
                "no frame",
                "no candidate slot/index",
                "no complete candidate competitor universe",
            ],
            "l3_blockers": [
                "no raw pre-height universe",
                "no detection score",
                "no quantized atomicMax key",
                "shadow capture has no commit",
            ],
        },
        "decision_funnel_status": "not_entered_due_to_capture_semantics_invalid",
        "inputs": {
            input_key(path, root): sha256(path)
            for path in (
                d0_packet_path,
                d0_capture_manifest_path,
                r1_packet_path,
                r1_hashes_path,
                r1_export_path,
                s0_packet_path,
                declaration,
                source,
            )
        },
    }


def write_packet(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "field_sufficiency.json").write_text(
        json.dumps(result["field_sufficiency"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    funnel_rows: Iterable[dict[str, str]] = (
        {
            "stage": stage,
            "event_count": "",
            "unique_candidate_count": "",
            "unique_lost_track_count": "",
            "sequence_distribution": "",
            "observability": "UNOBSERVABLE",
            "reason": "headline provenance is invalid; P4 not entered",
        }
        for stage in (
            "eligible_raw_pairs",
            "pass_height_gate",
            "pass_bdist_cutoff",
            "candidate_local_winners",
            "pass_margin",
            "claim_winners",
            "final_commits",
        )
    )
    with (output_dir / "decision_funnel.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "event_count",
                "unique_candidate_count",
                "unique_lost_track_count",
                "sequence_distribution",
                "observability",
                "reason",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(funnel_rows)
    metrics = {
        key: value
        for key, value in result.items()
        if key not in {"field_sufficiency", "inputs"}
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_hashes = {
        name: sha256(output_dir / name)
        for name in (
            "field_sufficiency.json",
            "decision_funnel.csv",
            "metrics.json",
        )
    }
    manifest = {
        "study": result["study"],
        "terminal": result["terminal"],
        "runner": "scripts/tools/audit_runtime_bridge_decision_path.py",
        "runner_sha256": sha256(Path(__file__)),
        "inputs": result["inputs"],
        "outputs": output_hashes,
        "files": output_hashes,
        "label_access": result["label_access"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO
        / "docs/modules/semantic/research/evidence/p0_runtime_bridge_decision_path_20260713",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    write_packet(result, output_dir)
    print(
        json.dumps(
            {"terminal": result["terminal"], "gt_or_fp_labels_accessed": False},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
