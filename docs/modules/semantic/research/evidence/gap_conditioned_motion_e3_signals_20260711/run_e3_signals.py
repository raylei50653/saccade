#!/usr/bin/env python3
"""Generate sealed E3 LOO fold signals for GCM-E2-POSITION-ONLY-v1.

Authorized scope (E2 §4 / §5.1 · thread · Phase B design §1 step 2):
  rebuild 7 LOO folds
  persist 28 parameter artifacts + 7 selection artifacts
  emit the full fold × pair × model score cube with evaluation_role

Score surface (A6-complete):
  every pair is scored under every LOO fold's frozen parameters for all
  four family members. Rows are tagged:
    evaluation_role=held_out  — pair.seq == fold held-out sequence
    evaluation_role=train     — pair.seq in the fold's six train sequences
  A6 training-side τ selection must use evaluation_role=train only.
  Held-out rows never enter fit, selection, or train-side threshold choice.

Explicitly out of scope:
  A1–A8 tables · V1–V5 verdict · calibration · family change ·
  production / hook / preset / baseline change · criterion edits
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import tempfile
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[6]
PACKET_DIR = Path(__file__).resolve().parent
E2_RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "gap_conditioned_motion_e2_family_20260711"
    / "run_e2_family_freeze.py"
)
PHASE_B_DESIGN = (
    REPO
    / "docs/modules/semantic/research"
    / "gap_conditioned_motion_phase_b_design_20260711.md"
)
# PR #113 merge = Phase B predeclaration seal (thread History).
PHASE_B_DESIGN_SEAL_COMMIT = "69b0e5be0c26d6fa9f460f90aef37e891555da67"
PHASE_B_DESIGN_REL = (
    "docs/modules/semantic/research/gap_conditioned_motion_phase_b_design_20260711.md"
)
CANONICAL_PAIRS = Path(
    "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
)
SOURCE_SHA256 = "0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17"
FREEZE_ID = "GCM-E2-POSITION-ONLY-v1"
PACKET_STATUS = "E3_SIGNALS_SEALED"
SEQUENCES = (
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
)
# Sealed E2 LOO lineage (model_family.json); E3 must match exactly.
SEALED_TRAIN_COUNTS = {
    "MOT17-02-SDP": 268,
    "MOT17-04-SDP": 328,
    "MOT17-05-SDP": 298,
    "MOT17-09-SDP": 326,
    "MOT17-10-SDP": 183,
    "MOT17-11-SDP": 320,
    "MOT17-13-SDP": 317,
}
SEALED_TRAIN_HASHES = {
    "MOT17-02-SDP": "9e7fe45468f37db1c53eb40b4d2f119674b9276a27543d055d05f306300cba0a",
    "MOT17-04-SDP": "9caca30ef899259101d64b7af65690e2ead48f9b1fb9d8e258f927eb90e0e020",
    "MOT17-05-SDP": "ada1a106fdb9163957be49d1c5820154578c13471ba3b3a81d9f7f63cdbcc4e7",
    "MOT17-09-SDP": "164034ec762604c73d488dd5e964d0886e01059af556c8ea36c5fb6ecd2ba01c",
    "MOT17-10-SDP": "8b1e67dd9aba0f2705349c7a23caf7681957c9e9439cff118d9609351c0d7eae",
    "MOT17-11-SDP": "0aa3abb410de384d4cc1d667ee25585dbcc3283da2186ec3201a00ce8a4ed784",
    "MOT17-13-SDP": "a88c8ed7c35a833b2d66c04344cb2fca3dc16969b4f8ecacdc55c96a5ee0fafb",
}
SCORE_FIELDS = (
    "freeze_id",
    "fold_id",
    "held_out_sequence",
    "evaluation_role",
    "model_id",
    "parameter_artifact_id",
    "selection_artifact_id",
    "selected_model_id",
    "is_selected_model",
    "seq",
    "lost_id",
    "cand_id",
    "gap",
    "gt_match",
    "gt_valid",
    "dx_h",
    "dy_h",
    "q_motion",
    "log_det_covariance",
    "gaussian_constant",
    "nll_motion",
    "source_pairs_sha256",
)
REQUIRED_PAIR_FIELDS = {
    "seq",
    "lost_id",
    "cand_id",
    "gt_match",
    "gt_valid",
    "gap",
    "lost_last_frame",
    "cand_first_frame",
    "lost_foot_x",
    "lost_foot_y",
    "cand_foot_x",
    "cand_foot_y",
    "h_ref",
}


def load_e2() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("gap_motion_e2", E2_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E2 runner: {E2_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_b_design_provenance() -> dict[str, str]:
    """Self-identify the frozen Phase B criterion seal this packet postdates."""
    if not PHASE_B_DESIGN.is_file():
        raise FileNotFoundError(f"Phase B design missing: {PHASE_B_DESIGN}")
    return {
        "path": PHASE_B_DESIGN_REL,
        "predeclaration_seal_commit": PHASE_B_DESIGN_SEAL_COMMIT,
        "content_sha256": sha256(PHASE_B_DESIGN),
    }


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _fold_slug(held_out: str) -> str:
    return held_out.replace("-", "_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_scoreable_pairs(pairs: Path) -> list[dict[str, Any]]:
    """Load every pair with a finite position-only observation (E0 gate)."""
    rows: list[dict[str, Any]] = []
    with pairs.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_PAIR_FIELDS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"missing E3 pair fields: {missing}")
        for source in reader:
            gap = int(source["gap"])
            h_ref = float(source["h_ref"])
            endpoints = np.asarray(
                [
                    float(source["lost_foot_x"]),
                    float(source["lost_foot_y"]),
                    float(source["cand_foot_x"]),
                    float(source["cand_foot_y"]),
                ],
                dtype=np.float64,
            )
            if (
                h_ref <= 0
                or not math.isfinite(h_ref)
                or not np.all(np.isfinite(endpoints))
            ):
                raise ValueError("invalid position row in sealed pair table")
            if gap < 1 or gap > 300:
                raise ValueError(f"gap out of sealed range: {gap}")
            if int(source["cand_first_frame"]) - int(source["lost_last_frame"]) != gap:
                raise ValueError("frame-window mismatch in sealed pair table")
            rows.append(
                {
                    "seq": source["seq"],
                    "lost_id": source["lost_id"],
                    "cand_id": source["cand_id"],
                    "gap": gap,
                    "gt_match": int(_as_bool(source["gt_match"])),
                    "gt_valid": int(_as_bool(source["gt_valid"])),
                    "dx_h": float((endpoints[2] - endpoints[0]) / h_ref),
                    "dy_h": float((endpoints[3] - endpoints[1]) / h_ref),
                }
            )
    return rows


def build_all_fold_artifacts(
    e2: types.ModuleType, pairs: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Rebuild the seven LOO folds via the lineage-aware E2 fold builder."""
    parameters_by_fold: dict[str, list[dict[str, Any]]] = {}
    selections_by_fold: dict[str, dict[str, Any]] = {}
    source_sha = e2.sha256(pairs)
    if source_sha != SOURCE_SHA256:
        raise ValueError(
            f"source SHA mismatch: expected {SOURCE_SHA256}, got {source_sha}"
        )

    for held_out in SEQUENCES:
        parameters, selection = e2.build_fold_artifacts(pairs, held_out)
        expected_count = SEALED_TRAIN_COUNTS[held_out]
        expected_hash = SEALED_TRAIN_HASHES[held_out]
        if selection["fit_row_count"] != expected_count:
            raise ValueError(
                f"train count mismatch for {held_out}: "
                f"{selection['fit_row_count']} != {expected_count}"
            )
        if selection["fit_row_key_sha256"] != expected_hash:
            raise ValueError(f"train-row lineage hash mismatch for {held_out}")
        if len(parameters) != len(e2.MODEL_ORDER):
            raise ValueError(f"expected 4 parameter artifacts for {held_out}")
        if set(selection["training_nll_by_model"]) != set(e2.MODEL_ORDER):
            raise ValueError(f"selection family incomplete for {held_out}")
        if held_out in selection["train_sequences"]:
            raise ValueError(f"held-out leaked into train_sequences for {held_out}")
        for parameter in parameters:
            if (
                parameter["held_out_sequence"] != held_out
                or held_out in parameter["train_sequences"]
                or parameter["fit_row_count"] != expected_count
                or parameter["fit_row_key_sha256"] != expected_hash
                or parameter["source_pairs_sha256"] != source_sha
                or parameter["freeze_id"] != FREEZE_ID
            ):
                raise ValueError(f"parameter artifact contract failed for {held_out}")
        parameters_by_fold[held_out] = parameters
        selections_by_fold[held_out] = selection
    return parameters_by_fold, selections_by_fold


def score_fold_cube(
    e2: types.ModuleType,
    pairs_rows: list[dict[str, Any]],
    parameters_by_fold: dict[str, list[dict[str, Any]]],
    selections_by_fold: dict[str, dict[str, Any]],
    source_sha: str,
) -> list[dict[str, Any]]:
    """Score every pair under every fold's parameters (full cube + role tag).

    evaluation_role distinguishes A6 train-side threshold rows from held-out
    evaluation rows. Held-out sequence pairs are never tagged train.
    """
    fold_sequences = tuple(sorted(parameters_by_fold))
    if set(fold_sequences) != set(selections_by_fold):
        raise ValueError("parameter and selection fold keys must match")

    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in pairs_rows:
        by_sequence.setdefault(str(row["seq"]), []).append(row)

    score_rows: list[dict[str, Any]] = []
    for held_out in fold_sequences:
        selection = selections_by_fold[held_out]
        train_sequences = set(selection["train_sequences"])
        if held_out in train_sequences:
            raise ValueError(f"held-out in train_sequences for {held_out}")
        selected_model_id = selection["selected_model_id"]

        # Deterministic order: train sequences (sorted) then held-out.
        score_sequences = sorted(train_sequences) + [held_out]
        for seq in score_sequences:
            seq_rows = by_sequence.get(seq)
            if not seq_rows:
                raise ValueError(f"no pairs for sequence {seq} under fold {held_out}")
            if seq == held_out:
                role = "held_out"
            elif seq in train_sequences:
                role = "train"
            else:
                raise ValueError(f"sequence {seq} not in fold {held_out} partition")

            displacements = np.asarray(
                [[row["dx_h"], row["dy_h"]] for row in seq_rows], dtype=np.float64
            )
            gaps = np.asarray([row["gap"] for row in seq_rows], dtype=np.float64)
            for parameter in parameters_by_fold[held_out]:
                scores = e2.score_model(parameter, displacements, gaps)
                model_id = parameter["model_id"]
                for index, pair in enumerate(seq_rows):
                    score_rows.append(
                        {
                            "freeze_id": FREEZE_ID,
                            "fold_id": parameter["fold_id"],
                            "held_out_sequence": held_out,
                            "evaluation_role": role,
                            "model_id": model_id,
                            "parameter_artifact_id": parameter["parameter_artifact_id"],
                            "selection_artifact_id": selection["selection_artifact_id"],
                            "selected_model_id": selected_model_id,
                            "is_selected_model": int(model_id == selected_model_id),
                            "seq": pair["seq"],
                            "lost_id": pair["lost_id"],
                            "cand_id": pair["cand_id"],
                            "gap": pair["gap"],
                            "gt_match": pair["gt_match"],
                            "gt_valid": pair["gt_valid"],
                            "dx_h": pair["dx_h"],
                            "dy_h": pair["dy_h"],
                            "q_motion": float(scores["q_motion"][index]),
                            "log_det_covariance": float(
                                scores["log_det_covariance"][index]
                            ),
                            "gaussian_constant": float(
                                scores["gaussian_constant"][index]
                            ),
                            "nll_motion": float(scores["nll_motion"][index]),
                            "source_pairs_sha256": source_sha,
                        }
                    )
    return score_rows


# Backward-compatible alias used by existing unit tests.
def score_pairs_for_folds(
    e2: types.ModuleType,
    pairs_rows: list[dict[str, Any]],
    parameters_by_fold: dict[str, list[dict[str, Any]]],
    selections_by_fold: dict[str, dict[str, Any]],
    source_sha: str,
) -> list[dict[str, Any]]:
    return score_fold_cube(
        e2, pairs_rows, parameters_by_fold, selections_by_fold, source_sha
    )


def _persist_fold_artifacts(
    output_dir: Path,
    parameters_by_fold: dict[str, list[dict[str, Any]]],
    selections_by_fold: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Write 28 parameter + 7 selection JSON files; return relative→sha map."""
    artifact_hashes: dict[str, str] = {}
    for held_out in SEQUENCES:
        slug = _fold_slug(held_out)
        selection = selections_by_fold[held_out]
        selection_rel = f"selections/LOO__{slug}.json"
        selection_path = output_dir / selection_rel
        _write_json(selection_path, selection)
        artifact_hashes[selection_rel] = sha256(selection_path)

        for parameter in parameters_by_fold[held_out]:
            model_id = parameter["model_id"]
            param_rel = f"parameters/LOO__{slug}__{model_id}.json"
            param_path = output_dir / param_rel
            _write_json(param_path, parameter)
            artifact_hashes[param_rel] = sha256(param_path)
    return artifact_hashes


def _write_score_table(output_dir: Path, score_rows: list[dict[str, Any]]) -> Path:
    """Write the sealed score cube as gzip (full cube exceeds GitHub's 100MB limit)."""
    import io

    path = output_dir / "pair_fold_model_scores.csv.gz"
    # mtime=0 keeps gzip headers byte-stable across rebuilds.
    with path.open("wb") as binary:
        with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as gz:
            buffer = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(SCORE_FIELDS))
            writer.writeheader()
            for row in score_rows:
                writer.writerow({field: row[field] for field in SCORE_FIELDS})
            buffer.flush()
            buffer.detach()
    return path


def _validate_score_surface(
    score_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    parameters_by_fold: dict[str, list[dict[str, Any]]],
    selections_by_fold: dict[str, dict[str, Any]],
    n_models: int,
) -> None:
    """Hard gates: cube completeness, role firewall, energy identity, lineage."""
    n_pairs = len(pair_rows)
    n_folds = len(parameters_by_fold)
    expected = n_pairs * n_folds * n_models
    if len(score_rows) != expected:
        raise ValueError(f"score cube size mismatch: {len(score_rows)} != {expected}")

    pair_keys = {
        (str(r["seq"]), str(r["lost_id"]), str(r["cand_id"])) for r in pair_rows
    }
    if len(pair_keys) != n_pairs:
        raise ValueError("duplicate pair keys in scoreable pairs")

    # (fold, pair) -> models
    fold_pair_models: dict[tuple[str, str, str, str], set[str]] = {}
    for row in score_rows:
        held = str(row["held_out_sequence"])
        seq = str(row["seq"])
        key = (held, seq, str(row["lost_id"]), str(row["cand_id"]))
        fold_pair_models.setdefault(key, set()).add(str(row["model_id"]))

        # Energy identity: nll = 1/2 (q + logdet + constant).
        rebuilt = 0.5 * (
            float(row["q_motion"])
            + float(row["log_det_covariance"])
            + float(row["gaussian_constant"])
        )
        if not math.isclose(
            float(row["nll_motion"]), rebuilt, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("energy identity failed for a score row")

        selection = selections_by_fold[held]
        parameter_ids = {
            p["model_id"]: p["parameter_artifact_id"] for p in parameters_by_fold[held]
        }
        if row["selection_artifact_id"] != selection["selection_artifact_id"]:
            raise ValueError(f"selection lineage mismatch for fold {held}")
        if row["parameter_artifact_id"] != parameter_ids[row["model_id"]]:
            raise ValueError(f"parameter lineage mismatch for fold {held}")
        if row["selected_model_id"] != selection["selected_model_id"]:
            raise ValueError(f"selected_model_id mismatch for fold {held}")

        if row["evaluation_role"] == "held_out":
            if seq != held:
                raise ValueError("held_out role must be the fold's held-out sequence")
        elif row["evaluation_role"] == "train":
            if seq == held or seq not in selection["train_sequences"]:
                raise ValueError("train role must be a fold train sequence only")
        else:
            raise ValueError(f"unknown evaluation_role: {row['evaluation_role']}")

    model_order = {p["model_id"] for p in next(iter(parameters_by_fold.values()))}
    if any(models != model_order for models in fold_pair_models.values()):
        raise ValueError("not every fold×pair retained all four frozen members")
    if len(fold_pair_models) != n_pairs * n_folds:
        raise ValueError("fold×pair coverage incomplete")


def _fold_summary(
    parameters_by_fold: dict[str, list[dict[str, Any]]],
    selections_by_fold: dict[str, dict[str, Any]],
    score_rows: list[dict[str, Any]],
    source_sha: str,
    phase_b_design: dict[str, str],
) -> dict[str, Any]:
    models_seen: set[str] = set()
    unique_pairs: set[tuple[str, str, str]] = set()
    per_fold: dict[str, dict[str, Any]] = {
        seq: {
            "score_row_count": 0,
            "held_out_score_row_count": 0,
            "train_score_row_count": 0,
            "held_out_pair_keys": set(),
            "train_pair_keys": set(),
            "train_sequences_seen": set(),
        }
        for seq in SEQUENCES
    }

    for row in score_rows:
        held = row["held_out_sequence"]
        bucket = per_fold[held]
        bucket["score_row_count"] += 1
        pair_key = (row["seq"], str(row["lost_id"]), str(row["cand_id"]))
        unique_pairs.add(pair_key)
        models_seen.add(row["model_id"])
        if row["evaluation_role"] == "held_out":
            bucket["held_out_score_row_count"] += 1
            bucket["held_out_pair_keys"].add(pair_key)
        else:
            bucket["train_score_row_count"] += 1
            bucket["train_pair_keys"].add(pair_key)
            bucket["train_sequences_seen"].add(row["seq"])

    folds = []
    for held_out in SEQUENCES:
        selection = selections_by_fold[held_out]
        bucket = per_fold[held_out]
        train_pair_count = len(bucket["train_pair_keys"])
        held_out_pair_count = len(bucket["held_out_pair_keys"])
        folds.append(
            {
                "fold_id": selection["fold_id"],
                "held_out_sequence": held_out,
                "train_sequences": selection["train_sequences"],
                "fit_row_count": selection["fit_row_count"],
                "fit_row_key_sha256": selection["fit_row_key_sha256"],
                "selected_model_id": selection["selected_model_id"],
                "selection_artifact_id": selection["selection_artifact_id"],
                "parameter_artifact_ids": {
                    p["model_id"]: p["parameter_artifact_id"]
                    for p in parameters_by_fold[held_out]
                },
                "training_nll_by_model": selection["training_nll_by_model"],
                "held_out_pair_count": held_out_pair_count,
                "train_pair_count": train_pair_count,
                "held_out_score_row_count": bucket["held_out_score_row_count"],
                "train_score_row_count": bucket["train_score_row_count"],
                "score_row_count": bucket["score_row_count"],
            }
        )

    return {
        "schema_version": 2,
        "freeze_id": FREEZE_ID,
        "status": PACKET_STATUS,
        "phase_b_authorized": False,
        "a1_a8_computed": False,
        "verdict": "NOT_YET_EVALUATED",
        "claim_ceiling": (
            "sealed LOO fold×pair×model score cube only; not Phase B; "
            "not V1-V5; not calibration; not production"
        ),
        "source": {
            "pairs_csv": str(CANONICAL_PAIRS),
            "sha256": source_sha,
        },
        "phase_b_design": phase_b_design,
        "counts": {
            "n_folds": len(SEQUENCES),
            "n_parameter_artifacts": sum(
                len(params) for params in parameters_by_fold.values()
            ),
            "n_selection_artifacts": len(selections_by_fold),
            "n_score_rows": len(score_rows),
            "n_held_out_score_rows": sum(f["held_out_score_row_count"] for f in folds),
            "n_train_score_rows": sum(f["train_score_row_count"] for f in folds),
            "n_unique_pairs_scored": len(unique_pairs),
            "n_models_per_pair_fold": len(models_seen),
            "models": sorted(models_seen),
        },
        "folds": folds,
        "output_contract": {
            "score_unit": (
                "full fold × pair × model cube; evaluation_role=held_out for "
                "the fold's held-out sequence pairs; evaluation_role=train for "
                "the six training sequences under the same fold-frozen "
                "parameters (A6 τ selection); all four members retained; "
                "selected_model_id is a marker only"
            ),
            "energy_terms_split": [
                "q_motion",
                "log_det_covariance",
                "gaussian_constant",
                "nll_motion",
            ],
            "a6_train_surface": (
                "filter evaluation_role=train; parameters/lineage already "
                "bound per row via parameter_artifact_id + selection_artifact_id"
            ),
            "forbidden_in_this_packet": [
                "A1-A8 tables",
                "V1-V5 verdict",
                "held-out calibration",
                "family redefinition",
                "winner-only score filtering",
                "using held_out rows for A6 threshold selection",
            ],
        },
    }


def _render(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    design = summary["phase_b_design"]
    lines = [
        f"freeze_id={summary['freeze_id']}",
        f"status={summary['status']}",
        f"source_sha256={summary['source']['sha256']}",
        f"phase_b_design_seal={design['predeclaration_seal_commit'][:12]}",
        f"phase_b_design_sha256={design['content_sha256'][:12]}",
        f"n_folds={counts['n_folds']}",
        f"n_parameter_artifacts={counts['n_parameter_artifacts']}",
        f"n_selection_artifacts={counts['n_selection_artifacts']}",
        f"n_score_rows={counts['n_score_rows']}",
        f"n_held_out_score_rows={counts['n_held_out_score_rows']}",
        f"n_train_score_rows={counts['n_train_score_rows']}",
        f"n_unique_pairs_scored={counts['n_unique_pairs_scored']}",
        f"models={','.join(counts['models'])}",
        f"phase_b_authorized={str(bool(summary['phase_b_authorized'])).lower()}",
        f"a1_a8_computed={str(bool(summary['a1_a8_computed'])).lower()}",
        f"verdict={summary['verdict']}",
    ]
    for fold in summary["folds"]:
        lines.append(
            "fold="
            f"{fold['held_out_sequence']}"
            f" train_gt={fold['fit_row_count']}"
            f" held_out_pairs={fold['held_out_pair_count']}"
            f" train_pairs={fold['train_pair_count']}"
            f" held_out_scores={fold['held_out_score_row_count']}"
            f" train_scores={fold['train_score_row_count']}"
            f" selected={fold['selected_model_id']}"
            f" hash={fold['fit_row_key_sha256'][:12]}"
        )
    return "\n".join(lines) + "\n"


def generate_packet(pairs: Path, output_dir: Path) -> dict[str, Any]:
    e2 = load_e2()
    source_sha = e2.sha256(pairs)
    if source_sha != SOURCE_SHA256:
        raise ValueError(
            f"source SHA mismatch: expected {SOURCE_SHA256}, got {source_sha}"
        )

    phase_b_design = phase_b_design_provenance()
    parameters_by_fold, selections_by_fold = build_all_fold_artifacts(e2, pairs)
    pair_rows = load_scoreable_pairs(pairs)
    score_rows = score_fold_cube(
        e2, pair_rows, parameters_by_fold, selections_by_fold, source_sha
    )

    if len(parameters_by_fold) * 4 != 28:
        raise ValueError("expected exactly 28 parameter artifacts")
    if len(selections_by_fold) != 7:
        raise ValueError("expected exactly 7 selection artifacts")

    _validate_score_surface(
        score_rows,
        pair_rows,
        parameters_by_fold,
        selections_by_fold,
        n_models=len(e2.MODEL_ORDER),
    )

    # Winner-only filtering is forbidden on both roles.
    role_pair_model_counts = Counter(
        (
            row["held_out_sequence"],
            row["evaluation_role"],
            row["seq"],
            str(row["lost_id"]),
            str(row["cand_id"]),
            row["model_id"],
        )
        for row in score_rows
    )
    if any(count != 1 for count in role_pair_model_counts.values()):
        raise ValueError("duplicate fold×role×pair×model score rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("parameters", "selections"):
        subdir = output_dir / sub
        if subdir.exists():
            for child in subdir.glob("*.json"):
                child.unlink()
        subdir.mkdir(parents=True, exist_ok=True)

    artifact_hashes = _persist_fold_artifacts(
        output_dir, parameters_by_fold, selections_by_fold
    )
    score_path = _write_score_table(output_dir, score_rows)
    summary = _fold_summary(
        parameters_by_fold,
        selections_by_fold,
        score_rows,
        source_sha,
        phase_b_design,
    )
    summary_path = output_dir / "fold_summary.json"
    _write_json(summary_path, summary)
    recorded_path = output_dir / "recorded_output.txt"
    recorded_path.write_text(_render(summary), encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "freeze_id": FREEZE_ID,
        "status": PACKET_STATUS,
        "source_pairs_csv": str(CANONICAL_PAIRS),
        "source_pairs_csv_sha256": source_sha,
        "e2_family_packet": (
            "docs/modules/semantic/research/evidence/"
            "gap_conditioned_motion_e2_family_20260711/manifest.json"
        ),
        "phase_b_design": phase_b_design,
        "runner_sha256": sha256(Path(__file__)),
        "e2_runner_sha256": sha256(E2_RUNNER),
        "phase_b_authorized": False,
        "a1_a8_computed": False,
        "verdict": "NOT_YET_EVALUATED",
        "counts": summary["counts"],
        "artifacts": {
            "fold_summary.json": sha256(summary_path),
            "recorded_output.txt": sha256(recorded_path),
            "pair_fold_model_scores.csv.gz": sha256(score_path),
            **artifact_hashes,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return summary


def verify(pairs: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        generate_packet(pairs, tmp_path)
        for name in (
            "fold_summary.json",
            "recorded_output.txt",
            "pair_fold_model_scores.csv.gz",
            "manifest.json",
        ):
            if (tmp_path / name).read_bytes() != (PACKET_DIR / name).read_bytes():
                raise SystemExit(f"verification failed: {name} differs")
        sealed_params = sorted((PACKET_DIR / "parameters").glob("*.json"))
        sealed_sels = sorted((PACKET_DIR / "selections").glob("*.json"))
        if len(sealed_params) != 28 or len(sealed_sels) != 7:
            raise SystemExit(
                f"sealed artifact count wrong: {len(sealed_params)} params, "
                f"{len(sealed_sels)} selections"
            )
        for sealed in sealed_params + sealed_sels:
            rel = sealed.relative_to(PACKET_DIR)
            rebuilt = tmp_path / rel
            if not rebuilt.exists() or rebuilt.read_bytes() != sealed.read_bytes():
                raise SystemExit(f"verification failed: {rel} differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=REPO / CANONICAL_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=PACKET_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    pairs = args.pairs.resolve()
    if args.verify:
        verify(pairs)
        print("E3 signal packet verification: PASS")
        return
    summary = generate_packet(pairs, args.output_dir)
    print(_render(summary), end="")


if __name__ == "__main__":
    main()
