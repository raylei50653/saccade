from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
E3_RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "gap_conditioned_motion_e3_signals_20260711"
    / "run_e3_signals.py"
)
E2_RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "gap_conditioned_motion_e2_family_20260711"
    / "run_e2_family_freeze.py"
)
PACKET = (
    REPO
    / "docs/modules/semantic/research/evidence"
    / "gap_conditioned_motion_e3_signals_20260711"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pairs(path: Path, *, held_out_offset: float = 0.0) -> None:
    fields = [
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
    ]
    sequences = ("SEQ-A", "SEQ-B", "SEQ-C")
    rows = []
    for seq_index, seq in enumerate(sequences):
        for row_index in range(4):
            gap = row_index + 1
            offset = held_out_offset if seq == "SEQ-C" else 0.0
            gt_match = "0" if row_index == 3 else "1"
            rows.append(
                {
                    "seq": seq,
                    "lost_id": str(10 * seq_index + row_index),
                    "cand_id": str(100 + 10 * seq_index + row_index),
                    "gt_match": gt_match,
                    "gt_valid": "1",
                    "gap": str(gap),
                    "lost_last_frame": "10",
                    "cand_first_frame": str(10 + gap),
                    "lost_foot_x": "0",
                    "lost_foot_y": "0",
                    "cand_foot_x": str(gap * (1.0 + 0.1 * row_index) + offset),
                    "cand_foot_y": str(gap * (-0.5 + 0.07 * seq_index) - offset),
                    "h_ref": "10",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_score_cube_has_train_and_held_out_roles(tmp_path):
    e2 = _load(E2_RUNNER, "gap_motion_e2")
    e3 = _load(E3_RUNNER, "gap_motion_e3")
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs)

    parameters_by_fold = {}
    selections_by_fold = {}
    for held_out in ("SEQ-A", "SEQ-B", "SEQ-C"):
        parameters, selection = e2.build_fold_artifacts(pairs, held_out)
        parameters_by_fold[held_out] = parameters
        selections_by_fold[held_out] = selection

    pair_rows = e3.load_scoreable_pairs(pairs)
    score_rows = e3.score_fold_cube(
        e2,
        pair_rows,
        parameters_by_fold,
        selections_by_fold,
        e2.sha256(pairs),
    )

    # Full cube: 3 folds × 12 pairs × 4 models.
    assert len(score_rows) == 3 * len(pair_rows) * 4

    fold_pair_models: dict[tuple[str, str, str, str], set[str]] = {}
    train_by_fold: dict[str, set[str]] = {}
    held_by_fold: dict[str, set[str]] = {}
    for row in score_rows:
        key = (
            row["held_out_sequence"],
            row["seq"],
            str(row["lost_id"]),
            str(row["cand_id"]),
        )
        fold_pair_models.setdefault(key, set()).add(row["model_id"])
        assert np.isclose(
            row["nll_motion"],
            0.5
            * (row["q_motion"] + row["log_det_covariance"] + row["gaussian_constant"]),
        )
        if row["evaluation_role"] == "held_out":
            assert row["seq"] == row["held_out_sequence"]
            held_by_fold.setdefault(row["held_out_sequence"], set()).add(row["seq"])
        else:
            assert row["evaluation_role"] == "train"
            assert row["seq"] != row["held_out_sequence"]
            train_by_fold.setdefault(row["held_out_sequence"], set()).add(row["seq"])

    assert all(models == set(e2.MODEL_ORDER) for models in fold_pair_models.values())
    # Every fold has train scores on the other two sequences.
    for held_out in ("SEQ-A", "SEQ-B", "SEQ-C"):
        assert held_by_fold[held_out] == {held_out}
        assert train_by_fold[held_out] == {"SEQ-A", "SEQ-B", "SEQ-C"} - {held_out}

    # A6 surface: train rows under fold-C use fold-C parameters, not own-LOO.
    fold_c_train = [
        r
        for r in score_rows
        if r["held_out_sequence"] == "SEQ-C" and r["evaluation_role"] == "train"
    ]
    assert fold_c_train
    assert all(r["fold_id"] == "LOO::SEQ-C" for r in fold_c_train)
    assert all(r["seq"] in {"SEQ-A", "SEQ-B"} for r in fold_c_train)
    assert any(r["is_selected_model"] == 0 for r in score_rows)
    assert any(r["gt_match"] == 0 for r in score_rows)


def test_held_out_mutation_does_not_change_train_artifacts_but_changes_scores(
    tmp_path,
):
    e2 = _load(E2_RUNNER, "gap_motion_e2")
    e3 = _load(E3_RUNNER, "gap_motion_e3")
    original = tmp_path / "original.csv"
    mutated = tmp_path / "mutated.csv"
    _write_pairs(original)
    _write_pairs(mutated, held_out_offset=1000.0)

    params_a, sel_a = e2.build_fold_artifacts(original, "SEQ-C")
    params_b, sel_b = e2.build_fold_artifacts(mutated, "SEQ-C")

    assert sel_a["fit_row_key_sha256"] == sel_b["fit_row_key_sha256"]
    assert sel_a["training_nll_by_model"] == sel_b["training_nll_by_model"]
    assert [p["drift_per_frame"] for p in params_a] == [
        p["drift_per_frame"] for p in params_b
    ]
    assert [p["base_covariance"] for p in params_a] == [
        p["base_covariance"] for p in params_b
    ]

    rows_a = [r for r in e3.load_scoreable_pairs(original) if r["seq"] == "SEQ-C"]
    rows_b = [r for r in e3.load_scoreable_pairs(mutated) if r["seq"] == "SEQ-C"]
    disp_a = np.asarray([[r["dx_h"], r["dy_h"]] for r in rows_a])
    disp_b = np.asarray([[r["dx_h"], r["dy_h"]] for r in rows_b])
    gaps = np.asarray([r["gap"] for r in rows_a], dtype=np.float64)
    assert not np.allclose(disp_a, disp_b)

    scores_a = e2.score_model(params_a[0], disp_a, gaps)
    scores_b = e2.score_model(params_b[0], disp_b, gaps)
    assert not np.allclose(scores_a["nll_motion"], scores_b["nll_motion"])


def test_train_scores_use_fold_params_not_own_loo(tmp_path):
    """A6 blocker: training sequences must be scored under fold-f params."""
    e2 = _load(E2_RUNNER, "gap_motion_e2")
    e3 = _load(E3_RUNNER, "gap_motion_e3")
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs)

    parameters_by_fold = {}
    selections_by_fold = {}
    for held_out in ("SEQ-A", "SEQ-B", "SEQ-C"):
        parameters, selection = e2.build_fold_artifacts(pairs, held_out)
        parameters_by_fold[held_out] = parameters
        selections_by_fold[held_out] = selection

    pair_rows = e3.load_scoreable_pairs(pairs)
    score_rows = e3.score_fold_cube(
        e2, pair_rows, parameters_by_fold, selections_by_fold, e2.sha256(pairs)
    )

    # SEQ-A under fold-C (train role) must use fold-C parameter IDs.
    fold_c_param_ids = {
        p["model_id"]: p["parameter_artifact_id"] for p in parameters_by_fold["SEQ-C"]
    }
    fold_a_param_ids = {
        p["model_id"]: p["parameter_artifact_id"] for p in parameters_by_fold["SEQ-A"]
    }
    # Different folds → different parameters (different train sets).
    assert fold_c_param_ids != fold_a_param_ids

    for row in score_rows:
        if row["held_out_sequence"] != "SEQ-C" or row["seq"] != "SEQ-A":
            continue
        assert row["evaluation_role"] == "train"
        assert row["parameter_artifact_id"] == fold_c_param_ids[row["model_id"]]
        assert row["parameter_artifact_id"] != fold_a_param_ids[row["model_id"]]
        assert (
            row["selection_artifact_id"]
            == selections_by_fold["SEQ-C"]["selection_artifact_id"]
        )


def test_sealed_packet_contract_counts_and_no_phase_b():
    e3 = _load(E3_RUNNER, "gap_motion_e3")
    manifest = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((PACKET / "fold_summary.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "E3_SIGNALS_SEALED"
    assert manifest["phase_b_authorized"] is False
    assert manifest["a1_a8_computed"] is False
    assert manifest["verdict"] == "NOT_YET_EVALUATED"
    assert summary["counts"]["n_parameter_artifacts"] == 28
    assert summary["counts"]["n_selection_artifacts"] == 7
    # Full cube: 24284 pairs × 7 folds × 4 models.
    assert summary["counts"]["n_score_rows"] == 679952
    assert summary["counts"]["n_held_out_score_rows"] == 97136
    assert summary["counts"]["n_train_score_rows"] == 582816
    assert summary["counts"]["n_unique_pairs_scored"] == 24284
    assert summary["counts"]["n_models_per_pair_fold"] == 4

    # Phase B design seal recorded for audit lineage.
    design = manifest["phase_b_design"]
    assert design["predeclaration_seal_commit"] == e3.PHASE_B_DESIGN_SEAL_COMMIT
    assert design["path"] == e3.PHASE_B_DESIGN_REL
    assert design["content_sha256"] == e3.sha256(e3.PHASE_B_DESIGN)
    assert summary["phase_b_design"] == design

    assert len(list((PACKET / "parameters").glob("*.json"))) == 28
    assert len(list((PACKET / "selections").glob("*.json"))) == 7

    forbidden_stems = {
        "a1_calibration",
        "a2_role_reversal",
        "a3_short_gap_retention",
        "a4_escape_tail",
        "a5_separability",
        "a6_conditional_sr",
        "a7_loo_transfer",
        "a8_m1_vs_m2",
        "phase_b_verdict",
        "v1_v5_verdict",
    }
    for path in PACKET.rglob("*"):
        if path.is_file():
            assert path.stem.lower() not in forbidden_stems

    fold_pair_model: dict[tuple[str, str, str, str], set[str]] = {}
    train_counts: dict[str, int] = {}
    held_counts: dict[str, int] = {}
    scores_path = PACKET / "pair_fold_model_scores.csv.gz"
    assert scores_path.is_file()
    with gzip.open(scores_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            assert set(e3.SCORE_FIELDS).issubset(row.keys())
            key = (
                row["held_out_sequence"],
                row["seq"],
                row["lost_id"],
                row["cand_id"],
            )
            fold_pair_model.setdefault(key, set()).add(row["model_id"])
            if row["evaluation_role"] == "held_out":
                assert row["seq"] == row["held_out_sequence"]
                held_counts[row["held_out_sequence"]] = (
                    held_counts.get(row["held_out_sequence"], 0) + 1
                )
            else:
                assert row["evaluation_role"] == "train"
                assert row["seq"] != row["held_out_sequence"]
                train_counts[row["held_out_sequence"]] = (
                    train_counts.get(row["held_out_sequence"], 0) + 1
                )

    assert len(fold_pair_model) == 24284 * 7
    assert all(
        models == set(e3.load_e2().MODEL_ORDER) for models in fold_pair_model.values()
    )
    # Per-fold train score counts match fold_summary.
    for fold in summary["folds"]:
        held = fold["held_out_sequence"]
        assert train_counts[held] == fold["train_score_row_count"]
        assert held_counts[held] == fold["held_out_score_row_count"]
        assert fold["train_score_row_count"] > 0
        assert fold["train_pair_count"] == 24284 - fold["held_out_pair_count"]


def test_sealed_loo_lineage_matches_e2_map():
    e3 = _load(E3_RUNNER, "gap_motion_e3")
    model_order = set(e3.load_e2().MODEL_ORDER)
    for held_out, expected_hash in e3.SEALED_TRAIN_HASHES.items():
        slug = e3._fold_slug(held_out)
        selection = json.loads(
            (PACKET / "selections" / f"LOO__{slug}.json").read_text(encoding="utf-8")
        )
        assert selection["fit_row_key_sha256"] == expected_hash
        assert selection["fit_row_count"] == e3.SEALED_TRAIN_COUNTS[held_out]
        assert held_out not in selection["train_sequences"]
        assert set(selection["training_nll_by_model"]) == model_order

        params = list((PACKET / "parameters").glob(f"LOO__{slug}__*.json"))
        assert len(params) == 4
        for path in params:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            assert artifact["fit_row_key_sha256"] == expected_hash
            assert artifact["held_out_sequence"] == held_out
            assert held_out not in artifact["train_sequences"]
