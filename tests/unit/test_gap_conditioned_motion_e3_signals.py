from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
E3_RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence/"
    / "gap_conditioned_motion_e3_signals_20260711/run_e3_signals.py"
)
E2_RUNNER = (
    REPO
    / "docs/modules/semantic/research/evidence/"
    / "gap_conditioned_motion_e2_family_20260711/run_e2_family_freeze.py"
)
PACKET = (
    REPO
    / "docs/modules/semantic/research/evidence/"
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
            # One FP row per sequence to prove non-GT scores are emitted.
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


def test_score_pairs_retains_four_models_for_every_pair(tmp_path):
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
        assert held_out not in selection["train_sequences"]
        assert set(selection["training_nll_by_model"]) == set(e2.MODEL_ORDER)

    pair_rows = e3.load_scoreable_pairs(pairs)
    score_rows = e3.score_pairs_for_folds(
        e2,
        pair_rows,
        parameters_by_fold,
        selections_by_fold,
        e2.sha256(pairs),
    )

    assert len(score_rows) == len(pair_rows) * 4
    by_pair: dict[tuple[str, str, str], set[str]] = {}
    for row in score_rows:
        key = (row["seq"], str(row["lost_id"]), str(row["cand_id"]))
        by_pair.setdefault(key, set()).add(row["model_id"])
        assert row["held_out_sequence"] == row["seq"]
        assert row["fold_id"] == f"LOO::{row['seq']}"
        assert np.isclose(
            row["nll_motion"],
            0.5
            * (row["q_motion"] + row["log_det_covariance"] + row["gaussian_constant"]),
        )

    assert all(models == set(e2.MODEL_ORDER) for models in by_pair.values())
    assert any(row["is_selected_model"] == 0 for row in score_rows)
    assert any(row["is_selected_model"] == 1 for row in score_rows)
    assert any(row["gt_match"] == 0 for row in score_rows)


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
    assert sel_a["selected_model_id"] == sel_b["selected_model_id"]
    # source_pairs_sha256 differs when the CSV bytes change, so artifact IDs
    # differ; train parameters themselves must not depend on held-out values.
    assert [p["fit_row_key_sha256"] for p in params_a] == [
        p["fit_row_key_sha256"] for p in params_b
    ]
    assert [p["training_total_nll"] for p in params_a] == [
        p["training_total_nll"] for p in params_b
    ]
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
    assert summary["counts"]["n_score_rows"] == 97136
    assert summary["counts"]["n_unique_pairs_scored"] == 24284
    assert summary["counts"]["n_models_per_pair_fold"] == 4
    assert summary["a1_a8_computed"] is False
    assert summary["phase_b_authorized"] is False

    assert len(list((PACKET / "parameters").glob("*.json"))) == 28
    assert len(list((PACKET / "selections").glob("*.json"))) == 7

    # Packet must not contain A1–A8 analysis table files.
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

    with (PACKET / "pair_fold_model_scores.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.DictReader(stream)
        counts: dict[tuple[str, str, str], int] = {}
        models: set[str] = set()
        for row in reader:
            key = (row["seq"], row["lost_id"], row["cand_id"])
            counts[key] = counts.get(key, 0) + 1
            models.add(row["model_id"])
            assert row["held_out_sequence"] == row["seq"]
            assert set(e3.SCORE_FIELDS).issubset(row.keys())
    assert len(counts) == 24284
    assert all(n == 4 for n in counts.values())
    assert models == set(e3.load_e2().MODEL_ORDER)


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
