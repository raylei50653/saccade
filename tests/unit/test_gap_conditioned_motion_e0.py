from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO / "docs/modules/semantic/research/evidence/"
    "gap_conditioned_motion_e0_20260711/run_e0_audit.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("gap_motion_e0", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pairs(path: Path, *, include_vectors: bool = False) -> None:
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
        "bridge_dist",
        "fwd_resid",
        "bwd_resid",
        "dir_cos",
        "lost_exit_speed",
        "cand_entry_speed",
    ]
    if include_vectors:
        fields += [
            "lost_exit_vx",
            "lost_exit_vy",
            "cand_entry_vx",
            "cand_entry_vy",
        ]
    row = {name: "0" for name in fields}
    row.update(
        seq="MOT17-02-SDP",
        lost_id="1",
        cand_id="2",
        gt_match="1",
        gt_valid="1",
        gap="10",
        lost_last_frame="5",
        cand_first_frame="15",
        lost_foot_x="10",
        lost_foot_y="20",
        cand_foot_x="12",
        cand_foot_y="24",
        h_ref="50",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def test_frozen_shape_is_partially_identifiable_without_vector_velocity(tmp_path):
    runner = _load_runner()
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs)

    result = runner.analyze(pairs)

    ident = result["identifiability"]
    assert ident["verdict"] == "PARTIALLY_IDENTIFIABLE"
    assert ident["m0_deterministic"]["identifiable"] is True
    assert ident["position_only_transition"]["identifiable"] is True
    assert ident["joint_position_velocity_transition"]["identifiable"] is False
    assert ident["contexts"]["loo_headline_eligible"] == ["global"]
    assert result["gap_bins"]["1-10"] == {"pairs": 1, "gt": 1, "fp": 0}


def test_joint_becomes_identifiable_only_when_vector_components_exist(tmp_path):
    runner = _load_runner()
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs, include_vectors=True)

    result = runner.analyze(pairs)

    assert result["identifiability"]["verdict"] == "IDENTIFIABLE"
    assert (
        result["identifiability"]["joint_position_velocity_transition"]["identifiable"]
        is True
    )


def test_gap_identity_mismatch_fails_position_gate(tmp_path):
    runner = _load_runner()
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs)
    text = pairs.read_text(encoding="utf-8").replace(",15,", ",16,")
    pairs.write_text(text, encoding="utf-8")

    result = runner.analyze(pairs)

    assert result["integrity"]["invalid_gap_identity"] == 1
    assert result["identifiability"]["verdict"] == "NOT_IDENTIFIABLE"
