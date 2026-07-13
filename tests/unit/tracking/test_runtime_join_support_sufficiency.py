"""Contracts for the two-phase RJ0 runtime-join support sufficiency audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/tools/audit_runtime_join_support_sufficiency.py"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("rj0_join_support", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(**override: Any) -> dict[str, Any]:
    event = {
        "event_ordinal": 1,
        "partition": "cohort_gap",
        "seq": "MOT17-02-SDP",
        "event_key": "MOT17-02-SDP|10|20",
        "event_key_version": "d0_event_key_v2_global",
        "lost_global_id": 10,
        "cand_global_id": 20,
        "lost_local_id_present": True,
        "cand_local_id_present": True,
        "runtime_coordinates_available": True,
        "runtime_dist_h": "0.1",
        "runtime_ema_lost": "10",
        "runtime_ema_cand": "10",
    }
    return {**event, **override}


def test_classification_is_outcome_blind_and_never_uses_local_id_fallback() -> None:
    runner = _load_runner()
    key = ("MOT17-02-SDP", 10, 20)
    universe = {key: {"offline_coordinates_available": True}}

    exact = runner.classify_event(
        _event(),
        offline_universe=universe,
        offline_nonunique=set(),
        duplicated_event_keys=set(),
    )
    assert exact[:2] == ("exact-key reconstructable", "exact_v2_global_pair_key")

    auxiliary = runner.classify_event(
        _event(event_key=""),
        offline_universe=universe,
        offline_nonunique=set(),
        duplicated_event_keys=set(),
    )
    assert auxiliary[:2] == (
        "deterministic auxiliary-key reconstructable",
        "canonical_v2_global_fields_without_event_key",
    )

    missing_pair = runner.classify_event(
        _event(),
        offline_universe={},
        offline_nonunique=set(),
        duplicated_event_keys=set(),
    )
    assert missing_pair[:2] == (
        "structurally unjoinable",
        "same_global_pair_absent_from_offline_universe",
    )

    unresolved = runner.classify_event(
        _event(event_key="", lost_global_id=-1, cand_global_id=-1),
        offline_universe=universe,
        offline_nonunique=set(),
        duplicated_event_keys=set(),
    )
    assert unresolved[:2] == (
        "structurally unjoinable",
        "unresolved_global_identity_no_local_id_fallback",
    )


def test_j3_reduces_multiple_events_to_one_lost_track() -> None:
    runner = _load_runner()
    records = [
        {
            "partition": "cohort_gap",
            "seq": "S",
            "lost_global_id": 1,
            "classification": "exact-key reconstructable",
            "reason": "exact_v2_global_pair_key",
        },
        {
            "partition": "cohort_gap",
            "seq": "S",
            "lost_global_id": 1,
            "classification": "exact-key reconstructable",
            "reason": "exact_v2_global_pair_key",
        },
        {
            "partition": "unemitted",
            "seq": "S",
            "lost_global_id": -1,
            "classification": "structurally unjoinable",
            "reason": "unresolved_global_identity_no_local_id_fallback",
        },
    ]
    reduced = runner.reduce_j3(records)
    cohort = reduced["partitions"]["cohort_gap"]
    assert cohort["events"] == 2
    assert cohort["identified_unique_lost_tracks"] == 1
    assert cohort["reconstructable_unique_lost_track_upper_bound"] == 1
    assert cohort["repeat_events_after_lost_track_reduction"] == 1
    assert reduced["reconstructable_unique_lost_track_upper_bound"] == 1
    assert reduced["n_max_zero_new_hurt"] == 117
    assert reduced["partitions"]["unemitted"]["identified_unique_lost_tracks"] == 0


def test_cp_floor_and_futility_are_mechanical() -> None:
    runner = _load_runner()
    assert runner.clopper_pearson_upper(3, 152) > 0.05
    assert runner.clopper_pearson_upper(3, 153) <= 0.05
    assert (
        runner.determine_terminal(
            j3={"n_max_zero_new_hurt": 152},
            reveal={"possible_merged_counts": [{"n": 152, "k": 3, "ucb": 0.06}]},
        )
        == "RJ0_EXPANSION_FUTILE"
    )
    assert (
        runner.determine_terminal(
            j3={"n_max_zero_new_hurt": 153},
            reveal={"possible_merged_counts": [{"n": 153, "k": 3, "ucb": 0.049}]},
            semantic_admissible=False,
        )
        == "RJ0_EXPANSION_INADMISSIBLE"
    )


def test_empty_frozen_reconstructable_stratum_does_not_read_gt() -> None:
    runner = _load_runner()
    reveal = runner.evaluate_reveal(
        inventory=[], pairs_path=Path("unused"), grid_path=Path("unused")
    )
    assert reveal["gt_label_accessed"] is False
    assert reveal["additional_gt_valid_match_unique_lost_tracks"] == 0
    assert reveal["additional_hurt_observable_range"] == [0, 0]
    assert reveal["possible_merged_counts"] == [
        {
            "n": 116,
            "k": 3,
            "ucb": pytest.approx(runner.clopper_pearson_upper(3, 116)),
        }
    ]
