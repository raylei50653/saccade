"""Contracts for the two-phase EK0 frozen-packet exact-key recoverability audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/tools/audit_frozen_packet_exact_key_recoverability.py"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("ek0_recoverability", RUNNER)
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


def _j3_row(**override: Any) -> dict[str, Any]:
    row = {
        "partition": "cohort_gap",
        "seq": "S",
        "lost_global_id": 1,
        "classification": "exact-key reconstructable",
        "reason": "exact_v2_global_pair_key",
    }
    return {**row, **override}


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


def test_j3_reduces_events_and_excludes_tracks_already_in_joined_partition() -> None:
    runner = _load_runner()
    records = [
        _j3_row(),
        _j3_row(),
        _j3_row(lost_global_id=2),
        _j3_row(
            partition="unemitted",
            lost_global_id=-1,
            classification="structurally unjoinable",
            reason="unresolved_global_identity_no_local_id_fallback",
        ),
    ]
    reduced = runner.reduce_j3(records, joined_tracks={("S", 1)})
    cohort = reduced["partitions"]["cohort_gap"]
    assert cohort["events"] == 3
    assert cohort["identified_unique_lost_tracks"] == 2
    assert cohort["identified_unique_lost_tracks_not_in_joined_partition"] == 1
    assert cohort["reconstructable_unique_lost_track_upper_bound"] == 2
    assert cohort["reconstructable_new_unique_lost_tracks"] == 1
    assert cohort["repeat_events_after_lost_track_reduction"] == 1
    assert reduced["reconstructable_unique_lost_track_upper_bound"] == 2
    assert reduced["reconstructable_tracks_already_in_joined_partition"] == 1
    assert reduced["reconstructable_new_unique_lost_tracks"] == 1
    # Track ("S", 1) is already base exposure: only one new track merges.
    assert reduced["n_max_zero_new_hurt"] == 117
    assert reduced["partitions"]["unemitted"]["identified_unique_lost_tracks"] == 0


def test_terminal_mapping_is_ordered_and_exhaustive() -> None:
    runner = _load_runner()
    assert runner.clopper_pearson_upper(3, 152) > 0.05
    assert runner.clopper_pearson_upper(3, 153) <= 0.05

    def j3(new: int, n_max: int) -> dict[str, int]:
        return {
            "reconstructable_new_unique_lost_tracks": new,
            "n_max_zero_new_hurt": n_max,
        }

    def reveal(counts: list[dict[str, Any]]) -> dict[str, Any]:
        return {"possible_merged_counts": counts}

    assert (
        runner.determine_terminal(
            j3=j3(0, 116), reveal=reveal([{"n": 116, "k": 3, "ucb": 0.0655}])
        )
        == "EK0_NO_RECOVERABLE_SUPPORT"
    )
    assert (
        runner.determine_terminal(
            j3=j3(10, 126), reveal=reveal([{"n": 126, "k": 3, "ucb": 0.06}])
        )
        == "EK0_RECOVERABLE_SUPPORT_BELOW_FLOOR"
    )
    # Reaches the floor in zero-hurt terms, but realized hurt keeps UCB > 0.05:
    # this must NOT be reported as below-floor futility.
    assert (
        runner.determine_terminal(
            j3=j3(40, 156), reveal=reveal([{"n": 156, "k": 5, "ucb": 0.066}])
        )
        == "EK0_RECOVERABLE_SUPPORT_UCB_NOT_MET"
    )
    assert (
        runner.determine_terminal(
            j3=j3(40, 156), reveal=reveal([{"n": 156, "k": 3, "ucb": 0.049}])
        )
        == "EK0_RECOVERABLE_SUPPORT_SUFFICIENT"
    )


def test_reveal_seal_refuses_swapped_runner_or_tampered_artifacts() -> None:
    runner = _load_runner()
    sealed = {
        "phase": "outcome_blind_sealed",
        "terminal": None,
        "declaration_sha256": "d" * 64,
        "runner_sha256": "r" * 64,
        "pre_gt_seal": {"gt_label_accessed": False, "inventory_sha256": "i" * 64},
        "files": {"inventory.csv": "i" * 64, "metrics.json": "m" * 64},
    }
    intact = dict(
        declaration_sha256="d" * 64,
        runner_sha256="r" * 64,
        metrics_sha256="m" * 64,
        inventory_sha256="i" * 64,
    )
    runner.verify_reveal_seal(sealed, **intact)

    for field, bad in (
        ("runner_sha256", "x" * 64),
        ("declaration_sha256", "x" * 64),
        ("metrics_sha256", "x" * 64),
        ("inventory_sha256", "x" * 64),
    ):
        with pytest.raises(runner.AuditInvalid):
            runner.verify_reveal_seal(sealed, **{**intact, field: bad})

    with pytest.raises(runner.AuditInvalid):
        runner.verify_reveal_seal({**sealed, "phase": "complete"}, **intact)


def test_empty_new_track_stratum_does_not_read_gt() -> None:
    runner = _load_runner()
    reveal = runner.evaluate_reveal(
        inventory=[],
        pairs_path=Path("unused"),
        grid_path=Path("unused"),
        joined_tracks=set(),
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


def test_reconstructable_rows_overlapping_base_are_excluded_without_gt() -> None:
    runner = _load_runner()
    row = {
        "seq": "S",
        "lost_global_id": "1",
        "cand_global_id": "2",
        "classification": "exact-key reconstructable",
    }
    reveal = runner.evaluate_reveal(
        inventory=[row],
        pairs_path=Path("unused"),
        grid_path=Path("unused"),
        joined_tracks={("S", 1)},
    )
    assert reveal["gt_label_accessed"] is False
    assert reveal["reconstructable_rows_total"] == 1
    assert reveal["reconstructable_tracks_overlapping_base_excluded"] == 1
    assert reveal["additional_gt_valid_match_unique_lost_tracks"] == 0
