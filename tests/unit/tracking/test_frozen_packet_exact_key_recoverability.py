"""Contracts for the single-phase EK0 frozen-packet consistency audit."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/tools/audit_frozen_packet_exact_key_recoverability.py"

CAPTURE_HEADER = (
    "event_key,event_key_version,partition,seq,lost_global_id,cand_global_id,"
    "lost_local_id,cand_local_id,dist_h,ema_lost,ema_cand"
)
PAIRS_HEADER = "seq,lost_id,cand_id,dist_h,h_lost_raw,h_cand_raw"

CONSISTENT_ROWS = {
    "matched": "S|1|2,d0_event_key_v2_global,matched,S,1,2,5,6,0.1,10,10",
    "cohort_gap": "S|3|4,d0_event_key_v2_global,cohort_gap,S,3,4,7,8,0.2,10,10",
    "unemitted": ",d0_event_key_v2_global,unemitted,S,-1,-1,9,10,0.3,10,10",
}
CROSS_LABEL_ROWS = {
    # unemitted shape (unresolved identity, no key) under a cohort_gap label.
    "cohort_gap": ",d0_event_key_v2_global,cohort_gap,S,-1,-1,7,8,0.2,10,10",
    # cohort_gap shape (resolved identity) under an unemitted label.
    "unemitted": ",d0_event_key_v2_global,unemitted,S,5,6,9,10,0.3,10,10",
}


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


def _classify(runner: Any, event: dict[str, Any], **override: Any) -> tuple[str, str]:
    kwargs: dict[str, Any] = {
        "offline_universe": {},
        "offline_nonunique": set(),
        "duplicated_event_keys": set(),
        **override,
    }
    return runner.classify_event(event, **kwargs)[:2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_fixture(
    root: Path,
    module: Any,
    *,
    cohort_pair_in_offline_universe: bool = False,
    row_overrides: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write a tiny synthetic frozen packet and freeze the module onto it.

    One event per partition, each matching its exporter shape by default.
    `row_overrides` swaps a partition's row for a defective shape;
    `cohort_pair_in_offline_universe` enumerates the cohort_gap pair offline,
    contradicting its label.
    """
    study_dir = root / "study"
    substrate_dir = root / "substrate"
    study_dir.mkdir()
    substrate_dir.mkdir()
    rows = {**CONSISTENT_ROWS, **(row_overrides or {})}
    with gzip.open(study_dir / "capture.csv.gz", "wt", encoding="utf-8") as stream:
        stream.write(CAPTURE_HEADER + "\n" + "\n".join(rows.values()) + "\n")
    pairs_rows = ["S,1,2,0.1,10,10"]
    if cohort_pair_in_offline_universe:
        pairs_rows.append("S,3,4,0.2,10,10")
    (study_dir / "pairs.csv").write_text(
        PAIRS_HEADER + "\n" + "\n".join(pairs_rows) + "\n", encoding="utf-8"
    )
    (substrate_dir / "_global_id_map.txt").write_text("S 1 1\n", encoding="utf-8")
    (substrate_dir / "MOT17-99.txt").write_text("1,1,0,0,1,1,1,-1,-1,-1\n")
    partition = {"matched": 1, "cohort_gap": 1, "unemitted": 1}
    (study_dir / "capture.csv.gz.manifest.json").write_text(
        json.dumps(
            {
                "event_key_version": "d0_event_key_v2_global",
                "event_key_fields": ["seq", "lost_global_id", "cand_global_id"],
                "partition": partition,
                "provenance": {"shadow": True},
                "overflow_events": 0,
            }
        ),
        encoding="utf-8",
    )
    module.EXPECTED_PARTITION = partition
    module.EXPECTED_INPUT_HASHES = {
        "pairs.csv": _sha256(study_dir / "pairs.csv"),
        "capture.csv.gz": _sha256(study_dir / "capture.csv.gz"),
        "capture.csv.gz.manifest.json": _sha256(
            study_dir / "capture.csv.gz.manifest.json"
        ),
        "_global_id_map.txt": _sha256(substrate_dir / "_global_id_map.txt"),
        "substrate_mot_concat": _sha256(substrate_dir / "MOT17-99.txt"),
    }
    return study_dir, substrate_dir


def test_classification_verifies_each_partition_label_shape() -> None:
    runner = _load_runner()
    key = ("MOT17-02-SDP", 10, 20)
    universe = {key: {"offline_coordinates_available": True}}

    exact = _classify(runner, _event(), offline_universe=universe)
    assert exact == ("exact-key reconstructable", "exact_v2_global_pair_key")

    consistent_gap = _classify(runner, _event())
    assert consistent_gap == (
        "structurally unjoinable",
        "same_global_pair_absent_from_offline_universe",
    )

    consistent_unemitted = _classify(
        runner,
        _event(
            partition="unemitted", event_key="", lost_global_id=-1, cand_global_id=-1
        ),
    )
    assert consistent_unemitted == (
        "structurally unjoinable",
        "unresolved_global_identity_no_local_id_fallback",
    )

    # Cross-label shapes are packet defects, not structural unjoinability.
    unemitted_shape_in_cohort_gap = _classify(
        runner, _event(event_key="", lost_global_id=-1, cand_global_id=-1)
    )
    assert unemitted_shape_in_cohort_gap == (
        "partition label inconsistent",
        "cohort_gap_row_has_unresolved_global_identity",
    )

    cohort_gap_shape_in_unemitted = _classify(
        runner, _event(partition="unemitted", event_key="")
    )
    assert cohort_gap_shape_in_unemitted == (
        "partition label inconsistent",
        "unemitted_row_has_resolved_global_identity",
    )

    keyless_cohort_gap = _classify(runner, _event(event_key=""))
    assert keyless_cohort_gap == (
        "partition label inconsistent",
        "cohort_gap_row_missing_canonical_event_key",
    )

    # An enumerated pair refutes the cohort_gap label even without usable
    # frozen coordinates.
    pair_without_coordinates = _classify(
        runner,
        _event(),
        offline_universe={key: {"offline_coordinates_available": False}},
    )
    assert pair_without_coordinates == (
        "partition label inconsistent",
        "cohort_gap_pair_enumerated_without_frozen_coordinates",
    )


def test_j3_reduction_is_descriptive_and_terminal_mapping_exhaustive() -> None:
    runner = _load_runner()
    consistent = runner.reduce_j3(
        [
            {
                "partition": "cohort_gap",
                "seq": "S",
                "lost_global_id": 1,
                "classification": "structurally unjoinable",
                "reason": "same_global_pair_absent_from_offline_universe",
            },
            {
                "partition": "cohort_gap",
                "seq": "S",
                "lost_global_id": 1,
                "classification": "structurally unjoinable",
                "reason": "same_global_pair_absent_from_offline_universe",
            },
            {
                "partition": "unemitted",
                "seq": "S",
                "lost_global_id": -1,
                "classification": "structurally unjoinable",
                "reason": "unresolved_global_identity_no_local_id_fallback",
            },
        ]
    )
    cohort = consistent["partitions"]["cohort_gap"]
    assert cohort["events"] == 2
    assert cohort["identified_unique_lost_tracks"] == 1
    assert cohort["repeat_events_after_lost_track_reduction"] == 1
    assert consistent["reconstructable_events"] == 0
    assert consistent["provenance_ambiguous_events"] == 0
    assert consistent["partition_label_inconsistent_events"] == 0
    assert runner.determine_terminal(consistent) == "EK0_NO_RECOVERABLE_SUPPORT"
    # No exposure/floor/UCB machinery may exist in the audit.
    assert "n_max_zero_new_hurt" not in consistent
    assert not hasattr(runner, "clopper_pearson_upper")

    for defect in (
        "reconstructable_events",
        "provenance_ambiguous_events",
        "partition_label_inconsistent_events",
    ):
        assert (
            runner.determine_terminal(dict(consistent, **{defect: 1}))
            == "EK0_PACKET_INCONSISTENT"
        )


def test_consistent_fixture_lands_no_recoverable_support(tmp_path: Path) -> None:
    runner = _load_runner()
    study, substrate = _build_fixture(tmp_path, runner)
    out = tmp_path / "out"
    metrics = runner.run_audit(study_dir=study, substrate_dir=substrate, output_dir=out)
    assert metrics["terminal"] == "EK0_NO_RECOVERABLE_SUPPORT"
    assert metrics["gt_label_accessed"] is False
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["terminal"] == "EK0_NO_RECOVERABLE_SUPPORT"
    assert manifest["runner_sha256"] == _sha256(RUNNER)
    assert (
        manifest["input_hashes"]["capture.csv.gz.manifest.json"]
        == runner.EXPECTED_INPUT_HASHES["capture.csv.gz.manifest.json"]
    )
    assert manifest["files"]["inventory.csv"] == _sha256(out / "inventory.csv")
    assert manifest["files"]["metrics.json"] == _sha256(out / "metrics.json")


def test_rescuable_row_lands_packet_inconsistent(tmp_path: Path) -> None:
    runner = _load_runner()
    study, substrate = _build_fixture(
        tmp_path, runner, cohort_pair_in_offline_universe=True
    )
    metrics = runner.run_audit(
        study_dir=study, substrate_dir=substrate, output_dir=tmp_path / "out"
    )
    assert metrics["terminal"] == "EK0_PACKET_INCONSISTENT"
    assert metrics["j3_reduction"]["reconstructable_events"] == 1


@pytest.mark.parametrize("cross_labeled", sorted(CROSS_LABEL_ROWS))
def test_cross_label_shape_lands_packet_inconsistent(
    tmp_path: Path, cross_labeled: str
) -> None:
    runner = _load_runner()
    study, substrate = _build_fixture(
        tmp_path,
        runner,
        row_overrides={cross_labeled: CROSS_LABEL_ROWS[cross_labeled]},
    )
    metrics = runner.run_audit(
        study_dir=study, substrate_dir=substrate, output_dir=tmp_path / "out"
    )
    assert metrics["terminal"] == "EK0_PACKET_INCONSISTENT"
    assert metrics["j3_reduction"]["partition_label_inconsistent_events"] == 1


@pytest.mark.parametrize(
    "mutated", ["pairs.csv", "capture.csv.gz", "capture.csv.gz.manifest.json"]
)
def test_mutated_frozen_input_is_invalid(tmp_path: Path, mutated: str) -> None:
    runner = _load_runner()
    study, substrate = _build_fixture(tmp_path, runner)
    target = study / mutated
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(runner.AuditInvalid, match="hash mismatch"):
        runner.run_audit(
            study_dir=study, substrate_dir=substrate, output_dir=tmp_path / "out"
        )


def test_completed_packet_is_immutable(tmp_path: Path) -> None:
    runner = _load_runner()
    study, substrate = _build_fixture(tmp_path, runner)
    out = tmp_path / "out"
    runner.run_audit(study_dir=study, substrate_dir=substrate, output_dir=out)
    sealed = {name: _sha256(out / name) for name in ("manifest.json", "metrics.json")}

    # A plain rerun must fail closed before touching the packet.
    with pytest.raises(runner.AuditInvalid, match="immutable"):
        runner.run_audit(study_dir=study, substrate_dir=substrate, output_dir=out)

    # Even a failing run routed through main() must not clobber the packet.
    (study / "pairs.csv").write_bytes((study / "pairs.csv").read_bytes() + b"x")
    exit_code = runner.main(
        [
            "--output-dir",
            str(out),
            "--study-dir",
            str(study),
            "--substrate-dir",
            str(substrate),
        ]
    )
    assert exit_code == 1
    for name, digest in sealed.items():
        assert _sha256(out / name) == digest
