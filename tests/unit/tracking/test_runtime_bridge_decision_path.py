"""P0 must fail closed before outcome access when capture provenance cannot be aligned.

The original P0 run hard-coded the `s` preset as "the headline" and read the
resulting `px` / `dir_bonus` delta as proof of a *foreign* capture. The audited
D0/R1/S0 evidence is sealed on `m`, whose preset genuinely resolves to
`px=0.4` / `dir_bonus=0.0` — so that delta proved nothing (declaration
Correction 1).

What actually forces the terminal is preset-independent: `h_lo`, `h_hi`,
`spatial_gate` and `max_speed` are never stamped into capture provenance, so
alignment fails against *any* policy target. These tests pin both halves — the
matching knobs under `m`, and the terminal that fires regardless.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/tools/audit_runtime_bridge_decision_path.py"

SEALED_PRESET = "mamba_whole_graph_m"  # the preset D0/R1/S0 are captured under
FOREIGN_PRESET = "mamba_whole_graph"  # the `s` preset P0 originally assumed


def _load_runner():
    spec = importlib.util.spec_from_file_location("p0_audit", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_d0_capture_dir(tmp_path: Path) -> Path:
    """Create only the outcome-free D0 fields read before P0 fails closed.

    These are the real stamped values: they are `m`'s, and they are all the
    bridge provenance a capture manifest carries.
    """
    d0_dir = tmp_path / "d0"
    d0_dir.mkdir()
    with gzip.open(d0_dir / "capture.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("seq,lost_global_id,cand_global_id,bdist\\n")
    (d0_dir / "capture.csv.gz.manifest.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "bridge": {
                        "at": 4,
                        "dir_bonus": 0.0,
                        "min_lost": 2,
                        "px": 0.4,
                        "ttl": 120,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return d0_dir


@pytest.mark.parametrize("preset", [SEALED_PRESET, FOREIGN_PRESET])
def test_p0_fails_closed_before_label_access_under_any_policy_target(
    tmp_path: Path, preset: str
) -> None:
    runner = _load_runner()
    result = runner.audit(
        ROOT, policy_preset=preset, d0_capture_dir=_synthetic_d0_capture_dir(tmp_path)
    )

    assert result["terminal"] == "P0_CAPTURE_SEMANTICS_INVALID"
    assert result["label_access"] == {
        "gt_or_fp_labels_accessed": False,
        "p5": "not_entered",
    }
    assert result["provenance"]["d0_alignment"]["all_fields_match"] is False
    assert result["provenance"]["r1_frozen_preset"].endswith("mamba_whole_graph_m.yaml")


def test_terminal_is_forced_by_unstamped_knobs_not_by_a_foreign_config(
    tmp_path: Path,
) -> None:
    """Against the preset the evidence is actually sealed on, the stamped knobs agree.

    Only the knobs the capture never stamps disagree — which is why the terminal
    is over-determined and its foreign-capture reading is wrong.
    """
    runner = _load_runner()
    result = runner.audit(
        ROOT,
        policy_preset=SEALED_PRESET,
        d0_capture_dir=_synthetic_d0_capture_dir(tmp_path),
    )
    status = {
        row["headline_knob"]: row["status"]
        for row in result["provenance"]["d0_alignment"]["comparisons"]
    }

    assert status["relink_bridge_px"] == "match"
    assert status["relink_bridge_dir_bonus"] == "match"
    for unstamped in (
        "relink_bridge_h_lo",
        "relink_bridge_h_hi",
        "relink_bridge_spatial_gate",
        "relink_bridge_max_speed",
    ):
        assert status[unstamped] == "mismatch_or_absent"


def test_p0_keeps_candidate_and_commit_replay_below_l2(tmp_path: Path) -> None:
    runner = _load_runner()
    result = runner.audit(
        ROOT,
        policy_preset=SEALED_PRESET,
        d0_capture_dir=_synthetic_d0_capture_dir(tmp_path),
    )
    matrix = {row["stage"]: row for row in result["field_sufficiency"]}

    assert matrix["D_pair_cutoff"]["complete"] is False
    assert matrix["E_candidate_local_ranking"]["complete"] is False
    assert matrix["F_claim_competition"]["complete"] is False
    assert matrix["G_commit"]["complete"] is False
    assert (
        result["replay"]["counterfactual_ceiling_if_headline_alignment_existed"]
        == "L1_pair_cutoff_replay"
    )
