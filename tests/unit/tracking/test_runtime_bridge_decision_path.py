"""P0's terminal must be derived from the *kind* of evidence found, not named ahead.

The original audit had one status for every way alignment could fail —
`mismatch_or_absent` — and one terminal named after the strongest of them. So a
stamped field that *contradicts* the audited policy and a field the capture never
stamps at all produced the same verdict, and the weaker evidence inherited the
stronger claim: an absence of evidence was published as evidence of invalidity.

These tests hold the two apart:

  * `_alignment` must report `mismatch` and `absent` as different statuses;
  * `derive_terminal` must rank contradiction above absence;
  * and the audit must therefore reach *different* terminals for the two presets —
    `s` (which the capture genuinely contradicts) and `m` (which it merely cannot
    certify).

The synthetic fixture below cannot exercise the last point end-to-end: a made-up
capture file has a made-up hash, which is itself a contradiction and masks the
policy comparison. So the mechanism is pinned hermetically here, and the
end-to-end split is asserted against the real sealed artifacts where they exist.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/tools/audit_runtime_bridge_decision_path.py"
REAL_CAPTURE = ROOT / "out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z"

SEALED_PRESET = "mamba_whole_graph_m"  # the preset D0/R1/S0 are captured under
FOREIGN_PRESET = "mamba_whole_graph"  # the `s` preset P0 originally assumed

# Exactly the bridge provenance a capture manifest carries — these are `m`'s values.
STAMPED_PROVENANCE = {"at": 4, "dir_bonus": 0.0, "min_lost": 2, "px": 0.4, "ttl": 120}
NEVER_STAMPED = [
    "relink_bridge_h_lo",
    "relink_bridge_h_hi",
    "relink_bridge_spatial_gate",
    "relink_bridge_max_speed",
]


def _load_runner():
    spec = importlib.util.spec_from_file_location("p0_audit", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_d0_capture_dir(tmp_path: Path) -> Path:
    """Only the outcome-free D0 fields read before P0 fails closed."""
    d0_dir = tmp_path / "d0"
    d0_dir.mkdir()
    with gzip.open(d0_dir / "capture.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("seq,lost_global_id,cand_global_id,bdist\\n")
    (d0_dir / "capture.csv.gz.manifest.json").write_text(
        json.dumps({"provenance": {"bridge": dict(STAMPED_PROVENANCE)}}),
        encoding="utf-8",
    )
    return d0_dir


# --------------------------------------------------------------------------- #
# The defect itself: two kinds of evidence, two statuses.                       #
# --------------------------------------------------------------------------- #
def test_alignment_separates_a_contradiction_from_an_absence() -> None:
    runner = _load_runner()

    against_m = runner._alignment(
        "D0", dict(STAMPED_PROVENANCE), runner.policy_target(SEALED_PRESET)
    )
    against_s = runner._alignment(
        "D0", dict(STAMPED_PROVENANCE), runner.policy_target(FOREIGN_PRESET)
    )

    # `m` is the policy the capture actually stamped: the stamped knobs agree.
    assert against_m["mismatched"] == []
    # `s` is a different policy: the same stamped knobs now contradict it.
    assert against_s["mismatched"] == ["relink_bridge_px", "relink_bridge_dir_bonus"]

    # The unstamped knobs are absent under *both* — they never contradict anything,
    # because there is nothing there to contradict with.
    assert against_m["unstamped"] == NEVER_STAMPED
    assert against_s["unstamped"] == NEVER_STAMPED


def test_derive_terminal_ranks_contradiction_above_absence() -> None:
    runner = _load_runner()
    nothing: dict[str, object] = {"knobs": [], "flag": False}

    contradicted, *_ = runner.derive_terminal({"knobs": ["relink_bridge_px"]}, nothing)
    unverifiable, *_ = runner.derive_terminal(
        nothing, {"unstamped": ["relink_bridge_h_lo"]}
    )
    clean, *_ = runner.derive_terminal(nothing, nothing)

    assert contradicted == "P0_CAPTURE_SEMANTICS_INVALID"
    assert unverifiable == "P0_CAPTURE_SEMANTICS_UNVERIFIABLE"
    assert clean == "P0_PAIR_CUTOFF_ONLY"

    # A contradiction is not softened by co-occurring absences: the ordering is
    # what carries the meaning.
    both, *_ = runner.derive_terminal(
        {"knobs": ["relink_bridge_px"]}, {"unstamped": ["relink_bridge_h_lo"]}
    )
    assert both == "P0_CAPTURE_SEMANTICS_INVALID"


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_the_two_presets_reach_different_terminals_on_the_real_artifacts() -> None:
    """End-to-end: `s` is contradicted, `m` is merely uncertifiable.

    This is the whole correction in one assertion. P0's original terminal was
    *correct for the preset it declared* — its error was the scope, not the
    inference.
    """
    runner = _load_runner()

    against_s = runner.audit(ROOT, policy_preset=FOREIGN_PRESET)
    against_m = runner.audit(ROOT, policy_preset=SEALED_PRESET)

    assert against_s["terminal"] == "P0_CAPTURE_SEMANTICS_INVALID"
    assert against_s["terminal_basis"]["contradictions"]["d0_mismatched_knobs"] == [
        "relink_bridge_px",
        "relink_bridge_dir_bonus",
    ]

    assert against_m["terminal"] == "P0_CAPTURE_SEMANTICS_UNVERIFIABLE"
    assert against_m["terminal_basis"]["contradicted"] is False
    assert (
        against_m["terminal_basis"]["absences"]["d0_unstamped_knobs"] == NEVER_STAMPED
    )


# --------------------------------------------------------------------------- #
# Outcome-blindness and replay level: unchanged by the retype.                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", [SEALED_PRESET, FOREIGN_PRESET])
def test_p0_never_reads_a_label_whatever_the_terminal(
    tmp_path: Path, preset: str
) -> None:
    runner = _load_runner()
    result = runner.audit(
        ROOT, policy_preset=preset, d0_capture_dir=_synthetic_d0_capture_dir(tmp_path)
    )

    assert result["terminal"] != "P0_PAIR_CUTOFF_ONLY"
    assert result["label_access"] == {
        "gt_or_fp_labels_accessed": False,
        "p5": "not_entered",
    }
    assert result["provenance"]["r1_frozen_preset"].endswith("mamba_whole_graph_m.yaml")


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
        result["replay"]["counterfactual_ceiling_if_provenance_were_complete"]
        == "L1_pair_cutoff_replay"
    )
