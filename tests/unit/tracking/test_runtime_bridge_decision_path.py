"""P0's terminal must be derived from the *kind* of evidence found, not named ahead.

The original audit had one status for every way alignment could fail —
`mismatch_or_absent` — and one terminal named after the strongest of them. So a
stamped field that *contradicts* the audited policy and a field the capture never
stamps at all produced the same verdict, and the weaker evidence inherited the
stronger claim: an absence of evidence was published as evidence of invalidity.

These tests hold the two apart:

  * `_alignment` must report `mismatch` and `absent` as different statuses;
  * `derive_terminal` must rank contradiction above absence;
  * the audit must therefore reach *different* terminals for the two presets —
    `s` (which the capture genuinely contradicts) and `m` (which it merely cannot
    certify);
  * and every terminal in the partition must be reachable *from evidence*. A
    verdict no input can produce is a verdict named in advance, which is the
    defect this study exists to correct.

The synthetic fixture below cannot exercise the last point end-to-end: a made-up
capture file has a made-up hash, which is itself a contradiction and masks the
policy comparison. So the mechanism is pinned hermetically here, and the
end-to-end split is asserted against the real sealed artifacts where they exist.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/tools/audit_runtime_bridge_decision_path.py"
REAL_CAPTURE = ROOT / "out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z"
KERNEL_SRC = ROOT / "src/tracking/tracker_gpu.cu"  # the source the audit hashes

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
# The clean terminal must be reachable *from evidence*.                         #
#                                                                               #
# `absences` once carried `capture_kernel_source_hash_absent: True` as a literal —
# a fact a human read off today's manifest and transcribed into the audit. It made
# `unverifiable` true under every possible input, so `P0_PAIR_CUTOFF_ONLY` could   #
# never be reached and a future capture that *did* stamp its kernel would go on   #
# being reported as uncertifiable. That is the same move the study corrects (a    #
# verdict fixed in code, not derived), and the unit test above could not see it:  #
# it calls `derive_terminal` with a hand-built dict, never the one `audit` sends. #
# So the partition is exercised end-to-end, through `audit`, instead.             #
# --------------------------------------------------------------------------- #
def test_kernel_source_evidence_splits_absence_from_contradiction() -> None:
    runner = _load_runner()

    absent = runner.kernel_source_evidence({"git_commit": "abc"}, "aa" * 32)
    assert absent == {"stamped": None, "absent": True, "differs": None}

    agrees = runner.kernel_source_evidence(
        {runner.CAPTURE_KERNEL_SOURCE_KEY: "aa" * 32}, "aa" * 32
    )
    assert agrees["absent"] is False and agrees["differs"] is None

    # Stamped, and it names a *different* kernel: that is a fact about the capture,
    # not a gap in the audit. It must contradict, never merely fail to verify.
    differs = runner.kernel_source_evidence(
        {runner.CAPTURE_KERNEL_SOURCE_KEY: "bb" * 32}, "aa" * 32
    )
    assert differs["absent"] is False and differs["differs"] == "bb" * 32


def _fully_stamped_capture_dir(runner, tmp_path: Path) -> Path:
    """The real capture, byte-for-byte, with the provenance a complete one would carry.

    The bytes must be the real ones or the packet hashes break — and a broken hash
    is itself a contradiction, which would mask the very thing under test.
    """
    d0_dir = tmp_path / "d0_complete"
    d0_dir.mkdir()
    shutil.copy(REAL_CAPTURE / "capture.csv.gz", d0_dir / "capture.csv.gz")

    manifest = json.loads(
        (REAL_CAPTURE / "capture.csv.gz.manifest.json").read_text(encoding="utf-8")
    )
    resolved = runner.resolve_policy(SEALED_PRESET)
    manifest["provenance"]["bridge"].update(
        {
            "h_lo": resolved["relink_bridge_h_lo"],
            "h_hi": resolved["relink_bridge_h_hi"],
            "spatial_gate": resolved["relink_bridge_spatial_gate"],
            "max_speed": resolved["relink_bridge_max_speed"],
        }
    )
    # A complete capture stamps the kernel that produced it — here, the source the
    # audit itself hashes, so a complete stamp is what the audit sees.
    manifest["provenance"][runner.CAPTURE_KERNEL_SOURCE_KEY] = runner.sha256(KERNEL_SRC)
    (d0_dir / "capture.csv.gz.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return d0_dir


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_every_absence_answers_to_evidence_and_the_clean_terminal_is_reachable(
    tmp_path: Path,
) -> None:
    """Stamp everything the D0 capture omits, and its absences must clear."""
    runner = _load_runner()
    capture_dir = _fully_stamped_capture_dir(runner, tmp_path)

    result = runner.audit(ROOT, policy_preset=SEALED_PRESET, d0_capture_dir=capture_dir)
    basis = result["terminal_basis"]
    absences = basis["absences"]

    # D0's side is now fully stamped, kernel included, and the audit sees it. Under
    # the literal, `capture_kernel_source_hash_absent` stayed `True` here.
    assert absences["d0_unstamped_knobs"] == []
    assert absences["capture_kernel_source_hash_absent"] is False
    assert basis["contradictions"]["capture_kernel_source_differs"] is None
    assert basis["contradicted"] is False

    # What still blocks the clean terminal is a gap in a *real artifact* — R1's
    # export stamps no height gate either — not a constant in the audit. Every
    # absence is now traceable to evidence, which is what the literal destroyed.
    assert absences["r1_unstamped_knobs"] == NEVER_STAMPED
    assert result["terminal"] == "P0_CAPTURE_SEMANTICS_UNVERIFIABLE"

    # Clear that last one and the partition does reach its clean terminal. Under the
    # literal this was unreachable under *every* input — a verdict named in advance
    # by being permanently withheld.
    terminal, _, unverifiable = runner.derive_terminal(
        basis["contradictions"], dict(absences, r1_unstamped_knobs=[])
    )
    assert unverifiable is False
    assert terminal == "P0_PAIR_CUTOFF_ONLY"


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
