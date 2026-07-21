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

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import csv
import gzip
import hashlib
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
    key = runner.CAPTURE_KERNEL_SOURCE_KEY
    at_capture = "aa" * 32

    absent = runner.kernel_source_evidence({"git_commit": "abc"}, at_capture)
    assert absent["absent"] is True and absent["differs"] is None

    agrees = runner.kernel_source_evidence({key: at_capture}, at_capture)
    assert agrees["absent"] is False and agrees["differs"] is None

    # Stamped, and it names a kernel the capture's own commit does not contain:
    # a fact about the capture, not a gap in the audit. It must contradict.
    differs = runner.kernel_source_evidence({key: "bb" * 32}, at_capture)
    assert differs["absent"] is False and differs["differs"] == "bb" * 32

    # The capture's commit cannot be resolved, so there is nothing to compare it
    # against. That is an absence — never a contradiction.
    unresolvable = runner.kernel_source_evidence({key: at_capture}, None)
    assert unresolvable["absent"] is True and unresolvable["differs"] is None


def test_a_capture_is_not_falsified_by_later_edits_to_the_kernel() -> None:
    """The comparand is the capture-time source, never the working tree's.

    Comparing a stamp against today's `tracker_gpu.cu` would turn every later edit
    to that file into a false `..._SEMANTICS_INVALID` for an untouched historical
    capture. The 2026-07-12 capture already runs a kernel a thousand lines removed
    from HEAD, so this is live, not hypothetical.
    """
    runner = _load_runner()
    at_capture, today = "aa" * 32, "ee" * 32
    assert at_capture != today

    honest = runner.kernel_source_evidence(
        {runner.CAPTURE_KERNEL_SOURCE_KEY: at_capture}, at_capture
    )
    assert honest["differs"] is None, "an untouched capture must survive kernel drift"


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_source_proofs_are_read_from_the_kernel_the_capture_ran() -> None:
    """Static proofs grepped from HEAD certify a decision path that never ran."""
    runner = _load_runner()
    result = runner.audit(ROOT, policy_preset=SEALED_PRESET)
    provenance = result["provenance"]

    assert provenance["source_proofs_verified_against"] == "capture_time_kernel"
    # The kernel has moved since the capture; the audit must notice and still not
    # treat that drift as evidence against the capture.
    assert provenance["kernel_source_drifted_since_capture"] is True
    assert (
        provenance["capture_time_kernel_source_sha256"]
        != provenance["current_kernel_source_sha256"]
    )
    assert result["terminal_basis"]["contradictions"]["source_proofs_missing"] == []
    assert result["terminal_basis"]["contradicted"] is False


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
    # A complete capture stamps the kernel *it ran* — the source at its own commit,
    # not the one in the tree today (which has since moved on).
    at_capture = runner.kernel_source_at_capture(
        ROOT, manifest["provenance"]["git_commit"]
    )
    assert at_capture is not None, "capture-time kernel source must be resolvable"
    manifest["provenance"][runner.CAPTURE_KERNEL_SOURCE_KEY] = hashlib.sha256(
        at_capture
    ).hexdigest()
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


# --------------------------------------------------------------------------- #
# A terminal types the whole packet — and admission is not observability.       #
#                                                                               #
# Two defects, in sequence. The funnel CSV first said "headline provenance is    #
# invalid" unconditionally, so an `..._UNVERIFIABLE` packet still shipped the    #
# proposition it had withdrawn. Fixing that with one terminal-wide string then   #
# created the mirror bug: under a clean terminal *every* stage claimed to be     #
# awaiting computation — including `eligible_raw_pairs`, `claim_winners` and     #
# `final_commits`, which the field matrix in the same packet calls unobservable  #
# whatever the provenance says, because the capture never recorded the fields.   #
#                                                                               #
# The first version of these tests could not see either: it drove a synthetic    #
# capture whose bytes cannot match the sealed packet hash, so both presets fell  #
# to INVALID and the UNVERIFIABLE branch never ran. (It even indexed a key that  #
# does not exist, `row["consequence"]`, and CI stayed green.) So the terminals    #
# are now reached for real: UNVERIFIABLE from the sealed artifacts, and clean    #
# from a fixture that stamps everything they omit.                               #
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_evidence_dir(runner, tmp_path: Path, *, reseal: bool = True) -> Path:
    """The real evidence tree, with R1's export stamping what it omits.

    The clean terminal is unreachable against the sealed packets — R1's export is
    missing the same four knobs the D0 capture is. Without an injectable copy no
    test can drive it, and a terminal no test can produce is one no reader should
    trust.

    But an injected packet must be **internally sealed**, not merely edited. R1's
    export is pinned in two places, and the ledger that pins it is itself pinned by
    the manifest, so a fixture that rewrites the export and leaves the hashes alone
    is a *tampered* packet. Driving the clean terminal with one would prove only
    that tampering reaches clean — the opposite of the property under test, and
    exactly what the partition calls a contradiction.

    So the seal is recomputed outward: export → hash ledger → manifest.

    `reseal=False` deliberately skips that, to prove the audit rejects the packet
    the honest fixture would otherwise be mistaken for.
    """
    evidence = tmp_path / ("evidence" if reseal else "evidence_tampered")
    shutil.copytree(ROOT / "docs/modules/semantic/research/evidence", evidence)
    packet = evidence / "r1_temporal_reduction_capture_20260712"

    export_path = packet / "export_manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    resolved = runner.resolve_policy(SEALED_PRESET)
    export["provenance"]["bridge"].update(
        {
            "h_lo": resolved["relink_bridge_h_lo"],
            "h_hi": resolved["relink_bridge_h_hi"],
            "spatial_gate": resolved["relink_bridge_spatial_gate"],
            "max_speed": resolved["relink_bridge_max_speed"],
        }
    )
    export_path.write_text(json.dumps(export), encoding="utf-8")
    if not reseal:
        return evidence

    ledger_path = packet / "frozen_packet_hashes.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["local_artifacts"]["export_manifest"] = _sha256(export_path)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["export_manifest.json"] = _sha256(export_path)
    manifest["files"]["frozen_packet_hashes.json"] = _sha256(ledger_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return evidence


def _funnel(runner, result, out: Path) -> list[dict[str, str]]:
    runner.write_packet(result, out)
    with (out / "decision_funnel.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_an_unverifiable_packet_never_also_calls_the_capture_invalid(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    result = runner.audit(ROOT, policy_preset=SEALED_PRESET)
    assert result["terminal"] == "P0_CAPTURE_SEMANTICS_UNVERIFIABLE"  # the real one

    funnel = _funnel(runner, result, tmp_path / "packet")
    assert len(funnel) == 7

    stopped = [
        result["replay"]["observed_level"],
        result["decision_funnel_status"],
        *(row["reason"] for row in funnel),
        *(str(row["missing_consequence"]) for row in result["field_sufficiency"]),
    ]
    assert not [text for text in stopped if "invalid" in text.lower()], (
        "a packet that cannot verify its capture must not also call it invalid"
    )
    # Admission failed, so the whole funnel is stopped — uniformly, and by the
    # terminal.
    assert {row["observability"] for row in funnel} == {"UNOBSERVABLE"}


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_a_clean_terminal_releases_only_the_pair_cutoff_stage(tmp_path: Path) -> None:
    """`P0_PAIR_CUTOFF_ONLY` means the pair cutoff — not the whole funnel."""
    runner = _load_runner()
    result = runner.audit(
        ROOT,
        policy_preset=SEALED_PRESET,
        d0_capture_dir=_fully_stamped_capture_dir(runner, tmp_path),
        evidence_dir=_clean_evidence_dir(runner, tmp_path),
    )
    assert result["terminal"] == "P0_PAIR_CUTOFF_ONLY"  # reached, not simulated
    assert result["terminal_basis"]["unverifiable"] is False
    assert result["replay"]["observed_level"] == "L1_pair_cutoff_replay"

    funnel = {row["stage"]: row for row in _funnel(runner, result, tmp_path / "packet")}

    # Exactly the stage the replay level names — and no other. Stamping provenance
    # cannot conjure a frame column, a candidate slot or an atomicMax key, so the
    # stages after D stay shut whatever the terminal says.
    assert funnel["pass_bdist_cutoff"]["observability"] == "PENDING_P4"
    assert [s for s, r in funnel.items() if r["observability"] == "PENDING_P4"] == [
        "pass_bdist_cutoff"
    ]
    for stage in ("eligible_raw_pairs", "claim_winners", "final_commits"):
        assert funnel[stage]["observability"] == "UNOBSERVABLE", stage

    # And each stopped row cites *its own* blocker, not one terminal-wide string.
    assert "atomicMax" in funnel["claim_winners"]["reason"]
    assert "shadow" in funnel["final_commits"]["reason"]


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_the_funnel_never_outruns_the_field_matrix_beside_it(tmp_path: Path) -> None:
    """No funnel row may claim computability its own matrix row denies.

    This is the invariant both bugs broke, stated once. It must hold under every
    terminal, so it is asserted under both the real one and the clean one.
    """
    runner = _load_runner()
    cases = {
        "real": runner.audit(ROOT, policy_preset=SEALED_PRESET),
        "clean": runner.audit(
            ROOT,
            policy_preset=SEALED_PRESET,
            d0_capture_dir=_fully_stamped_capture_dir(runner, tmp_path),
            evidence_dir=_clean_evidence_dir(runner, tmp_path),
        ),
    }
    for name, result in cases.items():
        matrix = {row["stage"]: row for row in result["field_sufficiency"]}
        for row in _funnel(runner, result, tmp_path / f"packet_{name}"):
            governing = matrix[runner.FUNNEL_STAGE_FIELD[row["stage"]]]
            if row["observability"] == "PENDING_P4":
                assert governing["complete"] is True, (
                    f"{name}/{row['stage']}: funnel says computable, field matrix "
                    f"says {governing['stage']} is incomplete"
                )


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_a_tampered_r1_export_cannot_reach_any_clean_terminal(tmp_path: Path) -> None:
    """Editing the sealed export and leaving its hashes alone must contradict.

    The audit verified the D0 capture's bytes but not R1's, and read R1's export for
    its bridge provenance on trust. So four fields written into the sealed
    `export_manifest.json`, with no hash updated, carried the audit all the way to
    `P0_PAIR_CUTOFF_ONLY` — the clean fixture was passing *because* it tampered.
    The partition is explicit that a broken packet hash is a contradiction.
    """
    runner = _load_runner()
    result = runner.audit(
        ROOT,
        policy_preset=SEALED_PRESET,
        d0_capture_dir=_fully_stamped_capture_dir(runner, tmp_path),
        evidence_dir=_clean_evidence_dir(runner, tmp_path, reseal=False),
    )

    assert result["terminal"] == "P0_CAPTURE_SEMANTICS_INVALID"
    assert result["terminal_basis"]["contradicted"] is True
    # Both pins fail together: the export no longer matches the manifest that seals
    # it, nor the ledger that inventories it.
    assert set(result["terminal_basis"]["contradictions"]["r1_packet_seal_broken"]) == {
        "export_vs_manifest",
        "export_vs_hash_ledger",
    }


@pytest.mark.packet_bound
@pytest.mark.skipif(
    not (REAL_CAPTURE / "capture.csv.gz").exists(),
    reason="sealed D0 capture not present in this checkout",
)
def test_the_real_r1_packet_is_sealed() -> None:
    """The guard must pass on the artifacts as committed, or it is unusable."""
    runner = _load_runner()
    result = runner.audit(ROOT, policy_preset=SEALED_PRESET)

    assert result["provenance"]["r1_packet_seal"] == {
        "export_vs_manifest": True,
        "export_vs_hash_ledger": True,
        "hash_ledger_vs_manifest": True,
    }
    assert result["terminal_basis"]["contradictions"]["r1_packet_seal_broken"] == []
