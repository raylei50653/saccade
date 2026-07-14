#!/usr/bin/env python3
"""P0: outcome-blind runtime bridge decision-path identifiability audit.

This source/artifact verifier intentionally reads no CSV row values: the D0
capture header is sufficient to decide whether the frozen packet could replay
the production decision graph.  In particular it never accesses pairs.csv or
any GT/FP field.  It writes a field matrix, an explicitly unobserved funnel,
and exactly one ordered terminal.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Final, Iterable


REPO = Path(__file__).resolve().parents[2]
STAMP = "20260713"

# Ordered terminal partition — first match wins, mirroring § 5's "takes precedence".
#
# The original partition had one label for every way alignment could fail, and it
# was named after the strongest of them. So "the capture records a *different*
# policy" (ontic: its semantics really are not the target's) and "the capture does
# not record the policy *at all*" (epistemic: nothing can be concluded) collapsed
# into one verdict, and the weaker evidence inherited the stronger claim. That is
# the defect this partition removes: the terminal is now *derived* from which kind
# of evidence was found, never named in advance (declaration Correction 1 § C1.7).
TERMINAL_CONTRADICTED = "P0_CAPTURE_SEMANTICS_INVALID"
TERMINAL_UNVERIFIABLE = "P0_CAPTURE_SEMANTICS_UNVERIFIABLE"
TERMINAL_CLEAN = "P0_PAIR_CUTOFF_ONLY"

# How a capture certifies which kernel source produced it. The name is not coined
# here: it is the key D0's own fidelity packet already stamps
# (evidence/d0_bridge_estimator_fidelity_20260711/manifest.json).
#
# The D0 *runtime* capture does not stamp it — its provenance carries a
# `git_commit`, which names a tree, not the file that ran. That is a fact about
# today's artifacts, so it is *read* from them. Writing it into the audit as a
# constant (`..._absent: True`, as this once did) re-commits the very error the
# study corrects: it fixes a verdict in code instead of deriving it from evidence,
# and it floors the terminal at UNVERIFIABLE under every possible input.
CAPTURE_KERNEL_SOURCE_KEY = "kernel_source_sha256"

KERNEL_SOURCE_REL = "src/tracking/tracker_gpu.cu"

# Every field that states *why* the audit stopped is derived from the terminal.
# They used to be fixed strings, and when the terminal was retyped they were left
# behind: a packet could carry `P0_CAPTURE_SEMANTICS_UNVERIFIABLE` while its
# decision funnel still read "headline provenance is invalid" — the withdrawn
# proposition, restated in a file nobody re-read. A terminal is not a label on one
# field; it types the whole packet.
TERMINAL_NARRATIVE: Final[dict[str, dict[str, str]]] = {
    TERMINAL_CONTRADICTED: {
        "observed_level": "not_assignable_due_to_capture_semantics_invalid",
        "funnel_status": "not_entered_due_to_capture_semantics_invalid",
        "observability": "UNOBSERVABLE",
        "funnel_reason": "capture provenance contradicts the audited policy; P4 not entered",
        "cutoff_consequence": "scalar cutoff is mechanically evaluable, but the capture records a policy other than the audited one",
        "scalar_consequence": "scalar formula terms are observable only for a population the capture attributes to another policy",
    },
    TERMINAL_UNVERIFIABLE: {
        "observed_level": "not_assignable_while_capture_provenance_is_incomplete",
        "funnel_status": "not_entered_while_capture_provenance_is_incomplete",
        "observability": "UNOBSERVABLE",
        "funnel_reason": "capture provenance is incomplete, so no policy can be certified; P4 not entered",
        "cutoff_consequence": "scalar cutoff is mechanically evaluable, but not attributable to a certified policy while provenance is incomplete",
        "scalar_consequence": "scalar formula terms are observable only for the emitted survivor population, under a configuration the capture cannot certify",
    },
    TERMINAL_CLEAN: {
        "observed_level": "L1_pair_cutoff_replay",
        "funnel_status": "admissible_p4_not_executed_by_this_runner",
        # Not `OBSERVABLE`: admission passing does not mean this runner counted
        # anything. It emits no funnel figures, so it must not claim to have any.
        "observability": "PENDING_P4",
        "funnel_reason": "provenance is complete and the funnel is computable; P4 must compute it — this admission runner does not",
        "cutoff_consequence": "scalar cutoff is mechanically evaluable and attributable to the audited policy",
        "scalar_consequence": "scalar formula terms are observable for the emitted survivor population under the audited policy",
    },
}


def kernel_source_at_capture(root: Path, git_commit: Any) -> bytes | None:
    """The audited kernel source as of the capture's *own* commit.

    Both the capture's stamped hash and this audit's static source proofs describe
    the kernel that **ran** — not the one in today's working tree. Reading either
    from the current checkout is a category error with two teeth:

      * every later edit to `tracker_gpu.cu` would make an untouched historical
        capture look like it stamped the wrong kernel (a false contradiction), and
      * the source proofs would be grepped from code the capture never executed,
        silently certifying a decision path that was never the one under audit.

    Not hypothetical: the 2026-07-12 capture ran a kernel a thousand lines removed
    from HEAD, and its proofs were being read from HEAD.

    Returns None when the capture's commit or that path within it cannot be
    resolved — which is an *absence* (nothing can be compared), never a
    contradiction.
    """
    if not git_commit:
        return None
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "cat-file",
            "blob",
            f"{git_commit}:{KERNEL_SOURCE_REL}",
        ],
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


if str(REPO / "scripts" / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "tools"))

from resolved_bridge_policy_config import (  # noqa: E402
    fingerprint as policy_fingerprint,
    resolve as resolve_policy,
)

# The policy target is a *parameter*, never an assumption. The original P0 run
# hard-coded the `s` preset while calling it "the headline", then judged
# `m`-sealed evidence against it; see the declaration's Correction 1. Callers
# must now name the preset, and its knob values are resolved from that preset
# rather than transcribed.
POLICY_KNOBS = (
    "relink_bridge_enabled",
    "relink_bridge_px",
    "relink_bridge_margin",
    "relink_bridge_h_lo",
    "relink_bridge_h_hi",
    "relink_bridge_spatial_gate",
    "relink_bridge_max_speed",
    "relink_bridge_dir_bonus",
    "reid_mode",
)


def policy_target(preset: str) -> dict[str, Any]:
    """Resolve the audited policy from a named preset (no hidden default)."""
    resolved = resolve_policy(preset)
    policy: dict[str, Any] = {
        "preset": f"configs/presets/{preset}.yaml",
        "resolved_bridge_policy_config_v1": policy_fingerprint(preset),
    }
    policy.update({knob: resolved[knob] for knob in POLICY_KNOBS})
    return policy


SOURCE_PROOFS = {
    "stage_a": "if (!active[cand] || trk_to_det[cand] < 0) return;",
    "stage_b_height": "if (bridge_h_hi > 0.0f)",
    "stage_c_bdist": "float bdist = w * 0.5f * (fwd_r + bwd_r) + (1.0f - w) * dist_h;",
    "stage_d_cutoff": "bool ok = bdist <= bridge_px;",
    "stage_e_margin": "(second_dist - best_dist) < bridge_margin",
    "stage_f_claim": "atomicMax(&bridge_claim[best_lost], key);",
    "stage_g_commit": "track_ids[cand] = track_ids[lost];",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def csv_header_gzip(path: Path) -> list[str]:
    # Do not iterate rows: P0 must remain outcome-blind until its stratum freezes.
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"empty capture: {path}")
    return header


def input_key(path: Path, root: Path) -> str:
    """Use a stable explicit key for a test-injected input outside the repo."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return f"external_injected/{path.name}"


def _eq(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-8
        except (TypeError, ValueError):
            return False
    return actual == expected


def _alignment(
    name: str, bridge: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    """Compare stamped provenance against the audited policy, three-valued.

    `mismatch` and `absent` are different *kinds* of evidence and must never share
    a status. A stamped field that differs proves the capture ran another policy.
    An unstamped field proves nothing at all — it only removes the ability to
    check. Fusing them (the original `mismatch_or_absent`) is what let an absence
    of evidence be reported as evidence of invalidity.
    """
    mapping = {
        "relink_bridge_px": "px",
        "relink_bridge_dir_bonus": "dir_bonus",
        "relink_bridge_h_lo": "h_lo",
        "relink_bridge_h_hi": "h_hi",
        "relink_bridge_spatial_gate": "spatial_gate",
        "relink_bridge_max_speed": "max_speed",
    }
    comparisons = []
    for policy_key, artifact_key in mapping.items():
        if artifact_key not in bridge:
            status = "absent"
        elif _eq(bridge[artifact_key], policy[policy_key]):
            status = "match"
        else:
            status = "mismatch"
        comparisons.append(
            {
                "policy_knob": policy_key,
                "artifact_field": artifact_key,
                "expected": policy[policy_key],
                "actual": bridge.get(artifact_key),
                "status": status,
            }
        )

    def _of(status: str) -> list[str]:
        return [c["policy_knob"] for c in comparisons if c["status"] == status]

    return {
        "artifact": name,
        "mismatched": _of("mismatch"),  # ontic: contradicts the audited policy
        "unstamped": _of("absent"),  # epistemic: cannot be checked at all
        "all_fields_match": not _of("mismatch") and not _of("absent"),
        "comparisons": comparisons,
    }


def kernel_source_evidence(
    provenance: dict[str, Any], capture_source_sha256: str | None
) -> dict[str, Any]:
    """Which kind of evidence the capture offers about the kernel that produced it.

    The same three-way split as `_alignment`, for the same reason. An unstamped
    kernel hash is an *absence*: the capture cannot say what ran, so nothing about
    the kernel can be concluded. A stamped hash that disagrees with the source at
    the capture's **own commit** is a *contradiction*: the capture records a kernel
    its own commit does not contain.

    The comparand is the capture-time source, never the working tree's. A capture
    is not falsified by edits made to the file after it ran.
    """
    stamped = provenance.get(CAPTURE_KERNEL_SOURCE_KEY)
    if capture_source_sha256 is None:
        return {
            "stamped": stamped,
            "absent": True,
            "differs": None,
            "reason": "capture-time kernel source could not be resolved from its commit",
        }
    if stamped is None:
        return {
            "stamped": None,
            "absent": True,
            "differs": None,
            "reason": f"capture stamps no {CAPTURE_KERNEL_SOURCE_KEY}",
        }
    stamped = str(stamped)
    if stamped != capture_source_sha256:
        return {
            "stamped": stamped,
            "absent": False,
            "differs": stamped,
            "reason": "capture stamps a kernel its own commit does not contain",
        }
    return {"stamped": stamped, "absent": False, "differs": None, "reason": None}


def derive_terminal(
    contradictions: dict[str, Any], absences: dict[str, Any]
) -> tuple[str, bool, bool]:
    """Ordered partition over the *kind* of evidence found, not over a stop condition.

    Contradiction outranks absence: if the artifacts positively record something
    incompatible with the audited policy, that is a fact about the capture. If they
    merely fail to record it, the only fact is about the audit's reach. Naming the
    verdict in advance — as the original partition did — is what let the second be
    reported as the first.
    """
    contradicted = any(bool(value) for value in contradictions.values())
    unverifiable = any(bool(value) for value in absences.values())

    if contradicted:
        return TERMINAL_CONTRADICTED, contradicted, unverifiable
    if unverifiable:
        return TERMINAL_UNVERIFIABLE, contradicted, unverifiable
    return TERMINAL_CLEAN, contradicted, unverifiable


def _field_matrix(header: set[str], narrative: dict[str, str]) -> list[dict[str, Any]]:
    def row(
        stage: str, required: list[str], artifact: str, complete: bool, consequence: str
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "required_fields": required,
            "existing_artifact": artifact,
            "complete": complete,
            "missing_consequence": consequence,
        }

    return [
        row(
            "source_snapshot_alignment",
            ["capture kernel source SHA-256", "headline kernel source SHA-256"],
            "D0/R1 provenance records only a git commit",
            False,
            "a commit label is not an exact source-file identity; capture/source code equivalence is unproven",
        ),
        row(
            "A_pair_eligibility",
            [
                "all candidate/lost state before height gate",
                "frame",
                "candidate slot",
                "lost slot",
            ],
            "D0 v2 capture header",
            False,
            "capture is emitted after eligibility and pre-score continues; raw exclusions are unobservable",
        ),
        row(
            "B_pre_score_gates",
            [
                "all raw pairs",
                "EMA heights",
                "height-gate result",
                "speed/centre state",
            ],
            "D0 has ema_lost/ema_cand only on emitted survivors",
            False,
            "height-gate attrition and disabled-path counterfactuals cannot be observed",
        ),
        row(
            "C_score_construction",
            [
                "fwd_r",
                "bwd_r",
                "dist_h",
                "s_lost",
                "w",
                "bdist",
                "directional provenance",
            ],
            "D0 v2 capture header",
            {"fwd_r", "bwd_r", "dist_h", "s_lost", "w", "bdist"}.issubset(header),
            narrative["scalar_consequence"],
        ),
        row(
            "D_pair_cutoff",
            ["bdist", "production threshold", "headline preset stamp"],
            "D0 v2 capture header + manifest",
            False,
            narrative["cutoff_consequence"],
        ),
        row(
            "E_candidate_local_ranking",
            ["frame", "candidate slot/index", "complete lost competitor set", "bdist"],
            "D0 v2 export",
            False,
            "exported event key omits frame and slots; best/second-best and margin are unreconstructable",
        ),
        row(
            "F_claim_competition",
            [
                "lost slot",
                "all proposing candidates",
                "detection score",
                "quantized key",
                "candidate index",
            ],
            "D0/R1 canonical packets",
            False,
            "atomicMax winner cannot be replayed",
        ),
        row(
            "G_commit",
            ["winning claim", "post-commit candidate ID", "lost-slot deactivation"],
            "shadow capture",
            False,
            "shadow deliberately suppresses the only bridge writes; final commit is not observed",
        ),
    ]


def audit(
    root: Path = REPO,
    *,
    policy_preset: str,
    d0_capture_dir: Path | None = None,
) -> dict[str, Any]:
    evidence = root / "docs/modules/semantic/research/evidence"
    d0_packet_path = evidence / "d0_runtime_shadow_fidelity_20260712/manifest.json"
    r1_packet_path = evidence / "r1_temporal_reduction_capture_20260712/manifest.json"
    r1_hashes_path = (
        evidence / "r1_temporal_reduction_capture_20260712/frozen_packet_hashes.json"
    )
    r1_export_path = (
        evidence / "r1_temporal_reduction_capture_20260712/export_manifest.json"
    )
    s0_packet_path = evidence / "s0_safe_domain_runtime_transfer_20260713/manifest.json"
    d0_dir = d0_capture_dir or (
        root / "out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z"
    )
    d0_capture = d0_dir / "capture.csv.gz"
    d0_capture_manifest_path = d0_dir / "capture.csv.gz.manifest.json"
    declaration = (
        root
        / "docs/modules/semantic/research/runtime_bridge_decision_path_identifiability_declaration_20260713.md"
    )
    source = root / KERNEL_SOURCE_REL

    for path in (
        d0_packet_path,
        r1_packet_path,
        r1_hashes_path,
        r1_export_path,
        s0_packet_path,
        d0_capture,
        d0_capture_manifest_path,
        declaration,
        source,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    d0_packet = load_json(d0_packet_path)
    d0_capture_manifest = load_json(d0_capture_manifest_path)
    r1_packet = load_json(r1_packet_path)
    r1_hashes = load_json(r1_hashes_path)
    r1_export = load_json(r1_export_path)
    s0_packet = load_json(s0_packet_path)
    header = csv_header_gzip(d0_capture)
    header_set = set(header)

    # The static proofs must be read from the kernel the capture *ran*, not from
    # today's working tree — otherwise the audit certifies a decision path the
    # capture never executed. When that source cannot be recovered, the proofs are
    # unverifiable (an absence), not missing (a contradiction).
    capture_provenance = dict(d0_capture_manifest["provenance"])
    capture_source = kernel_source_at_capture(
        root, capture_provenance.get("git_commit")
    )
    capture_source_sha = (
        hashlib.sha256(capture_source).hexdigest()
        if capture_source is not None
        else None
    )
    source_proofs: dict[str, bool | None]
    if capture_source is None:
        source_proofs = {name: None for name in SOURCE_PROOFS}
    else:
        capture_source_text = capture_source.decode("utf-8", errors="replace")
        source_proofs = {
            name: text in capture_source_text for name, text in SOURCE_PROOFS.items()
        }

    expected_d0_hash = str(d0_packet["frozen_inputs"]["capture.csv.gz"])
    actual_d0_hash = sha256(d0_capture)
    d0_hash_ok = expected_d0_hash == actual_d0_hash
    s0_hash_ok = str(s0_packet["input_hashes"]["capture.csv.gz"]) == actual_d0_hash
    policy = policy_target(policy_preset)
    d0_alignment = _alignment(
        "D0", dict(d0_capture_manifest["provenance"]["bridge"]), policy
    )
    r1_alignment = _alignment("R1", dict(r1_export["provenance"]["bridge"]), policy)
    r1_preset = str(r1_hashes["scope"]["preset"])
    current_source_sha = sha256(source)
    kernel = kernel_source_evidence(capture_provenance, capture_source_sha)

    # Positive contradictions: the artifacts record something incompatible with the
    # audited policy or with themselves. These license the ontic verdict.
    contradictions = {
        "d0_mismatched_knobs": d0_alignment["mismatched"],
        "r1_mismatched_knobs": r1_alignment["mismatched"],
        "r1_frozen_preset_differs": (
            r1_preset if r1_preset != policy["preset"] else None
        ),
        "d0_packet_hash_broken": not d0_hash_ok,
        "s0_capture_hash_broken": not s0_hash_ok,
        # `is False` — a proof that could not be *checked* is None, and an unchecked
        # proof is an absence. `not ok` would have promoted it to a contradiction.
        "source_proofs_missing": [n for n, ok in source_proofs.items() if ok is False],
        "capture_kernel_source_differs": kernel["differs"],
    }
    # Absences: nothing is contradicted, but nothing can be checked either. These
    # license only the epistemic verdict — never the ontic one.
    absences = {
        "d0_unstamped_knobs": d0_alignment["unstamped"],
        "r1_unstamped_knobs": r1_alignment["unstamped"],
        "capture_kernel_source_hash_absent": kernel["absent"],
        "source_proofs_unverifiable": [
            n for n, ok in source_proofs.items() if ok is None
        ],
    }
    terminal, contradicted, unverifiable = derive_terminal(contradictions, absences)
    narrative = TERMINAL_NARRATIVE[terminal]

    matrix = _field_matrix(header_set, narrative)
    return {
        "study": "p0_runtime_bridge_decision_path_20260713",
        "terminal": terminal,
        # The terminal is a derived value: this is what derived it. A reader can
        # recompute the verdict from `terminal_basis` alone, and can see which kind
        # of evidence carried it — which the fused `mismatch_or_absent` could not.
        "terminal_basis": {
            "ordered_partition": [
                TERMINAL_CONTRADICTED,
                TERMINAL_UNVERIFIABLE,
                TERMINAL_CLEAN,
            ],
            "contradictions": contradictions,
            "absences": absences,
            "contradicted": contradicted,
            "unverifiable": unverifiable,
        },
        "policy_target": policy,
        "label_access": {"gt_or_fp_labels_accessed": False, "p5": "not_entered"},
        "source_proofs": source_proofs,
        "provenance": {
            "d0_capture_sha256": actual_d0_hash,
            # The kernel the capture ran, the kernel it says it ran, and the kernel
            # in the tree today are three different questions. The audit compares
            # the first two; the third is drift, reported and never a verdict.
            "capture_time_kernel_source_sha256": capture_source_sha,
            "capture_stamped_kernel_source_sha256": kernel["stamped"],
            "capture_kernel_source_sha256_present": not kernel["absent"],
            "capture_kernel_source_note": kernel["reason"],
            "current_kernel_source_sha256": current_source_sha,
            "kernel_source_drifted_since_capture": (
                capture_source_sha is not None
                and capture_source_sha != current_source_sha
            ),
            "source_proofs_verified_against": (
                "capture_time_kernel" if capture_source is not None else None
            ),
            "d0_capture_git_commit": capture_provenance.get("git_commit"),
            "r1_capture_git_commit": r1_export["provenance"].get("git_commit"),
            "d0_packet_hash_match": d0_hash_ok,
            "s0_inherits_same_capture_hash": s0_hash_ok,
            "d0_alignment": d0_alignment,
            "r1_alignment": r1_alignment,
            "r1_frozen_preset": r1_preset,
            "r1_packet_terminal": r1_packet.get("terminal"),
        },
        "d0_header": header,
        "field_sufficiency": matrix,
        "replay": {
            "observed_level": narrative["observed_level"],
            "counterfactual_ceiling_if_provenance_were_complete": "L1_pair_cutoff_replay",
            "l2_blockers": [
                "no frame",
                "no candidate slot/index",
                "no complete candidate competitor universe",
            ],
            "l3_blockers": [
                "no raw pre-height universe",
                "no detection score",
                "no quantized atomicMax key",
                "shadow capture has no commit",
            ],
        },
        "decision_funnel_status": narrative["funnel_status"],
        "inputs": {
            input_key(path, root): sha256(path)
            for path in (
                d0_packet_path,
                d0_capture_manifest_path,
                r1_packet_path,
                r1_hashes_path,
                r1_export_path,
                s0_packet_path,
                declaration,
                source,
            )
        },
    }


def write_packet(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "field_sufficiency.json").write_text(
        json.dumps(result["field_sufficiency"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # The funnel's own account of why it is empty is derived from the terminal, like
    # every other such field. It used to read "headline provenance is invalid" no
    # matter what — so a packet terminating in `..._UNVERIFIABLE` shipped a CSV that
    # still asserted the invalidity the packet had just withdrawn. One artifact,
    # two incompatible propositions.
    narrative = TERMINAL_NARRATIVE[result["terminal"]]
    funnel_rows: Iterable[dict[str, str]] = (
        {
            "stage": stage,
            "event_count": "",
            "unique_candidate_count": "",
            "unique_lost_track_count": "",
            "sequence_distribution": "",
            "observability": narrative["observability"],
            "reason": narrative["funnel_reason"],
        }
        for stage in (
            "eligible_raw_pairs",
            "pass_height_gate",
            "pass_bdist_cutoff",
            "candidate_local_winners",
            "pass_margin",
            "claim_winners",
            "final_commits",
        )
    )
    with (output_dir / "decision_funnel.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "event_count",
                "unique_candidate_count",
                "unique_lost_track_count",
                "sequence_distribution",
                "observability",
                "reason",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(funnel_rows)
    metrics = {
        key: value
        for key, value in result.items()
        if key not in {"field_sufficiency", "inputs"}
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_hashes = {
        name: sha256(output_dir / name)
        for name in (
            "field_sufficiency.json",
            "decision_funnel.csv",
            "metrics.json",
        )
    }
    manifest = {
        "study": result["study"],
        "terminal": result["terminal"],
        "runner": "scripts/tools/audit_runtime_bridge_decision_path.py",
        "runner_sha256": sha256(Path(__file__)),
        "inputs": result["inputs"],
        "outputs": output_hashes,
        "files": output_hashes,
        "label_access": result["label_access"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--policy-preset",
        required=True,
        help=(
            "preset stem under configs/presets/ whose resolved policy the frozen "
            "evidence is audited against; there is no default, because assuming "
            "one is the error Correction 1 records"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO
        / "docs/modules/semantic/research/evidence/p0_runtime_bridge_decision_path_20260713",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    # The 2026-07-13 packet is sealed evidence: a re-run emits a new packet, it
    # never edits the old one.
    if (output_dir / "manifest.json").exists():
        raise SystemExit(
            f"refusing to overwrite a sealed packet at {output_dir}; "
            "pass --output-dir <new path>"
        )
    result = audit(root, policy_preset=args.policy_preset)
    write_packet(result, output_dir)
    print(
        json.dumps(
            {"terminal": result["terminal"], "gt_or_fp_labels_accessed": False},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
