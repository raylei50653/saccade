#!/usr/bin/env python3
"""Replay and validate the sealed H0 bridge-decision trace from capture alone."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from export_headline_bridge_decision_trace import (
    canonical_semantic_packet,
    semantic_digest,
)


NOT_EVALUATED = 0
PASS = 1
REJECT = 2
DISABLED = 3
NOT_COMPUTED = 0
COMPUTED_FINITE = 1
COMPUTED_NAN = 4
NATIVE_SENTINEL = struct.unpack("!f", struct.pack("!f", 1e30))[0]


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def scalar_value(value: Mapping[str, Any], *, field: str) -> float:
    if int(value.get("status", NOT_COMPUTED)) != COMPUTED_FINITE:
        raise ValueError(f"{field} is not a computed finite float32")
    return struct.unpack("!f", int(value["bits"]).to_bytes(4, "big"))[0]


def optional_scalar(value: Mapping[str, Any]) -> float | None:
    if int(value.get("status", NOT_COMPUTED)) != COMPUTED_FINITE:
        return None
    return struct.unpack("!f", int(value["bits"]).to_bytes(4, "big"))[0]


def close_f32(actual: float, expected: float, *, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=5e-6, abs_tol=5e-7):
        raise ValueError(f"{field} replay mismatch: {actual!r} != {expected!r}")


def _scalar_values(row: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for value in row.values():
        if isinstance(value, Mapping) and {"bits", "status"} <= set(value):
            yield value


def _pair_key_for_candidate(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (row["seq"], row["frame"], row["cand_slot"], row["cand_instance_uid"])


def verify_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute scalar/ranking/claim/commit decisions without MOT or labels."""
    packet = canonical_semantic_packet(capture)
    streams: Mapping[str, list[dict[str, Any]]] = packet["streams"]
    for stream, rows in streams.items():
        for row in rows:
            if int(row.get("schema_version", 0)) != 1:
                raise ValueError(f"{stream} has an unexpected record schema version")
            for scalar in _scalar_values(row):
                if int(scalar.get("status", NOT_COMPUTED)) == COMPUTED_NAN:
                    raise ValueError(f"{stream} contains a computed NaN policy scalar")

    pairs = streams["pair_records"]
    candidates = streams["candidate_records"]
    claims = streams["claim_records"]
    commits = streams["commit_records"]

    # Scalar construction is checked only after pre-score gates, precisely when
    # native code computes bdist.  All arithmetic is rounded to float32.
    for pair in pairs:
        bdist = optional_scalar(pair["bdist_after_direction"])
        before = optional_scalar(pair["bdist_before_direction"])
        if bdist is None or before is None:
            continue
        w = scalar_value(pair["w"], field="w")
        fwd_r = scalar_value(pair["fwd_r"], field="fwd_r")
        bwd_r = scalar_value(pair["bwd_r"], field="bwd_r")
        dist_h = scalar_value(pair["dist_h"], field="dist_h")
        replay_before = f32(
            f32(w * f32(0.5 * f32(fwd_r + bwd_r))) + f32((1.0 - w) * dist_h)
        )
        close_f32(before, replay_before, field="bdist_before_direction")
        alpha = scalar_value(pair["directional_alpha"], field="directional_alpha")
        cross = optional_scalar(pair["directional_cross_bdist"])
        replay_after = (
            before
            if cross is None or alpha == 0.0
            else f32(f32(before * f32(1.0 - alpha)) + f32(cross * alpha))
        )
        close_f32(bdist, replay_after, field="bdist_after_direction")

    pairs_by_candidate: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(
        list
    )
    for pair in pairs:
        pairs_by_candidate[_pair_key_for_candidate(pair)].append(pair)

    proposal_keys: set[tuple[object, ...]] = set()
    for candidate in candidates:
        key = (
            candidate["seq"],
            candidate["frame"],
            candidate["cand_slot"],
            candidate["cand_instance_uid"],
        )
        competitor_rows = pairs_by_candidate[key]
        if int(candidate["structural_competitors"]) != len(competitor_rows):
            raise ValueError("candidate structural competitor conservation failed")
        pre_score = sum(
            int(pair["height_verdict"]) in (PASS, DISABLED)
            and int(pair["speed_verdict"]) in (PASS, DISABLED)
            and int(pair["spatial_verdict"]) in (PASS, DISABLED)
            for pair in competitor_rows
        )
        if int(candidate["pre_score_passes"]) != pre_score:
            raise ValueError("candidate pre-score count disagrees with pair records")
        eligible = [
            pair for pair in competitor_rows if int(pair["final_pair_eligible"]) == PASS
        ]
        if int(candidate["final_pair_eligible_count"]) != len(eligible):
            raise ValueError(
                "candidate final-eligible count disagrees with pair records"
            )
        if not eligible:
            if int(candidate["proposal_emitted"]) != REJECT:
                raise ValueError("candidate without eligible pairs emitted a proposal")
            continue
        ranked = sorted(
            eligible,
            key=lambda pair: scalar_value(
                pair["bdist_after_direction"], field="pair bdist"
            ),
        )
        best = scalar_value(candidate["best_bdist"], field="candidate best_bdist")
        close_f32(
            best,
            scalar_value(ranked[0]["bdist_after_direction"], field="ranked bdist"),
            field="candidate best_bdist",
        )
        if int(candidate["best_lost_slot"]) != int(ranked[0]["lost_slot"]):
            raise ValueError("candidate best lost slot disagrees with pair ranking")
        second = scalar_value(
            candidate["second_best_bdist"], field="candidate second_best_bdist"
        )
        if len(ranked) == 1:
            close_f32(second, NATIVE_SENTINEL, field="native singleton second_best")
            if int(candidate["no_second_competitor"]) != 1:
                raise ValueError(
                    "singleton candidate lost its no_second_competitor tag"
                )
        else:
            close_f32(
                second,
                scalar_value(
                    ranked[1]["bdist_after_direction"], field="runner-up bdist"
                ),
                field="candidate second_best_bdist",
            )
            if int(candidate["second_lost_slot"]) != int(ranked[1]["lost_slot"]):
                raise ValueError(
                    "candidate second lost slot disagrees with pair ranking"
                )
        margin = scalar_value(candidate["margin"], field="candidate margin")
        close_f32(margin, f32(second - best), field="candidate margin")
        if int(candidate["proposal_emitted"]) == PASS:
            proposal_keys.add(
                (
                    candidate["seq"],
                    candidate["frame"],
                    candidate["cand_slot"],
                    candidate["cand_instance_uid"],
                    candidate["best_lost_slot"],
                    candidate["best_lost_instance_uid"],
                )
            )

    claim_keys = {
        (
            row["seq"],
            row["frame"],
            row["proposing_cand_slot"],
            row["proposing_cand_instance_uid"],
            row["proposed_lost_slot"],
            row["proposed_lost_instance_uid"],
        )
        for row in claims
    }
    if proposal_keys != claim_keys:
        raise ValueError("proposal-to-claim linkage is not injective")

    claims_by_lost: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        score = scalar_value(claim["detection_score"], field="claim detection_score")
        expected_sq = int(min(max(score, 0.0), 1.0) * 32767.0)
        expected_key = (expected_sq << 16) | (
            int(claim["proposing_cand_slot"]) & 0xFFFF
        )
        if (
            int(claim["sq"]) != expected_sq
            or int(claim["packed_atomic_key"]) != expected_key
        ):
            raise ValueError("claim packed atomic key replay mismatch")
        claims_by_lost[
            (
                claim["seq"],
                claim["frame"],
                claim["proposed_lost_slot"],
                claim["proposed_lost_instance_uid"],
            )
        ].append(claim)

    winners: dict[tuple[object, ...], dict[str, Any]] = {}
    for lost_key, lost_claims in claims_by_lost.items():
        winner = max(lost_claims, key=lambda claim: int(claim["packed_atomic_key"]))
        winners[lost_key] = winner
        for claim in lost_claims:
            should_win = claim is winner
            if int(claim["claim_won"]) != (PASS if should_win else REJECT):
                raise ValueError("claim winner replay mismatch")
            if int(claim["winning_cand_slot"]) != int(winner["proposing_cand_slot"]):
                raise ValueError("claim winner slot was not preserved")
            if int(claim["winning_cand_instance_uid"]) != int(
                winner["proposing_cand_instance_uid"]
            ):
                raise ValueError("claim winner immutable identity was not preserved")

    if len(commits) != len(winners):
        raise ValueError("claim winner to commit conservation failed")
    commit_by_key = {
        (
            row["seq"],
            row["frame"],
            row["cand_slot"],
            row["cand_instance_uid"],
            row["lost_slot"],
            row["lost_instance_uid"],
        ): row
        for row in commits
    }
    if len(commit_by_key) != len(commits):
        raise ValueError("commit stable keys are not unique")
    for winner in winners.values():
        key = (
            winner["seq"],
            winner["frame"],
            winner["proposing_cand_slot"],
            winner["proposing_cand_instance_uid"],
            winner["proposed_lost_slot"],
            winner["proposed_lost_instance_uid"],
        )
        commit = commit_by_key.get(key)
        if commit is None:
            raise ValueError("winning claim has no matching commit")
        if int(commit["commit_executed"]) != PASS:
            raise ValueError("claim winner did not execute commit")
        if int(commit["lost_slot_deactivated"]) != PASS:
            raise ValueError("commit did not deactivate its lost slot")
        if int(commit["cand_postcommit_track_id"]) != int(
            commit["lost_precommit_track_id"]
        ):
            raise ValueError("commit did not adopt lost visible track ID")
        if not (
            int(commit["lost_active_before"]) == 1
            and int(commit["lost_active_after"]) == 0
        ):
            raise ValueError("commit lost active-state transition is invalid")

    return {
        "semantic_digest_sha256": semantic_digest(capture),
        "stream_totals": packet["stream_totals"],
        "replay": "full_commit_decision_trace_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.input_json.read_text(encoding="utf-8"))
    print(json.dumps(verify_capture(capture), sort_keys=True))


if __name__ == "__main__":
    main()
