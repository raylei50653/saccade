#!/usr/bin/env python3
"""Verify R1's estimator replay without reading labels or fitting a score.

Input must be the JSONL payload emitted by
``export_r1_temporal_reduction_capture.py``. The verifier recomputes the
Consumer-A temporal reduction from only the recorded causal windows, EMA
state, horizon, and bridge configuration. It reports component errors,
predicate agreement, and an event-local order check; the sealed R1 declaration
defines how these measurements may be interpreted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from saccade.perception.eval.consumer_a_bridge_fidelity import (
    consumer_a_estimate_from_rings,
    production_safe,
)


REPO = Path(__file__).resolve().parents[2]
PAYLOAD_SCHEMA_VERSION = "r1_temporal_reduction_payload_v1"
ABS_TOLERANCE = 1e-5
REPLAY_FIELDS: tuple[str, ...] = (
    "bdist",
    "dist_h",
    "fwd_r",
    "bwd_r",
    "v_lost_x",
    "v_lost_y",
    "v_cand_x",
    "v_cand_y",
    "ax",
    "ay",
    "cx0",
    "cy0",
    "h_ref",
    "s_lost",
    "w",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("R1 replay needs at least one captured event")
    seen: set[str] = set()
    for row in rows:
        if row.get("payload_schema_version") != PAYLOAD_SCHEMA_VERSION:
            raise ValueError("unsupported R1 payload schema")
        key = str(row.get("event_key", ""))
        if not key or key in seen:
            raise ValueError("R1 payload has missing or duplicate event keys")
        seen.add(key)
    return rows


def _ring(block: object, *, name: str) -> list[tuple[float, float, float]]:
    if not isinstance(block, dict):
        raise ValueError(f"{name} reduction missing")
    samples = block.get("chronological_cx_cy_h")
    consumed = int(block.get("consumed_samples", 0))
    if not isinstance(samples, list) or len(samples) != 4 or consumed not in {1, 4}:
        raise ValueError(f"invalid {name} reduction window")
    return [tuple(float(v) for v in sample) for sample in samples[:consumed]]


def replay_record(row: dict[str, Any]) -> dict[str, float | int]:
    terms = row.get("kernel_terms")
    if not isinstance(terms, dict):
        raise ValueError("R1 record lacks kernel_terms")
    lost = _ring(row.get("lost_reduction"), name="lost")
    candidate = _ring(row.get("candidate_reduction"), name="candidate")
    if len(candidate) != 4:
        raise ValueError("candidate R1 window must contain four consumed samples")
    estimate = consumer_a_estimate_from_rings(
        lost,
        candidate,
        gap=int(terms["gap"]),
        bridge_at=int(terms["bridge_at"]),
        ema_lost=float(terms["ema_lost"]),
        ema_cand=float(terms["ema_cand"]),
        anchor_mode=int(terms["anchor_mode"]),
        rate_gate=float(terms["anchor_rate"]),
        bridge_dir_bonus=float(terms["bridge_dir_bonus"]),
    )
    return {
        "bdist": estimate.bdist,
        "dist_h": estimate.dist_h,
        "fwd_r": estimate.fwd_r,
        "bwd_r": estimate.bwd_r,
        "v_lost_x": estimate.v_lost_x,
        "v_lost_y": estimate.v_lost_y,
        "v_cand_x": estimate.v_cand_x,
        "v_cand_y": estimate.v_cand_y,
        "ax": estimate.ax,
        "ay": estimate.ay,
        "cx0": estimate.cx0,
        "cy0": estimate.cy0,
        "h_ref": estimate.h_ref,
        "s_lost": estimate.s_lost,
        "w": estimate.w,
        "gap": estimate.gap,
        "bridge_at": estimate.bridge_at,
        "la": estimate.la,
    }


def _estimate_values(
    row: dict[str, Any],
    *,
    lost: list[tuple[float, float, float]],
    candidate: list[tuple[float, float, float]],
) -> dict[str, float | int]:
    """Replay a declared causal mutation without treating it as R0 fidelity."""
    terms = row["kernel_terms"]
    assert isinstance(terms, dict)
    estimate = consumer_a_estimate_from_rings(
        lost,
        candidate,
        gap=int(terms["gap"]),
        bridge_at=int(terms["bridge_at"]),
        ema_lost=float(terms["ema_lost"]),
        ema_cand=float(terms["ema_cand"]),
        anchor_mode=int(terms["anchor_mode"]),
        rate_gate=float(terms["anchor_rate"]),
        bridge_dir_bonus=float(terms["bridge_dir_bonus"]),
    )
    return {
        "bdist": estimate.bdist,
        "dist_h": estimate.dist_h,
        "fwd_r": estimate.fwd_r,
        "bwd_r": estimate.bwd_r,
        "v_lost_x": estimate.v_lost_x,
        "v_lost_y": estimate.v_lost_y,
        "v_cand_x": estimate.v_cand_x,
        "v_cand_y": estimate.v_cand_y,
        "ax": estimate.ax,
        "ay": estimate.ay,
        "cx0": estimate.cx0,
        "cy0": estimate.cy0,
        "h_ref": estimate.h_ref,
        "s_lost": estimate.s_lost,
        "w": estimate.w,
    }


def _mutation_bucket(
    buckets: dict[str, dict[str, Any]],
    *,
    mutation: str,
    row: dict[str, Any],
    values: dict[str, float | int] | None,
    unavailable: str | None = None,
) -> None:
    terms = row["kernel_terms"]
    assert isinstance(terms, dict)
    branch = str(row["lost_reduction"]["branch"])
    key = f"{mutation}|{branch}|la={int(terms['la'])}"
    bucket = buckets.setdefault(
        key,
        {
            "mutation": mutation,
            "lost_branch": branch,
            "la": int(terms["la"]),
            "events": 0,
            "predicate_flips": 0,
            "unavailable": 0,
            "max_abs_delta": {field: 0.0 for field in REPLAY_FIELDS},
        },
    )
    bucket["events"] += 1
    if unavailable is not None:
        bucket["unavailable"] += 1
        return
    assert values is not None
    for field in REPLAY_FIELDS:
        bucket["max_abs_delta"][field] = max(
            float(bucket["max_abs_delta"][field]),
            abs(float(values[field]) - float(terms[field])),
        )
    if production_safe(
        float(values["bdist"]), float(terms["production_threshold"])
    ) != production_safe(float(terms["bdist"]), float(terms["production_threshold"])):
        bucket["predicate_flips"] += 1


def verify(path: Path, *, abs_tolerance: float = ABS_TOLERANCE) -> dict[str, Any]:
    """Return deterministic replay measurements; does not assign a terminal."""
    if not math.isfinite(abs_tolerance) or abs_tolerance <= 0.0:
        raise ValueError("abs_tolerance must be finite and positive")
    rows = _load(path)
    errors: dict[str, list[float]] = defaultdict(list)
    predicate_disagreements = 0
    grouped: dict[tuple[str, int], list[tuple[str, float, float]]] = defaultdict(list)
    mutation_buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        terms = row["kernel_terms"]
        assert isinstance(terms, dict)
        replay = replay_record(row)
        for field in REPLAY_FIELDS:
            errors[field].append(abs(float(replay[field]) - float(terms[field])))
        for field in ("gap", "bridge_at", "la"):
            if int(replay[field]) != int(terms[field]):
                raise ValueError(
                    f"structural replay mismatch for {field}: {row['event_key']}"
                )
        if production_safe(
            float(replay["bdist"]), float(terms["production_threshold"])
        ) != production_safe(
            float(terms["bdist"]), float(terms["production_threshold"])
        ):
            predicate_disagreements += 1
        # Candidate event is the natural local ranking group. This tool does not
        # invent a label or a global comparison universe.
        grouped[(str(row["seq"]), int(row["cand_local_id"]))].append(
            (str(row["event_key"]), float(terms["bdist"]), float(replay["bdist"]))
        )
        lost = _ring(row.get("lost_reduction"), name="lost")
        candidate = _ring(row.get("candidate_reduction"), name="candidate")
        # A cyclic shift is intentionally not an equivalent serialization: it
        # exercises the declared window-order sensitivity while keeping the
        # same sample multiset. It must never be read as an R0 substitute.
        shifted_lost = lost[1:] + lost[:1] if len(lost) == 4 else lost
        shifted_candidate = candidate[1:] + candidate[:1]
        _mutation_bucket(
            mutation_buckets,
            mutation="cyclic_window_shift",
            row=row,
            values=_estimate_values(
                row, lost=shifted_lost, candidate=shifted_candidate
            ),
        )
        # Removing the oldest candidate sample triggers the real kernel's
        # <4 early return, so it has no score to compare. Removing the oldest
        # lost sample switches a full lost window to the explicit short-lost
        # fallback. Both dispositions are recorded instead of imputed.
        _mutation_bucket(
            mutation_buckets,
            mutation="omit_oldest_candidate_sample",
            row=row,
            values=None,
            unavailable="candidate_early_return",
        )
        if len(lost) == 4:
            _mutation_bucket(
                mutation_buckets,
                mutation="omit_oldest_lost_sample",
                row=row,
                values=_estimate_values(row, lost=lost[-1:], candidate=candidate),
            )
        else:
            _mutation_bucket(
                mutation_buckets,
                mutation="omit_oldest_lost_sample",
                row=row,
                values=None,
                unavailable="no_older_lost_sample",
            )

    comparable_pairs = order_disagreements = near_ties = 0
    for values in grouped.values():
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                _, captured_l, replay_l = values[left]
                _, captured_r, replay_r = values[right]
                captured_delta = captured_l - captured_r
                replay_delta = replay_l - replay_r
                if abs(captured_delta) <= 2.0 * abs_tolerance:
                    near_ties += 1
                    continue
                comparable_pairs += 1
                if (captured_delta < 0.0) != (replay_delta < 0.0):
                    order_disagreements += 1

    field_summary = {
        field: {
            "max_abs_error": max(values),
            "within_abs_tolerance": max(values) <= abs_tolerance,
        }
        for field, values in errors.items()
    }
    return {
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "verifier": {
            "path": "scripts/tools/verify_r1_temporal_reduction_replay.py",
            "sha256": _sha256(Path(__file__)),
        },
        "events": len(rows),
        "abs_tolerance": abs_tolerance,
        "fields": field_summary,
        "all_terms_within_abs_tolerance": all(
            summary["within_abs_tolerance"] for summary in field_summary.values()
        ),
        "predicate_disagreements": predicate_disagreements,
        "event_local_order": {
            "groups": len(grouped),
            "comparable_pairs": comparable_pairs,
            "near_ties": near_ties,
            "order_disagreements": order_disagreements,
        },
        "causal_sensitivity": sorted(
            mutation_buckets.values(),
            key=lambda item: (
                str(item["mutation"]),
                str(item["lost_branch"]),
                int(item["la"]),
            ),
        ),
        "outcome_labels_read": False,
        "score_fit_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abs-tolerance", type=float, default=ABS_TOLERANCE)
    args = parser.parse_args(argv)

    payload = args.payload if args.payload.is_absolute() else REPO / args.payload
    output = args.output if args.output.is_absolute() else REPO / args.output
    result = verify(payload, abs_tolerance=args.abs_tolerance)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
