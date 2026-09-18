#!/usr/bin/env python
"""Derive production double-buffer exposed-cost attribution.

Combines a clean production run, optional --profile-frame-csv ledgers,
an nsys node-mode JSON (from nsys_frame_attribution.py --json), and
optional SACCADE_ASSOC_STATS dumps. Does not run eval.

Usage:
    uv run python scripts/benchmarks/production_db_attribution.py \\
      --production-dir $RUN/p0_production \\
      --ledger-dir $RUN/p1_frame_csv \\
      --nsys-json $RUN/d1_nsys_04.json \\
      --assoc-dir $RUN/d2_assoc_stats \\
      --out $RUN/derived.json
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

KERNEL_CATS: list[tuple[str, str]] = [
    ("scan", r"selective_scan"),
    ("trt", r"_trt$|__myl|implicit_convolve|sm90_xmma|sm100_xmma|cudnn"),
    ("gmc_downscale", r"chw_to_grayscale_downscale"),
    (
        "gmc_fft",
        r"fft|cufft|dpVector|cross_power|find_peak|peak_to_translation|write_identity_warp",
    ),
    ("gmc_other", r"zero_rects"),
    (
        "tracker_predict",
        r"predict_gmc_sinv_fused|predict_kernel|gmc_kernel|init_covariance",
    ),
    ("tracker_occlusion", r"compute_track_occlusion"),
    (
        "tracker_cost",
        r"stage1_cost_fused|count_stage1_candidates|compute_conditional_cost",
    ),
    ("tracker_sinkhorn", r"fused_sinkhorn_multistage"),
    ("tracker_auction", r"parallel_auction_shmem|commit_auction_results"),
    (
        "tracker_state",
        r"track_state_update|inline_kalman_update|spawn_new_tracks|collect_free_slots",
    ),
    (
        "tracker_bridge",
        r"relink_bidir|update_foot_history|archive_|age_relink|invalidate_tracked|retire_revived|relink_births",
    ),
    ("tracker_compact", r"compact_results"),
    ("tracker_stats", r"accumulate_assoc_|accumulate_private_"),
    ("nms", r"nms_|gather_compact|build_sort_keys|decode_sort_order"),
    ("private_append", r"append_private_continuation|mark_indices_bool"),
    (
        "decode_pre",
        r"nvjpeg|jpeg|decode|huffman|dct|yuv|nv12|letterbox|rgba2rgb|grayscale",
    ),
    ("pointwise", r"elementwise|pointwise|copy_|cast|fill|where|clamp|sigmoid|silu"),
    (
        "reduce_index",
        r"reduce|index|gather|scatter|topk|Topk|TopK|nonzero|masked|sort|cub|arange|cumsum|Device(Select|Compact|RadixSort|Reduce)",
    ),
    ("gemm_conv", r"gemm|conv|cutlass|fprop"),
    ("transpose_cat", r"nchwToNhwc|nhwcToNchw|transpose|cat_|CatArray|permute"),
    ("upsample_pool", r"upsample|pool"),
    ("triton", r"^triton_"),
]


def classify_kernel(name: str) -> str:
    import re

    for cat, pat in KERNEL_CATS:
        if re.search(pat, name):
            return cat
    return "other"


def overlap_ms(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def union_ms(ivals: list[tuple[float, float]]) -> float:
    if not ivals:
        return 0.0
    ivals = sorted(ivals)
    total = 0.0
    cs, ce = ivals[0]
    for s, e in ivals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + ce - cs


ASSOC_LOW_WORK_FRAME_FRACTION = 0.05
ASSOC_EXPOSED_STAGES = (
    "tracker_occlusion",
    "tracker_sinkhorn",
    "tracker_auction",
)


def classify_exposure(duration_ms: float, exposed_ms: float) -> str:
    if duration_ms <= 1e-9:
        return "fully hidden"
    frac = exposed_ms / duration_ms
    if frac < 0.05:
        return "fully hidden"
    if frac < 0.95:
        return "partially exposed"
    return "critical-path"


def classify_assoc_stage(
    *,
    frames: float | None,
    frames_with_assignment: float | None,
    frames_with_valid_topk: float | None,
    assignments: float | None,
) -> str:
    """Classify a pass by activity frequency, not by assignments == 0.

    A pass that assigns on <5% of frames is low-work even if the assignment
    count is nonzero.
    """
    n = float(frames or 0)
    if n <= 0:
        return "固定 shape / fixed launch overhead"
    activity = (
        max(
            float(frames_with_assignment or 0),
            float(frames_with_valid_topk or 0),
        )
        / n
    )
    if activity < ASSOC_LOW_WORK_FRAME_FRACTION:
        return "常跑 + 幾乎沒工作"
    if float(assignments or 0) > 0:
        return "常跑 + 有效工作"
    return "固定 shape / fixed launch overhead"


def removal_ceiling(
    period_ms: float | None, removable_ms: float | None
) -> tuple[float | None, float | None]:
    """Upper-bound FPS if ``removable_ms`` left the production period.

    Returns (None, None) when the slice is not a valid remainder of the
    period — including the detector container, whose span *is* the period.
    """
    if period_ms is None or removable_ms is None:
        return None, None
    period = float(period_ms)
    removable = float(removable_ms)
    if removable <= 0 or period <= removable:
        return None, None
    leftover = period - removable
    # A container that *is* the period leaves a sub-ms leftover and a
    # fantasy FPS ceiling (e.g. 2.65 ms of 2.875 ms → 4435 FPS).
    if leftover < 1.0:
        return None, None
    return round(removable, 4), round(1000.0 / leftover, 1)


def lookup_stage_ms(
    nsys: dict[str, Any], name: str, field: str = "duration_ms"
) -> float | None:
    cats = nsys.get("category_ms_per_frame") or {}
    if name in cats and cats[name] is not None:
        return float(cats[name])
    for row in nsys.get("exposed_stages") or []:
        if row.get("stage") == name:
            val = row.get(field)
            if val is not None:
                return float(val)
    return None


def mean_or_none(vals: list[float]) -> float | None:
    return float(st.mean(vals)) if vals else None


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - k) + xs[hi] * (k - lo))


def load_production_dir(path: Path) -> dict[str, Any]:
    fps_path = path / "_fps_summary.txt"
    sequences: dict[str, Any] = {}
    overall_fps = None
    if fps_path.is_file():
        for line in fps_path.read_text().splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            seq, *rest = line.split("\t")
            parts = dict(item.split("=", 1) for item in rest if "=" in item)
            row = {
                "fps": float(parts["fps"]) if "fps" in parts else None,
                "mean_ms": float(parts["mean_ms"]) if "mean_ms" in parts else None,
                "frames": int(float(parts["frames"])) if "frames" in parts else None,
            }
            if seq.upper() == "OVERALL":
                overall_fps = row["fps"]
            else:
                sequences[seq] = row
    for lat in sorted(path.glob("_latency_profile_*.json")):
        payload = json.loads(lat.read_text())
        seq = payload.get("seq") or lat.name.replace("_latency_profile_", "").replace(
            ".json", ""
        )
        sequences.setdefault(seq, {})
        sequences[seq].update(
            {
                "fps": payload.get("throughput_fps", sequences[seq].get("fps")),
                "mean_ms": payload.get("mean_ms", sequences[seq].get("mean_ms")),
                "p95_ms": payload.get("p95_ms"),
                "p99_ms": payload.get("p99_ms"),
                "frames": payload.get("frames"),
                "throughput_seconds": payload.get("throughput_seconds"),
            }
        )
    frame_periods = [
        1000.0 / s["fps"]
        for s in sequences.values()
        if isinstance(s.get("fps"), (int, float)) and s["fps"] > 0
    ]
    if overall_fps and overall_fps > 0:
        period = 1000.0 / overall_fps
    else:
        period = mean_or_none(frame_periods)
    return {
        "dir": str(path),
        "overall_fps": overall_fps,
        "sequences": sequences,
        "mean_frame_period_ms": period,
    }


def load_ledgers(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("_frame_ledger_*.csv"))
    by_seq: dict[str, Any] = {}
    timing_cols = [
        "total_ms",
        "fetch_ms",
        "detect_ms",
        "post_ms",
        "gmc_ms",
        "track_ms",
        "output_ms",
        "post_graph_count_wait_ms",
        "post_pre_nms_ms",
        "post_finalize_ms",
        "post_nms_count_sync_ms",
        "post_final_count_sync_ms",
        "post_filter_count_sync_ms",
    ]
    count_cols = ["n_dets_raw", "n_dets_after_nms", "n_dets_final", "n_tracks"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    source: dict[str, str] = {}
    # Later sequences rewrite a cumulative ledger; the last file has every seq.
    ledger_file = files[-1]
    with ledger_file.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        seq = row.get("seq") or ledger_file.name
        grouped[seq].append(row)
        source[seq] = str(ledger_file)
    for seq, rows in grouped.items():
        summary: dict[str, Any] = {"frames": len(rows), "file": source[seq]}
        for col in timing_cols + count_cols:
            vals = [float(r.get(col, 0) or 0) for r in rows]
            summary[col] = {
                "mean": round(float(st.mean(vals)), 4) if vals else 0.0,
                "p50": round(float(percentile(vals, 50) or 0.0), 4),
                "p95": round(float(percentile(vals, 95) or 0.0), 4),
                "max": round(float(max(vals)), 4) if vals else 0.0,
            }
        by_seq[seq] = summary
    return {"dir": str(path), "sequences": by_seq, "note": "rows grouped by seq column"}


def load_assoc_dir(path: Path) -> dict[str, Any]:
    # ``association`` (tracker per-stage counters) is optional: main only
    # emits ``private_continuation`` + ``bridge`` because tracker_gpu.{hpp,cu}
    # are frozen H0/GCTM inputs. Dumps from the branch-only instrumentation
    # (commit e03f7d81) still parse; missing blocks derive to zeros.
    files = sorted(path.glob("_assoc_workload_*.json"))
    by_seq: dict[str, Any] = {}
    for f in files:
        payload = json.loads(f.read_text())
        seq = payload.get("seq") or f.name
        assoc = payload.get("association") or {}
        private = payload.get("private_continuation") or {}
        frames = float(assoc.get("frames") or 0) or None
        derived_stages = []
        for stg in assoc.get("stages") or []:
            name = stg.get("name")
            entering = float(stg.get("unmatched_tracks_entering") or 0)
            topk = float(stg.get("tracks_with_valid_topk") or 0)
            assigned = float(stg.get("assignments") or 0)
            n = frames or 1.0
            derived_stages.append(
                {
                    "name": name,
                    "per_frame_unmatched_tracks": entering / n,
                    "per_frame_valid_topk": topk / n,
                    "per_frame_assignments": assigned / n,
                    "frames_with_assignment": stg.get("frames_with_assignment"),
                    "frames_with_valid_topk": stg.get("frames_with_valid_topk"),
                    "assignment_rate_among_topk": (assigned / topk if topk else 0.0),
                    "class": classify_assoc_stage(
                        frames=frames,
                        frames_with_assignment=stg.get("frames_with_assignment"),
                        frames_with_valid_topk=stg.get("frames_with_valid_topk"),
                        assignments=assigned,
                    ),
                }
            )
        n = frames or 1.0
        by_seq[seq] = {
            "file": str(f),
            "association": assoc,
            "private_continuation": private,
            "bridge": payload.get("bridge"),
            "per_frame": {
                "active_tracks": (float(assoc.get("sum_active") or 0) / n),
                "confirmed": (float(assoc.get("sum_confirmed") or 0) / n),
                "tentative": (float(assoc.get("sum_tentative") or 0) / n),
                "cand_n": (float(assoc.get("sum_cand_n") or 0) / n),
                "matched": (float(assoc.get("sum_matched") or 0) / n),
                "dets": (float(assoc.get("sum_num_dets") or 0) / n),
                "dets_hi": (float(assoc.get("sum_dets_hi") or 0) / n),
                "dets_mid": (float(assoc.get("sum_dets_mid") or 0) / n),
                "dets_lo": (float(assoc.get("sum_dets_lo") or 0) / n),
                "dets_below": (float(assoc.get("sum_dets_below") or 0) / n),
                "occ_ttl_pos": (float(assoc.get("sum_occ_ttl_pos") or 0) / n),
                "private_added": (
                    float(private.get("sum_added") or 0)
                    / float(private.get("invocations") or n)
                ),
                "private_candidates": (
                    float(private.get("sum_candidate_count") or 0)
                    / float(private.get("invocations") or n)
                ),
            },
            "stages": derived_stages,
        }
    return {"dir": str(path), "sequences": by_seq}


def merge_intervals(ivals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ivals:
        return []
    ivals = sorted(ivals)
    out = [ivals[0]]
    for s, e in ivals[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def nsys_overlap_from_spans(
    detect_spans: list[tuple[float, float]],
    work: dict[str, list[tuple[float, float]]],
) -> dict[str, Any]:
    """Overlap vs a detect-busy union inside scan-anchored frame windows.

    ``detect_spans`` here are frame windows (period start, period end), not
    merged detect-graph blobs. Under double-buffer, consecutive detect graphs
    overlap, so grouping detect-graph intervals with a 0.5 ms gap merges
    frames and must not be used as the window.
    """
    if len(detect_spans) < 1:
        return {"error": "need >= 1 frame window"}
    detect_source = work.get("detect_graph") or detect_spans
    periods = [e - s for s, e in detect_spans]
    detect_durs: list[float] = []
    per_cat: dict[str, dict[str, float]] = {}
    for ws, we in detect_spans:
        det_clipped = []
        for iv in detect_source:
            if iv[1] <= ws or iv[0] >= we:
                continue
            det_clipped.append((max(iv[0], ws), min(iv[1], we)))
        det_union = merge_intervals(det_clipped)
        detect_durs.append(union_ms(det_union))
        for cat, ivals in work.items():
            if cat == "detect_graph":
                continue
            dur = 0.0
            hidden = 0.0
            for iv in ivals:
                if iv[1] <= ws or iv[0] >= we:
                    continue
                clipped = (max(iv[0], ws), min(iv[1], we))
                if clipped[1] <= clipped[0]:
                    continue
                dur += clipped[1] - clipped[0]
                for du in det_union:
                    hidden += overlap_ms(clipped, du)
            exposed = max(0.0, dur - hidden)
            bucket = per_cat.setdefault(
                cat, {"duration_ms": 0.0, "hidden_ms": 0.0, "exposed_ms": 0.0}
            )
            bucket["duration_ms"] += dur
            bucket["hidden_ms"] += hidden
            bucket["exposed_ms"] += exposed
    n = float(len(detect_spans))
    rows = []
    for cat, b in sorted(per_cat.items(), key=lambda x: -x[1]["exposed_ms"]):
        dur = b["duration_ms"] / n
        hid = b["hidden_ms"] / n
        exp = b["exposed_ms"] / n
        rows.append(
            {
                "stage": cat,
                "duration_ms": round(dur, 4),
                "hidden_ms": round(hid, 4),
                "exposed_ms": round(exp, 4),
                "class": classify_exposure(dur, exp),
            }
        )
    return {
        "n_periods": int(n),
        "detect_span_mean_ms": mean_or_none(detect_durs),
        "detect_period_mean_ms": mean_or_none(periods),
        "stages": rows,
    }


def decompose_period(
    production_period_ms: float | None,
    nsys: dict[str, Any],
) -> dict[str, Any]:
    gpu_busy = nsys.get("gpu_union_busy_ms_per_frame")
    detect = nsys.get("detect_span_mean_ms")
    tail = nsys.get("tail_mean_ms")
    tail_busy = nsys.get("tail_other_work_busy_ms")
    exposed_tracker = 0.0
    hidden_tracker = 0.0
    for row in nsys.get("exposed_stages") or []:
        name = row.get("stage") or ""
        if name.startswith("tracker") or name in {
            "gmc_downscale",
            "gmc_fft",
            "gmc_other",
            "nms",
            "private_append",
        }:
            exposed_tracker += float(row.get("exposed_ms") or 0)
            hidden_tracker += float(row.get("hidden_ms") or 0)
    remainder = None
    if production_period_ms is not None and detect is not None:
        remainder = round(float(production_period_ms) - float(detect), 4)
    return {
        "production_frame_period_ms": production_period_ms,
        "diagnostic_detect_span_ms": detect,
        "diagnostic_gpu_union_busy_ms": gpu_busy,
        "outside_detect_remainder_ms": remainder,
        "exposed_detector_ms": detect,
        "exposed_tracker_like_ms": round(exposed_tracker, 4),
        "hidden_tracker_like_ms": round(hidden_tracker, 4),
        "sync_or_host_tail_ms": None
        if tail is None or tail_busy is None
        else round(float(tail) - float(tail_busy), 4),
        "identity": (
            "production_frame_period ≈ detect_span + outside_detect_remainder"
        ),
        "note": (
            "outside_detect_remainder_ms is P-layer period minus D-layer detect "
            "span. It is a cross-run calibrated residual, not GPU idle. Do not "
            "subtract diagnostic GPU-union busy from the production period."
        ),
    }


def rank_bottlenecks(
    production: dict[str, Any],
    nsys: dict[str, Any] | None,
    assoc: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    del assoc  # ranking uses nsys exposed cost; assoc is classification-only
    period = production.get("mean_frame_period_ms")
    nsys = nsys or {}
    detect = nsys.get("detect_span_mean_ms")
    scan = lookup_stage_ms(nsys, "scan")
    remainder = None
    if period is not None and detect is not None:
        remainder = float(period) - float(detect)

    assoc_duration = 0.0
    assoc_exposed = 0.0
    for row in nsys.get("exposed_stages") or []:
        if row.get("stage") in ASSOC_EXPOSED_STAGES:
            assoc_duration += float(row.get("duration_ms") or 0)
            assoc_exposed += float(row.get("exposed_ms") or 0)

    out: list[dict[str, Any]] = []
    if detect is not None:
        slice_ms, slice_fps = removal_ceiling(period, scan)
        out.append(
            {
                "rank": "Primary",
                "name": "detector whole-graph (TRT + scan + postprocess)",
                "class_label": "D. detector compute",
                "observed_cost_ms": detect,
                "exposed_cost_ms": detect,
                "trigger_frequency": "every frame",
                "workload_scaling": "resolution-sensitive (05 vs 1080p); occupancy-insensitive",
                "overlap": "defines the period under saturated DB",
                "removal_applicable": False,
                "removal_upper_bound_ms": None,
                "removal_upper_bound_fps": None,
                "attackable_slice": {
                    "name": "selective_scan",
                    "observed_cost_ms": scan,
                    "exposed_cost_ms": scan,
                    "removal_applicable": True,
                    "removal_upper_bound_ms": slice_ms,
                    "removal_upper_bound_fps": slice_fps,
                    "note": (
                        "Largest attackable slice inside the detector container. "
                        "The container itself is not removable."
                    ),
                },
            }
        )
    if remainder is not None:
        rem_ms, rem_fps = removal_ceiling(period, remainder)
        out.append(
            {
                "rank": "Secondary",
                "name": "outside-detect remainder (P period minus D detect span)",
                "class_label": "B. synchronization / scheduling plus C. memory staging / traffic",
                "observed_cost_ms": round(remainder, 4),
                "exposed_cost_ms": round(remainder, 4),
                "trigger_frequency": "every frame",
                "workload_scaling": "1080p DtoD staging vs 640x480; host sync opportunity",
                "overlap": "cross-run residual after the detect span",
                "removal_applicable": True,
                "removal_upper_bound_ms": rem_ms,
                "removal_upper_bound_fps": rem_fps,
                "uncertainty": (
                    "P-layer period and D-layer detect span come from different "
                    "runs. This is not GPU idle and not production_period − GPU-union busy."
                ),
            }
        )
    if assoc_exposed > 0:
        a_ms, a_fps = removal_ceiling(period, assoc_exposed)
        out.append(
            {
                "rank": "Tertiary",
                "name": "fixed-capacity tracker association (occlusion + sinkhorn + auction)",
                "class_label": "A. fixed-capacity computation",
                "observed_cost_ms": round(assoc_duration, 4),
                "exposed_cost_ms": round(assoc_exposed, 4),
                "trigger_frequency": "every frame, Tcap=2048 Dcap=1024 launches",
                "workload_scaling": "auction/sinkhorn/occlusion nearly occupancy-insensitive",
                "overlap": "partially exposed",
                "removal_applicable": True,
                "removal_upper_bound_ms": a_ms,
                "removal_upper_bound_fps": a_fps,
            }
        )
    if not out:
        out.append(
            {
                "rank": "Primary",
                "name": "UNRESOLVED",
                "class_label": "G. no single bottleneck — overlap frontier already saturated",
                "observed_cost_ms": None,
                "exposed_cost_ms": None,
                "uncertainty": "missing nsys and/or production period",
            }
        )
    return out


def derive(
    production: dict[str, Any],
    ledger: dict[str, Any] | None = None,
    nsys: dict[str, Any] | None = None,
    assoc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    period = production.get("mean_frame_period_ms")
    return {
        "schema": "saccade-production-db-critical-path-v2",
        "contract": "docs/research/pipeline/production_db_critical_path_contract.md",
        "production": production,
        "host_ledger": ledger,
        "nsys": nsys,
        "association": assoc,
        "period_decomposition": decompose_period(period, nsys or {}),
        "bottlenecks": rank_bottlenecks(production, nsys, assoc),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production-dir", type=Path, required=True)
    p.add_argument("--ledger-dir", type=Path, default=None)
    p.add_argument("--nsys-json", type=Path, default=None)
    p.add_argument("--assoc-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    production = load_production_dir(args.production_dir)
    ledger = load_ledgers(args.ledger_dir) if args.ledger_dir else None
    nsys = json.loads(args.nsys_json.read_text()) if args.nsys_json else None
    assoc = load_assoc_dir(args.assoc_dir) if args.assoc_dir else None
    payload = derive(production, ledger, nsys, assoc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    for b in payload["bottlenecks"]:
        print(
            f"{b.get('rank')}: {b.get('name')} "
            f"exposed={b.get('exposed_cost_ms')} ms "
            f"class={b.get('class_label')}"
        )


if __name__ == "__main__":
    main()
