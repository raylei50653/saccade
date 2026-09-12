#!/usr/bin/env python3
"""Measurement-only replay of output-layer identity-repair chaining.

Current main has no production ``--cheb-gr-postproc-order`` surface. This
harness stacks offline handover and Cheb-GR tracklet merge on a frozen
pre-interpolation MOT substrate, re-extracting embeddings from each stage's
output before the next stage runs. Shipping presets, evaluator dispatch, and
production config are not modified.

Arms
----
  base
  handover_only
  merge_only
  handover_then_merge
  merge_then_handover

Usage
-----
  # Substrate must be tracker output with interpolation OFF (quality filter
  # is a no-op at shipping defaults). Capture with:
  uv run python scripts/eval/mot17.py \\
      --preset mamba_whole_graph_m --detector SDP --double-buffer \\
      --no-gpu-decode --no-interpolate-tracklets \\
      --output results/olr_reval_YYYYMMDD/substrate

  uv run python scripts/eval/experiments/run_output_layer_repair_chaining.py \\
      --substrate results/olr_reval_YYYYMMDD/substrate \\
      --out results/olr_reval_YYYYMMDD \\
      --artifact-dir docs/modules/semantic/research/evidence/output_layer_repair_chaining_revalidation_YYYYMMDD \\
      --substrate-commit <substrate SHA> \\
      --expected-substrate-sha256 docs/modules/semantic/research/evidence/.../substrate_sha256.json \\
      --repeats 3 --repeat-arms merge_only,handover_then_merge,merge_then_handover
"""
# status: experiment

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

# Historical 2026-09-05 operating points. Thresholds are not retuned here.
DEFAULT_MERGE = {
    "max_cost": 0.45,
    "max_gap": 60,
    "min_overlap_frames": 1,
    "n_samples": 50,
    "pool_frac": 0.3,
    "cheb_lambda": 2.0,
    "k2": 6,
    "max_fwd": 50,
    "fuse_lambda": 0.3,
}
DEFAULT_HANDOVER = {
    "max_cost": 0.45,
    "max_gap": 60,
    "decide_n": 5,
    "min_head_samples": 2,
    "margin": 0.05,
    "n_samples": 50,
    "pool_frac": 0.3,
    "cheb_lambda": 2.0,
    "k2": 6,
    "max_fwd": 50,
    "fuse_lambda": 0.3,
    "bank_mode": "spread",
    "bank_n": 0,
    "appearance_occlusion_cov": 0.4,
    "neighbor_iou_max": 0.0,
}
# Shipping mamba_whole_graph_m interpolation (applied AFTER identity stages).
DEFAULT_INTERP = {
    "max_gap": 35,
    "min_track_len": 5,
    "min_h": 0.0,
}

ARM_STAGES: dict[str, tuple[str, ...]] = {
    "base": (),
    "handover_only": ("handover",),
    "merge_only": ("merge",),
    "handover_then_merge": ("handover", "merge"),
    "merge_then_handover": ("merge", "handover"),
}

StageFn = Callable[..., tuple[list[str], dict[str, Any]]]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        )
    except subprocess.CalledProcessError:
        return False
    return bool(out.strip())


def verify_substrate_hashes(
    substrate: Path, sequences: list[str], expected_path: Path
) -> None:
    expected = json.loads(expected_path.read_text())
    mismatches: list[str] = []
    for seq in sequences:
        name = f"{seq}.txt"
        got = sha256_file(substrate / name)
        want = expected.get(name, {}).get("sha256") or expected.get(seq)
        if want is None:
            mismatches.append(f"{name}: missing from {expected_path}")
        elif got != want:
            mismatches.append(f"{name}: got {got}, expected {want}")
    if mismatches:
        raise SystemExit("substrate SHA-256 mismatch:\n  " + "\n  ".join(mismatches))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_track_ids(lines: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        ids.add(int(float(stripped.split(",")[1])))
    return ids


def load_seq_lines(substrate: Path, seq: str) -> list[str]:
    path = substrate / f"{seq}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing substrate file: {path}")
    return [ln.rstrip("\n") for ln in path.read_text().splitlines() if ln.strip()]


def write_seq_lines(out_dir: Path, seq: str, lines: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seq}.txt"
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def connected_component_count(accepted_pairs: list[tuple[int, int]]) -> int:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in accepted_pairs:
        union(int(a), int(b))
    return len({find(x) for x in parent}) if parent else 0


def summarize_merge_log(decision_log: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("verdict", "")) for row in decision_log)
    accepted_pairs = [
        (int(row["a_id"]), int(row["b_id"]))
        for row in decision_log
        if row.get("kind") == "pair" and row.get("verdict") == "accepted"
    ]
    return {
        "accepted": int(counts.get("accepted", 0)),
        "has_embedding": int(counts.get("has_embedding", 0)),
        "no_embedding": int(counts.get("no_embedding", 0)),
        "reject_cost": int(counts.get("reject_cost", 0)),
        "reject_temporal": int(counts.get("reject_temporal", 0)),
        "reject_same_component": int(counts.get("reject_same_component", 0)),
        "reject_component_overlap": int(counts.get("reject_component_overlap", 0)),
        "component_count": connected_component_count(accepted_pairs),
    }


def apply_merge_stage(
    lines: list[str],
    *,
    seq_img_dir: str,
    extractor: Any,
    params: dict[str, Any],
    extract_fn: Callable[..., dict[int, Any]] | None = None,
    merge_fn: Callable[..., tuple[list[str], dict[str, Any]]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Re-extract tracklet embeddings from ``lines``, then merge."""
    from saccade.perception.eval.cheb_gr_merge import (
        cheb_gr_merge_output_tracklets,
        extract_tracklet_embeddings,
    )

    extract = extract_fn or extract_tracklet_embeddings
    merge = merge_fn or cheb_gr_merge_output_tracklets
    embeddings = extract(
        lines,
        seq_img_dir,
        extractor,
        n_samples=int(params["n_samples"]),
        crop_hw=getattr(extractor, "input_hw", (224, 224)),
        appearance_occlusion_gate=True,
        appearance_occlusion_cov=float(params.get("appearance_occlusion_cov", 0.4)),
    )
    decision_log: list[dict[str, Any]] = []
    out, stats = merge(
        lines,
        embeddings,
        enabled=True,
        max_cost=float(params["max_cost"]),
        max_gap=int(params["max_gap"]),
        min_overlap_frames=int(params["min_overlap_frames"]),
        pool_frac=float(params["pool_frac"]),
        cheb_lambda=float(params["cheb_lambda"]),
        k2=int(params["k2"]),
        max_fwd=int(params["max_fwd"]),
        fuse_lambda=float(params["fuse_lambda"]),
        decision_log=decision_log,
    )
    diag = summarize_merge_log(decision_log)
    record = {
        "stage": "merge",
        "input_track_ids": sorted(unique_track_ids(lines)),
        "output_track_ids": sorted(unique_track_ids(out)),
        "stats": {
            k: int(v) if isinstance(v, (int, np.integer)) else v
            for k, v in stats.items()
        },
        "diagnostics": diag,
        "accepted_links": int(stats.get("merges", diag["accepted"])),
    }
    return out, record


def apply_handover_stage(
    lines: list[str],
    *,
    seq_img_dir: str,
    extractor: Any,
    params: dict[str, Any],
    extract_fn: Callable[..., tuple[dict[int, Any], dict[int, Any]]] | None = None,
    handover_fn: Callable[..., tuple[list[str], dict[str, Any]]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Re-extract head/bank embeddings from ``lines``, then handover."""
    from saccade.perception.eval.cheb_gr_online import (
        causal_handover_lines,
        extract_handover_embeddings,
    )

    extract = extract_fn or extract_handover_embeddings
    handover = handover_fn or causal_handover_lines
    head_embs, bank_embs = extract(
        lines,
        seq_img_dir,
        extractor,
        decide_n=int(params["decide_n"]),
        n_samples=int(params["n_samples"]),
        crop_hw=getattr(extractor, "input_hw", (224, 224)),
        appearance_occlusion_cov=float(params["appearance_occlusion_cov"]),
        neighbor_iou_max=float(params["neighbor_iou_max"]),
        bank_mode=str(params["bank_mode"]),
        bank_n=int(params["bank_n"]),
    )
    out, stats = handover(
        lines,
        head_embs,
        bank_embs,
        enabled=True,
        max_cost=float(params["max_cost"]),
        max_gap=int(params["max_gap"]),
        decide_n=int(params["decide_n"]),
        min_head_samples=int(params["min_head_samples"]),
        margin=float(params["margin"]),
        pool_frac=float(params["pool_frac"]),
        cheb_lambda=float(params["cheb_lambda"]),
        k2=int(params["k2"]),
        max_fwd=int(params["max_fwd"]),
        fuse_lambda=float(params["fuse_lambda"]),
    )
    record = {
        "stage": "handover",
        "input_track_ids": sorted(unique_track_ids(lines)),
        "output_track_ids": sorted(unique_track_ids(out)),
        "stats": {
            k: int(v) if isinstance(v, (int, np.integer)) else v
            for k, v in stats.items()
        },
        "diagnostics": {
            "handovers": int(stats.get("handovers", 0)),
            "events": int(stats.get("events", 0)),
            "events_with_candidates": int(stats.get("events_with_candidates", 0)),
            "reject_cost": int(stats.get("reject_cost", 0)),
            "reject_margin": int(stats.get("reject_margin", 0)),
            "reject_min_head": int(stats.get("reject_min_head", 0)),
            "reject_no_head": int(stats.get("reject_no_head", 0)),
            "ids_before": int(stats.get("ids_before", 0)),
            "ids_after": int(stats.get("ids_after", 0)),
        },
        "accepted_links": int(stats.get("handovers", 0)),
    }
    return out, record


def apply_stages(
    lines: list[str],
    stages: tuple[str, ...],
    *,
    seq_img_dir: str,
    extractor: Any,
    merge_params: dict[str, Any],
    handover_params: dict[str, Any],
    merge_stage: StageFn | None = None,
    handover_stage: StageFn | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply identity stages in order. Each stage rebuilds its own inputs."""
    current = list(lines)
    records: list[dict[str, Any]] = []
    merge_stage = merge_stage or apply_merge_stage
    handover_stage = handover_stage or apply_handover_stage
    for stage in stages:
        if stage == "merge":
            current, rec = merge_stage(
                current,
                seq_img_dir=seq_img_dir,
                extractor=extractor,
                params=merge_params,
            )
        elif stage == "handover":
            current, rec = handover_stage(
                current,
                seq_img_dir=seq_img_dir,
                extractor=extractor,
                params=handover_params,
            )
        else:
            raise ValueError(f"unknown stage {stage!r}")
        rec["stage_index"] = len(records)
        records.append(rec)
    return current, records


def finalize_lines(
    lines: list[str],
    *,
    interpolate: bool,
    interp_params: dict[str, Any],
) -> tuple[list[str], dict[str, int]]:
    from saccade.perception.eval.post_merge import interpolate_tracklets

    if not interpolate:
        return list(lines), {
            "gaps_filled": 0,
            "frames_added": 0,
            "tracks_interpolated": 0,
        }
    out, stats = interpolate_tracklets(
        list(lines),
        max_gap=int(interp_params["max_gap"]),
        min_track_len=int(interp_params["min_track_len"]),
        min_h=float(interp_params["min_h"]),
    )
    return out, {k: int(v) for k, v in stats.items()}


def score_output_dir(
    output_dir: Path,
    *,
    data_root: str,
    split: str,
    sequences: list[str],
) -> dict[str, Any]:
    """Full-precision motmetrics + TrackEval HOTA family."""
    import motmetrics as mm

    from saccade.perception.eval.metrics import (
        _calculate_hota,
        run_motmetrics_evaluation,
    )

    seq_csv = ",".join(sequences)
    printed = run_motmetrics_evaluation(
        data_root, split, str(output_dir), seq_csv, detector="SDP"
    )
    accs, names = [], []
    jobs: list[tuple[str, str, str]] = []
    for seq in sequences:
        gt_path = Path(data_root) / split / seq / "gt" / "gt.txt"
        ts_path = output_dir / f"{seq}.txt"
        gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(str(ts_path), fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5))
        names.append(seq)
        jobs.append((seq, str(gt_path), str(ts_path)))
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        names=names,
        metrics=[
            "idf1",
            "mota",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mostly_tracked",
            "mostly_lost",
        ],
        generate_overall=True,
    )
    overall = summary.loc["OVERALL"]
    hota = _calculate_hota(data_root, split, str(output_dir), jobs)
    final_tracks = 0
    for seq in sequences:
        final_tracks += len(unique_track_ids(load_seq_lines(output_dir, seq)))
    raw = {
        "IDF1": float(overall["idf1"]) * 100.0,
        "MOTA": float(overall["mota"]) * 100.0,
        "IDs": int(overall["num_switches"]),
        "FP": int(overall["num_false_positives"]),
        "FN": int(overall["num_misses"]),
        "final_tracks": int(final_tracks),
    }
    if hota is not None:
        raw["HOTA"] = float(hota["HOTA"]) * 100.0
        raw["DetA"] = float(hota["DetA"]) * 100.0
        raw["AssA"] = float(hota["AssA"]) * 100.0
    return {
        "raw": raw,
        "printed": printed,
        "per_sequence": {
            seq: {
                "IDF1": float(summary.loc[seq]["idf1"]) * 100.0,
                "MOTA": float(summary.loc[seq]["mota"]) * 100.0,
                "IDs": int(summary.loc[seq]["num_switches"]),
            }
            for seq in sequences
        },
    }


def aggregate_stage_records(
    seq_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_stage: dict[str, dict[str, int]] = {}
    total_links = 0
    ordered: list[str] = []
    for recs in seq_records.values():
        for rec in recs:
            name = rec["stage"]
            if name not in ordered:
                ordered.append(name)
            slot = by_stage.setdefault(
                name,
                {
                    "accepted_links": 0,
                    "no_embedding": 0,
                    "reject_cost": 0,
                    "reject_temporal": 0,
                    "reject_same_component": 0,
                    "reject_component_overlap": 0,
                    "component_count": 0,
                    "handovers": 0,
                    "events": 0,
                    "merges": 0,
                },
            )
            slot["accepted_links"] += int(rec.get("accepted_links", 0))
            total_links += int(rec.get("accepted_links", 0))
            diag = rec.get("diagnostics") or {}
            for key in (
                "no_embedding",
                "reject_cost",
                "reject_temporal",
                "reject_same_component",
                "reject_component_overlap",
                "component_count",
                "handovers",
                "events",
            ):
                if key in diag:
                    slot[key] += int(diag[key])
            stats = rec.get("stats") or {}
            if "merges" in stats:
                slot["merges"] += int(stats["merges"])
    return {
        "stage_order": ordered,
        "accepted_links": total_links,
        "per_stage": by_stage,
    }


def runtime_identity_blob() -> dict[str, Any]:
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
    except Exception as exc:  # pragma: no cover - environment probe
        gpu = {"error": str(exc)}
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": gpu,
        "env": {
            key: os.environ.get(key)
            for key in (
                "SACCADE_DOUBLE_BUFFER",
                "SACCADE_DETECT_BARRIER",
                "SACCADE_GPU_DECODE",
                "SACCADE_MAIN_NMS_GRAPHED",
                "SACCADE_STREAM_MODE",
            )
        },
    }


def run_arm(
    arm: str,
    *,
    substrate: Path,
    out_root: Path,
    data_root: Path,
    split: str,
    sequences: list[str],
    extractor: Any,
    merge_params: dict[str, Any],
    handover_params: dict[str, Any],
    interp_params: dict[str, Any],
    interpolate: bool,
    repeat_index: int = 0,
) -> dict[str, Any]:
    stages = ARM_STAGES[arm]
    tag = arm if repeat_index == 0 else f"{arm}_r{repeat_index}"
    arm_dir = out_root / "arms" / tag
    seq_records: dict[str, list[dict[str, Any]]] = {}
    interp_totals = Counter()
    for seq in sequences:
        raw = load_seq_lines(substrate, seq)
        seq_img_dir = str(data_root / split / seq / "img1")
        repaired, records = apply_stages(
            raw,
            stages,
            seq_img_dir=seq_img_dir,
            extractor=extractor,
            merge_params=merge_params,
            handover_params=handover_params,
        )
        final, interp_stats = finalize_lines(
            repaired, interpolate=interpolate, interp_params=interp_params
        )
        write_seq_lines(arm_dir, seq, final)
        seq_records[seq] = records
        interp_totals.update(interp_stats)
        stage_bits = " → ".join(rec["stage"] for rec in records) or "(none)"
        links = sum(int(rec.get("accepted_links", 0)) for rec in records)
        print(
            f"  [{tag}] {seq}: stages={stage_bits or 'base'} "
            f"ids={len(unique_track_ids(raw))}→{len(unique_track_ids(final))} "
            f"accepted={links} interp_frames={interp_stats.get('frames_added', 0)}"
        )
    metrics = score_output_dir(
        arm_dir, data_root=str(data_root), split=split, sequences=sequences
    )
    stage_agg = aggregate_stage_records(seq_records)
    return {
        "arm": arm,
        "repeat_index": repeat_index,
        "tag": tag,
        "stage_order": list(stages),
        "output_dir": str(arm_dir),
        "metrics": metrics,
        "stages": stage_agg,
        "interpolation": dict(interp_totals),
        "seq_stage_records": {
            seq: [
                {
                    "stage": rec["stage"],
                    "stage_index": rec["stage_index"],
                    "accepted_links": rec["accepted_links"],
                    "n_input_ids": len(rec["input_track_ids"]),
                    "n_output_ids": len(rec["output_track_ids"]),
                    "diagnostics": rec.get("diagnostics", {}),
                    "stats": rec.get("stats", {}),
                }
                for rec in recs
            ]
            for seq, recs in seq_records.items()
        },
        "mot_sha256": {seq: sha256_file(arm_dir / f"{seq}.txt") for seq in sequences},
    }


def round1(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def markdown_table(arm_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| arm | IDF1 | MOTA | HOTA | AssA | IDs | final tracks | accepted links |",
        "| --- | ---: | ---: | ---: | ---: | --: | -----------: | -------------: |",
    ]
    for row in arm_rows:
        raw = row["metrics"]["raw"]
        lines.append(
            "| {arm} | {idf1} | {mota} | {hota} | {assa} | {ids} | {tracks} | {links} |".format(
                arm=row["arm"],
                idf1=round1(raw.get("IDF1")),
                mota=round1(raw.get("MOTA")),
                hota=round1(raw.get("HOTA")),
                assa=round1(raw.get("AssA")),
                ids=int(raw.get("IDs", 0)),
                tracks=int(raw.get("final_tracks", 0)),
                links=int(row["stages"]["accepted_links"]),
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay output-layer handover/merge chaining on a frozen substrate."
    )
    p.add_argument("--substrate", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Tracked evidence directory for JSON/manifest (MOT files stay under --out).",
    )
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument("--seqs", default=",".join(SEQS))
    p.add_argument("--cheb-gr-model", default="mobilenetv4_reid")
    p.add_argument("--cheb-gr-engine", default="")
    p.add_argument("--merge-max-cost", type=float, default=DEFAULT_MERGE["max_cost"])
    p.add_argument(
        "--handover-min-head", type=int, default=DEFAULT_HANDOVER["min_head_samples"]
    )
    p.add_argument("--handover-margin", type=float, default=DEFAULT_HANDOVER["margin"])
    p.add_argument(
        "--handover-max-cost", type=float, default=DEFAULT_HANDOVER["max_cost"]
    )
    p.add_argument("--no-interpolate", action="store_true")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--repeat-arms",
        default="merge_only,handover_then_merge,merge_then_handover",
        help="Comma-separated arms to repeat as within-process replays "
        "(same TRTFeatureExtractor, beyond the first pass).",
    )
    p.add_argument(
        "--arms",
        default=",".join(ARM_STAGES),
        help="Comma-separated arms to run.",
    )
    p.add_argument(
        "--substrate-commit",
        required=True,
        help="Git commit that produced the frozen tracker substrate. Recorded "
        "separately from the replay harness commit.",
    )
    p.add_argument(
        "--expected-substrate-sha256",
        type=Path,
        default=None,
        help="Optional JSON map of MOT filename → {sha256} to fail-closed "
        "if the substrate files have moved.",
    )
    p.add_argument(
        "--require-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse to start if the git worktree is dirty (default: true).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    substrate = args.substrate.resolve()
    out_root = args.out.resolve()
    artifact_dir = (
        args.artifact_dir.resolve() if args.artifact_dir else out_root / "artifacts"
    )
    sequences = [s.strip() for s in args.seqs.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARM_STAGES:
            raise SystemExit(f"unknown arm {arm!r}; choose from {list(ARM_STAGES)}")
    repeat_arms = {a.strip() for a in args.repeat_arms.split(",") if a.strip()}
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    merge_params = dict(DEFAULT_MERGE)
    merge_params["max_cost"] = float(args.merge_max_cost)
    handover_params = dict(DEFAULT_HANDOVER)
    handover_params["min_head_samples"] = int(args.handover_min_head)
    handover_params["margin"] = float(args.handover_margin)
    handover_params["max_cost"] = float(args.handover_max_cost)
    interp_params = dict(DEFAULT_INTERP)

    dirty = git_dirty()
    if args.require_clean and dirty:
        raise SystemExit(
            "git worktree is dirty; refuse to record replay provenance. "
            "Commit (or pass --no-require-clean)."
        )
    if args.expected_substrate_sha256 is not None:
        verify_substrate_hashes(
            substrate, sequences, args.expected_substrate_sha256.resolve()
        )

    from saccade.perception.feature_extractor import TRTFeatureExtractor

    extractor = TRTFeatureExtractor(
        engine_path=args.cheb_gr_engine,
        model_type=args.cheb_gr_model,
        max_batch=64,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    identity = {
        "substrate_commit": args.substrate_commit,
        "replay_harness_commit": git_sha(),
        "dirty": dirty,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "harness": "scripts/eval/experiments/run_output_layer_repair_chaining.py",
        "repeat_kind": "within_process",
        "substrate": str(substrate),
        "substrate_sha256": {
            seq: sha256_file(substrate / f"{seq}.txt") for seq in sequences
        },
        "preset_contract": {
            "preset": "mamba_whole_graph_m",
            "detector": "SDP",
            "sequences": sequences,
            "reid_mode": "off",
            "decode": "cpu JPEG via --no-gpu-decode (substrate capture)",
            "double_buffer": True,
            "interpolate_after_stages": not args.no_interpolate,
            "interpolation": interp_params,
            "cheb_gr_model": args.cheb_gr_model,
            "cheb_gr_engine": args.cheb_gr_engine
            or "models/embedding/mobilenetv4_reid_visclean_224.engine",
            "merge": merge_params,
            "handover": handover_params,
        },
        "runtime": runtime_identity_blob(),
        "argv": sys.argv,
    }

    results: list[dict[str, Any]] = []
    for arm in arms:
        print(f"\n=== arm {arm}  stages={list(ARM_STAGES[arm]) or ['(none)']} ===")
        results.append(
            run_arm(
                arm,
                substrate=substrate,
                out_root=out_root,
                data_root=data_root,
                split=args.split,
                sequences=sequences,
                extractor=extractor,
                merge_params=merge_params,
                handover_params=handover_params,
                interp_params=interp_params,
                interpolate=not args.no_interpolate,
                repeat_index=0,
            )
        )
        extra = args.repeats - 1 if arm in repeat_arms else 0
        for ridx in range(1, extra + 1):
            print(f"\n=== arm {arm} repeat {ridx} ===")
            results.append(
                run_arm(
                    arm,
                    substrate=substrate,
                    out_root=out_root,
                    data_root=data_root,
                    split=args.split,
                    sequences=sequences,
                    extractor=extractor,
                    merge_params=merge_params,
                    handover_params=handover_params,
                    interp_params=interp_params,
                    interpolate=not args.no_interpolate,
                    repeat_index=ridx,
                )
            )

    primary = [row for row in results if row["repeat_index"] == 0]
    variation: dict[str, Any] = {}
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_arm[row["arm"]].append(row)
    for arm, rows in by_arm.items():
        if len(rows) < 2:
            continue
        metric_keys = ["IDF1", "MOTA", "HOTA", "AssA", "IDs"]
        stats: dict[str, Any] = {"n": len(rows)}
        for key in metric_keys:
            vals = [
                float(r["metrics"]["raw"][key])
                for r in rows
                if key in r["metrics"]["raw"]
            ]
            if not vals:
                continue
            stats[key] = {
                "values": vals,
                "min": min(vals),
                "max": max(vals),
                "range": max(vals) - min(vals),
            }
        mot_equal = all(r["mot_sha256"] == rows[0]["mot_sha256"] for r in rows[1:])
        stats["mot_files_identical"] = mot_equal
        stats["repeat_kind"] = "within_process"
        variation[arm] = stats

    payload = {
        "schema": "output_layer_repair_chaining_revalidation/v1",
        "identity": identity,
        "table_markdown": markdown_table(primary),
        "primary": primary,
        "repeats": [row for row in results if row["repeat_index"] > 0],
        "variation": variation,
    }
    results_path = artifact_dir / "results.json"
    manifest_path = artifact_dir / "manifest.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "output_layer_repair_chaining_revalidation_manifest/v1",
                "identity": identity,
                "results": str(results_path),
                "commands": {
                    "substrate_capture": [
                        "uv run python scripts/eval/mot17.py",
                        "--preset mamba_whole_graph_m",
                        "--detector SDP",
                        "--double-buffer",
                        "--no-gpu-decode",
                        "--no-interpolate-tracklets",
                        f"--output {substrate}",
                    ],
                    "replay": sys.argv,
                },
                "arm_output_dirs": {row["tag"]: row["output_dir"] for row in results},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\n" + payload["table_markdown"])
    print(f"\nWrote {results_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
