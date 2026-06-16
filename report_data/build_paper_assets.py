#!/usr/bin/env python3
"""Rebuild paper-facing tables and figures from tracked experiment artifacts."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saccade.perception.eval.metrics import (
    _calculate_hota,
    _evaluate_single_sequence,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_data"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
GT_ROOT = ROOT / "datasets" / "MOT17" / "train"
SEQUENCES = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

EXPERIMENTS = [
    {
        "name": "legacy_v14",
        "family": "legacy",
        "seed": "",
        "result_dir": "results/mamba_v14_legacy_recheck",
        "checkpoint": "runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        "status": "development-only; all seven SDP sequences seen",
    },
    {
        "name": "replica_20260612",
        "family": "plain_gt2",
        "seed": "20260612",
        "result_dir": "results/mamba_v14replica_final",
        "checkpoint": "runs/mamba_gt_v14replica_final/best.ckpt",
        "status": "development-only; replication intentionally preserves leakage",
    },
    {
        "name": "replica_20260613",
        "family": "plain_gt2",
        "seed": "20260613",
        "result_dir": "results/mamba_v14replica_s13",
        "checkpoint": "runs/mamba_gt_v14replica_s13_final/best.ckpt",
        "status": "development-only; replication intentionally preserves leakage",
    },
    {
        "name": "replica_20260614",
        "family": "plain_gt2",
        "seed": "20260614",
        "result_dir": "results/mamba_v14replica_s14",
        "checkpoint": "runs/mamba_gt_v14replica_s14_final/best.ckpt",
        "status": "development-only; replication intentionally preserves leakage",
    },
    {
        "name": "t3t1_seed42",
        "family": "t3_to_t1",
        "seed": "42",
        "result_dir": "results/mamba_v14replica_t3t1",
        "checkpoint": "runs/mamba_gt_v14replica_t3_t1/best.ckpt",
        "status": "development-only; seed is not paired with replica_20260612",
    },
    {
        "name": "t3t1_20260613",
        "family": "t3_to_t1",
        "seed": "20260613",
        "result_dir": "results/mamba_v14replica_t3t1_s13",
        "checkpoint": "runs/mamba_gt_v14replica_t3_t1_s13/best.ckpt",
        "status": "development-only; paired curriculum ablation",
    },
    {
        "name": "t3t1_20260614",
        "family": "t3_to_t1",
        "seed": "20260614",
        "result_dir": "results/mamba_v14replica_t3t1_s14",
        "checkpoint": "runs/mamba_gt_v14replica_t3_t1_s14/best.ckpt",
        "status": "development-only; paired curriculum ablation",
    },
    {
        "name": "ssm_unfreeze_ft",
        "family": "ssm_unfreeze",
        "seed": "20260612",
        "result_dir": "results/mamba_v14replica_ssmft",
        "checkpoint": "runs/mamba_gt_v14replica_ssmft_n16/best.ckpt",
        "status": "development-only; warm-start unfreeze ablation",
    },
]

BRIDGE_EXPERIMENTS = [
    {
        "name": "bridge_off",
        "result_dir": "results/MOT17_bridge_off",
        "description": "interpolation on, bridge relink off",
    },
    {
        "name": "bridge_on",
        "result_dir": "results/MOT17_bridge_on",
        "description": "speed-weighted bridge with scale and margin gates",
    },
    {
        "name": "scale_gate_baseline",
        "result_dir": "results/MOT17_baseline_rerun",
        "description": "bridge baseline before scale gate",
    },
    {
        "name": "scale_gate_on",
        "result_dir": "results/MOT17_ablation_scale_gate",
        "description": "bridge plus height-ratio scale gate",
    },
]

CURRICULUM_BOUNDARIES = [
    {
        "name": "phase_a_t1_eval",
        "family": "temporal_inference_probe",
        "seed": "42",
        "result_dir": "results/mamba_v14replica_t3_T1eval",
        "checkpoint": "runs/mamba_gt_v14replica_t3/best_flowgate0.ckpt",
        "interpretation": "Phase-A checkpoint evaluated without streaming temporal inference",
    },
    {
        "name": "phase_a_t3_streaming",
        "family": "temporal_inference_probe",
        "seed": "42",
        "result_dir": "results/mamba_v14replica_t3_T3eval",
        "checkpoint": "runs/mamba_gt_v14replica_t3/best_flowgate0.ckpt",
        "interpretation": "Streaming temporal inference adds little IDF1 and large latency",
    },
    {
        "name": "t3t1_then_ssmft",
        "family": "curriculum_order",
        "seed": "42",
        "result_dir": "results/mamba_v14replica_t3t1_ssmft",
        "checkpoint": "runs/mamba_gt_v14replica_t3t1_ssmft/best.ckpt",
        "interpretation": "Full-gradient fine-tuning after shaping erases association gains",
    },
    {
        "name": "ssmft_then_t3t1",
        "family": "curriculum_order",
        "seed": "42",
        "result_dir": "results/mamba_v14replica_ssmft_t3t1",
        "checkpoint": "runs/mamba_gt_v14replica_ssmft_t3_t1/best.ckpt",
        "interpretation": "Shaping last partially restores association but not the T3-to-T1 optimum",
    },
    {
        "name": "soup_alpha_025",
        "family": "weight_interpolation",
        "seed": "mixed",
        "result_dir": "results/mamba_soup_a25",
        "checkpoint": "runs/mamba_soup_ssmft_t3t1_a25.ckpt",
        "interpretation": "No interpolation synergy",
    },
    {
        "name": "soup_alpha_050",
        "family": "weight_interpolation",
        "seed": "mixed",
        "result_dir": "results/mamba_soup_a50",
        "checkpoint": "runs/mamba_soup_ssmft_t3t1_a50.ckpt",
        "interpretation": "No interpolation synergy",
    },
    {
        "name": "soup_alpha_075",
        "family": "weight_interpolation",
        "seed": "mixed",
        "result_dir": "results/mamba_soup_a75",
        "checkpoint": "runs/mamba_soup_ssmft_t3t1_a75.ckpt",
        "interpretation": "Shallow valley before the narrow T3-to-T1 endpoint",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_latency(result_dir: Path) -> dict[str, Any]:
    latency = {
        "fps": 0.0,
        "mean_ms": 0.0,
        "frames": 0,
        "profile_sequence": "",
        "profile_frames": 0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
    }
    summary = result_dir / "_fps_summary.txt"
    if summary.exists():
        for line in summary.read_text().splitlines():
            if not line.startswith("OVERALL\t"):
                continue
            fields = dict(part.split("=", 1) for part in line.split("\t")[1:])
            latency.update(
                {
                    "fps": float(fields["fps"]),
                    "mean_ms": float(fields["mean_ms"]),
                    "frames": int(fields["frames"]),
                }
            )
            break

    profile_path = result_dir / "_latency_profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        latency.update(
            {
                "profile_sequence": profile.get("sequence", ""),
                "profile_frames": int(profile.get("frames", 0)),
                "p95_ms": float(profile.get("p95_ms", 0.0)),
                "p99_ms": float(profile.get("p99_ms", 0.0)),
            }
        )
    return latency


def _counts_to_metrics(counts: Mapping[str, float | int]) -> dict[str, float | int]:
    idtp = int(counts["idtp"])
    idfp = int(counts["idfp"])
    idfn = int(counts["idfn"])
    fp = int(counts["num_false_positives"])
    fn = int(counts["num_misses"])
    ids = int(counts["num_switches"])
    objects = max(int(counts["num_objects"]), 1)
    predictions = max(int(counts["num_predictions"]), 1)
    idf1_den = 2 * idtp + idfp + idfn
    return {
        "IDF1": 100.0 * 2 * idtp / idf1_den if idf1_den else 0.0,
        "MOTA": 100.0 * (1.0 - (fn + fp + ids) / objects),
        "IDs": ids,
        "FP": fp,
        "FN": fn,
        "Rcll": 100.0 * counts["num_detections"] / objects,
        "Prcn": 100.0 * counts["num_detections"] / predictions,
    }


def _evaluate(result_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    count_keys = [
        "idtp",
        "idfp",
        "idfn",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_objects",
        "num_detections",
        "num_predictions",
    ]
    totals = {key: 0 for key in count_keys}
    per_sequence: dict[str, dict[str, Any]] = {}
    jobs: list[tuple[str, str, str]] = []

    for sequence in SEQUENCES:
        gt_path = GT_ROOT / sequence / "gt" / "gt.txt"
        result_path = result_dir / f"{sequence}.txt"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        counts = _evaluate_single_sequence(
            sequence,
            str(gt_path),
            str(result_path),
        )
        per_sequence[sequence] = _counts_to_metrics(counts)
        for key in count_keys:
            totals[key] += int(counts[key])
        jobs.append((sequence, str(gt_path), str(result_path)))

    metrics = _counts_to_metrics(totals)
    with contextlib.redirect_stdout(io.StringIO()):
        hota = _calculate_hota(
            str(ROOT / "datasets" / "MOT17"),
            "train",
            str(result_dir),
            jobs,
        )
    if hota:
        metrics.update({key: 100.0 * value for key, value in hota.items()})
    latency = _read_latency(result_dir)
    metrics.update(
        {
            "FPS": latency["fps"],
            "latency_ms": latency["mean_ms"],
            "frames": latency["frames"],
            "latency_profile_sequence": latency["profile_sequence"],
            "latency_profile_frames": latency["profile_frames"],
            "latency_p95_ms": latency["p95_ms"],
            "latency_p99_ms": latency["p99_ms"],
        }
    )
    return metrics, per_sequence


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_row(experiment: dict[str, str]) -> dict[str, Any]:
    path = ROOT / experiment["checkpoint"]
    row: dict[str, Any] = {
        "experiment": experiment["name"],
        "checkpoint": experiment["checkpoint"],
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() else "",
        "seed": experiment["seed"],
        "epoch": "",
        "best_loss": "",
        "clip_len": "",
        "clip_stride": "",
        "scan_stop_grad": "",
        "add_temporal": "",
        "head_parameters": "",
    }
    if not path.exists():
        return row
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    state = checkpoint.get("student", {})
    row.update(
        {
            "seed": args.get("seed", experiment["seed"]),
            "epoch": checkpoint.get("epoch", ""),
            "best_loss": checkpoint.get("best_loss", ""),
            "clip_len": args.get("clip_len", ""),
            "clip_stride": args.get("clip_stride", ""),
            "scan_stop_grad": args.get("scan_stop_grad", ""),
            "add_temporal": args.get("add_temporal", ""),
            "head_parameters": sum(
                tensor.numel() for tensor in state.values() if hasattr(tensor, "numel")
            ),
        }
    )
    return row


def _tracking_row(
    experiment: dict[str, str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment["name"],
        "family": experiment["family"],
        "seed": experiment["seed"],
        "IDF1": f"{metrics['IDF1']:.3f}",
        "MOTA": f"{metrics['MOTA']:.3f}",
        "HOTA": f"{metrics.get('HOTA', 0.0):.3f}",
        "DetA": f"{metrics.get('DetA', 0.0):.3f}",
        "AssA": f"{metrics.get('AssA', 0.0):.3f}",
        "IDs": metrics["IDs"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "Rcll": f"{metrics['Rcll']:.3f}",
        "Prcn": f"{metrics['Prcn']:.3f}",
        "FPS": f"{metrics['FPS']:.2f}",
        "latency_ms": f"{metrics['latency_ms']:.2f}",
        "latency_p95_ms": f"{metrics['latency_p95_ms']:.3f}",
        "latency_p99_ms": f"{metrics['latency_p99_ms']:.3f}",
        "latency_profile_sequence": metrics["latency_profile_sequence"],
        "latency_profile_frames": metrics["latency_profile_frames"],
        "latency_scope": "post_decode_gpu_frame_to_tracking_output",
        "result_dir": experiment["result_dir"],
        "checkpoint": experiment["checkpoint"],
    }


def _plot_curriculum(overall: dict[str, dict[str, Any]]) -> None:
    valid_pairs = [
        ("20260613", "replica_20260613", "t3t1_20260613"),
        ("20260614", "replica_20260614", "t3t1_20260614"),
    ]
    metrics = ["IDF1", "HOTA", "AssA", "MOTA"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    x = np.arange(len(metrics))
    width = 0.34
    plain_mean = [
        np.mean([overall[plain][metric] for _, plain, _ in valid_pairs])
        for metric in metrics
    ]
    shaped_mean = [
        np.mean([overall[shaped][metric] for _, _, shaped in valid_pairs])
        for metric in metrics
    ]
    axes[0].bar(x - width / 2, plain_mean, width, label="Plain GT2")
    axes[0].bar(x + width / 2, shaped_mean, width, label="T3 to T1")
    axes[0].set_xticks(x, metrics)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Valid paired seeds: mean")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for seed, plain, shaped in valid_pairs:
        delta = [overall[shaped][metric] - overall[plain][metric] for metric in metrics]
        axes[1].plot(metrics, delta, marker="o", label=seed)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("T3 to T1 delta (percentage points)")
    axes[1].set_title("Per-seed paired deltas")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "mamba_t3t1_paired_metrics.png", dpi=180)
    plt.close(fig)


def _plot_per_sequence(
    per_sequence: dict[str, dict[str, dict[str, Any]]],
) -> None:
    valid_pairs = [
        ("20260613", "replica_20260613", "t3t1_20260613"),
        ("20260614", "replica_20260614", "t3t1_20260614"),
    ]
    labels = [
        sequence.removeprefix("MOT17-").removesuffix("-SDP") for sequence in SEQUENCES
    ]
    x = np.arange(len(SEQUENCES))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for seed, plain, shaped in valid_pairs:
        delta = [
            per_sequence[shaped][sequence]["IDF1"]
            - per_sequence[plain][sequence]["IDF1"]
            for sequence in SEQUENCES
        ]
        ax.plot(x, delta, marker="o", label=seed)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("MOT17 SDP sequence")
    ax.set_ylabel("IDF1 delta (percentage points)")
    ax.set_title("T3 to T1 curriculum: valid paired-seed sequence deltas")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "mamba_t3t1_per_sequence_idf1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    overall: dict[str, dict[str, Any]] = {}
    per_sequence: dict[str, dict[str, dict[str, Any]]] = {}
    tracking_rows: list[dict[str, Any]] = []

    for experiment in EXPERIMENTS:
        result_dir = ROOT / experiment["result_dir"]
        metrics, sequence_metrics = _evaluate(result_dir)
        overall[experiment["name"]] = metrics
        per_sequence[experiment["name"]] = sequence_metrics
        row = _tracking_row(experiment, metrics)
        row["protocol_status"] = experiment["status"]
        tracking_rows.append(row)

    boundary_metrics: dict[str, dict[str, Any]] = {}
    boundary_rows: list[dict[str, Any]] = []
    for experiment in CURRICULUM_BOUNDARIES:
        metrics, _ = _evaluate(ROOT / experiment["result_dir"])
        boundary_metrics[experiment["name"]] = metrics
        row = _tracking_row(experiment, metrics)
        row["interpretation"] = experiment["interpretation"]
        boundary_rows.append(row)

    class _PairSpec(TypedDict):
        label: str
        plain: str
        shaped: str
        paired: bool
        reason: str

    pair_specs: list[_PairSpec] = [
        {
            "label": "unpaired_original",
            "plain": "replica_20260612",
            "shaped": "t3t1_seed42",
            "paired": False,
            "reason": "plain seed 20260612; T3-to-T1 seed 42",
        },
        {
            "label": "seed_20260613",
            "plain": "replica_20260613",
            "shaped": "t3t1_20260613",
            "paired": True,
            "reason": "same student-chain seed and same GT1 lineage",
        },
        {
            "label": "seed_20260614",
            "plain": "replica_20260614",
            "shaped": "t3t1_20260614",
            "paired": True,
            "reason": "same student-chain seed and same GT1 lineage",
        },
    ]
    pair_rows: list[dict[str, Any]] = []
    for spec in pair_specs:
        plain = overall[spec["plain"]]
        shaped = overall[spec["shaped"]]
        pair_rows.append(
            {
                **spec,
                "plain_seed": next(
                    item["seed"]
                    for item in EXPERIMENTS
                    if item["name"] == spec["plain"]
                ),
                "shaped_seed": next(
                    item["seed"]
                    for item in EXPERIMENTS
                    if item["name"] == spec["shaped"]
                ),
                "delta_IDF1": f"{shaped['IDF1'] - plain['IDF1']:.3f}",
                "delta_MOTA": f"{shaped['MOTA'] - plain['MOTA']:.3f}",
                "delta_HOTA": f"{shaped.get('HOTA', 0.0) - plain.get('HOTA', 0.0):.3f}",
                "delta_DetA": f"{shaped.get('DetA', 0.0) - plain.get('DetA', 0.0):.3f}",
                "delta_AssA": f"{shaped.get('AssA', 0.0) - plain.get('AssA', 0.0):.3f}",
                "delta_IDs": shaped["IDs"] - plain["IDs"],
                "delta_FP": shaped["FP"] - plain["FP"],
                "delta_FN": shaped["FN"] - plain["FN"],
                "delta_FPS": f"{shaped['FPS'] - plain['FPS']:.2f}",
            }
        )

    sequence_rows: list[dict[str, Any]] = []
    for spec in pair_specs:
        for sequence in SEQUENCES:
            plain = per_sequence[spec["plain"]][sequence]
            shaped = per_sequence[spec["shaped"]][sequence]
            sequence_rows.append(
                {
                    "pair": spec["label"],
                    "paired": spec["paired"],
                    "sequence": sequence,
                    "plain_IDF1": f"{plain['IDF1']:.3f}",
                    "t3t1_IDF1": f"{shaped['IDF1']:.3f}",
                    "delta_IDF1": f"{shaped['IDF1'] - plain['IDF1']:.3f}",
                    "plain_MOTA": f"{plain['MOTA']:.3f}",
                    "t3t1_MOTA": f"{shaped['MOTA']:.3f}",
                    "delta_MOTA": f"{shaped['MOTA'] - plain['MOTA']:.3f}",
                    "delta_IDs": shaped["IDs"] - plain["IDs"],
                    "delta_FP": shaped["FP"] - plain["FP"],
                    "delta_FN": shaped["FN"] - plain["FN"],
                }
            )

    bridge_rows: list[dict[str, Any]] = []
    for experiment in BRIDGE_EXPERIMENTS:
        metrics, _ = _evaluate(ROOT / experiment["result_dir"])
        bridge_rows.append(
            {
                "experiment": experiment["name"],
                "description": experiment["description"],
                "IDF1": f"{metrics['IDF1']:.3f}",
                "MOTA": f"{metrics['MOTA']:.3f}",
                "HOTA": f"{metrics.get('HOTA', 0.0):.3f}",
                "DetA": f"{metrics.get('DetA', 0.0):.3f}",
                "AssA": f"{metrics.get('AssA', 0.0):.3f}",
                "IDs": metrics["IDs"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "FPS": f"{metrics['FPS']:.2f}",
                "result_dir": experiment["result_dir"],
            }
        )

    checkpoint_rows = [
        _checkpoint_row(experiment)
        for experiment in [*EXPERIMENTS, *CURRICULUM_BOUNDARIES]
    ]
    _write_csv(TABLES / "mamba_tracking_overall.csv", tracking_rows)
    _write_csv(TABLES / "mamba_t3t1_pairs.csv", pair_rows)
    _write_csv(TABLES / "mamba_t3t1_per_sequence.csv", sequence_rows)
    _write_csv(TABLES / "mamba_curriculum_boundaries.csv", boundary_rows)
    _write_csv(TABLES / "mamba_checkpoint_provenance.csv", checkpoint_rows)
    _write_csv(TABLES / "bridge_ablation.csv", bridge_rows)

    payload = {
        "generated_from": {
            "gt": "datasets/MOT17/train",
            "sequences": SEQUENCES,
            "metric_code": "src/saccade/perception/eval/metrics.py",
        },
        "overall": overall,
        "curriculum_boundaries": boundary_metrics,
        "per_sequence": per_sequence,
        "pairs": pair_rows,
        "checkpoints": checkpoint_rows,
    }
    (OUT / "paper_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _plot_curriculum(overall)
    _plot_per_sequence(per_sequence)


if __name__ == "__main__":
    main()
