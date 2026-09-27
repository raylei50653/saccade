#!/usr/bin/env python3
"""Decision equivalence of the sparse vs dense Cheb-GR merge distance paths.

For each sequence the tracklet embeddings are extracted **once** from a frozen
substrate. The same embeddings then go through three merge runs at the fixed
operating point:

  dense           distance_impl="dense" (pre-optimization reference)
  sparse          distance_impl="sparse", default row blocking
  sparse_blocked  distance_impl="sparse", row blocking forced
                  (``--forced-block-elems``) so the blocked regime is exercised
                  even where one block would fit

Each sparse run is compared with dense on:

- output MOT lines (sha256) and stats;
- the accepted pairs, in decision-log pair order;
- per-pair verdicts for temporally eligible pairs, and max |Δcost|;
- decision margins of the dense run: the distance from any eligible cost to
  ``max_cost``, and the smallest gap between distinct candidate costs.

Sequences whose dense run cannot fit (``--dense-max-samples``) are run sparse
only and marked ``dense: skipped``.

Usage
-----
  uv run python scripts/eval/experiments/merge_impl_equivalence.py \\
      --substrate results/.../substrate_mot17 --data-root datasets/MOT17 \\
      --seqs MOT17-02-SDP,... --out <evidence>/equivalence_mot17.json
"""
# status: experiment

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

DEFAULT_BLOCK_ELEMS = 1 << 28  # tracklet_distance_pairs default

MERGE = {
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


def _sha(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def _run(
    lines: list[str], emb: dict[int, Any], impl: str, block_elems: int | None
) -> dict[str, Any]:
    import saccade.perception.eval.cheb_gr_merge as m

    original = m.tracklet_distance_pairs
    if block_elems is not None:
        m.tracklet_distance_pairs = functools.partial(  # type: ignore[assignment]
            original, max_block_elems=block_elems
        )
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        log: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        out, stats = m.cheb_gr_merge_output_tracklets(
            lines,
            emb,
            enabled=True,
            max_cost=MERGE["max_cost"],
            max_gap=MERGE["max_gap"],
            min_overlap_frames=MERGE["min_overlap_frames"],
            pool_frac=MERGE["pool_frac"],
            cheb_lambda=MERGE["cheb_lambda"],
            k2=MERGE["k2"],
            max_fwd=MERGE["max_fwd"],
            fuse_lambda=MERGE["fuse_lambda"],
            decision_log=log,
            distance_impl=impl,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        peak = peak_allocated - base_mem
    finally:
        m.tracklet_distance_pairs = original  # type: ignore[assignment]
    pairs = [r for r in log if r["kind"] == "pair"]
    return {
        "output_lines": out,
        "out_sha256": _sha(out),
        "stats": {k: int(v) for k, v in stats.items()},
        "accepted": [
            (int(r["a_id"]), int(r["b_id"]))
            for r in pairs
            if r["verdict"] == "accepted"
        ],
        "pairs": {
            (int(r["a_id"]), int(r["b_id"])): (r["cost"], r["verdict"]) for r in pairs
        },
        "wall_s": wall,
        "peak_gpu_bytes": int(peak),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
        "baseline_allocated_bytes": int(base_mem),
    }


def _compare(dense: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    eligible = [k for k, (c, _) in other["pairs"].items() if c is not None]
    verdict_mismatch = [
        k for k in eligible if other["pairs"][k][1] != dense["pairs"][k][1]
    ]
    diffs = [abs(other["pairs"][k][0] - dense["pairs"][k][0]) for k in eligible]
    pregated = [k for k, (c, _) in other["pairs"].items() if c is None]
    relabeled = sum(1 for k in pregated if dense["pairs"][k][1] == "reject_cost")
    return {
        "out_identical": other["out_sha256"] == dense["out_sha256"],
        "stats_identical": other["stats"] == dense["stats"],
        "accepted_identical_in_order": other["accepted"] == dense["accepted"],
        "eligible_pairs": len(eligible),
        "eligible_verdict_mismatches": len(verdict_mismatch),
        "eligible_cost_max_abs_diff": max(diffs) if diffs else 0.0,
        "eligible_cost_exact_equal": sum(1 for d in diffs if d == 0.0),
        "pregated_pairs": len(pregated),
        "pregated_dense_reject_cost_relabeled": relabeled,
    }


def _margins(dense: dict[str, Any], eligible: list[tuple[int, int]]) -> dict[str, Any]:
    max_cost = MERGE["max_cost"]
    costs = [float(dense["pairs"][k][0]) for k in eligible]
    thr = min((abs(c - max_cost) for c in costs), default=None)
    cand = sorted(c for c in costs if c <= max_cost)
    gaps = [b - a for a, b in zip(cand, cand[1:]) if b > a]
    ties = sum(1 for a, b in zip(cand, cand[1:]) if b == a)
    return {
        "min_abs_cost_minus_max_cost": thr,
        "min_positive_candidate_cost_gap": min(gaps) if gaps else None,
        "candidate_exact_cost_ties": ties,
        "candidates": len(cand),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--substrate", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--split", default="train")
    p.add_argument("--seqs", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cheb-gr-model", default="mobilenetv4_reid")
    p.add_argument("--cheb-gr-engine", default="")
    p.add_argument("--dense-max-samples", type=int, default=27000)
    p.add_argument("--forced-block-elems", type=int, default=1 << 20)
    args = p.parse_args(argv)
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("F2 qualification requires highest FP32 matmul precision")
    numeric_contract = {
        "dtype": "float32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "autocast": torch.is_autocast_enabled("cuda"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "default_block_elems": DEFAULT_BLOCK_ELEMS,
        "reduction": "stable-key ordered sequential group sum",
        "topk": "unchanged torch.topk; ties qualified against dense on this stack",
    }

    from saccade.perception.eval.cheb_gr_merge import extract_tracklet_embeddings
    from saccade.perception.feature_extractor import TRTFeatureExtractor

    extractor = TRTFeatureExtractor(
        engine_path=args.cheb_gr_engine, model_type=args.cheb_gr_model, max_batch=64
    )
    rows: dict[str, Any] = {}
    for seq in [s.strip() for s in args.seqs.split(",") if s.strip()]:
        lines = [
            ln
            for ln in (args.substrate / f"{seq}.txt").read_text().splitlines()
            if ln.strip()
        ]
        emb = extract_tracklet_embeddings(
            lines,
            str(args.data_root / args.split / seq / "img1"),
            extractor,
            n_samples=MERGE["n_samples"],
            crop_hw=getattr(extractor, "input_hw", (224, 224)),
            appearance_occlusion_gate=True,
            appearance_occlusion_cov=0.4,
        )
        if any(v.dtype != torch.float32 for v in emb.values()):
            raise RuntimeError("Expected frozen FP32 embeddings")
        embedding_hash = hashlib.sha256()
        for tid, features in sorted(emb.items()):
            embedding_hash.update(str(tid).encode() + b"\0")
            embedding_hash.update(features.cpu().contiguous().numpy().tobytes())
        n_samples = int(sum(int(v.shape[0]) for v in emb.values()))
        row: dict[str, Any] = {
            "tracklets_with_embedding": sum(1 for v in emb.values() if v.shape[0]),
            "samples": n_samples,
            "embedding_sha256": embedding_hash.hexdigest(),
            "substrate_sha256": hashlib.sha256(
                (args.substrate / f"{seq}.txt").read_bytes()
            ).hexdigest(),
            "substrate_lines_sha256": _sha(lines),
            "row_chunk": max(1, DEFAULT_BLOCK_ELEMS // max(1, 2 * n_samples)),
            "graph_nodes": 2 * n_samples,
            # Rows per block = elems // nodes; blocked iff that is < nodes.
            "sparse_blocked_regime": (DEFAULT_BLOCK_ELEMS // max(1, 2 * n_samples))
            < 2 * n_samples,
            "sparse_blocked_forced_regime": (
                args.forced_block_elems // max(1, 2 * n_samples)
            )
            < 2 * n_samples,
        }
        runs: dict[str, dict[str, Any]] = {}
        if n_samples <= args.dense_max_samples:
            runs["dense"] = _run(lines, emb, "dense", None)
        runs["sparse"] = _run(lines, emb, "sparse", None)
        repeat = _run(lines, emb, "sparse", None)
        row["sparse_repeat_costs_exact"] = repeat["pairs"] == runs["sparse"]["pairs"]
        row["sparse_repeat_output_exact"] = (
            repeat["out_sha256"] == runs["sparse"]["out_sha256"]
        )
        if "dense" in runs:
            runs["sparse_blocked"] = _run(lines, emb, "sparse", args.forced_block_elems)
        from saccade.perception.eval.post_merge import interpolate_tracklets

        final, _ = interpolate_tracklets(
            runs["sparse"]["output_lines"], max_gap=35, min_track_len=5, min_h=0.0
        )
        row["final_mot_sha256"] = _sha(final)
        for name, r in runs.items():
            row[name] = {
                "out_sha256": r["out_sha256"],
                "stats": r["stats"],
                "accepted": len(r["accepted"]),
                "wall_s": r["wall_s"],
                "peak_gpu_bytes": r["peak_gpu_bytes"],
                "peak_allocated_bytes": r["peak_allocated_bytes"],
                "peak_reserved_bytes": r["peak_reserved_bytes"],
                "baseline_allocated_bytes": r["baseline_allocated_bytes"],
            }
        if "dense" in runs:
            for name in ("sparse", "sparse_blocked"):
                row[name]["vs_dense"] = _compare(runs["dense"], runs[name])
            eligible = [
                k for k, (c, _) in runs["sparse"]["pairs"].items() if c is not None
            ]
            row["dense_margins"] = _margins(runs["dense"], eligible)
        else:
            row["dense"] = "skipped (samples > --dense-max-samples)"
        rows[seq] = row
        summary = (
            row["sparse"].get("vs_dense", {}).get("out_identical", "n/a"),
            row.get("sparse_blocked", {})
            .get("vs_dense", {})
            .get("out_identical", "n/a"),
        )
        print(
            f"{seq}: S={n_samples} out_identical(sparse,blocked)={summary} "
            f"wall dense={row['dense']['wall_s'] if isinstance(row['dense'], dict) else '-'} "
            f"sparse={row['sparse']['wall_s']:.1f}s peak sparse="
            f"{row['sparse']['peak_gpu_bytes'] / 2**30:.2f}GiB",
            flush=True,
        )
        del emb
        torch.cuda.empty_cache()

    payload = {
        "schema": "merge_impl_equivalence/v3",
        "numeric_contract": numeric_contract,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        ),
        "substrate": str(args.substrate),
        "merge": MERGE,
        "forced_block_elems": args.forced_block_elems,
        "dense_max_samples": args.dense_max_samples,
        "device": torch.cuda.get_device_name(0),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/saccade/perception/eval/cheb_gr_merge.py",
                "src/saccade/perception/reid/cheb_gr.py",
                "scripts/eval/experiments/merge_impl_equivalence.py",
            )
        },
        "sequences": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    passed = all(
        r["sparse_repeat_costs_exact"]
        and r["sparse_repeat_output_exact"]
        and all(
            all(
                r[mode]["vs_dense"][key]
                for key in (
                    "out_identical",
                    "stats_identical",
                    "accepted_identical_in_order",
                )
            )
            and r[mode]["vs_dense"]["eligible_verdict_mismatches"] == 0
            for mode in ("sparse", "sparse_blocked")
            if "vs_dense" in r.get(mode, {})
        )
        for r in rows.values()
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
