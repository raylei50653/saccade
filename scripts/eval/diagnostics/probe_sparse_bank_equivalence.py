#!/usr/bin/env python3
"""Probe: can a sparse per-ID key-embedding bank replace the dense Cheb-GR bank?

Two questions, answered on a frozen no-handover substrate (zero tracker
nondeterminism):

1. **Signal importance** — which per-sample signals (det score, box height,
   neighbor IoU, temporal position) predict that a crop embedding is
   *reliable* (high cosine to its track's prototype)? These are the signals an
   online per-ID key-embedding maintainer would select on.
2. **Sparse equivalence** — starting from the dense reference bank (the
   current GO operating point's temporally-spread 50 samples), how far can the
   bank be compressed (subset selection or prototype averaging) before the
   offline Cheb-GR handover decisions and IDF1 diverge?

Bank strategies (head evidence is never compressed):
  ref        dense reference bank (spread-50)
  spread-N   temporal re-sampling to N rows (pure sparsity baseline)
  recent-N   last N clean samples before death (visclean-gated FIFO)
  stride-K   every K-th clean sample (sparse extraction schedule; boxes stay
             real, intermediate frames would copy the last embedding — the
             deduplicated bank content is exactly these rows)
  stridefifo-K-N  every K-th clean sample, keep the last N (sparse extraction
             + bounded FIFO retention: the cheapest online form)
  schedfifo-K-N  production CleanFifoBank scheduler (dense birth window into
             the FIFO + stride-K after), no event overrides — isolates
             scheduler semantics from stridefifo's post-hoc slicing
  evfifo-K-N[-ambP][-wW][-po]  schedfifo + event overrides: pre-occlusion /
             death snapshot always on (``-wW``: last W clean crops instead
             of 1); ``-ambP`` adds the ambiguity trigger at neighbor IoU >=
             P/100; ``-po`` adds the post-occlusion snapshot.

``--requery-band B`` additionally re-scores borderline decisions (|cost-gate|
or margin within B) with a dense fallback bank (``--requery-source``:
whole-life ref or online-feasible recent tail; ``--requery-top K`` densifies
only the top-K candidates) — the decision-side stabilizer for sparse banks.
  dupfill-N  spread-N unique rows, each duplicated to fill n_samples rows
             (models naively feeding the copied per-frame embeddings into the
             graph without dedup)
  mean1      single L2-normalized mean prototype
  segmean-K  K temporal-segment mean prototypes (diffusion-style maintenance)
  topscore-N N rows with highest det score
  lowpol-N   N rows with lowest same-frame neighbor IoU

Usage
-----
  .venv/bin/python scripts/eval/diagnostics/probe_sparse_bank_equivalence.py \
      --substrate results/diag_m_no_reid_current_20260704 \
      --out-json results/probe_sparse_bank_m.json
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from saccade.perception.eval.cheb_gr_merge import (  # noqa: E402
    _extract_native_crops_trt,
    temporal_sample_indices,
)
from saccade.perception.eval.cheb_gr_online import causal_handover_lines  # noqa: E402
from saccade.perception.eval.clean_fifo_bank import plan_clean_fifo_crops  # noqa: E402
from saccade.perception.eval.helpers import front_occlusion_mask_xyxy  # noqa: E402
from saccade.perception.eval.post_merge import (  # noqa: E402
    _build_output_tracklets,
    _parse_mot_lines,
)
from saccade.perception.eval.utils import box_iou_tuple  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]
SIGNALS = ("score", "box_h", "neighbor_iou", "time_frac")

_CROP = tuple[int, tuple[float, float, float, float]]


def parse_plan_strategy(strat: str) -> dict[str, Any] | None:
    """schedfifo-K-N / evfifo-K-N[-ambP][-wW][-po] -> planner kwargs.

    ``ambP``: ambiguity trigger at neighbor IoU >= P/100. ``wW``: pre-occlusion
    window of W clean crops (default 1). ``po``: post-occlusion snapshot.
    """
    parts = strat.split("-")
    if parts[0] not in ("schedfifo", "evfifo"):
        return None
    spec = {
        "stride": int(parts[1]),
        "fifo_n": int(parts[2]),
        "preocc_snapshot": parts[0] == "evfifo",
        "ambiguity_iou": 0.0,
        "preocc_window": 1,
        "postocc_snapshot": False,
    }
    for token in parts[3:]:
        if parts[0] != "evfifo":
            raise SystemExit(f"unknown strategy {strat!r}")
        if token.startswith("amb"):
            spec["ambiguity_iou"] = int(token[3:]) / 100.0
        elif token == "po":
            spec["postocc_snapshot"] = True
        elif token.startswith("w"):
            spec["preocc_window"] = int(token[1:])
        else:
            raise SystemExit(f"unknown strategy {strat!r}")
    return spec


def extract_bank_with_meta(
    lines: list[str],
    seq_dir: str,
    extractor: Any,
    *,
    decide_n: int,
    n_samples: int,
    cov: float,
    batch: int = 256,
    tail_n: int = 20,
    strides: tuple[int, ...] = (),
    plan_bank_crops: dict[str, dict[int, list[_CROP]]] | None = None,
    job_strategies: tuple[str, ...] = (),
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, list[dict]],
    dict[int, torch.Tensor],
    dict[int, dict[int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]],
    dict[str, int],
]:
    """extract_handover_embeddings + per-bank-row metadata (same contract).

    Additionally returns ``tails`` (last ``tail_n`` clean rows per track, for
    recent-N FIFO banks), ``stride_banks[K]`` (every K-th clean row, for
    sparse-extraction-schedule banks), ``plan_banks[strategy]`` (embeddings
    for externally planned bank crops, e.g. CleanFifoPlan schedules) and
    ``jobs[strategy]`` (unique head+bank extraction count = online ReID job
    count; ``__all__`` = every detection, ``__clean__`` = every clean crop).
    All share one deduplicated crop pool.
    """
    crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))
    records = _parse_mot_lines(lines)
    tracklets = _build_output_tracklets(records, velocity_samples=5)
    start_by_id = {t.track_id: t.start for t in tracklets}
    span_by_id = {t.track_id: (t.start, t.end) for t in tracklets}

    by_frame_idx: dict[int, list[int]] = defaultdict(list)
    for ri, r in enumerate(records):
        by_frame_idx[r.frame].append(ri)

    dirty: set[int] = set()
    neighbor_iou = np.zeros(len(records), dtype=np.float32)
    for idxs in by_frame_idx.values():
        boxes_t = torch.tensor(
            [
                (
                    records[i].x,
                    records[i].y,
                    records[i].x + records[i].w,
                    records[i].y + records[i].h,
                )
                for i in idxs
            ],
            dtype=torch.float32,
        )
        mask = front_occlusion_mask_xyxy(boxes_t, cov)
        dirty.update(i for i, d in zip(idxs, mask.tolist()) if d)
        boxes = [tuple(b) for b in boxes_t.tolist()]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                iou = float(box_iou_tuple(boxes[a], boxes[b]))
                if iou > neighbor_iou[idxs[a]]:
                    neighbor_iou[idxs[a]] = iou
                if iou > neighbor_iou[idxs[b]]:
                    neighbor_iou[idxs[b]] = iou

    clean_by_id: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for ri, r in enumerate(records):
        if ri not in dirty:
            clean_by_id[r.track_id].append((ri, r))

    pool_idx: dict[tuple, int] = {}
    pool: list[tuple[int, int, tuple[float, float, float, float]]] = []

    def add(rs: list[Any]) -> list[int]:
        out = []
        for r in rs:
            # Corner-format key so externally planned crops (already corner
            # boxes computed from the same records) dedup against these rows.
            key = (r.track_id, r.frame, r.x, r.y, r.x + r.w, r.y + r.h)
            if key not in pool_idx:
                pool_idx[key] = len(pool)
                pool.append((r.track_id, r.frame, (r.x, r.y, r.x + r.w, r.y + r.h)))
            out.append(pool_idx[key])
        return out

    head_rows: dict[int, list[int]] = {}
    bank_rows: dict[int, list[int]] = {}
    bank_meta: dict[int, list[dict]] = {}
    tail_rows: dict[int, list[int]] = {}
    stride_rows: dict[int, dict[int, list[int]]] = {k: {} for k in strides}
    for tid, pairs in clean_by_id.items():
        pairs.sort(key=lambda p: p[1].frame)
        items = [r for _, r in pairs]
        birth = start_by_id[tid]
        head_rows[tid] = add([r for r in items if r.frame < birth + decide_n])
        scores = np.asarray([r.score for r in items], dtype=np.float32)
        sel = temporal_sample_indices(len(items), n_samples, scores=scores)
        bank_rows[tid] = add([items[j] for j in sel])
        tail_rows[tid] = add(items[-tail_n:])
        for k in strides:
            stride_rows[k][tid] = add(items[::k])
        start, end = span_by_id[tid]
        life = max(end - start, 1)
        bank_meta[tid] = [
            {
                "frame": items[j].frame,
                "score": float(items[j].score),
                "box_h": float(items[j].h),
                "neighbor_iou": float(neighbor_iou[pairs[j][0]]),
                "time_frac": (items[j].frame - start) / life,
            }
            for j in sel
        ]

    plan_rows: dict[str, dict[int, list[int]]] = {}
    for strat, crops_by_tid in (plan_bank_crops or {}).items():
        rows_by_tid: dict[int, list[int]] = {}
        for tid, crops in crops_by_tid.items():
            row_idxs: list[int] = []
            for frame, (x1, y1, x2, y2) in crops:
                key = (tid, frame, x1, y1, x2, y2)
                if key not in pool_idx:
                    pool_idx[key] = len(pool)
                    pool.append((tid, frame, (x1, y1, x2, y2)))
                row_idxs.append(pool_idx[key])
            if row_idxs:
                rows_by_tid[tid] = row_idxs
        plan_rows[strat] = rows_by_tid

    n_clean = sum(len(v) for v in clean_by_id.values())
    jobs: dict[str, int] = {"__all__": len(records), "__clean__": n_clean}
    head_all: set[int] = set()
    for rows in head_rows.values():
        head_all.update(rows)
    for strat in job_strategies:
        if strat.startswith(("stride-", "stridefifo-")):
            k = int(strat.split("-")[1])
            sel = set(head_all)
            for rows in stride_rows[k].values():
                sel.update(rows)
            jobs[strat] = len(sel)
        elif strat not in plan_rows:
            # Dense post-hoc strategies (ref/spread/recent/mean/...) select
            # from the full clean pool -> online cost = every clean crop.
            jobs[strat] = n_clean

    if not pool:
        return {}, {}, {}, {}, {}, {}, jobs

    by_frame_s: dict[int, list[int]] = defaultdict(list)
    for si, (_, fr, _) in enumerate(pool):
        by_frame_s[fr].append(si)
    feats = _extract_native_crops_trt(
        pool,
        by_frame_s,
        seq_dir,
        extractor,
        crop_hw=crop_hw,
        im_ext=".jpg",
        batch=batch,
    )
    if feats is None:
        raise SystemExit("native crop extraction unavailable (TRT path required)")

    def gather(rows: dict[int, list[int]]) -> dict[int, torch.Tensor]:
        return {
            tid: feats[torch.tensor(idxs, device=feats.device)]
            for tid, idxs in rows.items()
            if idxs
        }

    return (
        gather(head_rows),
        gather(bank_rows),
        bank_meta,
        gather(tail_rows),
        {k: gather(rows) for k, rows in stride_rows.items()},
        {strat: gather(rows) for strat, rows in plan_rows.items()},
        jobs,
    )


def compress_bank(bank: torch.Tensor, meta: list[dict], strategy: str) -> torch.Tensor:
    name, _, arg = strategy.partition("-")
    n = bank.shape[0]
    if name == "ref":
        return bank
    if name == "spread":
        k = min(int(arg), n)
        scores = np.asarray([m["score"] for m in meta], dtype=np.float32)
        sel = temporal_sample_indices(n, k, scores=scores)
        return bank[torch.tensor(sel, device=bank.device)]
    if name == "mean1":
        return F.normalize(bank.mean(dim=0, keepdim=True), dim=1)
    if name == "segmean":
        k = min(int(arg), n)
        protos = [
            F.normalize(chunk.mean(dim=0, keepdim=True), dim=1)
            for chunk in torch.tensor_split(bank, k, dim=0)
            if chunk.shape[0] > 0
        ]
        return torch.cat(protos, dim=0)
    if name in ("topscore", "lowpol"):
        k = min(int(arg), n)
        key = "score" if name == "topscore" else "neighbor_iou"
        vals = np.asarray([m[key] for m in meta], dtype=np.float32)
        order = np.argsort(-vals if name == "topscore" else vals, kind="stable")[:k]
        sel = np.sort(order)
        return bank[torch.tensor(sel.copy(), device=bank.device)]
    raise SystemExit(f"unknown strategy {strategy!r}")


def accept_set(log_rows: list[dict]) -> set[tuple[int, int, int]]:
    return {
        (int(r["seq_idx"]), int(r["newborn_id"]), int(r["candidate_id"]))
        for r in log_rows
        if r["accepted"]
    }


def score_dir(output_dir: Path, args) -> Any:
    accs, names = [], []
    for seq in args.seqs:
        gt_path = PROJECT_ROOT / args.data_root / args.split / seq / "gt" / "gt.txt"
        gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
        hyp_df = mm.io.loadtxt(
            str(output_dir / f"{seq}.txt"), fmt="mot15-2D", min_confidence=-1.0
        )
        accs.append(mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5))
        names.append(seq)
    mh = mm.metrics.create()
    return mh.compute_many(
        accs, names=names, metrics=["idf1", "num_switches"], generate_overall=True
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--substrate", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--engine", default="models/embedding/mobilenetv4_reid_visclean_224.engine"
    )
    ap.add_argument("--model-type", default="mobilenetv4_reid")
    ap.add_argument("--decide-n", type=int, default=5)
    ap.add_argument("--max-cost", type=float, default=0.45)
    ap.add_argument("--min-head", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max-gap", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--cov", type=float, default=0.4)
    ap.add_argument("--tail-n", type=int, default=20)
    ap.add_argument(
        "--requery-band",
        type=float,
        default=0.0,
        help="Borderline half-width around max_cost/margin that triggers a "
        "dense-bank re-query for non-ref strategies (0 = off).",
    )
    ap.add_argument(
        "--requery-top",
        type=int,
        default=0,
        help="Densify only the top-K sparse-ranked candidates on re-query "
        "(0 = all in-graph banks).",
    )
    ap.add_argument(
        "--requery-source",
        choices=("ref", "recent"),
        default="ref",
        help="Fallback bank for re-queries: 'ref' = whole-life spread-50 "
        "(offline upper bound), 'recent' = dense tail-20 (what an online "
        "raw-crop ring buffer can actually reconstruct).",
    )
    ap.add_argument(
        "--strategies",
        nargs="*",
        default=[
            "ref",
            "spread-10",
            "spread-5",
            "spread-3",
            "mean1",
            "segmean-3",
            "segmean-5",
            "topscore-5",
            "lowpol-5",
        ],
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--dump-decisions",
        type=Path,
        default=None,
        help="Dump per-strategy handover decision-log rows to this JSON path.",
    )
    args = ap.parse_args()

    extractor = TRTFeatureExtractor(
        engine_path=str(PROJECT_ROOT / args.engine),
        model_type=args.model_type,
        max_batch=64,
    )

    strides_needed = tuple(
        sorted(
            {
                int(s.split("-")[1])
                for s in args.strategies
                if s.startswith(("stride-", "stridefifo-"))
            }
        )
    )
    plan_specs = {
        s: spec for s in args.strategies if (spec := parse_plan_strategy(s)) is not None
    }
    lines_by_seq: dict[str, list[str]] = {}
    heads: dict[str, dict[int, torch.Tensor]] = {}
    banks: dict[str, dict[int, torch.Tensor]] = {}
    metas: dict[str, dict[int, list[dict]]] = {}
    tails: dict[str, dict[int, torch.Tensor]] = {}
    stride_banks: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
    plan_banks: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    jobs_by_seq: dict[str, dict[str, int]] = {}
    for seq in args.seqs:
        lines = (args.substrate / f"{seq}.txt").read_text().splitlines()
        lines_by_seq[seq] = lines
        records = _parse_mot_lines(lines)
        plan_crops: dict[str, dict[int, list[_CROP]]] = {}
        plan_jobs: dict[str, int] = {}
        for strat, spec in plan_specs.items():
            plan = plan_clean_fifo_crops(
                records,
                appearance_occlusion_cov=args.cov,
                fifo_n=spec["fifo_n"],
                stride=spec["stride"],
                decide_n=args.decide_n,
                preocc_snapshot=spec["preocc_snapshot"],
                ambiguity_iou=spec["ambiguity_iou"],
                preocc_window=spec["preocc_window"],
                postocc_snapshot=spec["postocc_snapshot"],
            )
            plan_crops[strat] = plan.bank_crops
            job_keys = {
                (tid, frame, *box)
                for crops_map in (plan.head_crops, plan.job_crops)
                for tid, crops in crops_map.items()
                for frame, box in crops
            }
            plan_jobs[strat] = len(job_keys)
        (
            heads[seq],
            banks[seq],
            metas[seq],
            tails[seq],
            stride_banks[seq],
            plan_banks[seq],
            jobs_by_seq[seq],
        ) = extract_bank_with_meta(
            lines,
            str(PROJECT_ROOT / args.data_root / args.split / seq / "img1"),
            extractor,
            decide_n=args.decide_n,
            n_samples=args.n_samples,
            cov=args.cov,
            tail_n=args.tail_n,
            strides=strides_needed,
            plan_bank_crops=plan_crops,
            job_strategies=tuple(args.strategies),
        )
        jobs_by_seq[seq].update(plan_jobs)
        print(f"extracted {seq}: {len(banks[seq])} banks")

    # ── 1. Signal importance: per-sample cosine-to-prototype vs signals ──────
    sig_vals: dict[str, list[float]] = {s: [] for s in SIGNALS}
    reliab: list[float] = []
    for seq in args.seqs:
        for tid, bank in banks[seq].items():
            if bank.shape[0] < 8:
                continue
            proto = F.normalize(bank.mean(dim=0, keepdim=True), dim=1)
            cos = (bank @ proto.T).squeeze(1).cpu().numpy()
            for m, c in zip(metas[seq][tid], cos):
                reliab.append(float(c))
                for s in SIGNALS:
                    sig_vals[s].append(float(m[s]))
    reliab_arr = np.asarray(reliab)
    print(
        f"\n=== signal importance (n={len(reliab_arr)} bank samples, "
        "spearman vs cosine-to-prototype) ==="
    )
    signal_corr = {}
    for s in SIGNALS:
        rho = spearman(np.asarray(sig_vals[s]), reliab_arr)
        signal_corr[s] = rho
        print(f"  {s:>14}: {rho:+.3f}")

    # ── 2. Sparse-bank equivalence ────────────────────────────────────────────
    results: dict[str, dict[str, Any]] = {}
    logs: dict[str, list[dict]] = {}
    for strat in args.strategies:
        out_dir = Path(f"{args.substrate}_sparse_{strat}")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_rows: list[dict] = []
        handovers = 0
        requeries = 0
        for si, seq in enumerate(args.seqs):
            if strat in plan_specs:
                sparse_bank = plan_banks[seq][strat]
            elif strat.startswith("recent-"):
                n = int(strat.split("-", 1)[1])
                sparse_bank = {tid: t[-n:] for tid, t in tails[seq].items()}
            elif strat.startswith("stridefifo-"):
                _, k_s, n_s = strat.split("-")
                sparse_bank = {
                    tid: t[-int(n_s) :]
                    for tid, t in stride_banks[seq][int(k_s)].items()
                }
            elif strat.startswith("stride-"):
                k = int(strat.split("-", 1)[1])
                sparse_bank = stride_banks[seq][k]
            elif strat.startswith("dupfill-"):
                n = int(strat.split("-", 1)[1])
                sparse_bank = {}
                for tid, bank in banks[seq].items():
                    base = compress_bank(bank, metas[seq][tid], f"spread-{n}")
                    reps = max(1, args.n_samples // base.shape[0])
                    sparse_bank[tid] = base.repeat_interleave(reps, dim=0)
            else:
                sparse_bank = {
                    tid: compress_bank(bank, metas[seq][tid], strat)
                    for tid, bank in banks[seq].items()
                }
            rows: list[dict] = []
            out_lines, stats = causal_handover_lines(
                lines_by_seq[seq],
                heads[seq],
                sparse_bank,
                enabled=True,
                max_cost=args.max_cost,
                max_gap=args.max_gap,
                decide_n=args.decide_n,
                min_head_samples=args.min_head,
                margin=args.margin,
                requery_bank_embs=(
                    (
                        banks[seq]
                        if args.requery_source == "ref"
                        else {tid: t[-20:] for tid, t in tails[seq].items()}
                    )
                    if args.requery_band > 0.0 and strat != "ref"
                    else None
                ),
                requery_band=args.requery_band,
                requery_top=args.requery_top,
                decision_log=rows,
            )
            (out_dir / f"{seq}.txt").write_text("\n".join(out_lines) + "\n")
            log_rows.extend({"seq_idx": si, **r} for r in rows)
            handovers += stats["handovers"]
            requeries += stats["requeries"]
        logs[strat] = log_rows
        results[strat] = {
            "handovers": handovers,
            "requeries": requeries,
            "out_dir": str(out_dir),
        }
        print(f"applied {strat}: {handovers} handovers ({requeries} requeries)")

    if args.dump_decisions:
        args.dump_decisions.write_text(
            json.dumps({"seqs": args.seqs, "logs": logs}, indent=1) + "\n"
        )
        print(f"wrote decision logs to {args.dump_decisions}")

    ref_accepts = accept_set(logs["ref"])
    ref_costs = {
        (int(r["seq_idx"]), int(r["newborn_id"])): float(r["best_cost"])
        for r in logs["ref"]
    }
    total_all = sum(jobs_by_seq[seq]["__all__"] for seq in args.seqs)
    total_jobs = {
        strat: sum(jobs_by_seq[seq][strat] for seq in args.seqs)
        for strat in args.strategies
    }
    print(
        f"\n=== equivalence vs dense reference (all-detection jobs = {total_all}) ==="
    )
    print(
        f"{'strategy':>18} {'handover':>8} {'kept':>5} {'lost':>5} {'new':>5} "
        f"{'jaccard':>8} {'|dcost|med':>10} {'IDF1':>7} {'dIDF1':>7} {'IDs':>5} "
        f"{'jobs':>7} {'jobs%':>6}"
    )
    ref_metrics = score_dir(Path(results["ref"]["out_dir"]), args)
    ref_idf1 = float(ref_metrics.loc["OVERALL", "idf1"]) * 100
    for strat in args.strategies:
        acc = accept_set(logs[strat])
        kept = len(acc & ref_accepts)
        lost = len(ref_accepts - acc)
        new = len(acc - ref_accepts)
        union = len(acc | ref_accepts)
        jac = kept / union if union else 1.0
        dcosts = [
            abs(
                float(r["best_cost"])
                - ref_costs[(int(r["seq_idx"]), int(r["newborn_id"]))]
            )
            for r in logs[strat]
            if (int(r["seq_idx"]), int(r["newborn_id"])) in ref_costs
        ]
        dmed = float(np.median(dcosts)) if dcosts else float("nan")
        metrics = score_dir(Path(results[strat]["out_dir"]), args)
        idf1 = float(metrics.loc["OVERALL", "idf1"]) * 100
        ids = int(metrics.loc["OVERALL", "num_switches"])
        jobs_frac = total_jobs[strat] / total_all if total_all else float("nan")
        results[strat].update(
            {
                "kept": kept,
                "lost": lost,
                "new": new,
                "jaccard": jac,
                "median_abs_dcost": dmed,
                "idf1": idf1,
                "d_idf1": idf1 - ref_idf1,
                "num_switches": ids,
                "jobs": total_jobs[strat],
                "jobs_frac_of_all_detections": jobs_frac,
            }
        )
        print(
            f"{strat:>18} {results[strat]['handovers']:>8} {kept:>5} {lost:>5} "
            f"{new:>5} {jac:>8.3f} {dmed:>10.4f} {idf1:>7.2f} "
            f"{idf1 - ref_idf1:>+7.2f} {ids:>5} {total_jobs[strat]:>7} "
            f"{jobs_frac * 100:>5.1f}%"
        )

    if args.out_json:
        payload = {
            "schema": "sparse_bank_equivalence_probe/v1",
            "substrate": str(args.substrate),
            "operating_point": {
                "decide_n": args.decide_n,
                "max_cost": args.max_cost,
                "min_head": args.min_head,
                "margin": args.margin,
                "max_gap": args.max_gap,
                "n_samples": args.n_samples,
            },
            "signal_spearman_vs_prototype_cosine": signal_corr,
            "total_detection_jobs": total_all,
            "total_clean_jobs": sum(jobs_by_seq[seq]["__clean__"] for seq in args.seqs),
            "strategies": results,
        }
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
