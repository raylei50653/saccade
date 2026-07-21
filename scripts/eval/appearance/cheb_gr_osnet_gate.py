#!/usr/bin/env python3
"""Cheb-GR correctness gate on Market-1501 using OSNet features.

Runs the OSNet TensorRT extractor over the standard Market-1501 query /
bounding_box_test split, then compares mAP/CMC of the raw-cosine baseline vs
Cheb-GR re-ranking. OSNet is a strong, purpose-built ReID backbone, so this is a
meaningful gate: Cheb-GR should *raise* mAP. (The zero-training FPN harness gives
~0.5% mAP — random — and cannot validate re-ranking.)

Usage:
    uv run python scripts/eval/appearance/cheb_gr_osnet_gate.py \
        --cheb-lambda 1.0 --gconv-layers 2
"""
# status: diagnostic

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402
from saccade.perception.reid.cheb_gr import (  # noqa: E402
    cheb_gr_jaccard_distance,
    cheb_gr_kreciprocal,
    cheb_gr_rerank_distance,
)


def _parse_pid(filename: str) -> int:
    return int(Path(filename).stem.split("_")[0])


def _parse_cam(filename: str) -> int:
    parts = Path(filename).stem.split("_")
    for p in parts[1:]:
        if p.startswith("c"):
            cam_str = p[1:]
            s_idx = cam_str.find("s")
            if s_idx > 0:
                cam_str = cam_str[:s_idx]
            return int(cam_str)
    return -1


def _load_image(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """RGB uint8 [3, H, W] resized to (H, W)."""
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))


def extract_features(
    extractor: TRTFeatureExtractor,
    paths: list[Path],
    *,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> np.ndarray:
    hw = extractor.input_hw
    n = len(paths)
    out = np.empty((n, extractor.feature_dim), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, n, batch_size):
            batch_paths = paths[i : i + batch_size]
            arrs = list(pool.map(lambda p: _load_image(p, hw), batch_paths))
            batch = (
                torch.from_numpy(np.stack(arrs))
                .to(device)
                .float()
                .div_(255.0)
                .contiguous()
            )
            feats = extractor.extract(batch)
            out[i : i + len(batch_paths)] = feats.cpu().numpy()
            if (i // batch_size) % 50 == 0:
                print(f"  {min(i + batch_size, n):5d}/{n}", flush=True)
    return out


def standard_kreciprocal(
    q_embs: np.ndarray,
    g_embs: np.ndarray,
    *,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Canonical k-reciprocal re-ranking (Zhong et al., CVPR 2017).

    Returns the re-ranked query x gallery distance [Nq, Ng]. Used as the
    fixed-k baseline to compare against the Chebyshev-adaptive variant.
    """
    feats = np.concatenate([q_embs, g_embs], axis=0).astype(np.float32)
    query_num = q_embs.shape[0]
    all_num = feats.shape[0]
    # Squared-Euclidean distance for L2-normalized features = 2 - 2*cos.
    original_dist = 2.0 - 2.0 * (feats @ feats.T)
    np.clip(original_dist, 0.0, None, out=original_dist)
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist, dtype=np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        fwd = initial_rank[i, : k1 + 1]
        bwd = initial_rank[fwd, : k1 + 1]
        fi = np.where(bwd == i)[0]
        krecip = fwd[fi]
        krecip_exp = krecip
        for cand in krecip:
            c_fwd = initial_rank[cand, : int(round(k1 / 2)) + 1]
            c_bwd = initial_rank[c_fwd, : int(round(k1 / 2)) + 1]
            c_fi = np.where(c_bwd == cand)[0]
            c_krecip = c_fwd[c_fi]
            if len(np.intersect1d(c_krecip, krecip)) > 2.0 / 3.0 * len(c_krecip):
                krecip_exp = np.append(krecip_exp, c_krecip)
        krecip_exp = np.unique(krecip_exp)
        w = np.exp(-original_dist[i, krecip_exp])
        V[i, krecip_exp] = (w / np.sum(w)).astype(np.float16)

    original_dist = original_dist[:query_num]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe
    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]

    jaccard = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float16)
        nz = np.where(V[i, :] != 0)[0]
        imgs = [inv_index[ind] for ind in nz]
        for j in range(len(nz)):
            temp_min[0, imgs[j]] += np.minimum(V[i, nz[j]], V[imgs[j], nz[j]])
        jaccard[i] = 1.0 - temp_min / (2.0 - temp_min)

    final = jaccard * (1.0 - lambda_value) + original_dist * lambda_value
    return final[:query_num, query_num:].astype(np.float32)


def compute_mAP_cmc(
    q_embs: np.ndarray,
    g_embs: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 50,
    score_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    n_q = len(q_embs)
    sim = score_matrix if score_matrix is not None else q_embs @ g_embs.T
    # Ranking is the heaviest step; do the argsort on GPU when available.
    if torch.cuda.is_available():
        indices = (
            torch.from_numpy(np.ascontiguousarray(sim))
            .cuda()
            .argsort(dim=1, descending=True)
            .cpu()
            .numpy()
        )
    else:
        indices = np.argsort(-sim, axis=1)

    aps: list[float] = []
    cmc = np.zeros(max_rank, dtype=np.int32)
    for q_idx in range(n_q):
        pid = q_pids[q_idx]
        cam = q_cams[q_idx]
        ranked = indices[q_idx]
        good = (g_pids[ranked] == pid) & (g_cams[ranked] != cam)
        junk = (
            (g_pids[ranked] == -1)
            | (g_pids[ranked] == 0)
            | ((g_pids[ranked] == pid) & (g_cams[ranked] == cam))
        )
        if good.sum() == 0:
            continue
        valid = ~junk
        good_filtered = good[valid]
        good_cumsum = np.cumsum(good_filtered)
        for k in range(max_rank):
            if k < len(good_cumsum) and good_cumsum[k] > 0:
                cmc[k] += 1
        n_good = good_filtered.sum()
        if n_good == 0:
            continue
        scores = sim[q_idx, ranked[valid]]
        labels = good[valid].astype(np.float32)
        order = np.argsort(-scores)
        labels_sorted = labels[order]
        tp = np.cumsum(labels_sorted)
        fp = np.cumsum(1 - labels_sorted)
        precision = tp / (tp + fp + 1e-12)
        ap = sum(precision[t] for t in np.where(labels_sorted == 1)[0]) / n_good
        aps.append(float(ap))

    mAP = float(np.mean(aps)) * 100 if aps else 0.0
    cmc_rates = {f"rank-{k + 1}": cmc[k] / n_q * 100 for k in [0, 4, 9, 19]}
    return {"mAP": mAP, **cmc_rates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="osnet",
        choices=[
            "osnet",
            "siglip2_reid",
            "transreid",
            "fastreid",
            "dinov2",
            "siglip2",
            "mobilenetv4_reid",
        ],
        help="ReID extractor engine to use (default: osnet).",
    )
    parser.add_argument(
        "--engine",
        default="",
        help="Override engine path; empty uses the default for --model-type.",
    )
    parser.add_argument("--market-root", default="datasets/Market-1501-v15.09.15")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cheb-lambda", type=float, default=1.0)
    parser.add_argument("--gconv-layers", type=int, default=2)
    parser.add_argument("--residual-gamma", type=float, default=0.0)
    parser.add_argument("--k-max", type=int, default=0)
    parser.add_argument("--fuse-lambda", type=float, default=1.0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Extract once, then grid-search Cheb-GR hyperparameters vs baseline.",
    )
    parser.add_argument(
        "--sweep-variant",
        default="both",
        choices=["feat", "jaccard", "both"],
        help="Which re-rank variant(s) to sweep (jaccard is slow).",
    )
    parser.add_argument(
        "--kreciprocal",
        action="store_true",
        help="Run canonical fixed-k k-reciprocal re-ranking as a sanity baseline.",
    )
    parser.add_argument(
        "--cheb-krecip",
        action="store_true",
        help="Run faithful Cheb-GR (Chebyshev-adaptive k-reciprocal) sweep.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    market_root = Path(args.market_root)
    if not market_root.is_absolute():
        market_root = project_root / market_root

    query_paths = sorted((market_root / "query").glob("*.jpg"))
    gallery_paths = sorted((market_root / "bounding_box_test").glob("*.jpg"))
    q_pids = np.array([_parse_pid(p.name) for p in query_paths], dtype=np.int32)
    g_pids = np.array([_parse_pid(p.name) for p in gallery_paths], dtype=np.int32)
    q_cams = np.array([_parse_cam(p.name) for p in query_paths], dtype=np.int32)
    g_cams = np.array([_parse_cam(p.name) for p in gallery_paths], dtype=np.int32)
    print(
        f"Device: {device}  Queries: {len(query_paths)}  Gallery: {len(gallery_paths)}"
    )

    extractor = TRTFeatureExtractor(
        engine_path=str((project_root / args.engine).resolve()) if args.engine else "",
        device=str(device),
        model_type=args.model_type,
    )

    print("\n[Phase 1] Extracting OSNet query embeddings...")
    t0 = time.perf_counter()
    q_embs = extract_features(
        extractor,
        query_paths,
        batch_size=args.batch_size,
        device=device,
        workers=args.workers,
    )
    print(f"  {q_embs.shape} in {time.perf_counter() - t0:.1f}s")

    print("\n[Phase 2] Extracting OSNet gallery embeddings...")
    t0 = time.perf_counter()
    g_embs = extract_features(
        extractor,
        gallery_paths,
        batch_size=args.batch_size,
        device=device,
        workers=args.workers,
    )
    print(f"  {g_embs.shape} in {time.perf_counter() - t0:.1f}s")

    print("\n[Phase 3] Baseline (raw cosine)...")
    base = compute_mAP_cmc(q_embs, g_embs, q_pids, g_pids, q_cams, g_cams)
    print("=" * 56)
    print("Market-1501 — OSNet (cosine baseline)")
    print("=" * 56)
    for k, v in base.items():
        print(f"  {k:>10s}: {v:6.2f}%")

    if args.cheb_krecip:
        qd = torch.from_numpy(q_embs).to(device)
        gd = torch.from_numpy(g_embs).to(device)
        d_orig_blk = (0.5 * (1.0 - qd @ gd.t())).cpu().numpy()
        print("\n[Phase 4] Faithful Cheb-GR (Chebyshev-adaptive k-reciprocal, GPU)...")
        print(f"  baseline mAP = {base['mAP']:.2f}%  (fixed-k ref: ~91.3%)")
        print("  " + "-" * 56)
        print(f"  {'mAP':>7} {'Δ':>8}  c_lam  k2  max_fwd  fuse  ({'s':>3})")
        for c_lam in (2.0, 3.0, 4.0, 5.0):
            for max_fwd in (0,):
                for k2 in (6,):
                    t0 = time.perf_counter()
                    dj = (
                        cheb_gr_kreciprocal(
                            qd,
                            gd,
                            cheb_lambda=c_lam,
                            k2=k2,
                            max_fwd=max_fwd,
                            fuse_lambda=1.0,
                        )
                        .cpu()
                        .numpy()
                    )
                    dt = time.perf_counter() - t0
                    for fuse in (1.0, 0.7, 0.5):
                        scores = -(fuse * dj + (1.0 - fuse) * d_orig_blk)
                        m = compute_mAP_cmc(
                            q_embs,
                            g_embs,
                            q_pids,
                            g_pids,
                            q_cams,
                            g_cams,
                            score_matrix=scores,
                        )
                        print(
                            f"  {m['mAP']:7.2f} {m['mAP'] - base['mAP']:+8.2f}  "
                            f"{c_lam:4.1f}  {k2}  {max_fwd:5d}  {fuse:4.1f}  ({dt:4.0f})"
                        )
        return

    if args.kreciprocal:
        print("\n[Phase 4] Standard fixed-k k-reciprocal (Zhong CVPR2017) sanity...")
        print(f"  baseline mAP = {base['mAP']:.2f}%")
        print("  " + "-" * 40)
        print(f"  {'mAP':>7} {'Δ':>8}  k1  k2  lambda")
        for k1, k2, lam in [(20, 6, 0.3), (20, 6, 0.1), (30, 6, 0.3), (15, 4, 0.3)]:
            t0 = time.perf_counter()
            fd = standard_kreciprocal(q_embs, g_embs, k1=k1, k2=k2, lambda_value=lam)
            m = compute_mAP_cmc(
                q_embs, g_embs, q_pids, g_pids, q_cams, g_cams, score_matrix=-fd
            )
            print(
                f"  {m['mAP']:7.2f} {m['mAP'] - base['mAP']:+8.2f}  {k1:2d}  {k2}  "
                f"{lam:4.1f}   ({time.perf_counter() - t0:.0f}s)"
            )
        return

    q = torch.from_numpy(q_embs).to(device)
    g = torch.from_numpy(g_embs).to(device)

    def _rerank_map(lam: float, layers: int, gamma: float, fuse: float) -> float:
        dist = cheb_gr_rerank_distance(
            q,
            g,
            cheb_lambda=lam,
            gconv_layers=layers,
            residual_gamma=gamma,
            k_max=args.k_max,
            fuse_lambda=fuse,
        )
        scores = (-dist).cpu().numpy()
        m = compute_mAP_cmc(
            q_embs, g_embs, q_pids, g_pids, q_cams, g_cams, score_matrix=scores
        )
        return m["mAP"]

    # d_orig block reused for Jaccard fuse blending (cosine distance in [0, 1]).
    d_orig_block = (0.5 * (1.0 - q @ g.t())).cpu().numpy()

    def _map_from_dist(scores: np.ndarray) -> float:
        m = compute_mAP_cmc(
            q_embs, g_embs, q_pids, g_pids, q_cams, g_cams, score_matrix=scores
        )
        return m["mAP"]

    def _jaccard_raw(lam: float, layers: int) -> np.ndarray:
        """Raw Jaccard distance [Nq, Ng] (fuse=1.0), reused across fuse values."""
        dist = cheb_gr_jaccard_distance(
            q,
            g,
            cheb_lambda=lam,
            gconv_layers=layers,
            residual_gamma=0.0,
            k_max=args.k_max,
            fuse_lambda=1.0,
        )
        return dist.cpu().numpy()

    if args.sweep:
        print("\n[Phase 4] Cheb-GR sweep (features extracted once)...")
        # Sanity: fuse=0 must reproduce the baseline exactly.
        print(
            f"  sanity fuse=0.0 -> mAP={_rerank_map(1.0, 1, 0.0, 0.0):.2f}% "
            f"(should == {base['mAP']:.2f})"
        )
        results = []
        # Variant A: feature-propagation re-rank.
        if args.sweep_variant in ("feat", "both"):
            for lam in (1.0, 1.5, 2.0, 3.0):
                for layers in (1, 2):
                    for gamma in (0.0, 0.3):
                        for fuse in (1.0, 0.5, 0.3):
                            mp = _rerank_map(lam, layers, gamma, fuse)
                            results.append((mp, "feat", lam, layers, gamma, fuse))
        # Variant B: Chebyshev k-reciprocal Jaccard re-rank (layers=0 => no refine).
        # Compute the costly raw Jaccard once per (lambda, layers); sweep fuse cheaply.
        if args.sweep_variant in ("jaccard", "both"):
            for lam in (1.0, 1.5, 2.0, 2.5, 3.0):
                for layers in (0, 1):
                    dj = _jaccard_raw(lam, layers)
                    for fuse in (1.0, 0.7, 0.5, 0.3):
                        scores = -(fuse * dj + (1.0 - fuse) * d_orig_block)
                        results.append(
                            (_map_from_dist(scores), "jacc", lam, layers, 0.0, fuse)
                        )
        results.sort(reverse=True)
        print(f"\n  baseline mAP = {base['mAP']:.2f}%")
        print("  " + "-" * 56)
        print(f"  {'mAP':>7} {'Δ':>8}  variant lambda  L  gamma  fuse")
        for mp, var, lam, layers, gamma, fuse in results[:24]:
            print(
                f"  {mp:7.2f} {mp - base['mAP']:+8.2f}  {var:>5}  {lam:5.1f}  "
                f"{layers}  {gamma:4.1f}  {fuse:4.1f}"
            )
        return

    print("\n[Phase 4] Cheb-GR re-ranking...")
    t0 = time.perf_counter()
    dist = cheb_gr_rerank_distance(
        q,
        g,
        cheb_lambda=args.cheb_lambda,
        gconv_layers=args.gconv_layers,
        residual_gamma=args.residual_gamma,
        k_max=args.k_max,
        fuse_lambda=args.fuse_lambda,
    )
    scores = (-dist).cpu().numpy()
    rr = compute_mAP_cmc(
        q_embs, g_embs, q_pids, g_pids, q_cams, g_cams, score_matrix=scores
    )
    print(f"  re-rank + eval in {time.perf_counter() - t0:.1f}s")
    print("=" * 56)
    print(
        f"Market-1501 — Cheb-GR (lambda={args.cheb_lambda} L={args.gconv_layers} "
        f"gamma={args.residual_gamma} k_max={args.k_max} fuse={args.fuse_lambda})"
    )
    print("=" * 56)
    for k, v in rr.items():
        print(f"  {k:>10s}: {v:6.2f}%  ({v - base.get(k, 0.0):+.2f})")
    print("=" * 56)


if __name__ == "__main__":
    main()
