"""Cheb-GR: Chebyshev's Theorem-guided Graph Re-ranking (CVPR 2025).

Reference: Yang, Li, Du, Ye. "Cheb-GR: Rethinking K-nearest Neighbor Search in
Re-ranking for Person Re-identification." CVPR 2025.

Re-ranking pipeline (no trainable weights — propagation only):

  1. Pairwise Euclidean distance        D_ij = ||f_i - f_j||_2
  2. Per-node distance statistics       mu_i = mean_j D_ij ,  sigma_i = std_j D_ij
  3. Chebyshev adaptive threshold       T_i  = mu_i - lambda * sigma_i
       (Chebyshev's inequality: P(D_ij <= mu_i - lambda*sigma_i) <= 1/lambda^2,
        i.e. only statistically-close neighbours survive — no fixed k.)
  4. Graph construction                 W_ij = sim(f_i, f_j) if D_ij <= T_i else 0
  5. Symmetric normalization            W~   = Dd^{-1/2} W Dd^{-1/2},  Dd_ii = sum_j W_ij
  6. Graph-conv refinement              F^{l+1} = W~ F^l   (or residual: F + gamma * W~ F)
  7. Re-rank distance recomputed on the refined features.

All features are assumed L2-normalized on input.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "cheb_gr_refine",
    "cheb_gr_rerank_distance",
    "cheb_gr_jaccard_distance",
    "cheb_gr_kreciprocal",
]


def _pairwise_euclidean(a: Tensor, b: Tensor) -> Tensor:
    """Euclidean distance matrix [Na, Nb] (clamped to be non-negative)."""
    # ||a - b||^2 = |a|^2 + |b|^2 - 2 a.b ; cdist is numerically cleaner.
    return torch.cdist(a, b, p=2.0)


def cheb_gr_refine(
    feats: Tensor,
    *,
    cheb_lambda: float = 1.0,
    gconv_layers: int = 2,
    residual_gamma: float = 0.0,
    k_max: int = 0,
) -> Tensor:
    """Refine L2-normalized features [N, D] via Cheb-GR graph convolution.

    Args:
        feats: [N, D] L2-normalized features.
        cheb_lambda: lambda in the Chebyshev threshold T_i = mu_i - lambda*sigma_i.
        gconv_layers: number of graph-convolution propagation layers (L).
        residual_gamma: if > 0, use residual update F + gamma * (W~ F); else F = W~ F.
        k_max: optional safety cap — keep at most k_max nearest neighbours per row
            (0 disables the cap; pure Chebyshev-adaptive selection).

    Returns:
        Refined features [N, D] (not re-normalized; rank with cosine downstream).
    """
    n = feats.shape[0]
    if n <= 1 or gconv_layers <= 0:
        return feats.clone()

    # Intermediates are released eagerly — N x N matrices dominate memory.
    dist = _pairwise_euclidean(feats, feats)  # [N, N]
    mu = dist.mean(dim=1, keepdim=True)  # [N, 1]
    sigma = dist.std(dim=1, unbiased=False, keepdim=True)  # [N, 1]
    thresh = mu - cheb_lambda * sigma  # [N, 1]

    mask = dist <= thresh  # [N, N] per-row adaptive neighbourhood
    if k_max > 0 and k_max < n:
        # Keep only the k_max smallest-distance entries per row, then intersect.
        topk_idx = torch.topk(dist, k_max, dim=1, largest=False).indices
        keep = torch.zeros_like(mask)
        keep.scatter_(1, topk_idx, True)
        mask = mask & keep
    del dist

    weight = (feats @ feats.t()) * mask  # W = sim where neighbour else 0
    del mask
    # Symmetrize so the symmetric normalization below is well-defined
    # (T_i differs per row, making the raw neighbourhood asymmetric).
    weight = 0.5 * (weight + weight.t())

    deg = weight.sum(dim=1)  # [N]
    dinv = torch.rsqrt(deg.clamp_min(1e-12))  # D^{-1/2}
    weight.mul_(dinv.unsqueeze(1)).mul_(dinv.unsqueeze(0))  # W~ in place
    w_norm = weight

    refined = feats
    for _ in range(gconv_layers):
        prop = w_norm @ refined
        if residual_gamma > 0.0:
            refined = refined + residual_gamma * prop
        else:
            refined = prop
    return refined


def cheb_gr_rerank_distance(
    query_feats: Tensor,
    gallery_feats: Tensor,
    *,
    cheb_lambda: float = 1.0,
    gconv_layers: int = 2,
    residual_gamma: float = 0.0,
    k_max: int = 0,
    fuse_lambda: float = 1.0,
) -> Tensor:
    """Cheb-GR re-ranked query-vs-gallery distance matrix [Nq, Ng].

    The graph is built over the combined query+gallery set, features are refined
    jointly, then the query x gallery block of (1 - cosine) distance is returned.

    Args:
        query_feats: [Nq, D] L2-normalized.
        gallery_feats: [Ng, D] L2-normalized.
        fuse_lambda: blend with the original distance:
            D = fuse_lambda * D_rerank + (1 - fuse_lambda) * D_orig.
            1.0 = pure re-rank, 0.0 = original distance.
    """
    nq = query_feats.shape[0]
    combined = torch.cat([query_feats, gallery_feats], dim=0)
    refined = cheb_gr_refine(
        combined,
        cheb_lambda=cheb_lambda,
        gconv_layers=gconv_layers,
        residual_gamma=residual_gamma,
        k_max=k_max,
    )
    refined = torch.nn.functional.normalize(refined, dim=1)
    qr = refined[:nq]
    gr = refined[nq:]
    dist_rerank = 1.0 - qr @ gr.t()

    if fuse_lambda >= 1.0:
        return dist_rerank
    dist_orig = 1.0 - query_feats @ gallery_feats.t()
    return fuse_lambda * dist_rerank + (1.0 - fuse_lambda) * dist_orig


def cheb_gr_jaccard_distance(
    query_feats: Tensor,
    gallery_feats: Tensor,
    *,
    cheb_lambda: float = 1.0,
    gconv_layers: int = 1,
    residual_gamma: float = 0.0,
    k_max: int = 0,
    fuse_lambda: float = 0.3,
) -> Tensor:
    """k-reciprocal Jaccard re-ranking with Chebyshev-adaptive neighbourhoods.

    This is the "rethinking k-NN" variant: the fixed-k reciprocal neighbour set of
    classic k-reciprocal re-ranking (Zhong et al., CVPR 2017) is replaced by the
    Chebyshev-adaptive neighbourhood (D_ij <= mu_i - lambda*sigma_i). Features are
    optionally refined first by Cheb-GR graph convolution.

    Each sample is encoded as V_i = softmax-like exp(-D) over its reciprocal
    neighbours (row-normalised to sum 1). The Jaccard distance between two
    L1-normalised encodings simplifies to  d_J = L1 / (1 + 0.5*L1)  with
    L1 = sum_k |V_i,k - V_j,k|, which is cheap via cdist(p=1).

    Returns [Nq, Ng] fused distance:
        D = fuse_lambda * d_jaccard + (1 - fuse_lambda) * d_original.
    """
    nq = query_feats.shape[0]
    combined = torch.cat([query_feats, gallery_feats], dim=0)
    n = combined.shape[0]

    base = combined
    if gconv_layers > 0:
        base = torch.nn.functional.normalize(
            cheb_gr_refine(
                combined,
                cheb_lambda=cheb_lambda,
                gconv_layers=gconv_layers,
                residual_gamma=residual_gamma,
                k_max=k_max,
            ),
            dim=1,
        )

    dist = _pairwise_euclidean(base, base)  # [N, N]
    mu = dist.mean(dim=1, keepdim=True)
    sigma = dist.std(dim=1, unbiased=False, keepdim=True)
    nbr = dist <= (mu - cheb_lambda * sigma)
    nbr = nbr | torch.eye(
        n, dtype=torch.bool, device=base.device
    )  # always include self
    recip = nbr & nbr.t()  # k-reciprocal (here: Chebyshev-reciprocal) set

    v = torch.exp(-dist) * recip
    del dist, nbr, recip
    v = v / v.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # Jaccard distance = 1 - sum_min / sum_max. Since each V row sums to 1,
    # sum_max = 2 - sum_min, so d_J = 1 - s/(2 - s) with s = histogram intersection.
    # Computed per query row to avoid an [Nq, Ng, N] blow-up (cdist p=1 would).
    vg = v[nq:]  # [Ng, N]
    ng = vg.shape[0]
    d_jaccard = torch.empty((nq, ng), device=base.device)
    for i in range(nq):
        idx = v[i].nonzero(as_tuple=True)[0]  # [m] support of query i
        if idx.numel() == 0:
            d_jaccard[i] = 1.0
            continue
        s = torch.minimum(v[i, idx].unsqueeze(0), vg[:, idx]).sum(dim=1)  # [Ng]
        d_jaccard[i] = 1.0 - s / (2.0 - s).clamp_min(1e-12)

    if fuse_lambda >= 1.0:
        return d_jaccard
    d_orig = 0.5 * (1.0 - query_feats @ gallery_feats.t())  # cosine dist in [0, 1]
    return fuse_lambda * d_jaccard + (1.0 - fuse_lambda) * d_orig


def cheb_gr_kreciprocal(
    query_feats: Tensor,
    gallery_feats: Tensor,
    *,
    cheb_lambda: float = 2.0,
    k2: int = 6,
    max_fwd: int = 50,
    fuse_lambda: float = 0.3,
) -> Tensor:
    """Faithful Cheb-GR re-ranking on GPU (all-torch, no Python neighbour loops).

    Full k-reciprocal machinery (Zhong et al., CVPR 2017) with the fixed-k1
    forward cutoff replaced by the Chebyshev-adaptive neighbourhood
    (D_ij <= mu_i - lambda*sigma_i), capped at `max_fwd` for tractability, plus
    the k2 local query expansion that drives most of k-reciprocal's gain.

    Returns [Nq, Ng] fused distance:
        D = (1 - fuse_lambda) * d_original + fuse_lambda * d_jaccard.
    (Here `fuse_lambda` is the weight on the Jaccard term, matching the variant
    above; classic k-reciprocal's lambda is `1 - fuse_lambda`.)
    """
    nq = query_feats.shape[0]
    combined = torch.cat([query_feats, gallery_feats], dim=0)
    n = combined.shape[0]
    device = combined.device

    # Squared-Euclidean distance for unit vectors = 2 - 2*cos, built in place
    # to avoid the extra [N, N] temporaries of (2 - 2*G).clamp_min(0).
    dist = combined @ combined.t()  # [N, N]
    dist.mul_(-2.0).add_(2.0).clamp_min_(0.0)
    mu = dist.mean(dim=1, keepdim=True)
    sigma = dist.std(dim=1, unbiased=False, keepdim=True)

    fwd = dist <= (mu - cheb_lambda * sigma)  # Chebyshev-adaptive neighbourhood
    fwd.fill_diagonal_(True)  # always include self (no [N, N] eye temporary)
    if 0 < max_fwd < n:
        cap = torch.zeros_like(fwd)
        cap.scatter_(1, torch.topk(dist, max_fwd, dim=1, largest=False).indices, True)
        fwd &= cap
        del cap
    recip = fwd & fwd.t()  # k-reciprocal set
    del fwd

    knn = torch.topk(dist, max(k2, 1), dim=1, largest=False).indices  # for QE
    # Reuse the dist buffer as V = exp(-dist) * recip (no extra [N, N] alloc).
    v = dist.neg_().exp_().mul_(recip)
    del recip
    v = v / v.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # k2 local query expansion: V_i <- mean of V over i's k2 nearest neighbours.
    if k2 > 1:
        rows = torch.arange(n, device=device).repeat_interleave(k2)
        a = torch.sparse_coo_tensor(
            torch.stack([rows, knn.reshape(-1)]),
            torch.full((n * k2,), 1.0 / k2, device=device),
            (n, n),
        ).coalesce()
        v = torch.sparse.mm(a, v)

    # Jaccard distance via histogram intersection. V is sparse (only reciprocal
    # neighbours are non-zero), so each query touches just its support columns
    # instead of the full [Ng, N] dense minimum -- ~N/nnz_i less memory traffic.
    vg = v[nq:]
    ng = vg.shape[0]
    d_jaccard = torch.empty((nq, ng), device=device)
    for i in range(nq):
        idx = v[i].nonzero(as_tuple=True)[0]  # [m] support of query i
        if idx.numel() == 0:
            d_jaccard[i] = 1.0
            continue
        s = torch.minimum(v[i, idx].unsqueeze(0), vg[:, idx]).sum(dim=1)  # [Ng]
        d_jaccard[i] = 1.0 - s / (2.0 - s).clamp_min(1e-12)

    if fuse_lambda >= 1.0:
        return d_jaccard
    d_orig = 0.5 * (1.0 - query_feats @ gallery_feats.t())
    return fuse_lambda * d_jaccard + (1.0 - fuse_lambda) * d_orig
