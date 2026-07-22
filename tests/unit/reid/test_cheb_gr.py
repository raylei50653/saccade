"""Unit tests for Cheb-GR re-ranking core (CPU, no GPU required)."""

# scope: reid
# function: behavior
# lifecycle: active

from __future__ import annotations

import torch
import torch.nn.functional as F

from saccade.perception.reid.cheb_gr import (
    cheb_gr_jaccard_distance,
    cheb_gr_kreciprocal,
    cheb_gr_refine,
    cheb_gr_rerank_distance,
)


def _two_clusters(
    d: int = 16, per: int = 6, sep: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two well-separated L2-normalized clusters; returns (feats, labels)."""
    torch.manual_seed(0)
    c0 = torch.zeros(d)
    c0[0] = sep
    c1 = torch.zeros(d)
    c1[1] = sep
    a = c0 + 0.1 * torch.randn(per, d)
    b = c1 + 0.1 * torch.randn(per, d)
    feats = F.normalize(torch.cat([a, b], dim=0), dim=1)
    labels = torch.cat([torch.zeros(per), torch.ones(per)]).long()
    return feats, labels


def test_refine_shape_and_finite() -> None:
    feats, _ = _two_clusters()
    out = cheb_gr_refine(feats, cheb_lambda=1.0, gconv_layers=2)
    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_rerank_distance_block_shape() -> None:
    q, _ = _two_clusters(per=4)
    g, _ = _two_clusters(per=8)
    dist = cheb_gr_rerank_distance(q, g, gconv_layers=2)
    assert dist.shape == (q.shape[0], g.shape[0])
    assert torch.isfinite(dist).all()


def test_degenerate_single_and_zero_layers() -> None:
    feats, _ = _two_clusters()
    # N == 1 → unchanged
    single = feats[:1]
    assert torch.allclose(cheb_gr_refine(single), single)
    # gconv_layers == 0 → unchanged (returns a copy)
    out0 = cheb_gr_refine(feats, gconv_layers=0)
    assert torch.allclose(out0, feats)


def test_chebyshev_neighbours_respect_clusters() -> None:
    """The adaptive neighbourhood (D_ij <= mu_i - lambda*sigma_i) should pick
    same-cluster samples and exclude cross-cluster ones."""
    feats, labels = _two_clusters(sep=8.0)
    dist = torch.cdist(feats, feats, p=2.0)
    mu = dist.mean(dim=1, keepdim=True)
    sigma = dist.std(dim=1, unbiased=False, keepdim=True)
    mask = dist <= (mu - 1.0 * sigma)
    # remove self so we test true neighbours only
    eye = torch.eye(len(feats), dtype=torch.bool)
    mask = mask & ~eye
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    # every selected neighbour shares the cluster label
    assert (mask & ~same).sum() == 0
    # and at least some same-cluster neighbours are selected
    assert (mask & same).sum() > 0


def test_rerank_pulls_same_id_closer() -> None:
    """After Cheb-GR, the nearest gallery item to each query shares its cluster."""
    feats, labels = _two_clusters(per=6, sep=8.0)
    q = feats
    g = feats
    dist = cheb_gr_rerank_distance(q, g, cheb_lambda=1.0, gconv_layers=2)
    # ignore self-match on the diagonal
    dist = dist + torch.eye(len(feats)) * 10.0
    nearest = dist.argmin(dim=1)
    assert (labels[nearest] == labels).all()


def test_jaccard_distance_shape_and_range() -> None:
    q, _ = _two_clusters(per=4)
    g, _ = _two_clusters(per=8)
    d = cheb_gr_jaccard_distance(q, g, cheb_lambda=2.0, gconv_layers=0, fuse_lambda=1.0)
    assert d.shape == (q.shape[0], g.shape[0])
    assert torch.isfinite(d).all()
    assert (d >= -1e-4).all() and (d <= 1.0 + 1e-4).all()


def test_jaccard_pulls_same_id_closer() -> None:
    feats, labels = _two_clusters(per=6, sep=8.0)
    d = cheb_gr_jaccard_distance(
        feats, feats, cheb_lambda=2.0, gconv_layers=0, fuse_lambda=0.5
    )
    d = d + torch.eye(len(feats)) * 10.0  # ignore self
    nearest = d.argmin(dim=1)
    assert (labels[nearest] == labels).all()


def test_kreciprocal_shape_and_pulls_same_id() -> None:
    feats, labels = _two_clusters(per=6, sep=8.0)
    d = cheb_gr_kreciprocal(
        feats, feats, cheb_lambda=1.0, k2=3, max_fwd=8, fuse_lambda=0.5
    )
    assert d.shape == (len(feats), len(feats))
    assert torch.isfinite(d).all()
    d = d + torch.eye(len(feats)) * 10.0
    nearest = d.argmin(dim=1)
    assert (labels[nearest] == labels).all()


def test_kreciprocal_max_fwd_bounds_neighbours() -> None:
    q, _ = _two_clusters(per=5)
    g, _ = _two_clusters(per=7)
    # Should run without error for a small cap and various k2.
    d = cheb_gr_kreciprocal(q, g, cheb_lambda=2.0, k2=1, max_fwd=4, fuse_lambda=1.0)
    assert d.shape == (q.shape[0], g.shape[0])
    assert (d >= -1e-4).all()


def test_fuse_lambda_endpoints() -> None:
    q, _ = _two_clusters(per=4)
    g, _ = _two_clusters(per=5)
    pure = cheb_gr_rerank_distance(q, g, fuse_lambda=1.0)
    orig = cheb_gr_rerank_distance(q, g, fuse_lambda=0.0)
    expected_orig = 1.0 - q @ g.t()
    assert torch.allclose(orig, expected_orig, atol=1e-5)
    assert not torch.allclose(pure, orig)
