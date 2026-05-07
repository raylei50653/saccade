"""Phase 2A: Verify C++ extract_with_stability matches Python last_vit_pipeline_dual.

Skipped automatically when:
  - CUDA GPU not available
  - SigLIP2 TRT engine not found (models/embedding/google_siglip2-base-patch16-224.engine)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

REID_ENGINE = ROOT / "models/embedding/google_siglip2-base-patch16-224.engine"

skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)
skip_no_engine = pytest.mark.skipif(
    not REID_ENGINE.exists(), reason="SigLIP2 TRT engine not found"
)


# ---------------------------------------------------------------------------
# Python reference pipeline (matches validate_last_vit_phase0.py)
# ---------------------------------------------------------------------------


def _gauss_filter(x: torch.Tensor, sigma: float) -> torch.Tensor:
    C = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.arange(C // 2 + 1, device=x.device, dtype=torch.float32) / C
    gauss_w = torch.exp(-(freqs**2) / (2.0 * sigma**2))
    return torch.fft.irfft(X * gauss_w, n=C, dim=-1)


def _stability_scores(x: torch.Tensor, x_filt: torch.Tensor) -> torch.Tensor:
    diff_sq = (x - x_filt).pow(2).sum(dim=-1)
    norm_sq = x.pow(2).sum(dim=-1).clamp(min=1e-8)
    return (1.0 - diff_sq / norm_sq).clamp(0.0, 1.0)


def _topk_pool(
    x: torch.Tensor, scores: torch.Tensor, top_k_ratio: float
) -> torch.Tensor:
    B, N, C = x.shape
    K = max(1, int(N * top_k_ratio))
    topk_idx = scores.topk(K, dim=-1).indices
    topk_feats = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
    return F.normalize(topk_feats.mean(dim=1), dim=-1)


def _last_vit_py(
    lhs: torch.Tensor,
    sigma_embed: float,
    sigma_gate: float,
    top_k_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python reference: returns (embed [B, C], stab_mean [B])."""
    x = lhs.float()
    x_filt_embed = _gauss_filter(x, sigma_embed)
    x_filt_gate = _gauss_filter(x, sigma_gate)
    scores_gate = _stability_scores(x, x_filt_gate)  # [B, N]
    scores_embed = _stability_scores(x, x_filt_embed)  # [B, N]
    embedding = _topk_pool(x, scores_embed, top_k_ratio)  # [B, C] L2-norm
    stab_mean = scores_gate.mean(dim=-1)  # [B]
    return embedding, stab_mean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def py_extractor():
    import saccade.perception.feature_extractor as _fe_mod

    _fe_mod.HAS_CPP_EXT = False
    from saccade.perception.feature_extractor import TRTFeatureExtractor

    return TRTFeatureExtractor(model_type="siglip2")


@pytest.fixture(scope="module")
def cpp_extractor():
    import saccade_perception_ext as ext

    return ext.FeatureExtractor(str(REID_ENGINE), ext.ModelType.SIGLIP2, 16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crops(n: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.rand(n, 3, 224, 224, device="cuda", dtype=torch.float32, generator=g)


def _py_ref(extractor, crops, sigma_embed, sigma_gate, top_k_ratio):
    n = crops.size(0)
    extractor._extract_chunk(crops)  # fills output_buffers in-place
    torch.cuda.synchronize()
    lhs = extractor.output_buffers["last_hidden_state"][:n].float().clone()
    return _last_vit_py(lhs, sigma_embed, sigma_gate, top_k_ratio)


def _cpp_run(ext, crops, sigma_embed, sigma_gate, top_k_ratio):
    n = crops.size(0)
    inp = crops.clone().contiguous()  # C++ normalizes in-place; protect original
    embed = torch.empty(n, 768, device="cuda", dtype=torch.float32)
    stab = torch.empty(n, device="cuda", dtype=torch.float32)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    ext.extract_with_stability(
        inp.data_ptr(),
        n,
        embed.data_ptr(),
        stab.data_ptr(),
        stream_ptr,
        sigma_embed,
        sigma_gate,
        top_k_ratio,
    )
    torch.cuda.synchronize()
    return embed, stab


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SIGMA_PAIRS = [(0.015, 0.040), (0.010, 0.030)]
BATCH_SIZES = [1, 8, 16]

# Thresholds based on float32 FFT numerical tolerance:
#   - cuFFT vs PyTorch FFT may differ by ~1e-5 per element; accumulated across
#     pooling the delta propagates to ~1e-4 in cosine space.
EMBED_COS_MIN = 0.9990  # cosine similarity lower bound
STAB_ABS_MAX = 2e-3  # absolute diff on mean stability score


@skip_no_gpu
@skip_no_engine
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("sigma_embed,sigma_gate", SIGMA_PAIRS)
def test_embedding_cosine_sim(
    py_extractor, cpp_extractor, batch_size, sigma_embed, sigma_gate
):
    """C++ embedding must be nearly identical to Python reference (cosine sim ≥ threshold)."""
    crops = _make_crops(batch_size)
    embed_py, _ = _py_ref(py_extractor, crops, sigma_embed, sigma_gate, 0.5)
    embed_cpp, _ = _cpp_run(cpp_extractor, crops, sigma_embed, sigma_gate, 0.5)

    cos = F.cosine_similarity(embed_py, embed_cpp, dim=-1)
    assert cos.min().item() >= EMBED_COS_MIN, (
        f"cosine sim too low: min={cos.min():.6f} "
        f"(σ_e={sigma_embed}, σ_g={sigma_gate}, B={batch_size})"
    )


@skip_no_gpu
@skip_no_engine
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("sigma_embed,sigma_gate", SIGMA_PAIRS)
def test_stability_score_close(
    py_extractor, cpp_extractor, batch_size, sigma_embed, sigma_gate
):
    """C++ per-image stability mean must match Python reference (abs diff ≤ threshold)."""
    crops = _make_crops(batch_size, seed=99)
    _, stab_py = _py_ref(py_extractor, crops, sigma_embed, sigma_gate, 0.5)
    _, stab_cpp = _cpp_run(cpp_extractor, crops, sigma_embed, sigma_gate, 0.5)

    diff = (stab_py - stab_cpp).abs()
    assert diff.max().item() <= STAB_ABS_MAX, (
        f"stability diff too high: max={diff.max():.6f} "
        f"(σ_e={sigma_embed}, σ_g={sigma_gate}, B={batch_size})"
    )


@skip_no_gpu
@skip_no_engine
def test_output_shapes(py_extractor, cpp_extractor):
    """Sanity-check output shapes for a batch of 4."""
    crops = _make_crops(4)
    embed_cpp, stab_cpp = _cpp_run(cpp_extractor, crops, 0.015, 0.040, 0.5)
    assert embed_cpp.shape == (4, 768), f"embed shape: {embed_cpp.shape}"
    assert stab_cpp.shape == (4,), f"stab shape:  {stab_cpp.shape}"
    assert embed_cpp.is_cuda and stab_cpp.is_cuda


@skip_no_gpu
@skip_no_engine
def test_embedding_is_l2_normalized(cpp_extractor):
    """Output embeddings must be unit-norm (L2 normalization applied by C++)."""
    crops = _make_crops(8)
    embed, _ = _cpp_run(cpp_extractor, crops, 0.015, 0.040, 0.5)
    norms = embed.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-5, (
        f"embeddings not unit-norm: max deviation {(norms - 1.0).abs().max():.2e}"
    )


@skip_no_gpu
@skip_no_engine
def test_stability_range(cpp_extractor):
    """Stability scores must be in [0, 1]."""
    crops = _make_crops(8)
    _, stab = _cpp_run(cpp_extractor, crops, 0.015, 0.040, 0.5)
    assert stab.min().item() >= -1e-5, f"stab below 0: min={stab.min():.4f}"
    assert stab.max().item() <= 1.0 + 1e-5, f"stab above 1: max={stab.max():.4f}"


@skip_no_gpu
@skip_no_engine
@pytest.mark.parametrize("top_k_ratio", [0.25, 0.5, 0.75])
def test_top_k_ratio_sweep(cpp_extractor, top_k_ratio):
    """Different top_k_ratio values must all produce valid embeddings."""
    crops = _make_crops(4)
    embed, stab = _cpp_run(cpp_extractor, crops, 0.015, 0.040, top_k_ratio)
    norms = embed.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-4, (
        f"top_k={top_k_ratio}: embeddings not unit-norm"
    )
    assert stab.min().item() >= -1e-5 and stab.max().item() <= 1.0 + 1e-5, (
        f"top_k={top_k_ratio}: stability out of [0,1]"
    )
