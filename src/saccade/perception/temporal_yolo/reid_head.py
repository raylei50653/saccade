"""
ReID projection head for FPN-based appearance embeddings.

Takes 896-dim (P3+P4+P5 avg-pool) raw FPN features and projects to a
128-dim discriminative embedding trained with supervised contrastive loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM_IN = 896  # P3(128) + P4(256) + P5(512)
EMB_DIM_OUT = 128


class ReIDHead(nn.Module):
    """896 → 256 → 128 projection with BN+ReLU."""

    def __init__(
        self,
        in_dim: int = EMB_DIM_IN,
        hidden: int = 256,
        out_dim: int = EMB_DIM_OUT,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, in_dim) → (N, out_dim) L2-normalised."""
        return F.normalize(self.proj(x), dim=1)


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (SupCon).

    Args:
        embeddings: (N, D) L2-normalised float32
        labels:     (N,)  int64 track IDs
        temperature: logit scale; lower = sharper separation

    Returns:
        scalar loss (mean over samples with ≥1 positive)
    """
    N = embeddings.shape[0]
    if N < 2:
        return embeddings.sum() * 0.0

    sim = (embeddings @ embeddings.T) / temperature  # (N, N)

    # Numerical stability
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    sim = sim - sim_max

    # Positive mask: same label, excluding diagonal
    lab = labels.view(-1, 1)
    pos_mask = (lab == lab.T).float()
    pos_mask.fill_diagonal_(0.0)

    # Exclude self from log-sum-exp denominator
    eye = torch.eye(N, device=sim.device, dtype=torch.bool)
    exp_sim = sim.exp().masked_fill(eye, 0.0)

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8))

    n_pos = pos_mask.sum(dim=1)
    loss_per = -(pos_mask * log_prob).sum(dim=1) / n_pos.clamp(min=1e-8)

    valid = n_pos > 0
    if not valid.any():
        return embeddings.sum() * 0.0

    return loss_per[valid].mean()


def build_reid_head(
    in_dim: int = EMB_DIM_IN,
    hidden: int = 256,
    out_dim: int = EMB_DIM_OUT,
) -> ReIDHead:
    return ReIDHead(in_dim=in_dim, hidden=hidden, out_dim=out_dim)


def load_reid_head(path: str, device: str | torch.device = "cuda") -> ReIDHead:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("cfg", {})
    head = ReIDHead(
        in_dim=cfg.get("in_dim", EMB_DIM_IN),
        hidden=cfg.get("hidden", 256),
        out_dim=cfg.get("out_dim", EMB_DIM_OUT),
    )
    head.load_state_dict(ckpt["head"])
    head.to(device).eval()
    return head
