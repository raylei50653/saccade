"""
Mamba SSM Detection Head — replaces YOLO Detect with selective state-space models.

Each FPN level (P3/P4/P5) is processed through Mamba blocks operating on
the flattened spatial dimension as a sequence, then projected to detection
outputs (boxes + class scores).

Reference:
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Selective SSM scan (pure PyTorch — no CUDA kernel dependency)
# ---------------------------------------------------------------------------
def _selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    delta_softplus: bool = True,
) -> Tensor:
    return _selective_scan_cuda(u, delta, A, B, C, D)


def _selective_scan_cuda(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
) -> Tensor:
    import saccade_tracking_ext

    D_ptr = D.data_ptr() if D is not None and D.numel() > 0 else 0

    u = u.contiguous()
    delta = delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    y = torch.empty_like(u)

    saccade_tracking_ext.selective_scan_fwd(
        u.data_ptr(),
        delta.data_ptr(),
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        D_ptr,
        y.data_ptr(),
        u.shape[0],
        u.shape[1],
        u.shape[2],
        A.shape[0],
        1 if D_ptr != 0 else 0,
    )
    return y


@torch.jit.script
def _selective_scan_jit(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
) -> Tensor:
    B_dim, L_dim, D_dim = u.shape
    N = A.shape[1]
    delta = F.softplus(delta)

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
    B_exp = B.unsqueeze(2)  # (B, L, 1, N)
    deltaB_u = delta.unsqueeze(-1) * B_exp * u.unsqueeze(-1)  # (B, L, D, N)

    h = torch.zeros(B_dim, D_dim, N, device=u.device, dtype=u.dtype)
    ys = torch.zeros(B_dim, L_dim, D_dim, device=u.device, dtype=u.dtype)
    for t in range(L_dim):
        h = deltaA[:, t] * h + deltaB_u[:, t]
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
        ys[:, t] = y_t

    if D is not None:
        ys = ys + u * D
    return ys


# ---------------------------------------------------------------------------
# Mamba S6 block
# ---------------------------------------------------------------------------
class MambaBlock(nn.Module):
    """Single Mamba S6 block operating on 1D sequences."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)

        # A is a learned parameter matrix (diagonalized per channel)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        d_inner = self.d_model * self.expand

        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ = xz[..., :d_inner]
        z = xz[..., d_inner:]

        # Causal 1D conv
        x_t = x_.transpose(1, 2)  # (B, d_inner, L)
        x_t = self.conv1d(x_t)[..., :L]
        x_t = F.silu(x_t.transpose(1, 2))  # (B, L, d_inner)

        # SSM parameters
        x_db = self.x_proj(x_t)
        dt = self.dt_proj(x_db[..., : self.d_state])
        B_ssm = x_db[..., self.d_state : self.d_state * 2]
        C_ssm = x_db[..., self.d_state * 2 :]

        A = -torch.exp(self.A_log.float())  # (1, N)

        y = _selective_scan(
            x_t,
            dt,
            A,
            B_ssm,
            C_ssm,
            D=self.D,
        )

        y = y * F.silu(z)
        y = self.out_proj(y)

        return y


# ---------------------------------------------------------------------------
# Mamba detection head
# ---------------------------------------------------------------------------
class MambaDetectionHead(nn.Module):
    """Multi-scale Mamba head for YOLO-style detection.

    Each FPN level is downsampled to a coarse spatial grid before Mamba
    processing, then upsampled back for per-position detection predictions.
    This keeps the sequence length tractable (L ≤ 400) while preserving the
    SSM's ability to model long-range spatial dependencies at the coarse level.

    Optional embedding branch (emb_dim > 0) produces per-pixel appearance
    embeddings for JDE-style joint detection + ReID.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (128, 256, 512),
        d_model: int = 128,
        d_state: int = 16,
        num_blocks: int = 2,
        num_classes: int = 80,
        reg_max: int = 1,
        strides: tuple[int, ...] = (8, 16, 32),
        spatial_reduction: int = 4,  # downsample factor before Mamba
        emb_dim: int = 0,  # embedding dimension (0 = disabled)
    ):
        super().__init__()
        self.nl = len(in_channels)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.stride = torch.tensor(strides, dtype=torch.float32)
        self.no = num_classes + reg_max * 4
        self.spatial_reduction = spatial_reduction
        self.emb_dim = emb_dim

        self.input_proj = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels])
        self.downsample = nn.ModuleList(
            [
                nn.Conv2d(d_model, d_model, spatial_reduction, stride=spatial_reduction)
                for _ in range(self.nl)
            ]
        )

        self.mamba_blocks = nn.ModuleList(
            [
                nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(num_blocks)])
                for _ in range(self.nl)
            ]
        )
        self.cls_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(d_model * 2, d_model, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(d_model, num_classes, 1),
                )
                for _ in range(self.nl)
            ]
        )
        self.reg_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(d_model * 2, d_model, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(d_model, reg_max * 4, 1),
                )
                for _ in range(self.nl)
            ]
        )

        self.emb_head: nn.ModuleList | None = None
        if emb_dim > 0:
            self.emb_head = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(d_model * 2, d_model, 3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(d_model, emb_dim, 1),
                    )
                    for _ in range(self.nl)
                ]
            )

    def forward(
        self, feats: list[Tensor], return_embeddings: bool = False
    ) -> (
        tuple[list[Tensor], list[Tensor]]
        | tuple[list[Tensor], list[Tensor], list[Tensor]]
    ):
        cls_preds = []
        reg_preds = []
        emb_preds: list[Tensor] = []

        for i, x in enumerate(feats):
            B, C, H, W = x.shape

            x_proj = self.input_proj[i](x)  # (B, d_model, H, W)
            x_small = self.downsample[i](x_proj)  # (B, d_model, H/4, W/4)
            _, _, Hs, Ws = x_small.shape

            x_seq = x_small.flatten(2).transpose(1, 2)  # (B, Hs*Ws, d_model)

            for block in self.mamba_blocks[i]:
                x_seq = block(x_seq) + x_seq

            x_up = x_seq.transpose(1, 2).reshape(B, -1, Hs, Ws)
            x_up = F.interpolate(
                x_up, size=(H, W), mode="bilinear", align_corners=False
            )

            x_cat = torch.cat([x_proj, x_up], dim=1)  # (B, d_model*2, H, W)
            cls_preds.append(self.cls_head[i](x_cat))
            reg_preds.append(self.reg_head[i](x_cat))
            if self.emb_head is not None:
                emb_preds.append(self.emb_head[i](x_cat))

        if return_embeddings and self.emb_head is not None:
            return cls_preds, reg_preds, emb_preds
        return cls_preds, reg_preds

    def pool_embeddings(self, emb_preds: list[Tensor], boxes_xyxy: Tensor) -> Tensor:
        from torchvision.ops import roi_align

        if boxes_xyxy.numel() == 0:
            return torch.zeros((0, self.emb_dim * self.nl), device=emb_preds[0].device)

        _SPATIAL_SCALES = {0: 1 / 8, 1: 1 / 16, 2: 1 / 32}
        batch_boxes = torch.cat(
            [
                torch.zeros(len(boxes_xyxy), 1, device=boxes_xyxy.device),
                boxes_xyxy.float(),
            ],
            dim=1,
        )

        parts = []
        for i, emb in enumerate(emb_preds):
            pooled = roi_align(
                emb.float(),
                batch_boxes,
                output_size=1,
                spatial_scale=_SPATIAL_SCALES[i],
                aligned=True,
            )
            parts.append(pooled.flatten(1))
        return F.normalize(torch.cat(parts, dim=1), dim=1)

    def pool_embeddings_global(self, emb_preds: list[Tensor]) -> Tensor:
        parts = []
        for emb in emb_preds:
            pooled = emb.mean(dim=[2, 3])
            parts.append(pooled)
        return F.normalize(torch.cat(parts, dim=1), dim=1)

    def pool_embeddings_center(self, emb_preds: list[Tensor]) -> Tensor:
        parts = []
        for emb in emb_preds:
            h, w = emb.shape[2], emb.shape[3]
            center = emb[:, :, h // 2, w // 2]
            parts.append(center)
        return F.normalize(torch.cat(parts, dim=1), dim=1)


class EmbeddingProjector(nn.Module):
    """Projects per-scale concatenated emb_head features to a ReID embedding.

    Input:  (N, emb_dim * 3)  [P3 + P4 + P5 pooled features]
    Output: (N, out_dim)      L2-normalised
    """

    def __init__(self, emb_dim: int = 128, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        in_dim = emb_dim * 3
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.proj(x), dim=1)
