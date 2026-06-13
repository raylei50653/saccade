"""
Mamba SSM Detection Head — replaces YOLO Detect with selective state-space models.

Each FPN level (P3/P4/P5) is processed through Mamba blocks operating on
the flattened spatial dimension as a sequence, then projected to detection
outputs (boxes + class scores).

Reference:
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Escape hatch: force the differentiable training scan back onto the pure-Python
# JIT loop (e.g. to A/B against the CUDA backward kernel).
_DISABLE_CUDA_SCAN_BWD = os.environ.get("SACCADE_DISABLE_CUDA_SCAN_BWD", "0") == "1"


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
) -> Tensor:
    # softplus(delta) is applied inside both backends, so delta is passed raw.
    # The CUDA kernel dereferences raw device pointers, so it must only be given
    # CUDA tensors — CPU inputs (and the case where the extension is unavailable)
    # fall back to the pure-PyTorch scan instead of feeding host pointers to the
    # kernel (which corrupts the CUDA context with an illegal memory access).
    # The CUDA op (saccade::selective_scan_fwd) registers no autograd formula, so
    # it is inference-only. When grad is required (training), fall back to the
    # pure-PyTorch differentiable scan. Eval/inference is unaffected.
    grad_needed = torch.is_grad_enabled() and (
        u.requires_grad
        or delta.requires_grad
        or A.requires_grad
        or B.requires_grad
        or C.requires_grad
    )
    if grad_needed:
        # Training: use the analytic CUDA fwd/bwd kernels (the JIT scan is a
        # launch-bound Python loop). Falls back to JIT on CPU or if the C++
        # extension is unavailable.
        if u.is_cuda and not _DISABLE_CUDA_SCAN_BWD:
            try:
                return _SelectiveScanCudaFn.apply(u, delta, A, B, C, D)
            except (ImportError, RuntimeError):
                return _selective_scan_jit(u, delta, A, B, C, D)
        return _selective_scan_jit(u, delta, A, B, C, D)
    if not u.is_cuda:
        return _selective_scan_jit(u, delta, A, B, C, D)
    try:
        return _selective_scan_cuda(u, delta, A, B, C, D)
    except ImportError:
        return _selective_scan_jit(u, delta, A, B, C, D)


# Custom operator registration for traceability in PyTorch 2.x / torch.compile / torch.export
try:

    @torch.library.custom_op("saccade::selective_scan_fwd", mutates_args=())
    def _saccade_selective_scan_op(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        a_per_channel: int,
        is_half: bool,
    ) -> torch.Tensor:
        import saccade_tracking_ext

        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        D_ptr = D.data_ptr() if D.numel() > 0 else 0
        y = torch.empty_like(u)

        # Launch on PyTorch's current stream so the kernel is recorded during
        # CUDA-graph capture (the legacy default stream is never captured).
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
            A.shape[-1],
            1 if D_ptr != 0 else 0,
            a_per_channel,
            is_half,
            torch.cuda.current_stream(u.device).cuda_stream,
        )
        return y

    @_saccade_selective_scan_op.register_fake
    def _(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        a_per_channel: int,
        is_half: bool,
    ) -> torch.Tensor:
        return torch.empty(u.shape, dtype=u.dtype, device=u.device)

except Exception:
    # Fallback to direct call in case custom_op is not supported or errors out
    def _saccade_selective_scan_op(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        a_per_channel: int,
        is_half: bool,
    ) -> torch.Tensor:
        import saccade_tracking_ext

        D_ptr = D.data_ptr() if D.numel() > 0 else 0
        y = torch.empty_like(u)
        saccade_tracking_ext.selective_scan_fwd(
            u.contiguous().data_ptr(),
            delta.contiguous().data_ptr(),
            A.contiguous().data_ptr(),
            B.contiguous().data_ptr(),
            C.contiguous().data_ptr(),
            D_ptr,
            y.data_ptr(),
            u.shape[0],
            u.shape[1],
            u.shape[2],
            A.shape[-1],
            1 if D_ptr != 0 else 0,
            a_per_channel,
            is_half,
            torch.cuda.current_stream(u.device).cuda_stream,
        )
        return y


def _selective_scan_cuda(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
) -> Tensor:
    # A is (rows, N): rows=1 → shared decay across channels; rows=d_inner →
    # per-channel dynamics. N (state dim) is always the last axis; the kernel
    # indexes A[d * N + n] when per-channel.
    N = A.shape[-1]
    D_dim = u.shape[2]
    a_per_channel = 1 if (A.dim() == 2 and A.shape[0] == D_dim) else 0

    # C may be rank-1 (shape ...,1) from legacy x_proj=d_state*2+1 checkpoints.
    # Broadcast to (B, L, N) so the kernel reads correct C values for all N states.
    if C.shape[-1] < N:
        C = C.expand(*C.shape[:-1], N).contiguous()
    else:
        C = C.contiguous()

    is_half = u.dtype == torch.float16
    D_tensor = D if D is not None else torch.empty(0, dtype=u.dtype, device=u.device)

    return _saccade_selective_scan_op(
        u,
        delta,
        A,
        B,
        C,
        D_tensor,
        a_per_channel,
        is_half,
    )


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


class _SelectiveScanCudaFn(torch.autograd.Function):
    """Differentiable selective scan backed by the C++ CUDA fwd/bwd kernels.

    Forward calls ``selective_scan_fwd``; backward calls ``selective_scan_bwd``
    (analytic reverse-time gradient). This replaces the pure-Python JIT scan on
    the training path — the JIT loop is launch-bound (one tiny kernel per
    sequence step), while these kernels run the whole recurrence on-device.

    delta is RAW (pre-softplus); softplus is applied inside both kernels. A is
    (1, N) shared or (D, N) per-channel. C may be rank-1 (..., 1) — it is
    broadcast to N for the kernel and the gradient summed back.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):  # type: ignore[override]
        import saccade_tracking_ext

        N = A.shape[-1]
        D_dim = u.shape[2]
        a_per_channel = 1 if (A.dim() == 2 and A.shape[0] == D_dim) else 0
        c_rank1 = C.shape[-1] < N

        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C_full = C.expand(*C.shape[:-1], N).contiguous() if c_rank1 else C.contiguous()
        has_D = D is not None and D.numel() > 0
        D_t = (
            D.contiguous() if has_D else torch.empty(0, dtype=u.dtype, device=u.device)
        )

        y = torch.empty_like(u)
        stream = torch.cuda.current_stream(u.device).cuda_stream
        saccade_tracking_ext.selective_scan_fwd(
            u.data_ptr(),
            delta.data_ptr(),
            A.data_ptr(),
            B.data_ptr(),
            C_full.data_ptr(),
            D_t.data_ptr() if has_D else 0,
            y.data_ptr(),
            u.shape[0],
            u.shape[1],
            D_dim,
            N,
            1 if has_D else 0,
            a_per_channel,
            False,
            stream,
        )

        ctx.save_for_backward(u, delta, A, B, C_full, D_t)
        ctx.a_per_channel = a_per_channel
        ctx.has_D = has_D
        ctx.c_rank1 = c_rank1
        ctx.c_orig_rank = C.shape[-1]
        return y

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        import saccade_tracking_ext

        u, delta, A, B, C_full, D_t = ctx.saved_tensors
        Bb, L, D_dim = u.shape
        N = A.shape[-1]
        grad_y = grad_y.contiguous()

        du = torch.empty_like(u)
        ddelta = torch.empty_like(delta)
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
        dC = torch.zeros_like(C_full)
        dD = torch.zeros_like(D_t) if ctx.has_D else torch.empty(0, device=u.device)
        h_buf = torch.empty(Bb, L, D_dim, N, dtype=u.dtype, device=u.device)

        stream = torch.cuda.current_stream(u.device).cuda_stream
        saccade_tracking_ext.selective_scan_bwd(
            grad_y.data_ptr(),
            u.data_ptr(),
            delta.data_ptr(),
            A.data_ptr(),
            B.data_ptr(),
            C_full.data_ptr(),
            D_t.data_ptr() if ctx.has_D else 0,
            h_buf.data_ptr(),
            du.data_ptr(),
            ddelta.data_ptr(),
            dA.data_ptr(),
            dB.data_ptr(),
            dC.data_ptr(),
            dD.data_ptr() if ctx.has_D else 0,
            Bb,
            L,
            D_dim,
            N,
            1 if ctx.has_D else 0,
            ctx.a_per_channel,
            stream,
        )

        # Fold rank-1 C gradient back to its original last-dim size.
        if ctx.c_rank1:
            dC = dC.sum(dim=-1, keepdim=True)
        dD_out = dD if ctx.has_D else None
        return du, ddelta, dA, dB, dC, dD_out


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
        full_rank_c: bool = False,
        per_channel_a: bool = False,
        scan_stop_grad: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.full_rank_c = full_rank_c
        self.per_channel_a = per_channel_a
        self.scan_stop_grad = scan_stop_grad
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        # SSM parameters: [dt(d_state) | B(d_state) | C(c_rank)]
        # full_rank_c=False: c_rank=1 (rank-1, broadcast to N) — compatible with v14 weights
        # full_rank_c=True:  c_rank=d_state (rank-N) — selective per-state output weighting
        c_rank = d_state if full_rank_c else 1
        self.x_proj = nn.Linear(d_inner, d_state * 2 + c_rank, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)

        # A: real S4D init, A[..., n] = n+1.
        # per_channel_a=False: shape (1, N) — single decay schedule shared by all
        #   d_inner channels (v14-compatible, fewer params).
        # per_channel_a=True:  shape (d_inner, N) — each channel learns its own
        #   SSM dynamics. Initialized identically across rows so a shared-A (1, N)
        #   checkpoint warm-starts (broadcast) to an exactly equivalent state; the
        #   kernel indexes A[d * N + n].
        rows = d_inner if per_channel_a else 1
        A = torch.arange(1, d_state + 1, dtype=torch.float32).expand(rows, d_state)
        self.A_log = nn.Parameter(torch.log(A).contiguous())
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

        A = -torch.exp(self.A_log.float())  # (1, N) shared or (d_inner, N) per-channel

        y = _selective_scan(
            x_t,
            dt,
            A,
            B_ssm,
            C_ssm,
            D=self.D,
        )

        # v14 historical regime: the raw CUDA scan had no grad_fn, so conv1d /
        # x_proj / dt_proj / A_log / D (and the x half of in_proj) stayed at
        # init; only the z gate and out_proj learned. detach reproduces that
        # gradient topology with the correct N=16 forward.
        if self.scan_stop_grad:
            y = y.detach()

        y = y * F.silu(z)
        y = self.out_proj(y)

        return y


# ---------------------------------------------------------------------------
# Temporal self-attention block (T=4 optimized)
# ---------------------------------------------------------------------------
class TemporalAttentionBlock(nn.Module):
    """Self-attention over T frames at each spatial location.

    Input/output: (B*H*W, T, d_model) — drop-in for MambaBlock in temporal path.
    ReZero init (out_proj.weight=0) ensures identity at start so spatial features
    are not disturbed before the block has learned anything useful.
    """

    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # ReZero init: starts as identity, gradients flow from step 1
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: Tensor) -> Tensor:  # (BHW, T, d_model)
        BHW, T, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(BHW, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BHW, heads, T, head_dim)
        q, k, v = qkv.unbind(0)  # each (BHW, heads, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BHW, heads, T, T)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (BHW, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(BHW, T, D)  # (BHW, T, d_model)
        return x + self.out_proj(out)


# ---------------------------------------------------------------------------
# Cross-scan Mamba: 4-directional spatial scanning (P2 optimization)
# ---------------------------------------------------------------------------
def _cross_scan_mamba(
    x_2d: Tensor,
    blocks: nn.ModuleList,
    H: int,
    W: int,
) -> Tensor:
    B, C = x_2d.shape[0], x_2d.shape[1]

    scans = torch.stack(
        [
            x_2d,
            torch.flip(x_2d, dims=[2, 3]),
            torch.flip(x_2d, dims=[3]),
            torch.flip(x_2d, dims=[2]),
        ],
        dim=0,
    )  # (4, B, C, H, W)

    scans_batched = scans.reshape(4 * B, C, H, W)
    seq = scans_batched.flatten(2).transpose(1, 2)  # (4*B, L, C)
    for block in blocks:
        seq = block(seq) + seq

    out = seq.transpose(1, 2).reshape(4, B, C, H, W)

    return torch.stack(
        [
            out[0],
            torch.flip(out[1], dims=[2, 3]),
            torch.flip(out[2], dims=[3]),
            torch.flip(out[3], dims=[2]),
        ]
    ).mean(dim=0)


class P3DetailFusion(nn.Module):
    """Encode a high-resolution image and align local 3x3 patches to P3."""

    def __init__(
        self,
        d_model: int,
        channels: int = 32,
        patch_size: int = 3,
        feature_channels: int = 0,
    ):
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError("detail patch_size must be odd")
        self.patch_size = patch_size
        # Keep the first stage compatible with the 32-channel YOLO stem while
        # allowing the second stage to test wider detail representations.
        hidden = min(channels, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.feature_adapter: nn.Module | None = None
        if feature_channels > 0:
            self.feature_adapter = nn.Conv2d(feature_channels, channels, 1)
        token_channels = channels * patch_size * patch_size
        self.cls_proj = nn.Conv2d(token_channels, d_model, 1)
        self.reg_proj = nn.Conv2d(token_channels, d_model, 1)
        self.cls_gate = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.Sigmoid())
        self.reg_gate = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.Sigmoid())

        # Exact v14 identity at initialization. The first optimizer step trains
        # these output projections; gradients then propagate into the encoder.
        nn.init.zeros_(self.cls_proj.weight)
        nn.init.zeros_(self.cls_proj.bias)
        nn.init.zeros_(self.reg_proj.weight)
        nn.init.zeros_(self.reg_proj.bias)

    @torch.no_grad()
    def initialize_from_yolo_stem(self, stem: nn.Module) -> bool:
        """Copy the first YOLO Conv+BN when its channel layout matches."""
        source_conv = getattr(stem, "conv", None)
        source_bn = getattr(stem, "bn", None)
        target_conv = self.encoder[0]
        target_bn = self.encoder[1]
        if (
            not isinstance(source_conv, nn.Conv2d)
            or not isinstance(source_bn, nn.BatchNorm2d)
            or source_conv.weight.shape != target_conv.weight.shape
            or source_bn.weight.shape != target_bn.weight.shape
        ):
            return False
        target_conv.weight.copy_(source_conv.weight)
        target_bn.load_state_dict(source_bn.state_dict())
        return True

    def _sample_tokens(
        self,
        detail_feat: Tensor,
        output_hw: tuple[int, int],
        valid_hw: Tensor | None,
        source_stride: int,
    ) -> Tensor:
        batch, channels, feat_h, feat_w = detail_feat.shape
        out_h, out_w = output_hw
        patch = self.patch_size

        if valid_hw is None:
            valid_hw = detail_feat.new_tensor(
                [[feat_h * source_stride, feat_w * source_stride]],
                dtype=torch.float32,
            ).expand(batch, -1)
        elif valid_hw.shape != (batch, 2):
            raise ValueError(
                f"detail_valid_hw must have shape ({batch}, 2), got {tuple(valid_hw.shape)}"
            )
        valid_hw = valid_hw.to(device=detail_feat.device, dtype=torch.float32)
        valid_feat_h = torch.ceil(valid_hw[:, 0] / source_stride).clamp_(1, feat_h)
        valid_feat_w = torch.ceil(valid_hw[:, 1] / source_stride).clamp_(1, feat_w)

        y_fraction = (
            torch.arange(out_h * patch, device=detail_feat.device, dtype=torch.float32)
            + 0.5
        ) / (out_h * patch)
        x_fraction = (
            torch.arange(out_w * patch, device=detail_feat.device, dtype=torch.float32)
            + 0.5
        ) / (out_w * patch)
        grid_y = (
            2.0 * (y_fraction[None, :, None] * valid_feat_h[:, None, None] / feat_h)
            - 1.0
        )
        grid_x = (
            2.0 * (x_fraction[None, None, :] * valid_feat_w[:, None, None] / feat_w)
            - 1.0
        )
        grid = torch.stack(
            [
                grid_x.expand(-1, out_h * patch, -1),
                grid_y.expand(-1, -1, out_w * patch),
            ],
            dim=-1,
        ).to(dtype=detail_feat.dtype)

        sampled = F.grid_sample(
            detail_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return (
            sampled.view(batch, channels, out_h, patch, out_w, patch)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(batch, channels * patch * patch, out_h, out_w)
        )

    def forward(
        self,
        detail_images: Tensor | None,
        global_context: Tensor,
        valid_hw: Tensor | None = None,
        detail_features: Tensor | None = None,
        feature_stride: int = 8,
    ) -> tuple[Tensor, Tensor]:
        if detail_features is not None:
            if self.feature_adapter is None:
                raise ValueError(
                    "detail_features were provided without detail_feature_channels"
                )
            detail_feat = self.feature_adapter(detail_features)
            source_stride = feature_stride
        else:
            if detail_images is None:
                raise ValueError("detail_images or detail_features must be provided")
            detail_feat = self.encoder(detail_images)
            source_stride = 4
        tokens = self._sample_tokens(
            detail_feat,
            global_context.shape[-2:],
            valid_hw,
            source_stride,
        )
        cls_delta = self.cls_proj(tokens) * self.cls_gate(global_context)
        reg_delta = self.reg_proj(tokens) * self.reg_gate(global_context)
        return cls_delta, reg_delta


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
        use_pixel_shuffle: bool = False,  # enable PixelShuffle learned upsampling (P1 optimization)
        use_cross_scan: bool = False,  # enable 4-directional cross-scan (P2 optimization)
        use_hybrid_head: bool = False,  # hybrid: conv blocks on P3, Mamba on P4/P5 (P3 optimization)
        use_temporal_mamba: bool = False,  # spatio-temporal SSM across T frames (P2 ST optimization)
        use_temporal_attention: bool = False,  # T=4 self-attention over frames (replaces SSM, no flow gate)
        per_channel_a: bool = False,  # per-channel SSM A (d_inner, N); warm-starts from shared (1, N)
        scan_stop_grad: bool = False,  # detach scan output (v14 frozen-SSM training regime)
        use_cuda_graph: bool = False,  # enable CUDA Graph capture and replay
        use_flatten: bool = False,  # skip downsample/upsample — flatten full FPN resolution
        use_detail_fusion: bool = False,
        detail_channels: int = 32,
        detail_patch_size: int = 3,
        detail_feature_channels: int = 0,
    ):
        super().__init__()
        self.nl = len(in_channels)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.stride = torch.tensor(strides, dtype=torch.float32)
        self.no = num_classes + reg_max * 4
        self.spatial_reduction = spatial_reduction
        self.emb_dim = emb_dim
        self.use_pixel_shuffle = use_pixel_shuffle
        self.use_cross_scan = use_cross_scan
        self.use_hybrid_head = use_hybrid_head
        self.use_temporal_mamba = use_temporal_mamba
        self.per_channel_a = per_channel_a
        self.scan_stop_grad = scan_stop_grad
        self.use_detail_fusion = use_detail_fusion
        self.use_flatten = use_flatten
        self.upsample_loaded = use_pixel_shuffle and not use_flatten

        self.input_proj = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels])
        if not use_flatten:
            self.downsample = nn.ModuleList(
                [
                    nn.Conv2d(
                        d_model, d_model, spatial_reduction, stride=spatial_reduction
                    )
                    for _ in range(self.nl)
                ]
            )
        else:
            self.downsample = nn.ModuleList([nn.Identity() for _ in range(self.nl)])

        self.mamba_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MambaBlock(
                            d_model,
                            d_state,
                            per_channel_a=per_channel_a,
                            scan_stop_grad=scan_stop_grad,
                        )
                        for _ in range(num_blocks)
                    ]
                )
                for _ in range(self.nl)
            ]
        )

        # P3: EfficientViT-style depthwise separable conv blocks for hybrid mode
        self.hybrid_conv: nn.ModuleList | None = None
        if use_hybrid_head:
            self.hybrid_conv = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
                        nn.Conv2d(d_model, d_model, 1),
                        nn.SiLU(),
                    )
                    for _ in range(num_blocks)
                ]
            )
        # P2-ST: temporal Mamba blocks — scan across frames for each spatial location
        self.temporal_blocks: nn.ModuleList | None = None
        self.flow_gate_conv: nn.ModuleList | None = None
        if use_temporal_mamba:
            self.temporal_blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            MambaBlock(
                                d_model,
                                d_state,
                                full_rank_c=True,
                                per_channel_a=per_channel_a,
                                scan_stop_grad=scan_stop_grad,
                            )
                            for _ in range(1)
                        ]
                    )
                    for _ in range(self.nl)
                ]
            )
            self.flow_gate_conv = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(d_model + 2, d_model, 3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(d_model, d_model, 3, padding=1),
                        nn.Sigmoid(),
                    )
                    for _ in range(self.nl)
                ]
            )
            # Init last conv so gate≈0: weight=0, bias=-10 → sigmoid(-10)≈0
            # → x_up*(1+0)=x_up (identity). Prevents random gate noise when
            # flow_gate_conv is never activated during cache-based training.
            for gate_seq in self.flow_gate_conv:
                last_conv = gate_seq[-2]  # Conv2d before Sigmoid
                nn.init.zeros_(last_conv.weight)
                nn.init.constant_(last_conv.bias, -10.0)
        # Temporal self-attention (T=4): replaces SSM temporal blocks, no flow gate needed
        self.use_temporal_attention = use_temporal_attention
        self.temporal_attn_blocks: nn.ModuleList | None = None
        if use_temporal_attention:
            self.temporal_attn_blocks = nn.ModuleList(
                [TemporalAttentionBlock(d_model) for _ in range(self.nl)]
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

        # P1: PixelShuffle learned upsampler layer (with 3x3 conv for spatial feature integration)
        if use_flatten:
            self.upsample = nn.ModuleList([nn.Identity() for _ in range(self.nl)])
        elif use_pixel_shuffle:
            self.upsample = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            d_model, d_model * (spatial_reduction**2), 3, padding=1
                        ),
                        nn.PixelShuffle(spatial_reduction),
                    )
                    for _ in range(self.nl)
                ]
            )
        else:
            self.upsample = None

        self.detail_fusion: P3DetailFusion | None = None
        if use_detail_fusion:
            self.detail_fusion = P3DetailFusion(
                d_model=d_model,
                channels=detail_channels,
                patch_size=detail_patch_size,
                feature_channels=detail_feature_channels,
            )

        self.use_cuda_graph = use_cuda_graph
        self._head_compile_enabled: bool = False
        self._head_modules_original: dict[str, nn.Module] = {}
        self._block_compile_enabled: bool = False
        self._block_modules_original: nn.ModuleList | None = None
        # Lazily-captured ``torch.cuda.make_graphed_callables``, keyed by
        # (input shapes, return_embeddings). make_graphed_callables owns the
        # graph pool + capture stream + static I/O, so its replay is robust to
        # the surrounding eval GPU work (GMC / tracker kernels). This replaces a
        # hand-rolled ``torch.cuda.CUDAGraph`` capture whose replay corrupted
        # cls outputs in the full eval pipeline — see
        # docs/research/pipeline/mamba_head_cuda_graph_eval_bug_20260602.md.
        self._graphed_callables: dict = {}

    def initialize_detail_from_yolo_stem(self, stem: nn.Module) -> bool:
        if self.detail_fusion is None:
            return False
        return self.detail_fusion.initialize_from_yolo_stem(stem)

    def load_state_dict(self, state_dict, strict=True):
        # Check if the keys of state_dict contain 'upsample' parameters.
        # This prevents breaking existing checkpoints that were trained without PixelShuffle.
        has_upsample = any("upsample" in k for k in state_dict.keys())
        if self.use_pixel_shuffle and self.upsample is not None:
            self.upsample_loaded = has_upsample
        else:
            self.upsample_loaded = False
            # Strip 'upsample' from state_dict to avoid unexpected keys in strict loading
            state_dict = {k: v for k, v in state_dict.items() if "upsample" not in k}

        # Warm-start A_log between shared (1, N) and per-channel (d_inner, N) layouts.
        # Shared→per-channel: broadcast the single decay schedule to every channel so
        #   a v14 (shared-A) checkpoint initializes an exactly equivalent per-channel
        #   model (init == v14), then fine-tuning learns per-channel deviations.
        # Per-channel→shared: collapse by averaging rows.
        own_state = self.state_dict()
        for k, v in list(state_dict.items()):
            if not k.endswith("A_log") or k not in own_state:
                continue
            tgt = own_state[k]
            if tgt.shape == v.shape:
                continue
            if v.shape[0] == 1 and tgt.shape[0] > 1 and v.shape[1:] == tgt.shape[1:]:
                state_dict[k] = v.expand_as(tgt).contiguous()
            elif tgt.shape[0] == 1 and v.shape[0] > 1 and tgt.shape[1:] == v.shape[1:]:
                state_dict[k] = v.mean(dim=0, keepdim=True).contiguous()

        if not strict:
            # When strict=False, also skip keys with shape mismatches (e.g. loading
            # rank-1 C checkpoint into a model with rank-N C temporal blocks, or vice versa).
            own_state = self.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not (k in own_state and own_state[k].shape != v.shape)
            }

        return super().load_state_dict(state_dict, strict=strict)

    def set_use_cuda_graph(self, enabled: bool) -> None:
        """Enable or disable CUDA Graph capture and replay."""
        self.use_cuda_graph = enabled
        if not enabled:
            self.clear_cuda_graphs()

    def clear_cuda_graphs(self) -> None:
        """Drop all graphed callables (releases their captured graph mempools)."""
        self._graphed_callables.clear()

    def set_head_compile(self, enabled: bool) -> None:
        """Enable or disable ``torch.compile`` on cls_head / reg_head modules.

        Each ``nn.Sequential`` (Conv2d → SiLU → Conv2d) is individually
        compiled with ``mode="default"`` so the fused kernel graph can be
        captured by the outer ``make_graphed_callables`` in the whole-graph
        path.
        """
        if enabled and not self._head_compile_enabled:
            self._head_modules_original["cls_head"] = self.cls_head
            self._head_modules_original["reg_head"] = self.reg_head
            self.cls_head = nn.ModuleList(
                [torch.compile(m, mode="default") for m in self.cls_head]
            )
            self.reg_head = nn.ModuleList(
                [torch.compile(m, mode="default") for m in self.reg_head]
            )
            self._head_compile_enabled = True
        elif not enabled and self._head_compile_enabled:
            if "cls_head" in self._head_modules_original:
                self.cls_head = self._head_modules_original.pop("cls_head")
            if "reg_head" in self._head_modules_original:
                self.reg_head = self._head_modules_original.pop("reg_head")
            self._head_compile_enabled = False

    def set_block_compile(self, enabled: bool) -> None:
        """Enable or disable ``torch.compile`` on MambaBlock modules.

        Compiles each ``MambaBlock.forward`` with ``mode="default"``. The
        custom ``selective_scan_fwd`` CUDA op is registered via
        ``@torch.library.custom_op`` so torch.compile treats it as a
        passthrough, fusing only the surrounding PyTorch ops.
        """
        if enabled and not self._block_compile_enabled:
            self._block_modules_original = self.mamba_blocks
            compiled_blocks = nn.ModuleList()
            for scale_blocks in self.mamba_blocks:
                compiled_blocks.append(
                    nn.ModuleList(
                        [torch.compile(b, mode="default") for b in scale_blocks]
                    )
                )
            self.mamba_blocks = compiled_blocks
            self._block_compile_enabled = True
        elif not enabled and self._block_compile_enabled:
            if self._block_modules_original is not None:
                self.mamba_blocks = self._block_modules_original
                self._block_modules_original = None
            self._block_compile_enabled = False

    def forward(
        self,
        feats: list[Tensor],
        return_embeddings: bool = False,
        T: int | None = None,
        flows: list[Tensor] | None = None,
        detail_images: Tensor | None = None,
        detail_valid_hw: Tensor | None = None,
        detail_features: Tensor | None = None,
        detail_feature_stride: int = 8,
    ) -> (
        tuple[list[Tensor], list[Tensor]]
        | tuple[list[Tensor], list[Tensor], list[Tensor]]
    ):
        # Fall back to eager execution under any of these conditions:
        # 1. CUDA Graph not explicitly enabled
        # 2. PyTorch is tracing (e.g. exporting to TorchScript)
        # 3. Inputs are not on CUDA
        # 4. Gradients are enabled (i.e. during training)
        is_tracing = False
        try:
            import torch.jit

            if torch.jit.is_tracing():
                is_tracing = True
        except Exception:
            pass

        if (
            not self.use_cuda_graph
            or is_tracing
            or not feats[0].is_cuda
            or torch.is_grad_enabled()
            # The graph only covers the static single-frame path. The temporal /
            # flow-gated path has data-dependent shapes (T_buf grows while the
            # ring buffer fills, flows toggle on once GMC has ≥2 mats), so it is
            # left eager.
            or T is not None
            or flows is not None
            or detail_images is not None
            or detail_features is not None
        ):
            return self._forward_eager(
                feats,
                return_embeddings,
                T,
                flows,
                detail_images,
                detail_valid_hw,
                detail_features,
                detail_feature_stride,
            )

        return self._graphed_forward(feats, return_embeddings)

    def _graphed_forward(
        self,
        feats: list[Tensor],
        return_embeddings: bool,
    ) -> (
        tuple[list[Tensor], list[Tensor]]
        | tuple[list[Tensor], list[Tensor], list[Tensor]]
    ):
        """Single-frame head replayed via ``torch.cuda.make_graphed_callables``.

        Collapses the ~278 eager kernel launches of one head pass into a single
        CUDA-graph replay. The graphed callable copies the live ``feats`` into
        its static input surface on each call, so passing fresh tensors per
        frame is fine. Captured lazily per input shape; make_graphed_callables
        manages the graph pool / capture stream / static I/O so replay is
        isolated from the surrounding eval GPU work.
        """
        feats = [f.contiguous() for f in feats]
        key = (tuple(tuple(f.shape) for f in feats), return_embeddings)

        graphed = self._graphed_callables.get(key)
        if graphed is None:
            if len(self._graphed_callables) >= 10:
                self.clear_cuda_graphs()

            sample = tuple(f.clone() for f in feats)

            def _head_fn(*fs: Tensor) -> tuple[Tensor, ...]:
                out = self._forward_eager(list(fs), return_embeddings, None, None)
                # Flatten (cls, reg[, emb]) lists into one tuple of tensors so
                # make_graphed_callables' pytree handling captures every output.
                flat: list[Tensor] = []
                for group in out:
                    flat.extend(group)
                return tuple(flat)

            print(
                f"🧠 [MambaHead] Capturing graphed callable for shapes "
                f"{[list(s) for s in key[0]]} (return_embeddings={return_embeddings})"
            )
            graphed = torch.cuda.make_graphed_callables(_head_fn, sample)
            self._graphed_callables[key] = graphed

        flat_out = graphed(*feats)
        n = self.nl
        cls_preds = list(flat_out[:n])
        reg_preds = list(flat_out[n : 2 * n])
        if return_embeddings and self.emb_head is not None:
            emb_preds = list(flat_out[2 * n :])
            return cls_preds, reg_preds, emb_preds
        return cls_preds, reg_preds

    def _forward_eager(
        self,
        feats: list[Tensor],
        return_embeddings: bool = False,
        T: int | None = None,
        flows: list[Tensor] | None = None,
        detail_images: Tensor | None = None,
        detail_valid_hw: Tensor | None = None,
        detail_features: Tensor | None = None,
        detail_feature_stride: int = 8,
    ) -> (
        tuple[list[Tensor], list[Tensor]]
        | tuple[list[Tensor], list[Tensor], list[Tensor]]
    ):
        cls_preds: list[Tensor] = []
        reg_preds: list[Tensor] = []
        emb_preds: list[Tensor] = []

        for i, x in enumerate(feats):
            B_merged, C, H, W = x.shape
            T_frames = T if T is not None else 1
            B = B_merged // T_frames

            x_proj = self.input_proj[i](x)
            x_small = self.downsample[i](x_proj)
            _, _, Hs, Ws = x_small.shape

            if self.use_hybrid_head and i == 0:
                x_proc = x_small
                for block in self.hybrid_conv:
                    x_proc = block(x_proc) + x_proc
                x_up = x_proc
            elif self.use_cross_scan:
                x_up = _cross_scan_mamba(x_small, self.mamba_blocks[i], Hs, Ws)
            else:
                x_seq = x_small.flatten(2).transpose(1, 2)
                for block in self.mamba_blocks[i]:
                    x_seq = block(x_seq) + x_seq
                x_up = x_seq.transpose(1, 2).reshape(B_merged, -1, Hs, Ws)

            if (
                self.use_pixel_shuffle
                and getattr(self, "upsample_loaded", False)
                and not self.use_flatten
            ):
                x_up = self.upsample[i](x_up)
            elif not self.use_flatten:
                x_up = F.interpolate(
                    x_up, size=(H, W), mode="bilinear", align_corners=False
                )

            if (
                self.use_temporal_mamba
                and self.temporal_blocks is not None
                and T_frames > 1
            ):
                if (
                    flows is not None
                    and self.flow_gate_conv is not None
                    and i < len(flows)
                ):
                    flow_i = flows[i]  # (B*T, 2, Hf, Wf) or (B*T, 2, H, W)
                    if flow_i.shape[2:] != (H, W):
                        flow_i = F.interpolate(
                            flow_i, size=(H, W), mode="bilinear", align_corners=False
                        )
                    gate_input = torch.cat([x_up, flow_i], dim=1)
                    gate = self.flow_gate_conv[i](gate_input)
                    x_up = x_up * (1.0 + gate)
                x_up_t = x_up.reshape(B, T_frames, -1, H, W).permute(0, 2, 3, 4, 1)
                x_up_t = x_up_t.reshape(B * H * W, T_frames, -1)
                for block in self.temporal_blocks[i]:
                    x_up_t = block(x_up_t) + x_up_t
                x_up = x_up_t.reshape(B, H, W, T_frames, -1).permute(0, 3, 4, 1, 2)
                x_up = x_up.reshape(B_merged, -1, H, W)

            # Temporal attention path (T=4 self-attention, no flow gate)
            if (
                self.use_temporal_attention
                and self.temporal_attn_blocks is not None
                and T_frames > 1
            ):
                x_up_t = x_up.reshape(B, T_frames, -1, H, W).permute(0, 2, 3, 4, 1)
                x_up_t = x_up_t.reshape(B * H * W, T_frames, -1)
                x_up_t = self.temporal_attn_blocks[i](x_up_t)
                x_up = x_up_t.reshape(B, H, W, T_frames, -1).permute(0, 3, 4, 1, 2)
                x_up = x_up.reshape(B_merged, -1, H, W)
            x_cls_proj = x_proj
            x_reg_proj = x_proj
            if i == 0 and (detail_images is not None or detail_features is not None):
                if self.detail_fusion is None:
                    raise ValueError(
                        "detail input was provided but use_detail_fusion is disabled"
                    )
                detail_batch = (
                    detail_features.shape[0]
                    if detail_features is not None
                    else detail_images.shape[0]  # type: ignore[union-attr]
                )
                if detail_batch != B_merged:
                    raise ValueError(
                        "detail input batch must match the merged feature batch: "
                        f"{detail_batch} != {B_merged}"
                    )
                cls_delta, reg_delta = self.detail_fusion(
                    detail_images,
                    x_up,
                    detail_valid_hw,
                    detail_features=detail_features,
                    feature_stride=detail_feature_stride,
                )
                x_cls_proj = x_proj + cls_delta
                x_reg_proj = x_proj + reg_delta

            x_cls = torch.cat([x_cls_proj, x_up], dim=1)
            x_reg = torch.cat([x_reg_proj, x_up], dim=1)
            cls_preds.append(self.cls_head[i](x_cls))
            reg_preds.append(self.reg_head[i](x_reg))
            if self.emb_head is not None:
                x_emb = torch.cat([x_proj, x_up], dim=1)
                emb_preds.append(self.emb_head[i](x_emb))

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
