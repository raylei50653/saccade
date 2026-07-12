"""Consumer-A bridge estimator fidelity — research contract (D0).

Default-off observational surface.  Does **not** change production bridge
decisions, thresholds, ordering, lifecycle, or presets.

Production surface named by the audit (read-only reference):
  ``src/tracking/tracker_gpu.cu`` :: ``relink_bidir_propose_kernel``
  primary quantity ``bdist`` vs production threshold ``0.4``.

Two statuses, never conflated
-----------------------------
* **Legacy v1 reconstruction packet (2026-07-11):** status
  ``D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE``.  It is a **kernel-formula
  reconstruction** from no-relink MOT tracklets — *not* runtime Consumer-A
  capture of ``foot_ring`` / ``ema_h`` / float32 kernel outputs.  That packet
  stays frozen and its constants below keep their original meaning; they are
  prefixed ``V1_LEGACY_`` so they cannot be read as the current status.
* **Issue #112 (current): COMPLETE.**  A shadow bridge (propose + capture with
  the commit kernel skipped) produces output byte-identical to a bridge-off run
  while emitting real float32 CUDA values, so runtime capture now exists and is
  joinable.  Terminal: **T2 PROXY_UNFAITHFUL** (the issue's own vocabulary:
  ``not_fidelity_aligned``) — ``score_m_bridge`` is an **offline quantity** and
  must not be used as an equivalent of production ``bdist``.
  See ``docs/modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md``
  and the binding ``docs/research/eval/runtime_quantity_fidelity_protocol.md``.

Reconstruction metrics remain diagnostics only and must never be labeled
``D4 exact captured Consumer-A``.

This module never mounts a production policy change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

# ── Production constants (must match headline preset / kernel defaults) ──────
PRODUCTION_BRIDGE_PX: Final[float] = 0.4
PRODUCTION_BRIDGE_AT: Final[int] = 4
PRODUCTION_BRIDGE_ANCHOR: Final[str] = "adaptive"  # mode 2
PRODUCTION_BRIDGE_ANCHOR_RATE: Final[float] = 0.03
PRODUCTION_EMA_ALPHA: Final[float] = 0.05  # tracker_gpu.cu update_foot_history_kernel
SPEED_WEIGHT_REF: Final[float] = 0.12  # s_lost reference for w

# Research audit defaults — production path must leave these off.
RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT: Final[bool] = False
NATIVE_CUDA_BRIDGE_FIDELITY_CAPTURE_IMPLEMENTED: Final[bool] = True

# ── Issue #112: CURRENT status ──────────────────────────────────────────────
# Runtime capture exists (shadow bridge: propose + capture, commit skipped) and
# the fidelity question is answered. Use these, not the V1_LEGACY_* constants,
# for any statement about Issue #112 today.
ISSUE_112_CURRENT_STATUS: Final[str] = "complete_runtime_shadow_capture"
ISSUE_112_TERMINAL: Final[str] = "T2_PROXY_UNFAITHFUL"  # issue: not_fidelity_aligned
# `score_m_bridge` is an offline quantity. It is NOT an equivalent of the
# production CUDA `bdist` and must not be cited as one.
SCORE_M_BRIDGE_IS_PRODUCTION_EQUIVALENT: Final[bool] = False

# Capture mode vocabulary (sealed packet must use reconstruction, not "exact").
CAPTURE_MODE_RECONSTRUCTION: Final[str] = "kernel_formula_reconstruction"
CAPTURE_MODE_RUNTIME_CUDA: Final[str] = "runtime_cuda_event_ring"  # reserved

# ── Event-key contract versions ─────────────────────────────────────────────
#
# v1 (LEGACY, frozen): the 5-field key used by the sealed reconstruction packet
# in ``run_d0_bridge_fidelity.py``. It is retained *only* so historical sealed
# packets keep validating under their original semantics. It must never be
# reinterpreted: two of its fields are unsound for runtime capture.
#
#   * ``lost_id`` / ``cand_id`` are tracker-**local** ids. The evaluator remaps
#     them to global ids before writing MOT output, and the offline pair cohort
#     is built from that MOT output -- so local ids do not address the cohort.
#     Joining on them yields *false* matches wherever the remap happens to be
#     the identity (observed: MOT17-02, MOT17-05).
#   * ``lost_last_frame`` / ``cand_first_frame`` are not real frame indices. The
#     kernel derives them from a capture-local counter minus a track age
#     (``fidelity_frame - la``); the tracker has no absolute frame counter, so
#     they underflow to negative values. They are unusable as identity.
#
# v2 (runtime shadow fidelity): global-id key, no frame fields.
EVENT_KEY_VERSION_V1_LEGACY: Final[str] = "d0_event_key_v1_local_legacy"
EVENT_KEY_VERSION_V2: Final[str] = "d0_event_key_v2_global"

EVENT_KEY_FIELDS_V2: Final[tuple[str, ...]] = (
    "seq",
    "lost_global_id",
    "cand_global_id",
)

# Fields the v2 packet must NOT emit: they looked valid but carried impossible
# (negative) frame indices. Dropped rather than deprecated-in-place.
EVENT_KEY_V1_UNSOUND_FIELDS: Final[tuple[str, ...]] = (
    "lost_last_frame",
    "cand_first_frame",
)

# Exhaustive, mutually exclusive partition of every captured runtime proposal.
# Fidelity may only be computed on MATCHED; the other two bound how far a
# fidelity conclusion extrapolates and must never enter an agreement
# denominator.
PARTITION_MATCHED: Final[str] = "matched"  # joins the offline cohort
PARTITION_COHORT_GAP: Final[str] = "cohort_gap"  # ids emitted, pair not enumerated
PARTITION_UNEMITTED: Final[str] = "unemitted"  # id never reached MOT output
PARTITIONS: Final[tuple[str, ...]] = (
    PARTITION_MATCHED,
    PARTITION_COHORT_GAP,
    PARTITION_UNEMITTED,
)


def event_key_v2(seq: str, lost_global_id: int, cand_global_id: int) -> str:
    """Canonical v2 event key. Global ids only -- local ids are a contract error."""
    return f"{seq}|{int(lost_global_id)}|{int(cand_global_id)}"


# ── V1 LEGACY (frozen) ──────────────────────────────────────────────────────
# Status of the sealed 2026-07-11 *reconstruction* packet, which had no runtime
# capture. These describe THAT PACKET, not Issue #112 today -- see
# ISSUE_112_CURRENT_STATUS above.
#
# The names are kept because the sealed packet's runner imports them verbatim;
# renaming them would mutate a frozen artifact. The V1_LEGACY_* aliases are the
# preferred spelling for any new code.
V1_LEGACY_ISSUE_112_STATUS: Final[str] = "incomplete_runtime_capture_unavailable"
V1_LEGACY_PACKET_STATUS_FAIL_CLOSED: Final[str] = "D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE"
V1_LEGACY_PRIMARY_FAIL_REASON: Final[str] = "runtime_capture_unavailable"
V1_LEGACY_LIVE_CUDA_EVENT_RING_IMPLEMENTED: Final[bool] = False

# Frozen import surface of the sealed v1 runner. Do not rename.
ISSUE_112_STATUS: Final[str] = V1_LEGACY_ISSUE_112_STATUS
PACKET_STATUS_FAIL_CLOSED: Final[str] = V1_LEGACY_PACKET_STATUS_FAIL_CLOSED
PRIMARY_FAIL_REASON: Final[str] = V1_LEGACY_PRIMARY_FAIL_REASON
LIVE_CUDA_EVENT_RING_IMPLEMENTED: Final[bool] = (
    V1_LEGACY_LIVE_CUDA_EVENT_RING_IMPLEMENTED
)

ANCHOR_MODE: Final[dict[str, int]] = {
    "center": 0,
    "foot": 1,
    "adaptive": 2,
}

HEADLINE_PRESET_REL: Final[str] = "configs/presets/mamba_whole_graph_m.yaml"


@dataclass(frozen=True)
class BridgeEstimate:
    """One kernel-formula reconstruction of the Consumer-A bridge score."""

    bdist: float
    dist_h: float
    fwd_r: float
    bwd_r: float
    la: int
    gap: int
    bridge_at: int
    h_ref: float
    ema_lost: float
    ema_cand: float
    v_lost_x: float
    v_lost_y: float
    v_cand_x: float
    v_cand_y: float
    ax: float
    ay: float
    cx0: float
    cy0: float
    s_lost: float
    w: float
    production_threshold: float = PRODUCTION_BRIDGE_PX


def production_safe(bdist: float, threshold: float = PRODUCTION_BRIDGE_PX) -> bool:
    """Production predicate: accept if bdist <= bridge_px."""
    return float(bdist) <= float(threshold)


def decide_bridge(
    bdist: float,
    *,
    threshold: float = PRODUCTION_BRIDGE_PX,
    audit_enabled: bool = RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT,
) -> bool:
    """Decision path used by unit tests: audit flag must not affect accept/reject."""
    del audit_enabled  # observational only
    return production_safe(bdist, threshold)


def bridge_vel4(samples: Sequence[float]) -> float:
    """CUDA ``bridge_vel4``: v = (3 y3 + y2 − y1 − 3 y0) / 10."""
    y0, y1, y2, y3 = (float(samples[i]) for i in range(4))
    return (3.0 * y3 + y2 - y1 - 3.0 * y0) / 10.0


def bridge_linres4(samples: Sequence[float]) -> float:
    """CUDA ``bridge_linres4`` residual sum of squares vs OLS line."""
    y = [float(samples[i]) for i in range(4)]
    ybar = 0.25 * sum(y)
    sxy = (
        -1.5 * (y[0] - ybar)
        - 0.5 * (y[1] - ybar)
        + 0.5 * (y[2] - ybar)
        + 1.5 * (y[3] - ybar)
    )
    slope = sxy / 5.0
    res = 0.0
    for i in range(4):
        fit = ybar + slope * (float(i) - 1.5)
        d = y[i] - fit
        res += d * d
    return res


def bridge_anchor4(
    ring4: Sequence[tuple[float, float, float]],
    *,
    anchor_mode: int,
    rate_gate: float,
    endpoint_idx: int,
) -> tuple[float, float, float, float]:
    """CUDA ``bridge_anchor4`` host replica.

    ``ring4`` is four chronological ``(cx, cy, h)`` samples (foot-ring stride-3).
    Returns ``(ax, ay, vx, vy)``.
    """
    if len(ring4) != 4:
        raise ValueError("bridge_anchor4 requires exactly 4 samples")
    cx = [float(p[0]) for p in ring4]
    cy = [float(p[1]) for p in ring4]
    h = [float(p[2]) for p in ring4]
    yt = [c - 0.5 * hh for c, hh in zip(cy, h)]
    yb = [c + 0.5 * hh for c, hh in zip(cy, h)]
    hbar = 0.25 * sum(h)
    vx = bridge_vel4(cx)
    ax = cx[endpoint_idx]
    use_edges = anchor_mode == 2
    if use_edges and rate_gate > 0.0:
        dh = (abs(h[1] - h[0]) + abs(h[2] - h[1]) + abs(h[3] - h[2])) / 3.0
        if dh / (hbar + 1e-3) <= rate_gate:
            use_edges = False
    if anchor_mode == 1:
        vy = bridge_vel4(yb)
        ay = yb[endpoint_idx]
    elif use_edges:
        hn = hbar * hbar + 1e-3
        wt = 1.0 / (bridge_linres4(yt) / hn + 0.01)
        wb = 1.0 / (bridge_linres4(yb) / hn + 0.01)
        ws = wt + wb
        vy = (wt * bridge_vel4(yt) + wb * bridge_vel4(yb)) / ws
        ay = (wt * yt[endpoint_idx] + wb * yb[endpoint_idx]) / ws
    else:
        vy = bridge_vel4(cy)
        ay = cy[endpoint_idx]
    return ax, ay, vx, vy


def ema_height(heights: Sequence[float], alpha: float = PRODUCTION_EMA_ALPHA) -> float:
    """Causal EMA matching ``update_foot_history_kernel`` (seed = first height)."""
    if not heights:
        raise ValueError("ema_height requires at least one height")
    e = float(heights[0])
    keep = 1.0 - alpha
    for h in heights[1:]:
        e = keep * e + alpha * float(h)
    return e


def window_mean_velocity(
    feet: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Offline builder ``_velocity``: mean per-frame foot velocity (unit dt)."""
    if len(feet) < 2:
        return 0.0, 0.0
    vx = vy = 0.0
    n = 0
    for (x0, y0), (x1, y1) in zip(feet[:-1], feet[1:]):
        vx += x1 - x0
        vy += y1 - y0
        n += 1
    return (vx / n, vy / n) if n else (0.0, 0.0)


def speed_weighted_bdist(
    *,
    lx: float,
    ly: float,
    cx0: float,
    cy0: float,
    vxl: float,
    vyl: float,
    vxc: float,
    vyc: float,
    horizon: float,
    h_ref: float,
    bridge_dir_bonus: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """CUDA speed-weighted bridge score, including the optional direction bonus.

    Returns ``(bdist, dist_h, fwd_r, bwd_r, s_lost, w)``.
    """
    href = max(float(h_ref), 1.0)
    la = float(horizon)
    fwd_x = lx + vxl * la
    fwd_y = ly + vyl * la
    bwd_x = cx0 - vxc * la
    bwd_y = cy0 - vyc * la
    fwd_r = math.hypot(fwd_x - cx0, fwd_y - cy0) / href
    bwd_r = math.hypot(bwd_x - lx, bwd_y - ly) / href
    dist_h = math.hypot(lx - cx0, ly - cy0) / href
    s_lost = math.hypot(vxl, vyl) / href
    w = math.sqrt(min(max(s_lost / SPEED_WEIGHT_REF, 0.0), 1.0))
    bdist = w * 0.5 * (fwd_r + bwd_r) + (1.0 - w) * dist_h
    # Keep this branch line-for-line aligned with relink_bidir_propose_kernel.
    # It is normally dormant in the active preset, but an R1 replay cannot
    # silently change the quantity when a capture records a non-zero bonus.
    if bridge_dir_bonus > 0.0:
        sl = math.hypot(vxl, vyl)
        sc = math.hypot(vxc, vyc)
        min_speed = min(sl, sc)
        speed_trust = min(min_speed / max(href * 0.005, 1e-3), 1.0)
        if speed_trust > 0.0:
            cos_sim = (vxl * vxc + vyl * vyc) / max(sl * sc, 1e-9)
            if cos_sim > 0.5:
                ux = vxl / sl + vxc / sc
                uy = vyl / sl + vyc / sc
                un = math.hypot(ux, uy)
                ux /= un
                uy /= un
                px, py = -uy, ux
                fe_x = (lx + vxl * la) - cx0
                fe_y = (ly + vyl * la) - cy0
                fwd_cross = abs(fe_x * px + fe_y * py) / href
                be_x = (cx0 - vxc * la) - lx
                be_y = (cy0 - vyc * la) - ly
                bwd_cross = abs(be_x * px + be_y * py) / href
                bdist_dir = 0.5 * (fwd_cross + bwd_cross)
                gap_scale = min(la / 30.0, 1.0)
                alpha = bridge_dir_bonus * cos_sim * cos_sim * speed_trust * gap_scale
                alpha = min(alpha, 1.0)
                bdist = bdist * (1.0 - alpha) + bdist_dir * alpha
    return bdist, dist_h, fwd_r, bwd_r, s_lost, w


def midpoint_bridge_dist(
    *,
    lx: float,
    ly: float,
    cx0: float,
    cy0: float,
    vxl: float,
    vyl: float,
    vxc: float,
    vyc: float,
    gap: float,
    h_ref: float,
) -> float:
    """Offline builder midpoint ``bridge_dist`` (not Consumer-A bdist)."""
    href = max(float(h_ref), 1.0)
    half = 0.5 * float(gap)
    mlx = lx + vxl * half
    mly = ly + vyl * half
    mcx = cx0 - vxc * half
    mcy = cy0 - vyc * half
    return math.hypot(mlx - mcx, mly - mcy) / href


def consumer_a_estimate_from_rings(
    lost_ring: Sequence[tuple[float, float, float]],
    cand_ring: Sequence[tuple[float, float, float]],
    *,
    gap: int,
    ema_lost: float,
    ema_cand: float,
    bridge_at: int = PRODUCTION_BRIDGE_AT,
    anchor_mode: int = ANCHOR_MODE[PRODUCTION_BRIDGE_ANCHOR],
    rate_gate: float = PRODUCTION_BRIDGE_ANCHOR_RATE,
    bridge_dir_bonus: float = 0.0,
) -> BridgeEstimate:
    """Kernel-formula reconstruction on pre-extracted rings + EMA heights.

    Matches ``relink_bidir_propose_kernel`` algebra on reconstructed inputs:
    * candidate requires >=4 ring samples (kernel early-return otherwise);
    * lost with >=4 uses ``bridge_anchor4`` on last-4;
    * lost with 1–3 samples anchors the last point with zero velocity.

    This is **not** a dump of live CUDA ``foot_ring`` / ``ema_h``.
    """
    if len(cand_ring) < 4:
        raise ValueError("Consumer-A path requires >=4 cand foot-ring samples")
    if len(lost_ring) < 1:
        raise ValueError("Consumer-A path requires >=1 lost foot-ring sample")
    la = int(gap) + int(bridge_at) - 1
    if len(lost_ring) >= 4:
        lx, ly, vxl, vyl = bridge_anchor4(
            lost_ring[-4:],
            anchor_mode=anchor_mode,
            rate_gate=rate_gate,
            endpoint_idx=3,
        )
    else:
        # tracker_gpu.cu short-lost branch: last point, zero velocity.
        lc_x, lc_y, lh = lost_ring[-1]
        lx = float(lc_x)
        ly = float(lc_y + 0.5 * lh) if anchor_mode == 1 else float(lc_y)
        vxl, vyl = 0.0, 0.0
    cx0, cy0, vxc, vyc = bridge_anchor4(
        cand_ring[:4],
        anchor_mode=anchor_mode,
        rate_gate=rate_gate,
        endpoint_idx=0,
    )
    h_ref = max(0.5 * (float(ema_lost) + float(ema_cand)), 1.0)
    bdist, dist_h, fwd_r, bwd_r, s_lost, w = speed_weighted_bdist(
        lx=lx,
        ly=ly,
        cx0=cx0,
        cy0=cy0,
        vxl=vxl,
        vyl=vyl,
        vxc=vxc,
        vyc=vyc,
        horizon=float(la),
        h_ref=h_ref,
        bridge_dir_bonus=float(bridge_dir_bonus),
    )
    return BridgeEstimate(
        bdist=bdist,
        dist_h=dist_h,
        fwd_r=fwd_r,
        bwd_r=bwd_r,
        la=la,
        gap=int(gap),
        bridge_at=int(bridge_at),
        h_ref=h_ref,
        ema_lost=float(ema_lost),
        ema_cand=float(ema_cand),
        v_lost_x=vxl,
        v_lost_y=vyl,
        v_cand_x=vxc,
        v_cand_y=vyc,
        ax=lx,
        ay=ly,
        cx0=cx0,
        cy0=cy0,
        s_lost=s_lost,
        w=w,
    )
