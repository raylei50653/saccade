#!/usr/bin/env python3
"""D0 — Consumer-A bridge estimator fidelity audit (Issue #112).

This PR path is **fail-closed on runtime capture unavailable**.

What is sealed here:
  kernel-formula reconstruction from frozen pairs + no-relink MOT substrate
  same-event exact-key join · S_A metrics as **reconstruction diagnostics**
  single three-value terminal verdict (forced not_fidelity_aligned while
  LIVE_CUDA_EVENT_RING_IMPLEMENTED is False)

What is NOT sealed:
  live CUDA Consumer-A event capture of foot_ring / ema_h / float32 bdist
  Issue #112 runtime fidelity completion
  D4 "exact captured Consumer-A" claim

Explicitly out of scope:
  A1–A8 tables · V1–V5 · threshold search · offline estimator repair ·
  production / preset / lifecycle / bridge_px changes · Phase B ·
  E_motion claims · runtime hook acceptance
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[6]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from saccade.perception.eval.consumer_a_bridge_fidelity import (  # noqa: E402
    ANCHOR_MODE,
    CAPTURE_MODE_RECONSTRUCTION,
    HEADLINE_PRESET_REL,
    ISSUE_112_STATUS,
    LIVE_CUDA_EVENT_RING_IMPLEMENTED,
    PACKET_STATUS_FAIL_CLOSED,
    PRIMARY_FAIL_REASON,
    PRODUCTION_BRIDGE_ANCHOR,
    PRODUCTION_BRIDGE_ANCHOR_RATE,
    PRODUCTION_BRIDGE_AT,
    PRODUCTION_BRIDGE_PX,
    RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT,
    bridge_anchor4,
    consumer_a_estimate_from_rings,
    ema_height,
    midpoint_bridge_dist,
    speed_weighted_bdist,
    window_mean_velocity,
)

PACKET_DIR = Path(__file__).resolve().parent
CANONICAL_PAIRS = Path(
    "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
)
SOURCE_SHA256 = "0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17"
SUBSTRATE_MOT_DIR = Path("results/MOT17_eval_m_b1_substrate_20260709T092543Z")
KERNEL_SOURCE = Path("src/tracking/tracker_gpu.cu")
HEADLINE_PRESET = Path(HEADLINE_PRESET_REL)
PACKET_STATUS = PACKET_STATUS_FAIL_CLOSED
PRIMARY_SUPPORT = "gt_valid && 1 <= gap <= 26"
# Metrics compare offline atoms to reconstruction quantities — not runtime CA.
RECONSTRUCTION_QUANTITY_NOTE = (
    "reconstruction_diagnostic_not_runtime_consumer_a_capture"
)
BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPS = 400
CLUSTER_QUANTILES = (0.50, 0.70, 0.85, 0.90, 0.95)
BOUNDARY_LO = 0.35
BOUNDARY_HI = 0.45
EVENT_KEY_FIELDS = (
    "seq",
    "lost_id",
    "cand_id",
    "lost_last_frame",
    "cand_first_frame",
)
CAPTURE_FIELDS = (
    "event_key",
    "seq",
    "lost_id",
    "cand_id",
    "lost_last_frame",
    "cand_first_frame",
    "gap",
    "bridge_at",
    "la",
    "bdist",
    "dist_h",
    "fwd_r",
    "bwd_r",
    "v_lost_x",
    "v_lost_y",
    "v_cand_x",
    "v_cand_y",
    "ax",
    "ay",
    "cx0",
    "cy0",
    "ema_lost",
    "ema_cand",
    "h_ref",
    "s_lost",
    "w",
    "production_threshold",
    "capture_mode",
    "evidence_role",
    "anchor_mode",
    "anchor_rate",
)
JOIN_FIELDS = (
    "event_key",
    "seq",
    "lost_id",
    "cand_id",
    "lost_last_frame",
    "cand_first_frame",
    "gap",
    "gt_match",
    "gt_valid",
    "join_status",
    "offline_bridge_dist",
    "offline_dist_h",
    "offline_fwd_r",
    "offline_bwd_r",
    "offline_h_ref",
    "offline_lost_exit_speed",
    "recon_bdist",
    "recon_dist_h",
    "recon_fwd_r",
    "recon_bwd_r",
    "recon_h_ref",
    "recon_la",
    "recon_ema_lost",
    "recon_ema_cand",
    "recon_v_lost_x",
    "recon_v_lost_y",
    "recon_v_cand_x",
    "recon_v_cand_y",
    "s1_aggregation",
    "s2_anchor_endpoints",
    "s3_velocity",
    "s4_horizon",
    "s5_normalization",
)

# Frozen terminal gates (predeclared; do not retune after seeing data).
GATE_SPEARMAN_THRESHOLD = {
    "overall": 0.95,
    "gt": 0.90,
    "fp": 0.95,
}
GATE_SPEARMAN_RANK_ONLY = {
    "overall": 0.85,
    "gt": 0.75,
    "fp": 0.85,
}
GATE_Q85_ABS = 0.04
GATE_PRED_AGREE = 0.95
GATE_GT_SAFE_UNSAFE_COUNT = 1
GATE_GT_SAFE_UNSAFE_RATE = 0.02
GATE_GAP_PRED_AGREE = 0.90
GATE_GAP_SPEARMAN_GT = 0.65
GATE_GT_COVERAGE = 1.0
GATE_OVERALL_COVERAGE = 0.98


# ── IO helpers ────────────────────────────────────────────────────────────────


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def event_key_from_row(row: dict[str, Any]) -> str:
    return "|".join(str(row[field]) for field in EVENT_KEY_FIELDS)


def _stable_float(value: Any, ndigits: int = 10) -> float:
    """Quantize floats so capture → reload → metrics is byte-stable."""
    x = float(value)
    if not math.isfinite(x):
        return float("nan")
    return round(x, ndigits)


def _fmt_cell(value: Any) -> str:
    """Deterministic cell serialization for byte-stable CSV rebuilds."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if not math.isfinite(x):
            return "nan"
        # Fixed decimal places (matches _stable_float) for round-trip equality.
        return f"{x:.10f}".rstrip("0").rstrip(".") if x != 0 else "0"
    return str(value)


def write_gzip_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as binary:
        with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as gz:
            buffer = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _fmt_cell(row.get(field, "")) for field in fieldnames}
                )
            buffer.flush()
            buffer.detach()
    return sha256(path)


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _fmt_cell(row.get(field, "")) for field in fieldnames}
            )
    return sha256(path)


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256(path)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_headline_preset_bridge() -> dict[str, Any]:
    """Load and validate bridge fields from the actual m headline preset file."""
    path = REPO / HEADLINE_PRESET
    if not path.is_file():
        raise FileNotFoundError(f"headline preset missing: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Fields written in mamba_whole_graph_m.yaml
    file_fields = {
        "relink_bridge_enabled": raw.get("relink_bridge_enabled"),
        "relink_bridge_px": raw.get("relink_bridge_px"),
        "relink_bridge_margin": raw.get("relink_bridge_margin"),
        "relink_bridge_h_lo": raw.get("relink_bridge_h_lo"),
        "relink_bridge_h_hi": raw.get("relink_bridge_h_hi"),
        "relink_bridge_spatial_gate": raw.get("relink_bridge_spatial_gate"),
        "relink_bridge_dir_bonus": raw.get("relink_bridge_dir_bonus"),
    }
    # Schema defaults used when preset omits keys (config.py TRACKING defaults).
    resolved = {
        **file_fields,
        "relink_bridge_at": raw.get("relink_bridge_at", PRODUCTION_BRIDGE_AT),
        "relink_bridge_anchor": raw.get(
            "relink_bridge_anchor", PRODUCTION_BRIDGE_ANCHOR
        ),
        "relink_bridge_anchor_rate": raw.get(
            "relink_bridge_anchor_rate", PRODUCTION_BRIDGE_ANCHOR_RATE
        ),
    }
    # Fail closed if headline production threshold drifts.
    if float(resolved["relink_bridge_px"]) != float(PRODUCTION_BRIDGE_PX):
        raise ValueError(
            f"headline preset bridge_px={resolved['relink_bridge_px']} "
            f"!= production constant {PRODUCTION_BRIDGE_PX}"
        )
    if float(resolved["relink_bridge_dir_bonus"]) != 0.0:
        raise ValueError(
            "headline preset dir_bonus must be 0.0 for D0 dir-bonus-off algebra"
        )
    return {
        "preset_path": str(HEADLINE_PRESET),
        "preset_file_sha256": sha256(path),
        "file_fields": file_fields,
        "resolved_bridge": resolved,
        "schema_default_note": (
            "bridge_at/anchor/anchor_rate omitted in yaml → schema defaults "
            f"at={PRODUCTION_BRIDGE_AT} anchor={PRODUCTION_BRIDGE_ANCHOR} "
            f"rate={PRODUCTION_BRIDGE_ANCHOR_RATE}"
        ),
    }


def preset_config_hash() -> str:
    """SHA-256 of the actual headline preset file bytes (not hand-written constants)."""
    return sha256(REPO / HEADLINE_PRESET)


# ── Track / capture ───────────────────────────────────────────────────────────


def load_tracks(path: Path) -> dict[int, list[tuple[int, float, float, float]]]:
    tracks: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frm, tid = int(parts[0]), int(parts[1])
            x, y, w, h = (float(parts[i]) for i in range(2, 6))
            tracks[tid].append((frm, x + w / 2.0, y + h / 2.0, h))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return dict(tracks)


def _traj_slice_for_pair(
    traj: list[tuple[int, float, float, float]],
    *,
    last_or_first: str,
    n: int,
) -> list[tuple[float, float, float]]:
    if last_or_first == "last":
        seg = traj[-n:] if len(traj) >= n else traj
    else:
        seg = traj[:n] if len(traj) >= n else traj
    return [(cx, cy, h) for _f, cx, cy, h in seg]


def _feet_from_ring(
    ring: Sequence[tuple[float, float, float]],
) -> list[tuple[float, float]]:
    return [(cx, cy + 0.5 * h) for cx, cy, h in ring]


def capture_consumer_a_for_pair(
    lost_traj: list[tuple[int, float, float, float]],
    cand_traj: list[tuple[int, float, float, float]],
    *,
    gap: int,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Kernel-formula **reconstruction** for one offline pair event.

    Not a live CUDA dump of ``foot_ring`` / ``ema_h`` / float32 ``bdist``.
    Candidate requires >=4 frames (kernel early return). Lost may be short.
    EMA is causal MOT-tracklet replay matching the kernel algebra only.
    Per-row git/kernel/preset hashes are omitted so capture is a pure function
    of pairs + substrate + estimator code (enables capture-origin --verify).
    """
    if len(lost_traj) < 1 or len(cand_traj) < 4:
        return None
    lost_ring = _traj_slice_for_pair(
        lost_traj, last_or_first="last", n=min(4, len(lost_traj))
    )
    cand_ring = _traj_slice_for_pair(cand_traj, last_or_first="first", n=4)
    if len(cand_ring) < 4:
        return None
    ema_lost = ema_height([h for _f, _cx, _cy, h in lost_traj])
    cand_prefix = cand_traj[:PRODUCTION_BRIDGE_AT]
    ema_cand = ema_height([h for _f, _cx, _cy, h in cand_prefix])
    est = consumer_a_estimate_from_rings(
        lost_ring,
        cand_ring,
        gap=gap,
        ema_lost=ema_lost,
        ema_cand=ema_cand,
    )
    return {
        "event_key": event_key_from_row(meta),
        "seq": meta["seq"],
        "lost_id": meta["lost_id"],
        "cand_id": meta["cand_id"],
        "lost_last_frame": meta["lost_last_frame"],
        "cand_first_frame": meta["cand_first_frame"],
        "gap": gap,
        "bridge_at": est.bridge_at,
        "la": est.la,
        "bdist": _stable_float(est.bdist),
        "dist_h": _stable_float(est.dist_h),
        "fwd_r": _stable_float(est.fwd_r),
        "bwd_r": _stable_float(est.bwd_r),
        "v_lost_x": _stable_float(est.v_lost_x),
        "v_lost_y": _stable_float(est.v_lost_y),
        "v_cand_x": _stable_float(est.v_cand_x),
        "v_cand_y": _stable_float(est.v_cand_y),
        "ax": _stable_float(est.ax),
        "ay": _stable_float(est.ay),
        "cx0": _stable_float(est.cx0),
        "cy0": _stable_float(est.cy0),
        "ema_lost": _stable_float(est.ema_lost),
        "ema_cand": _stable_float(est.ema_cand),
        "h_ref": _stable_float(est.h_ref),
        "s_lost": _stable_float(est.s_lost),
        "w": _stable_float(est.w),
        "production_threshold": PRODUCTION_BRIDGE_PX,
        "capture_mode": CAPTURE_MODE_RECONSTRUCTION,
        "evidence_role": "reconstruction_diagnostic",
        "anchor_mode": PRODUCTION_BRIDGE_ANCHOR,
        "anchor_rate": PRODUCTION_BRIDGE_ANCHOR_RATE,
    }


def decompose_estimators(
    lost_traj: list[tuple[int, float, float, float]],
    cand_traj: list[tuple[int, float, float, float]],
    *,
    gap: int,
    offline_h_ref: float,
) -> dict[str, float]:
    """Single-factor progressive steps on the reconstruction surface.

    Each SW step changes one declared factor vs the previous SW step:

    * S0  offline midpoint (frozen pairs ``bridge_dist``; not SW)
    * S1  aggregation only (offline geometry → speed-weighted)
    * S2  anchor endpoints only (CA adaptive positions; offline window-mean vel)
    * S3  velocity only (CA adaptive vel; keep S2 endpoints)
    * S4  horizon only (gap → la)
    * S5  normalization only (offline h_ref → reconstructed bilateral EMA)
    * S6  reconstruction (equals S5)

    Step deltas are descriptive attribution only — not single-cause claims.
    """
    lost_ring = _traj_slice_for_pair(
        lost_traj, last_or_first="last", n=min(4, len(lost_traj))
    )
    cand_ring = _traj_slice_for_pair(cand_traj, last_or_first="first", n=4)
    v_off_l = window_mean_velocity(_feet_from_ring(lost_ring))
    v_off_c = window_mean_velocity(_feet_from_ring(cand_ring))
    if len(lost_ring) >= 4:
        ax_ca, ay_ca, vxl_ca, vyl_ca = bridge_anchor4(
            lost_ring,
            anchor_mode=ANCHOR_MODE[PRODUCTION_BRIDGE_ANCHOR],
            rate_gate=PRODUCTION_BRIDGE_ANCHOR_RATE,
            endpoint_idx=3,
        )
    else:
        lc_x, lc_y, lh = lost_ring[-1]
        ax_ca = float(lc_x)
        ay_ca = float(lc_y)
        vxl_ca, vyl_ca = 0.0, 0.0
    cx_ca, cy_ca, vxc_ca, vyc_ca = bridge_anchor4(
        cand_ring,
        anchor_mode=ANCHOR_MODE[PRODUCTION_BRIDGE_ANCHOR],
        rate_gate=PRODUCTION_BRIDGE_ANCHOR_RATE,
        endpoint_idx=0,
    )
    lx_off, ly_off = _feet_from_ring(lost_ring)[-1]
    cx_off, cy_off = _feet_from_ring(cand_ring)[0]
    la = gap + PRODUCTION_BRIDGE_AT - 1
    h_off = max(float(offline_h_ref), 1.0)
    ema_lost = ema_height([h for _f, _cx, _cy, h in lost_traj])
    ema_cand = ema_height([h for _f, _cx, _cy, h in cand_traj[:PRODUCTION_BRIDGE_AT]])
    h_ca = max(0.5 * (ema_lost + ema_cand), 1.0)

    def _sw(
        lx: float,
        ly: float,
        cx0: float,
        cy0: float,
        vxl: float,
        vyl: float,
        vxc: float,
        vyc: float,
        horizon: float,
        href: float,
    ) -> float:
        bdist, *_rest = speed_weighted_bdist(
            lx=lx,
            ly=ly,
            cx0=cx0,
            cy0=cy0,
            vxl=vxl,
            vyl=vyl,
            vxc=vxc,
            vyc=vyc,
            horizon=horizon,
            h_ref=href,
        )
        return bdist

    s0_mid = midpoint_bridge_dist(
        lx=lx_off,
        ly=ly_off,
        cx0=cx_off,
        cy0=cy_off,
        vxl=v_off_l[0],
        vyl=v_off_l[1],
        vxc=v_off_c[0],
        vyc=v_off_c[1],
        gap=float(gap),
        h_ref=h_off,
    )
    s1 = _sw(
        lx_off,
        ly_off,
        cx_off,
        cy_off,
        v_off_l[0],
        v_off_l[1],
        v_off_c[0],
        v_off_c[1],
        float(gap),
        h_off,
    )
    s2 = _sw(
        ax_ca,
        ay_ca,
        cx_ca,
        cy_ca,
        v_off_l[0],
        v_off_l[1],
        v_off_c[0],
        v_off_c[1],
        float(gap),
        h_off,
    )
    s3 = _sw(
        ax_ca,
        ay_ca,
        cx_ca,
        cy_ca,
        vxl_ca,
        vyl_ca,
        vxc_ca,
        vyc_ca,
        float(gap),
        h_off,
    )
    s4 = _sw(
        ax_ca,
        ay_ca,
        cx_ca,
        cy_ca,
        vxl_ca,
        vyl_ca,
        vxc_ca,
        vyc_ca,
        float(la),
        h_off,
    )
    s5 = _sw(
        ax_ca,
        ay_ca,
        cx_ca,
        cy_ca,
        vxl_ca,
        vyl_ca,
        vxc_ca,
        vyc_ca,
        float(la),
        h_ca,
    )
    return {
        "s0_midpoint": _stable_float(s0_mid),
        "s1_aggregation": _stable_float(s1),
        "s2_anchor_endpoints": _stable_float(s2),
        "s3_velocity": _stable_float(s3),
        "s4_horizon": _stable_float(s4),
        "s5_normalization": _stable_float(s5),
        "s6_reconstruction": _stable_float(s5),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────


def _mask_s_a(gt_valid: np.ndarray, gap: np.ndarray) -> np.ndarray:
    return gt_valid & (gap >= 1) & (gap <= 26)


def spearman_with_cluster_ci(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Spearman rho with cluster bootstrap CI (cluster = sequence|lost_id)."""
    if x.size < 3:
        return {
            "rho": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": int(x.size),
            "n_clusters": 0,
        }
    rho = float(spearmanr(x, y).correlation)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    by_cluster: dict[str, np.ndarray] = {
        str(c): np.flatnonzero(clusters == c) for c in unique
    }
    keys = list(by_cluster.keys())
    for _ in range(reps):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([by_cluster[k] for k in chosen])
        if idx.size < 3:
            continue
        r = spearmanr(x[idx], y[idx]).correlation
        if r is not None and math.isfinite(float(r)):
            boots.append(float(r))
    if boots:
        lo, hi = np.quantile(boots, [0.025, 0.975])
    else:
        lo = hi = float("nan")
    return {
        "rho": rho,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(x.size),
        "n_clusters": int(unique.size),
    }


def quantiles(
    values: np.ndarray, qs: Sequence[float] = CLUSTER_QUANTILES
) -> dict[str, float]:
    if values.size == 0:
        return {f"q{int(q * 100)}": float("nan") for q in qs}
    out: dict[str, float] = {}
    for q in qs:
        out[f"q{int(q * 100)}"] = float(np.quantile(values, q))
    return out


def quantile_alignment(offline: np.ndarray, online: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in CLUSTER_QUANTILES:
        oq = float(np.quantile(offline, q)) if offline.size else float("nan")
        cq = float(np.quantile(online, q)) if online.size else float("nan")
        abs_err = (
            abs(oq - cq) if math.isfinite(oq) and math.isfinite(cq) else float("nan")
        )
        rel_err = (
            abs_err / max(abs(oq), 1e-12) if math.isfinite(abs_err) else float("nan")
        )
        rows.append(
            {
                "quantile": f"q{int(q * 100)}",
                "offline_quantile": oq,
                "consumer_a_quantile": cq,
                "absolute_error": abs_err,
                "relative_error": rel_err,
                "headline": q == 0.85,
            }
        )
    return rows


def predicate_confusion(
    offline: np.ndarray, online: np.ndarray, thr: float = PRODUCTION_BRIDGE_PX
) -> dict[str, Any]:
    o_safe = offline <= thr
    c_safe = online <= thr
    ss = int(np.sum(o_safe & c_safe))
    su = int(np.sum(o_safe & ~c_safe))
    us = int(np.sum(~o_safe & c_safe))
    uu = int(np.sum(~o_safe & ~c_safe))
    n = int(offline.size)
    agree = (ss + uu) / n if n else float("nan")
    rate = su / n if n else float("nan")
    return {
        "n": n,
        "offline_safe_online_safe": ss,
        "offline_safe_online_unsafe": su,
        "offline_unsafe_online_safe": us,
        "offline_unsafe_online_unsafe": uu,
        "predicate_agreement": agree,
        "offline_safe_online_unsafe_count": su,
        "offline_safe_online_unsafe_rate": rate,
    }


def slice_metrics(
    name: str,
    offline: np.ndarray,
    online: np.ndarray,
    gt_match: np.ndarray,
    clusters: np.ndarray,
    quantity: str,
) -> dict[str, Any]:
    sp = spearman_with_cluster_ci(offline, online, clusters)
    conf = predicate_confusion(offline, online) if quantity == "bdist" else None
    qrows = quantile_alignment(offline, online)
    q85 = next(r for r in qrows if r["quantile"] == "q85")
    return {
        "slice": name,
        "quantity": quantity,
        "n": int(offline.size),
        "n_gt": int(np.sum(gt_match)),
        "n_fp": int(np.sum(~gt_match)),
        "spearman_rho": sp["rho"],
        "spearman_ci_low": sp["ci_low"],
        "spearman_ci_high": sp["ci_high"],
        "n_clusters": sp["n_clusters"],
        "q85_offline": q85["offline_quantile"],
        "q85_consumer_a": q85["consumer_a_quantile"],
        "q85_abs_error": q85["absolute_error"],
        "q85_rel_error": q85["relative_error"],
        "predicate": conf,
        "quantiles": qrows,
        "offline_quantiles": quantiles(offline),
        "consumer_a_quantiles": quantiles(online),
        "quantile_monotone_offline": _monotone(quantiles(offline)),
        "quantile_monotone_consumer_a": _monotone(quantiles(online)),
    }


def _monotone(qmap: dict[str, float]) -> bool:
    order = ["q50", "q70", "q85", "q90", "q95"]
    vals = [qmap[k] for k in order]
    if any(not math.isfinite(v) for v in vals):
        return False
    return all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def evaluate_verdict(
    coverage: dict[str, Any],
    metrics_bdist: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Frozen three-value partition. Never invents a fourth verdict."""
    cov_pass = bool(coverage["gates_pass"])
    overall = metrics_bdist["S_A_overall"]
    gt = metrics_bdist["S_A_GT"]
    fp = metrics_bdist["S_A_FP"]

    def _rho(m: dict[str, Any]) -> float:
        return float(m["spearman_rho"])

    q85_err = float(overall["q85_abs_error"])
    pred_o = overall["predicate"]
    pred_g = gt["predicate"]
    pred_f = fp["predicate"]

    gap_pred_ok = True
    gap_rank_ok = True
    gap_details: list[dict[str, Any]] = []
    for key in ("gap_1_10", "gap_11_26"):
        m = metrics_bdist[key]
        n_gt = int(m["n_gt"])
        detail: dict[str, Any] = {"slice": key, "n_gt": n_gt}
        if n_gt >= 20:
            # GT-conditional metrics inside gap cell
            gtm = metrics_bdist[f"{key}_GT"]
            pa = float(gtm["predicate"]["predicate_agreement"])
            rho = float(gtm["spearman_rho"])
            detail["gt_predicate_agreement"] = pa
            detail["gt_spearman"] = rho
            if pa < GATE_GAP_PRED_AGREE:
                gap_pred_ok = False
            if rho < GATE_GAP_SPEARMAN_GT:
                gap_rank_ok = False
        gap_details.append(detail)

    thr_ok = (
        cov_pass
        and _rho(overall) >= GATE_SPEARMAN_THRESHOLD["overall"]
        and _rho(gt) >= GATE_SPEARMAN_THRESHOLD["gt"]
        and _rho(fp) >= GATE_SPEARMAN_THRESHOLD["fp"]
        and q85_err <= GATE_Q85_ABS
        and float(pred_o["predicate_agreement"]) >= GATE_PRED_AGREE
        and float(pred_g["predicate_agreement"]) >= GATE_PRED_AGREE
        and float(pred_f["predicate_agreement"]) >= GATE_PRED_AGREE
        and int(pred_g["offline_safe_online_unsafe_count"]) <= GATE_GT_SAFE_UNSAFE_COUNT
        and float(pred_g["offline_safe_online_unsafe_rate"]) <= GATE_GT_SAFE_UNSAFE_RATE
        and gap_pred_ok
    )
    mono = bool(
        overall["quantile_monotone_offline"] and overall["quantile_monotone_consumer_a"]
    )
    rank_ok = (
        (not thr_ok)
        and cov_pass
        and _rho(overall) >= GATE_SPEARMAN_RANK_ONLY["overall"]
        and _rho(gt) >= GATE_SPEARMAN_RANK_ONLY["gt"]
        and _rho(fp) >= GATE_SPEARMAN_RANK_ONLY["fp"]
        and gap_rank_ok
        and mono
    )
    if thr_ok:
        metric_verdict = "threshold_transfer_supported"
    elif rank_ok:
        metric_verdict = "rank_only_transfer_supported"
    else:
        metric_verdict = "not_fidelity_aligned"

    # Binding fail-closed: without live CUDA capture, Issue #112 cannot certify
    # runtime Consumer-A fidelity. Reconstruction metrics stay diagnostic only.
    primary_fail_reason: str | None
    if not LIVE_CUDA_EVENT_RING_IMPLEMENTED:
        verdict = "not_fidelity_aligned"
        primary_fail_reason = PRIMARY_FAIL_REASON
    else:
        verdict = metric_verdict
        primary_fail_reason = (
            None if metric_verdict != "not_fidelity_aligned" else "metric_gates_failed"
        )

    return {
        "verdict": verdict,
        "primary_fail_reason": primary_fail_reason,
        "metric_based_verdict_diagnostic_only": metric_verdict,
        "runtime_capture_available": LIVE_CUDA_EVENT_RING_IMPLEMENTED,
        "evidence_role": RECONSTRUCTION_QUANTITY_NOTE,
        "issue_112_status": ISSUE_112_STATUS,
        "coverage_gates_pass": cov_pass,
        "threshold_transfer_criteria_pass": thr_ok,
        "rank_only_criteria_pass": rank_ok,
        "gap_cell_details": gap_details,
        "checks": {
            "spearman_overall": _rho(overall),
            "spearman_gt": _rho(gt),
            "spearman_fp": _rho(fp),
            "q85_abs_error": q85_err,
            "pred_agree_overall": float(pred_o["predicate_agreement"]),
            "pred_agree_gt": float(pred_g["predicate_agreement"]),
            "pred_agree_fp": float(pred_f["predicate_agreement"]),
            "gt_safe_unsafe_count": int(pred_g["offline_safe_online_unsafe_count"]),
            "gt_safe_unsafe_rate": float(pred_g["offline_safe_online_unsafe_rate"]),
            "quantile_monotone_both": mono,
        },
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


_CAPTURE_FLOAT_FIELDS = {
    "bdist",
    "dist_h",
    "fwd_r",
    "bwd_r",
    "v_lost_x",
    "v_lost_y",
    "v_cand_x",
    "v_cand_y",
    "ax",
    "ay",
    "cx0",
    "cy0",
    "ema_lost",
    "ema_cand",
    "h_ref",
    "s_lost",
    "w",
    "production_threshold",
    "anchor_rate",
}
_CAPTURE_INT_FIELDS = {
    "gap",
    "bridge_at",
    "la",
    "lost_last_frame",
    "cand_first_frame",
}


def load_capture_csv(path: Path) -> list[dict[str, Any]]:
    if str(path).endswith(".gz"):
        opener = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        opener = path.open(newline="", encoding="utf-8")
    with opener as stream:
        raw_rows = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        out: dict[str, Any] = dict(row)
        for field in _CAPTURE_FLOAT_FIELDS:
            if field in out and out[field] not in ("", None):
                out[field] = float(out[field])
        for field in _CAPTURE_INT_FIELDS:
            if field in out and out[field] not in ("", None):
                out[field] = int(float(out[field]))
        out["event_key"] = out.get("event_key") or event_key_from_row(out)
        rows.append(out)
    return rows


def build_capture(
    pairs: list[dict[str, Any]],
    mot_dir: Path,
    *,
    kernel_sha: str,
    commit: str,
    config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # kernel_sha/commit/config_hash kept for call-site compatibility; capture
    # rows intentionally omit them (packet-level provenance only).
    del kernel_sha, commit, config_hash
    tracks_by_seq: dict[str, dict[int, list[tuple[int, float, float, float]]]] = {}
    capture_rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    for row in pairs:
        seq = str(row["seq"])
        if seq not in tracks_by_seq:
            tracks_by_seq[seq] = load_tracks(mot_dir / f"{seq}.txt")
        tracks = tracks_by_seq[seq]
        try:
            lid = int(row["lost_id"])
            cid = int(row["cand_id"])
        except ValueError:
            skip_reasons["bad_id"] += 1
            continue
        if lid not in tracks or cid not in tracks:
            skip_reasons["missing_track"] += 1
            continue
        gap = int(row["gap"])
        meta = {
            "seq": seq,
            "lost_id": str(lid),
            "cand_id": str(cid),
            "lost_last_frame": str(row["lost_last_frame"]),
            "cand_first_frame": str(row["cand_first_frame"]),
        }
        cap = capture_consumer_a_for_pair(tracks[lid], tracks[cid], gap=gap, meta=meta)
        if cap is None:
            skip_reasons["short_ring"] += 1
            continue
        capture_rows.append(cap)
    stats = {
        "n_pairs": len(pairs),
        "n_captured": len(capture_rows),
        "skip_reasons": dict(skip_reasons),
        "capture_mode": CAPTURE_MODE_RECONSTRUCTION,
        "evidence_role": "reconstruction_diagnostic",
        "live_cuda_event_ring_implemented": LIVE_CUDA_EVENT_RING_IMPLEMENTED,
        "research_audit_default": RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT,
        "issue_112_status": ISSUE_112_STATUS,
    }
    return capture_rows, stats


def join_events(
    pairs: list[dict[str, Any]],
    capture_rows: list[dict[str, Any]],
    tracks_cache: dict[str, dict[int, list[tuple[int, float, float, float]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in capture_rows:
        cap_by_key[str(row["event_key"])].append(row)

    offline_keys = [event_key_from_row(r) for r in pairs]
    offline_key_counts = Counter(offline_keys)
    duplicate_offline = sum(1 for _k, c in offline_key_counts.items() if c > 1)
    duplicate_capture = sum(1 for _k, rows in cap_by_key.items() if len(rows) > 1)

    join_rows: list[dict[str, Any]] = []
    matched = 0
    offline_only = 0
    ambiguous = 0

    for prow in pairs:
        key = event_key_from_row(prow)
        gap = int(prow["gap"])
        gt_valid = _as_bool(prow["gt_valid"])
        gt_match = _as_bool(prow["gt_match"])
        base = {
            "event_key": key,
            "seq": prow["seq"],
            "lost_id": prow["lost_id"],
            "cand_id": prow["cand_id"],
            "lost_last_frame": prow["lost_last_frame"],
            "cand_first_frame": prow["cand_first_frame"],
            "gap": gap,
            "gt_match": int(gt_match),
            "gt_valid": int(gt_valid),
            "offline_bridge_dist": float(prow["bridge_dist"]),
            "offline_dist_h": float(prow["dist_h"]),
            "offline_fwd_r": float(prow["fwd_resid"]),
            "offline_bwd_r": float(prow["bwd_resid"]),
            "offline_h_ref": float(prow["h_ref"]),
            "offline_lost_exit_speed": float(prow["lost_exit_speed"]),
        }
        hits = cap_by_key.get(key, [])
        if len(hits) > 1:
            ambiguous += 1
            base["join_status"] = "ambiguous"
            join_rows.append(base)
            continue
        if len(hits) == 0:
            offline_only += 1
            base["join_status"] = "offline_only"
            join_rows.append(base)
            continue
        crow = hits[0]
        matched += 1
        base.update(
            {
                "join_status": "exact_match",
                "recon_bdist": float(crow["bdist"]),
                "recon_dist_h": float(crow["dist_h"]),
                "recon_fwd_r": float(crow["fwd_r"]),
                "recon_bwd_r": float(crow["bwd_r"]),
                "recon_h_ref": float(crow["h_ref"]),
                "recon_la": int(crow["la"]),
                "recon_ema_lost": float(crow["ema_lost"]),
                "recon_ema_cand": float(crow["ema_cand"]),
                "recon_v_lost_x": float(crow["v_lost_x"]),
                "recon_v_lost_y": float(crow["v_lost_y"]),
                "recon_v_cand_x": float(crow["v_cand_x"]),
                "recon_v_cand_y": float(crow["v_cand_y"]),
            }
        )
        # Decomposition when tracks available.
        seq = str(prow["seq"])
        if seq not in tracks_cache:
            tracks_cache[seq] = load_tracks((REPO / SUBSTRATE_MOT_DIR / f"{seq}.txt"))
        tracks = tracks_cache[seq]
        lid, cid = int(prow["lost_id"]), int(prow["cand_id"])
        if (
            lid in tracks
            and cid in tracks
            and len(tracks[lid]) >= 1
            and len(tracks[cid]) >= 4
        ):
            decomp = decompose_estimators(
                tracks[lid],
                tracks[cid],
                gap=gap,
                offline_h_ref=float(prow["h_ref"]),
            )
            base["s1_aggregation"] = decomp["s1_aggregation"]
            base["s2_anchor_endpoints"] = decomp["s2_anchor_endpoints"]
            base["s3_velocity"] = decomp["s3_velocity"]
            base["s4_horizon"] = decomp["s4_horizon"]
            base["s5_normalization"] = decomp["s5_normalization"]
        join_rows.append(base)

    capture_keys = set(cap_by_key)
    offline_key_set = set(offline_keys)
    online_only = len(capture_keys - offline_key_set)

    # Coverage on S_A
    sa_pairs = [
        r for r in pairs if _as_bool(r["gt_valid"]) and 1 <= int(r["gap"]) <= 26
    ]
    sa_keys = {event_key_from_row(r) for r in sa_pairs}
    sa_gt_keys = {event_key_from_row(r) for r in sa_pairs if _as_bool(r["gt_match"])}
    matched_keys = {
        r["event_key"] for r in join_rows if r["join_status"] == "exact_match"
    }
    sa_matched = sa_keys & matched_keys
    sa_gt_matched = sa_gt_keys & matched_keys
    gt_cov = len(sa_gt_matched) / len(sa_gt_keys) if sa_gt_keys else 0.0
    overall_cov = len(sa_matched) / len(sa_keys) if sa_keys else 0.0
    fp_keys = sa_keys - sa_gt_keys
    fp_cov = len(fp_keys & matched_keys) / len(fp_keys) if fp_keys else 0.0

    gates_pass = (
        duplicate_offline == 0
        and duplicate_capture == 0
        and ambiguous == 0
        and gt_cov >= GATE_GT_COVERAGE - 1e-12
        and overall_cov >= GATE_OVERALL_COVERAGE - 1e-12
    )
    coverage = {
        "offline_eligible_rows": len(pairs),
        "offline_s_a_rows": len(sa_pairs),
        "offline_s_a_gt": len(sa_gt_keys),
        "offline_s_a_fp": len(fp_keys),
        "consumer_a_captured_rows": len(capture_rows),
        "exact_matched_rows": matched,
        "offline_only_rows": offline_only,
        "online_only_rows": online_only,
        "duplicate_keys_offline": duplicate_offline,
        "duplicate_keys_capture": duplicate_capture,
        "ambiguous_keys": ambiguous,
        "gt_match_coverage_s_a": gt_cov,
        "fp_match_coverage_s_a": fp_cov,
        "overall_match_coverage_s_a": overall_cov,
        "gates": {
            "duplicate_keys_zero": duplicate_offline == 0 and duplicate_capture == 0,
            "ambiguous_keys_zero": ambiguous == 0,
            "gt_coverage_s_a_100": gt_cov >= GATE_GT_COVERAGE - 1e-12,
            "overall_coverage_s_a_98": overall_cov >= GATE_OVERALL_COVERAGE - 1e-12,
        },
        "gates_pass": gates_pass,
    }
    return join_rows, coverage


def _arrays_from_join(matched: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "offline_bridge_dist": np.array(
            [float(r["offline_bridge_dist"]) for r in matched], dtype=np.float64
        ),
        "offline_dist_h": np.array(
            [float(r["offline_dist_h"]) for r in matched], dtype=np.float64
        ),
        "offline_fwd_r": np.array(
            [float(r["offline_fwd_r"]) for r in matched], dtype=np.float64
        ),
        "offline_bwd_r": np.array(
            [float(r["offline_bwd_r"]) for r in matched], dtype=np.float64
        ),
        "recon_bdist": np.array(
            [float(r["recon_bdist"]) for r in matched], dtype=np.float64
        ),
        "recon_dist_h": np.array(
            [float(r["recon_dist_h"]) for r in matched], dtype=np.float64
        ),
        "recon_fwd_r": np.array(
            [float(r["recon_fwd_r"]) for r in matched], dtype=np.float64
        ),
        "recon_bwd_r": np.array(
            [float(r["recon_bwd_r"]) for r in matched], dtype=np.float64
        ),
        "gt_match": np.array([bool(int(r["gt_match"])) for r in matched]),
        "gt_valid": np.array([bool(int(r["gt_valid"])) for r in matched]),
        "gap": np.array([int(r["gap"]) for r in matched], dtype=np.int32),
        "seq": np.array([str(r["seq"]) for r in matched]),
        "cluster": np.array(
            [f"{r['seq']}|{r['lost_id']}" for r in matched], dtype=object
        ),
        "s1": np.array(
            [float(r.get("s1_aggregation", np.nan)) for r in matched], dtype=np.float64
        ),
        "s2": np.array(
            [float(r.get("s2_anchor_endpoints", np.nan)) for r in matched],
            dtype=np.float64,
        ),
        "s3": np.array(
            [float(r.get("s3_velocity", np.nan)) for r in matched], dtype=np.float64
        ),
        "s4": np.array(
            [float(r.get("s4_horizon", np.nan)) for r in matched], dtype=np.float64
        ),
        "s5": np.array(
            [float(r.get("s5_normalization", np.nan)) for r in matched],
            dtype=np.float64,
        ),
        "offline_h_ref": np.array(
            [float(r["offline_h_ref"]) for r in matched], dtype=np.float64
        ),
        "recon_h_ref": np.array(
            [float(r["recon_h_ref"]) for r in matched], dtype=np.float64
        ),
        "recon_la": np.array([int(r["recon_la"]) for r in matched], dtype=np.int32),
    }


def compute_all_metrics(matched: list[dict[str, Any]]) -> dict[str, Any]:
    arr = _arrays_from_join(matched)
    sa = _mask_s_a(arr["gt_valid"], arr["gap"])
    quantities = {
        "bdist": ("offline_bridge_dist", "recon_bdist"),
        "dist_h": ("offline_dist_h", "recon_dist_h"),
        "fwd_r": ("offline_fwd_r", "recon_fwd_r"),
        "bwd_r": ("offline_bwd_r", "recon_bwd_r"),
    }

    def _select(mask: np.ndarray) -> list[dict[str, Any]]:
        idx = np.flatnonzero(mask)
        return [matched[i] for i in idx]

    metrics: dict[str, Any] = {"by_quantity": {}}
    for qname, (off_k, on_k) in quantities.items():
        qmetrics: dict[str, Any] = {}
        base_mask = sa
        slices = {
            "S_A_overall": base_mask,
            "S_A_GT": base_mask & arr["gt_match"],
            "S_A_FP": base_mask & ~arr["gt_match"],
            "gap_1_10": base_mask & (arr["gap"] >= 1) & (arr["gap"] <= 10),
            "gap_11_26": base_mask & (arr["gap"] >= 11) & (arr["gap"] <= 26),
            "gap_1_10_GT": base_mask
            & arr["gt_match"]
            & (arr["gap"] >= 1)
            & (arr["gap"] <= 10),
            "gap_11_26_GT": base_mask
            & arr["gt_match"]
            & (arr["gap"] >= 11)
            & (arr["gap"] <= 26),
        }
        for seq in sorted(set(arr["seq"][base_mask].tolist())):
            slices[f"seq::{seq}"] = base_mask & (arr["seq"] == seq)
        for sname, mask in slices.items():
            if not np.any(mask):
                qmetrics[sname] = {
                    "slice": sname,
                    "quantity": qname,
                    "n": 0,
                    "n_gt": 0,
                    "n_fp": 0,
                    "spearman_rho": float("nan"),
                    "spearman_ci_low": float("nan"),
                    "spearman_ci_high": float("nan"),
                    "n_clusters": 0,
                    "q85_offline": float("nan"),
                    "q85_consumer_a": float("nan"),
                    "q85_abs_error": float("nan"),
                    "q85_rel_error": float("nan"),
                    "predicate": predicate_confusion(np.array([]), np.array([]))
                    if qname == "bdist"
                    else None,
                    "quantiles": [],
                    "offline_quantiles": quantiles(np.array([])),
                    "consumer_a_quantiles": quantiles(np.array([])),
                    "quantile_monotone_offline": False,
                    "quantile_monotone_consumer_a": False,
                }
                continue
            qmetrics[sname] = slice_metrics(
                sname,
                arr[off_k][mask],
                arr[on_k][mask],
                arr["gt_match"][mask],
                arr["cluster"][mask],
                qname,
            )
        metrics["by_quantity"][qname] = qmetrics

    # Decomposition table vs reconstruction S6 on S_A (diagnostic only).
    decomp_rows: list[dict[str, Any]] = []
    sa_idx = np.flatnonzero(sa)
    if sa_idx.size:
        ref = arr["recon_bdist"][sa_idx]
        gt_sa = arr["gt_match"][sa_idx]
        cl_sa = arr["cluster"][sa_idx]
        step_specs = (
            ("S0_offline_midpoint", "offline_bridge_dist"),
            ("S1_aggregation_only", "s1"),
            ("S2_anchor_endpoints_only", "s2"),
            ("S3_velocity_only", "s3"),
            ("S4_horizon_only", "s4"),
            ("S5_normalization_only", "s5"),
            ("S6_kernel_formula_reconstruction", "ref"),
        )
        for step, key in step_specs:
            if key == "offline_bridge_dist":
                x = arr["offline_bridge_dist"][sa_idx]
            elif key == "ref":
                x = ref
            else:
                x = arr[key][sa_idx]
            finite = np.isfinite(x) & np.isfinite(ref)
            if finite.sum() < 3:
                decomp_rows.append(
                    {
                        "step": step,
                        "factor_changed": step.split("_", 1)[0],
                        "n": int(finite.sum()),
                        "spearman_rho_vs_reconstruction": float("nan"),
                        "q85_abs_error_vs_reconstruction": float("nan"),
                        "predicate_agreement_at_0.4": float("nan"),
                        "gt_step_safe_recon_unsafe_count": 0,
                    }
                )
                continue
            sp = spearman_with_cluster_ci(x[finite], ref[finite], cl_sa[finite])
            q_x = float(np.quantile(x[finite], 0.85))
            q_ref = float(np.quantile(ref[finite], 0.85))
            conf = predicate_confusion(x[finite], ref[finite])
            gt_f = gt_sa[finite]
            step_safe = x[finite] <= PRODUCTION_BRIDGE_PX
            ref_unsafe = ref[finite] > PRODUCTION_BRIDGE_PX
            gt_su = int(np.sum(gt_f & step_safe & ref_unsafe))
            decomp_rows.append(
                {
                    "step": step,
                    "factor_changed": step.split("_", 1)[0],
                    "n": int(finite.sum()),
                    "spearman_rho_vs_reconstruction": sp["rho"],
                    "q85_abs_error_vs_reconstruction": abs(q_x - q_ref),
                    "predicate_agreement_at_0.4": conf["predicate_agreement"],
                    "gt_step_safe_recon_unsafe_count": gt_su,
                }
            )
    metrics["estimator_decomposition"] = decomp_rows
    return metrics


def disagreement_localization(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sa = [r for r in matched if int(r["gt_valid"]) == 1 and 1 <= int(r["gap"]) <= 26]
    su = [
        r
        for r in sa
        if float(r["offline_bridge_dist"]) <= PRODUCTION_BRIDGE_PX
        and float(r["recon_bdist"]) > PRODUCTION_BRIDGE_PX
    ]
    if not sa:
        return rows

    def _qbin(val: float, ref: np.ndarray) -> str:
        qs = np.quantile(ref, [0.25, 0.5, 0.75])
        if val <= qs[0]:
            return "<=q25"
        if val <= qs[1]:
            return "q25-q50"
        if val <= qs[2]:
            return "q50-q75"
        return ">q75"

    offline_h = np.array([float(r["offline_h_ref"]) for r in sa])
    ca_h = np.array([float(r["recon_h_ref"]) for r in sa])
    offline_bd = np.array([float(r["offline_bridge_dist"]) for r in sa])
    ca_bd = np.array([float(r["recon_bdist"]) for r in sa])
    gap_arr = np.array([int(r["gap"]) for r in sa])

    for r in su:
        rows.append(
            {
                "event_key": r["event_key"],
                "seq": r["seq"],
                "gap": r["gap"],
                "gt_match": r["gt_match"],
                "offline_bridge_dist": r["offline_bridge_dist"],
                "recon_bdist": r["recon_bdist"],
                "offline_h_ref": r["offline_h_ref"],
                "recon_h_ref": r["recon_h_ref"],
                "recon_la": r["recon_la"],
                "true_gap": r["gap"],
                "offline_fwd_r": r["offline_fwd_r"],
                "offline_bwd_r": r["offline_bwd_r"],
                "recon_fwd_r": r["recon_fwd_r"],
                "recon_bwd_r": r["recon_bwd_r"],
                "offline_dist_h": r["offline_dist_h"],
                "recon_dist_h": r["recon_dist_h"],
                "gap_bin": "1-10" if int(r["gap"]) <= 10 else "11-26",
                "offline_h_ref_bin": _qbin(float(r["offline_h_ref"]), offline_h),
                "recon_h_ref_bin": _qbin(float(r["recon_h_ref"]), ca_h),
                "offline_bridge_bin": _qbin(
                    float(r["offline_bridge_dist"]), offline_bd
                ),
                "recon_bdist_bin": _qbin(float(r["recon_bdist"]), ca_bd),
                "regime": (
                    f"gap={('1-10' if int(r['gap']) <= 10 else '11-26')}"
                    f"|seq={r['seq']}"
                    f"|gt={r['gt_match']}"
                ),
            }
        )
    # Summary regime counts
    regime_counts = Counter(r["regime"] for r in rows)
    for r in rows:
        r["regime_count"] = regime_counts[r["regime"]]
        r["n_su_total"] = len(su)
        r["n_sa"] = len(sa)
        r["gap_pool_median"] = float(np.median(gap_arr))
    return rows


def flatten_metrics_tables(
    metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    seq_rows: list[dict[str, Any]] = []
    for qname, qmetrics in metrics["by_quantity"].items():
        for sname, m in qmetrics.items():
            pred = m.get("predicate") or {}
            row = {
                "quantity": qname,
                "slice": sname,
                "n": m["n"],
                "n_gt": m["n_gt"],
                "n_fp": m["n_fp"],
                "spearman_rho": m["spearman_rho"],
                "spearman_ci_low": m["spearman_ci_low"],
                "spearman_ci_high": m["spearman_ci_high"],
                "q85_offline": m["q85_offline"],
                "q85_consumer_a": m["q85_consumer_a"],
                "q85_abs_error": m["q85_abs_error"],
                "predicate_agreement": pred.get("predicate_agreement", ""),
                "offline_safe_online_unsafe_count": pred.get(
                    "offline_safe_online_unsafe_count", ""
                ),
                "offline_safe_online_unsafe_rate": pred.get(
                    "offline_safe_online_unsafe_rate", ""
                ),
            }
            if sname.startswith("seq::"):
                seq_rows.append(row)
            elif sname.startswith("gap_"):
                gap_rows.append(row)
            else:
                overall_rows.append(row)
    return overall_rows, gap_rows, seq_rows


def build_predicate_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bdist = metrics["by_quantity"]["bdist"]
    for sname, m in bdist.items():
        pred = m.get("predicate")
        if not pred:
            continue
        rows.append(
            {
                "slice": sname,
                "n": pred["n"],
                "offline_safe_online_safe": pred["offline_safe_online_safe"],
                "offline_safe_online_unsafe": pred["offline_safe_online_unsafe"],
                "offline_unsafe_online_safe": pred["offline_unsafe_online_safe"],
                "offline_unsafe_online_unsafe": pred["offline_unsafe_online_unsafe"],
                "predicate_agreement": pred["predicate_agreement"],
                "offline_safe_online_unsafe_count": pred[
                    "offline_safe_online_unsafe_count"
                ],
                "offline_safe_online_unsafe_rate": pred[
                    "offline_safe_online_unsafe_rate"
                ],
            }
        )
    return rows


def build_quantile_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qname, qmetrics in metrics["by_quantity"].items():
        for sname in ("S_A_overall", "S_A_GT", "S_A_FP", "gap_1_10", "gap_11_26"):
            m = qmetrics.get(sname)
            if not m:
                continue
            for qrow in m.get("quantiles") or []:
                rows.append(
                    {
                        "quantity": qname,
                        "slice": sname,
                        **qrow,
                    }
                )
    return rows


def boundary_diagnostics(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sa = [r for r in matched if int(r["gt_valid"]) == 1 and 1 <= int(r["gap"]) <= 26]
    band = [
        r
        for r in sa
        if BOUNDARY_LO <= float(r["recon_bdist"]) <= BOUNDARY_HI
        or BOUNDARY_LO <= float(r["offline_bridge_dist"]) <= BOUNDARY_HI
    ]
    if not band:
        return [
            {
                "band": f"[{BOUNDARY_LO},{BOUNDARY_HI}]",
                "row_count": 0,
                "n_gt": 0,
                "n_fp": 0,
                "predicate_disagreement": 0,
                "offline_safe_online_unsafe_rate": float("nan"),
            }
        ]
    o = np.array([float(r["offline_bridge_dist"]) for r in band])
    c = np.array([float(r["recon_bdist"]) for r in band])
    conf = predicate_confusion(o, c)
    n_gt = sum(int(r["gt_match"]) for r in band)
    return [
        {
            "band": f"[{BOUNDARY_LO},{BOUNDARY_HI}]",
            "row_count": len(band),
            "n_gt": n_gt,
            "n_fp": len(band) - n_gt,
            "predicate_disagreement": conf["offline_safe_online_unsafe"]
            + conf["offline_unsafe_online_safe"],
            "offline_safe_online_unsafe_rate": conf["offline_safe_online_unsafe_rate"],
            "predicate_agreement": conf["predicate_agreement"],
        }
    ]


def run_pipeline(
    pairs_path: Path,
    mot_dir: Path,
    output_dir: Path,
    *,
    capture_path: Path | None = None,
) -> dict[str, Any]:
    pairs_path = pairs_path if pairs_path.is_absolute() else REPO / pairs_path
    mot_dir = mot_dir if mot_dir.is_absolute() else REPO / mot_dir
    output_dir = output_dir if output_dir.is_absolute() else REPO / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_sha = sha256(pairs_path)
    if source_sha != SOURCE_SHA256:
        raise ValueError(
            f"source pairs SHA mismatch: got {source_sha}, expected {SOURCE_SHA256}"
        )

    kernel_path = REPO / KERNEL_SOURCE
    kernel_sha = sha256(kernel_path)
    commit = git_commit()
    config_hash = preset_config_hash()
    pairs = load_pairs(pairs_path)

    if capture_path is not None:
        # Optional external capture input (tests only). Production --verify
        # always rebuilds capture from pairs+substrate with capture_path=None.
        cap_p = capture_path if capture_path.is_absolute() else REPO / capture_path
        capture_rows = load_capture_csv(cap_p)
        capture_stats = {
            "n_pairs": len(pairs),
            "n_captured": len(capture_rows),
            "capture_mode": capture_rows[0].get(
                "capture_mode", CAPTURE_MODE_RECONSTRUCTION
            )
            if capture_rows
            else "empty",
            "evidence_role": "reconstruction_diagnostic",
            "live_cuda_event_ring_implemented": LIVE_CUDA_EVENT_RING_IMPLEMENTED,
            "research_audit_default": RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT,
            "issue_112_status": ISSUE_112_STATUS,
        }
    else:
        capture_rows, capture_stats = build_capture(
            pairs,
            mot_dir,
            kernel_sha=kernel_sha,
            commit=commit,
            config_hash=config_hash,
        )
        capture_stats = {
            "n_pairs": capture_stats["n_pairs"],
            "n_captured": capture_stats["n_captured"],
            "capture_mode": capture_stats["capture_mode"],
            "evidence_role": capture_stats.get(
                "evidence_role", "reconstruction_diagnostic"
            ),
            "live_cuda_event_ring_implemented": capture_stats[
                "live_cuda_event_ring_implemented"
            ],
            "research_audit_default": capture_stats["research_audit_default"],
            "issue_112_status": ISSUE_112_STATUS,
        }

    tracks_cache: dict[str, dict[int, list[tuple[int, float, float, float]]]] = {}
    join_rows, coverage = join_events(pairs, capture_rows, tracks_cache)
    matched = [r for r in join_rows if r["join_status"] == "exact_match"]

    # If coverage fails we still compute metrics on available matches.
    metrics = (
        compute_all_metrics(matched)
        if matched
        else {
            "by_quantity": {
                q: {
                    "S_A_overall": {
                        "slice": "S_A_overall",
                        "quantity": q,
                        "n": 0,
                        "n_gt": 0,
                        "n_fp": 0,
                        "spearman_rho": float("nan"),
                        "spearman_ci_low": float("nan"),
                        "spearman_ci_high": float("nan"),
                        "n_clusters": 0,
                        "q85_offline": float("nan"),
                        "q85_consumer_a": float("nan"),
                        "q85_abs_error": float("nan"),
                        "q85_rel_error": float("nan"),
                        "predicate": predicate_confusion(np.array([]), np.array([])),
                        "quantiles": [],
                        "offline_quantiles": quantiles(np.array([])),
                        "consumer_a_quantiles": quantiles(np.array([])),
                        "quantile_monotone_offline": False,
                        "quantile_monotone_consumer_a": False,
                    },
                    "S_A_GT": {
                        "slice": "S_A_GT",
                        "quantity": q,
                        "n": 0,
                        "n_gt": 0,
                        "n_fp": 0,
                        "spearman_rho": float("nan"),
                        "spearman_ci_low": float("nan"),
                        "spearman_ci_high": float("nan"),
                        "n_clusters": 0,
                        "q85_offline": float("nan"),
                        "q85_consumer_a": float("nan"),
                        "q85_abs_error": float("nan"),
                        "q85_rel_error": float("nan"),
                        "predicate": predicate_confusion(np.array([]), np.array([])),
                        "quantiles": [],
                        "offline_quantiles": quantiles(np.array([])),
                        "consumer_a_quantiles": quantiles(np.array([])),
                        "quantile_monotone_offline": False,
                        "quantile_monotone_consumer_a": False,
                    },
                    "S_A_FP": {
                        "slice": "S_A_FP",
                        "quantity": q,
                        "n": 0,
                        "n_gt": 0,
                        "n_fp": 0,
                        "spearman_rho": float("nan"),
                        "spearman_ci_low": float("nan"),
                        "spearman_ci_high": float("nan"),
                        "n_clusters": 0,
                        "q85_offline": float("nan"),
                        "q85_consumer_a": float("nan"),
                        "q85_abs_error": float("nan"),
                        "q85_rel_error": float("nan"),
                        "predicate": predicate_confusion(np.array([]), np.array([])),
                        "quantiles": [],
                        "offline_quantiles": quantiles(np.array([])),
                        "consumer_a_quantiles": quantiles(np.array([])),
                        "quantile_monotone_offline": False,
                        "quantile_monotone_consumer_a": False,
                    },
                    "gap_1_10": {
                        "slice": "gap_1_10",
                        "quantity": q,
                        "n": 0,
                        "n_gt": 0,
                        "n_fp": 0,
                        "spearman_rho": float("nan"),
                        "spearman_ci_low": float("nan"),
                        "spearman_ci_high": float("nan"),
                        "n_clusters": 0,
                        "q85_offline": float("nan"),
                        "q85_consumer_a": float("nan"),
                        "q85_abs_error": float("nan"),
                        "q85_rel_error": float("nan"),
                        "predicate": predicate_confusion(np.array([]), np.array([])),
                        "quantiles": [],
                        "offline_quantiles": quantiles(np.array([])),
                        "consumer_a_quantiles": quantiles(np.array([])),
                        "quantile_monotone_offline": False,
                        "quantile_monotone_consumer_a": False,
                    },
                    "gap_11_26": {
                        "slice": "gap_11_26",
                        "quantity": q,
                        "n": 0,
                        "n_gt": 0,
                        "n_fp": 0,
                        "spearman_rho": float("nan"),
                        "spearman_ci_low": float("nan"),
                        "spearman_ci_high": float("nan"),
                        "n_clusters": 0,
                        "q85_offline": float("nan"),
                        "q85_consumer_a": float("nan"),
                        "q85_abs_error": float("nan"),
                        "q85_rel_error": float("nan"),
                        "predicate": predicate_confusion(np.array([]), np.array([])),
                        "quantiles": [],
                        "offline_quantiles": quantiles(np.array([])),
                        "consumer_a_quantiles": quantiles(np.array([])),
                        "quantile_monotone_offline": False,
                        "quantile_monotone_consumer_a": False,
                    },
                    "gap_1_10_GT": {
                        "slice": "gap_1_10_GT",
                        "quantity": q,
                        "n": 0,
                        "n_gt": 0,
                        "n_fp": 0,
                        "spearman_rho": float("nan"),
                        "spearman_ci_low": float("nan"),
                        "spearman_ci_high": float("nan"),
                        "n_clusters": 0,
                        "q85_offline": float("nan"),
                        "q85_consumer_a": float("nan"),
                        "q85_abs_error": float("nan"),
                        "q85_rel_error": float("nan"),
                        "predicate": predicate_confusion(np.array([]), np.array([])),
                        "quantiles": [],
                        "offline_quantiles": quantiles(np.array([])),
                        "consumer_a_quantiles": quantiles(np.array([])),
                        "quantile_monotone_offline": False,
                        "quantile_monotone_consumer_a": False,
                    },
                    "gap_11_26_GT": {
                        "slice": "gap_11_26_GT",
                        "quantity": q,
                        "n": 0,
                        "n_gt": 0,
                        "n_fp": 0,
                        "spearman_rho": float("nan"),
                        "spearman_ci_low": float("nan"),
                        "spearman_ci_high": float("nan"),
                        "n_clusters": 0,
                        "q85_offline": float("nan"),
                        "q85_consumer_a": float("nan"),
                        "q85_abs_error": float("nan"),
                        "q85_rel_error": float("nan"),
                        "predicate": predicate_confusion(np.array([]), np.array([])),
                        "quantiles": [],
                        "offline_quantiles": quantiles(np.array([])),
                        "consumer_a_quantiles": quantiles(np.array([])),
                        "quantile_monotone_offline": False,
                        "quantile_monotone_consumer_a": False,
                    },
                }
                for q in ("bdist", "dist_h", "fwd_r", "bwd_r")
            },
            "estimator_decomposition": [],
        }
    )

    # Ensure required bdist slices exist for verdict
    by_quantity: dict[str, Any] = metrics["by_quantity"]  # type: ignore[assignment]
    bdist_m: dict[str, Any] = by_quantity["bdist"]
    verdict_info = evaluate_verdict(coverage, bdist_m)

    disagree = disagreement_localization(matched)
    overall_rows, gap_rows, seq_rows = flatten_metrics_tables(metrics)
    pred_rows = build_predicate_table(metrics)
    quant_rows = build_quantile_table(metrics)
    boundary_rows = boundary_diagnostics(matched)
    decomp_raw = metrics.get("estimator_decomposition")
    decomp_rows: list[dict[str, Any]] = (
        list(decomp_raw) if isinstance(decomp_raw, list) else []
    )

    # Write artifacts
    capture_sha = write_gzip_csv(
        output_dir / "consumer_a_capture.csv.gz", CAPTURE_FIELDS, capture_rows
    )
    join_sha = write_gzip_csv(
        output_dir / "same_event_join.csv.gz", JOIN_FIELDS, join_rows
    )

    metrics_summary = {
        "schema_version": 1,
        "primary_support": PRIMARY_SUPPORT,
        "production_threshold": PRODUCTION_BRIDGE_PX,
        "coverage": coverage,
        "capture_stats": capture_stats,
        "verdict": verdict_info["verdict"],
        "verdict_checks": verdict_info["checks"],
        "bdist_headline": {
            "S_A_overall": {
                k: bdist_m["S_A_overall"][k]
                for k in (
                    "n",
                    "spearman_rho",
                    "spearman_ci_low",
                    "spearman_ci_high",
                    "q85_abs_error",
                )
            },
            "S_A_GT": {
                k: bdist_m["S_A_GT"][k]
                for k in (
                    "n",
                    "spearman_rho",
                    "q85_abs_error",
                )
            },
            "S_A_FP": {
                k: bdist_m["S_A_FP"][k]
                for k in (
                    "n",
                    "spearman_rho",
                    "q85_abs_error",
                )
            },
            "predicate_overall": bdist_m["S_A_overall"].get("predicate"),
            "predicate_gt": bdist_m["S_A_GT"].get("predicate"),
            "predicate_fp": bdist_m["S_A_FP"].get("predicate"),
        },
        "estimator_decomposition": decomp_rows,
        "disagreement_n_su": len(disagree),
        "phase_b_authorized": False,
        "a1_a8_computed": False,
        "production_changed": False,
        "evidence_role": RECONSTRUCTION_QUANTITY_NOTE,
        "runtime_capture_available": LIVE_CUDA_EVENT_RING_IMPLEMENTED,
        "issue_112_status": ISSUE_112_STATUS,
        "primary_fail_reason": verdict_info.get("primary_fail_reason"),
        "metric_based_verdict_diagnostic_only": verdict_info.get(
            "metric_based_verdict_diagnostic_only"
        ),
    }
    write_json(output_dir / "metrics_summary.json", metrics_summary)

    write_csv(
        output_dir / "metrics_overall.csv",
        [
            "quantity",
            "slice",
            "n",
            "n_gt",
            "n_fp",
            "spearman_rho",
            "spearman_ci_low",
            "spearman_ci_high",
            "q85_offline",
            "q85_consumer_a",
            "q85_abs_error",
            "predicate_agreement",
            "offline_safe_online_unsafe_count",
            "offline_safe_online_unsafe_rate",
        ],
        overall_rows,
    )
    write_csv(
        output_dir / "metrics_by_gap.csv",
        [
            "quantity",
            "slice",
            "n",
            "n_gt",
            "n_fp",
            "spearman_rho",
            "spearman_ci_low",
            "spearman_ci_high",
            "q85_offline",
            "q85_consumer_a",
            "q85_abs_error",
            "predicate_agreement",
            "offline_safe_online_unsafe_count",
            "offline_safe_online_unsafe_rate",
        ],
        gap_rows,
    )
    write_csv(
        output_dir / "metrics_by_sequence.csv",
        [
            "quantity",
            "slice",
            "n",
            "n_gt",
            "n_fp",
            "spearman_rho",
            "spearman_ci_low",
            "spearman_ci_high",
            "q85_offline",
            "q85_consumer_a",
            "q85_abs_error",
            "predicate_agreement",
            "offline_safe_online_unsafe_count",
            "offline_safe_online_unsafe_rate",
        ],
        seq_rows,
    )
    write_csv(
        output_dir / "quantile_alignment.csv",
        [
            "quantity",
            "slice",
            "quantile",
            "offline_quantile",
            "consumer_a_quantile",
            "absolute_error",
            "relative_error",
            "headline",
        ],
        quant_rows,
    )
    write_csv(
        output_dir / "predicate_confusion.csv",
        [
            "slice",
            "n",
            "offline_safe_online_safe",
            "offline_safe_online_unsafe",
            "offline_unsafe_online_safe",
            "offline_unsafe_online_unsafe",
            "predicate_agreement",
            "offline_safe_online_unsafe_count",
            "offline_safe_online_unsafe_rate",
        ],
        pred_rows,
    )
    write_csv(
        output_dir / "boundary_diagnostics.csv",
        [
            "band",
            "row_count",
            "n_gt",
            "n_fp",
            "predicate_disagreement",
            "offline_safe_online_unsafe_rate",
            "predicate_agreement",
        ],
        boundary_rows,
    )
    write_csv(
        output_dir / "estimator_decomposition.csv",
        [
            "step",
            "factor_changed",
            "n",
            "spearman_rho_vs_reconstruction",
            "q85_abs_error_vs_reconstruction",
            "predicate_agreement_at_0.4",
            "gt_step_safe_recon_unsafe_count",
        ],
        decomp_rows,
    )
    write_csv(
        output_dir / "disagreement_localization.csv",
        [
            "event_key",
            "seq",
            "gap",
            "gt_match",
            "offline_bridge_dist",
            "recon_bdist",
            "offline_h_ref",
            "recon_h_ref",
            "recon_la",
            "true_gap",
            "offline_fwd_r",
            "offline_bwd_r",
            "recon_fwd_r",
            "recon_bwd_r",
            "offline_dist_h",
            "recon_dist_h",
            "gap_bin",
            "offline_h_ref_bin",
            "recon_h_ref_bin",
            "offline_bridge_bin",
            "recon_bdist_bin",
            "regime",
            "regime_count",
            "n_su_total",
            "n_sa",
            "gap_pool_median",
        ],
        disagree,
    )

    runner_sha = sha256(Path(__file__))
    module_sha = sha256(
        REPO / "src/saccade/perception/eval/consumer_a_bridge_fidelity.py"
    )
    preset_info = load_headline_preset_bridge()
    verdict_payload = {
        "schema_version": 2,
        "status": PACKET_STATUS,
        "verdict": verdict_info["verdict"],
        "primary_fail_reason": verdict_info.get("primary_fail_reason"),
        "metric_based_verdict_diagnostic_only": verdict_info.get(
            "metric_based_verdict_diagnostic_only"
        ),
        "issue_112_status": ISSUE_112_STATUS,
        "runtime_capture_available": LIVE_CUDA_EVENT_RING_IMPLEMENTED,
        "evidence_role": RECONSTRUCTION_QUANTITY_NOTE,
        "primary_support": PRIMARY_SUPPORT,
        "production_threshold": PRODUCTION_BRIDGE_PX,
        "source_pairs_sha256": source_sha,
        "consumer_a_source_sha256": kernel_sha,
        "runner_sha256": runner_sha,
        "module_sha256": module_sha,
        "capture_sha256": capture_sha,
        "join_sha256": join_sha,
        "phase_b_authorized": False,
        "a1_a8_computed": False,
        "production_changed": False,
        "coverage_gates_pass": verdict_info["coverage_gates_pass"],
        "verdict_checks": verdict_info["checks"],
        "gap_cell_details": verdict_info["gap_cell_details"],
        "capture_mode": CAPTURE_MODE_RECONSTRUCTION,
        "research_bridge_fidelity_audit_default": RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT,
        "live_cuda_event_ring_implemented": LIVE_CUDA_EVENT_RING_IMPLEMENTED,
        "git_commit": commit,
        "headline_preset_path": preset_info["preset_path"],
        "headline_preset_sha256": preset_info["preset_file_sha256"],
        "headline_preset_resolved_bridge": preset_info["resolved_bridge"],
        "preset_config_hash": preset_info["preset_file_sha256"],
        "substrate_mot_dir": str(SUBSTRATE_MOT_DIR),
    }
    write_json(output_dir / "verdict.json", verdict_payload)

    recorded_lines = [
        f"status={PACKET_STATUS}",
        f"verdict={verdict_info['verdict']}",
        f"primary_fail_reason={verdict_info.get('primary_fail_reason')}",
        f"issue_112_status={ISSUE_112_STATUS}",
        f"runtime_capture_available={LIVE_CUDA_EVENT_RING_IMPLEMENTED}",
        f"evidence_role={RECONSTRUCTION_QUANTITY_NOTE}",
        f"source_pairs_sha256={source_sha}",
        f"kernel_source_sha256={kernel_sha}",
        f"headline_preset_sha256={preset_info['preset_file_sha256']}",
        f"module_sha256={module_sha}",
        f"capture_sha256={capture_sha}",
        f"join_sha256={join_sha}",
        f"coverage_gates_pass={coverage['gates_pass']}",
        f"gt_coverage_s_a={coverage['gt_match_coverage_s_a']:.6f}",
        f"overall_coverage_s_a={coverage['overall_match_coverage_s_a']:.6f}",
        f"exact_matched_rows={coverage['exact_matched_rows']}",
        f"spearman_overall={verdict_info['checks']['spearman_overall']}",
        f"spearman_gt={verdict_info['checks']['spearman_gt']}",
        f"spearman_fp={verdict_info['checks']['spearman_fp']}",
        f"q85_abs_error={verdict_info['checks']['q85_abs_error']}",
        f"pred_agree_overall={verdict_info['checks']['pred_agree_overall']}",
        f"pred_agree_gt={verdict_info['checks']['pred_agree_gt']}",
        f"gt_safe_unsafe_count={verdict_info['checks']['gt_safe_unsafe_count']}",
        "phase_b_authorized=false",
        "a1_a8_computed=false",
        "production_changed=false",
        f"capture_mode={CAPTURE_MODE_RECONSTRUCTION}",
        f"research_audit_default={RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT}",
        f"git_commit={commit}",
    ]
    recorded_path = output_dir / "recorded_output.txt"
    recorded_path.write_text("\n".join(recorded_lines) + "\n", encoding="utf-8")

    # Manifest hashes all artifacts (recorded_output written above).
    artifact_names = [
        "consumer_a_capture.csv.gz",
        "same_event_join.csv.gz",
        "metrics_summary.json",
        "metrics_overall.csv",
        "metrics_by_gap.csv",
        "metrics_by_sequence.csv",
        "quantile_alignment.csv",
        "predicate_confusion.csv",
        "boundary_diagnostics.csv",
        "estimator_decomposition.csv",
        "disagreement_localization.csv",
        "verdict.json",
        "recorded_output.txt",
        "run_d0_bridge_fidelity.py",
    ]
    artifacts: dict[str, str] = {}
    for name in artifact_names:
        p = output_dir / name if name != "run_d0_bridge_fidelity.py" else Path(__file__)
        if name == "run_d0_bridge_fidelity.py":
            dest = output_dir / "run_d0_bridge_fidelity.py"
            if dest.resolve() != Path(__file__).resolve():
                dest.write_text(
                    Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
                )
            artifacts[name] = sha256(dest if dest.exists() else Path(__file__))
        else:
            artifacts[name] = sha256(p)

    manifest = {
        "schema_version": 2,
        "status": PACKET_STATUS,
        "freeze_id": "D0-BRIDGE-ESTIMATOR-FIDELITY-v1",
        "issue_112_status": ISSUE_112_STATUS,
        "runtime_capture_available": LIVE_CUDA_EVENT_RING_IMPLEMENTED,
        "evidence_role": RECONSTRUCTION_QUANTITY_NOTE,
        "primary_fail_reason": verdict_info.get("primary_fail_reason"),
        "source_pairs_csv": str(CANONICAL_PAIRS),
        "source_pairs_sha256": source_sha,
        "substrate_mot_dir": str(SUBSTRATE_MOT_DIR),
        "kernel_source": str(KERNEL_SOURCE),
        "kernel_source_sha256": kernel_sha,
        "runner_sha256": runner_sha,
        "consumer_a_bridge_fidelity_module": str(
            Path("src/saccade/perception/eval/consumer_a_bridge_fidelity.py")
        ),
        "consumer_a_bridge_fidelity_module_sha256": module_sha,
        "headline_preset_path": preset_info["preset_path"],
        "headline_preset_sha256": preset_info["preset_file_sha256"],
        "headline_preset_resolved_bridge": preset_info["resolved_bridge"],
        "production_threshold": PRODUCTION_BRIDGE_PX,
        "primary_support": PRIMARY_SUPPORT,
        "verdict": verdict_info["verdict"],
        "phase_b_authorized": False,
        "a1_a8_computed": False,
        "production_changed": False,
        "artifacts": artifacts,
        "git_commit": commit,
        "preset_config_hash": preset_info["preset_file_sha256"],
    }
    write_json(output_dir / "manifest.json", manifest)

    return {
        "verdict": verdict_info["verdict"],
        "coverage": coverage,
        "metrics_summary": metrics_summary,
        "verdict_payload": verdict_payload,
        "output_dir": str(output_dir),
        "capture_sha256": capture_sha,
        "join_sha256": join_sha,
        "source_sha256": source_sha,
        "kernel_sha256": kernel_sha,
    }


def verify_packet(output_dir: Path, pairs_path: Path, mot_dir: Path) -> None:
    """Rebuild capture from frozen pairs+substrate and require byte identity.

    Guarantees:
    * capture is regenerated from pairs + MOT substrate (not re-read sealed bytes)
    * all downstream artifacts match the sealed packet
    * kernel / module / headline preset provenance hashes match sealed verdict
    """
    output_dir = output_dir if output_dir.is_absolute() else REPO / output_dir
    pairs_path = pairs_path if pairs_path.is_absolute() else REPO / pairs_path
    mot_dir = mot_dir if mot_dir.is_absolute() else REPO / mot_dir

    sealed = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    # Provenance gates before rebuild
    if sha256(pairs_path) != SOURCE_SHA256:
        raise AssertionError("frozen pairs SHA mismatch")
    if sha256(REPO / KERNEL_SOURCE) != sealed["consumer_a_source_sha256"]:
        raise AssertionError("kernel source hash drift")
    if sha256(REPO / HEADLINE_PRESET) != sealed["headline_preset_sha256"]:
        raise AssertionError("headline preset hash drift")
    module_path = REPO / "src/saccade/perception/eval/consumer_a_bridge_fidelity.py"
    if sha256(module_path) != sealed["module_sha256"]:
        raise AssertionError("fidelity module hash drift")
    if sealed.get("live_cuda_event_ring_implemented") is not False:
        raise AssertionError("live capture must remain unimplemented in this packet")
    if sealed.get("issue_112_status") != ISSUE_112_STATUS:
        raise AssertionError("issue_112_status drift")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Full rebuild: regenerate capture from pairs + substrate (no capture_path).
        run_pipeline(pairs_path, mot_dir, tmp_path, capture_path=None)
        # Capture origin byte-compare (pure reconstruction content).
        for name in (
            "consumer_a_capture.csv.gz",
            "same_event_join.csv.gz",
            "metrics_summary.json",
            "metrics_overall.csv",
            "metrics_by_gap.csv",
            "metrics_by_sequence.csv",
            "quantile_alignment.csv",
            "predicate_confusion.csv",
            "boundary_diagnostics.csv",
            "estimator_decomposition.csv",
            "disagreement_localization.csv",
        ):
            a = sha256(output_dir / name)
            b = sha256(tmp_path / name)
            if a != b:
                raise AssertionError(f"--verify mismatch on {name}: {a} != {b}")
        # Verdict/manifest/recorded embed git_commit/runner_sha — compare after
        # normalizing to sealed provenance for fields that track HEAD.
        rebuilt = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
        for key in (
            "verdict",
            "status",
            "primary_fail_reason",
            "issue_112_status",
            "source_pairs_sha256",
            "consumer_a_source_sha256",
            "headline_preset_sha256",
            "module_sha256",
            "capture_sha256",
            "join_sha256",
            "phase_b_authorized",
            "a1_a8_computed",
            "production_changed",
            "production_threshold",
            "capture_mode",
            "runtime_capture_available",
        ):
            if rebuilt.get(key) != sealed.get(key):
                raise AssertionError(
                    f"--verify verdict field {key}: {rebuilt.get(key)!r} != {sealed.get(key)!r}"
                )
        # Capture sha must equal sealed (regenerated without sealed input).
        if rebuilt["capture_sha256"] != sealed["capture_sha256"]:
            raise AssertionError("regenerated capture_sha256 does not match sealed")

    verdict = sealed
    if verdict["status"] != PACKET_STATUS:
        raise AssertionError(f"status must be {PACKET_STATUS}")
    if verdict["verdict"] not in {
        "threshold_transfer_supported",
        "rank_only_transfer_supported",
        "not_fidelity_aligned",
    }:
        raise AssertionError(f"invalid verdict: {verdict['verdict']}")
    if verdict["phase_b_authorized"] is not False:
        raise AssertionError("phase_b_authorized must be false")
    if verdict["a1_a8_computed"] is not False:
        raise AssertionError("a1_a8_computed must be false")
    if verdict["production_changed"] is not False:
        raise AssertionError("production_changed must be false")
    if verdict["production_threshold"] != PRODUCTION_BRIDGE_PX:
        raise AssertionError("production threshold drift")
    if RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT is not False:
        raise AssertionError("audit default must be off")
    if LIVE_CUDA_EVENT_RING_IMPLEMENTED is not False:
        raise AssertionError("live CUDA capture must be unimplemented")
    if verdict["verdict"] != "not_fidelity_aligned":
        raise AssertionError(
            "capture-unavailable fail-closed requires not_fidelity_aligned"
        )
    if verdict.get("primary_fail_reason") != PRIMARY_FAIL_REASON:
        raise AssertionError("primary_fail_reason must be runtime_capture_unavailable")
    print("VERIFY_PASS")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=CANONICAL_PAIRS,
        help="Frozen pairs.csv (SHA-gated)",
    )
    parser.add_argument(
        "--mot-dir",
        type=Path,
        default=SUBSTRATE_MOT_DIR,
        help="Sealed no-relink substrate MOT dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKET_DIR,
        help="Evidence packet directory",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="Optional pre-built consumer_a_capture.csv[.gz]",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild from sealed capture and require byte-identical artifacts",
    )
    args = parser.parse_args(argv)

    if args.verify:
        verify_packet(args.output_dir, args.pairs, args.mot_dir)
        return 0

    result = run_pipeline(
        args.pairs,
        args.mot_dir,
        args.output_dir,
        capture_path=args.capture,
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "coverage_gates_pass": result["coverage"]["gates_pass"],
                "gt_coverage_s_a": result["coverage"]["gt_match_coverage_s_a"],
                "overall_coverage_s_a": result["coverage"][
                    "overall_match_coverage_s_a"
                ],
                "output_dir": result["output_dir"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
