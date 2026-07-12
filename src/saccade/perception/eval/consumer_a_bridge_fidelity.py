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
  and the binding ``docs/research/contracts/runtime_quantity_fidelity_protocol.md``.

Reconstruction metrics remain diagnostics only and must never be labeled
``D4 exact captured Consumer-A``.

This module never mounts a production policy change.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Sequence

# ── Float32 / FMA host helpers ───────────────────────────────────────────────


def _f32(value: float) -> float:
    """Round ``value`` to IEEE-754 binary32 (CUDA ``float``)."""
    packed = struct.pack("<f", float(value))
    # Explicit float() keeps mypy strict (struct.unpack is typed as Any).
    return float(struct.unpack("<f", packed)[0])


@lru_cache(maxsize=1)
def _fmaf_impl() -> Any:
    """Resolve libc ``fmaf`` once — CUDA contracts ``bridge_vel4`` to FMAs."""
    # Python 3.12 has no math.fma; match device bit-patterns via libm.
    for lib_name in ("libm.so.6", "libm.so", "libSystem.dylib", "msvcrt"):
        try:
            lib = ctypes.CDLL(lib_name)
            fn = lib.fmaf
            fn.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
            fn.restype = ctypes.c_float
            return fn
        except (OSError, AttributeError):
            continue
    raise RuntimeError(
        "libc fmaf is required for CUDA-faithful Consumer-A host replay "
        "(bridge_vel4 FMA contraction)"
    )


def _fmaf(a: float, b: float, c: float) -> float:
    """``fmaf(a, b, c)`` → binary32 fused multiply-add, matching CUDA FMAs."""
    return float(_fmaf_impl()(_f32(a), _f32(b), _f32(c)))


# ── Device replay backend (research-only) ────────────────────────────────────

REPLAY_BACKEND_DEVICE: Final[str] = "device_bridge_anchor4"
REPLAY_BACKEND_HOST_FMA: Final[str] = "host_binary32_fma"
REQUIRE_DEVICE_ENV: Final[str] = "SACCADE_RESEARCH_R1_REQUIRE_DEVICE_REPLAY"

_EVAL_DIR: Final[Path] = Path(__file__).resolve().parent
_DEVICE_SO_PATH: Final[Path] = _EVAL_DIR / "_cuda" / "libr1_bridge_replay.so"
_DEVICE_SRC_PATH: Final[Path] = _EVAL_DIR / "_cuda" / "r1_bridge_replay.cu"
_DEVICE_BUILD_META_PATH: Final[Path] = (
    _EVAL_DIR / "_cuda" / "libr1_bridge_replay.build.json"
)
_PRODUCTION_TRACKER_PATH: Final[Path] = (
    _EVAL_DIR.parents[2] / "tracking" / "tracker_gpu.cu"
)  # src/tracking/tracker_gpu.cu

# Programmatic override for tests / authority verifier. None → read env.
_REQUIRE_DEVICE_OVERRIDE: bool | None = None


def set_require_device_replay(required: bool | None) -> None:
    """Force device backend (True), allow host fallback (False), or env (None)."""
    global _REQUIRE_DEVICE_OVERRIDE
    _REQUIRE_DEVICE_OVERRIDE = required


def require_device_replay() -> bool:
    """Whether R0 must use the device helper (authority fail-closed)."""
    if _REQUIRE_DEVICE_OVERRIDE is not None:
        return _REQUIRE_DEVICE_OVERRIDE
    return os.environ.get(REQUIRE_DEVICE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_build_meta() -> dict[str, Any]:
    if not _DEVICE_BUILD_META_PATH.is_file():
        return {}
    try:
        payload = json.loads(_DEVICE_BUILD_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _gpu_runtime_info() -> dict[str, Any]:
    """Best-effort GPU identity for authority provenance (optional)."""
    info: dict[str, Any] = {
        "gpu_name": None,
        "gpu_compute_capability": None,
        "cuda_runtime_available": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        info["cuda_runtime_available"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["gpu_compute_capability"] = f"{major}.{minor}"
    except Exception:
        return info
    return info


@lru_cache(maxsize=1)
def _device_replay_lib_state() -> tuple[Any | None, str | None]:
    """Return ``(lib, load_error)``.  Host fallback is only silent when allowed."""
    if not _DEVICE_SO_PATH.is_file():
        return None, f"missing device helper: {_DEVICE_SO_PATH}"
    try:
        lib = ctypes.CDLL(str(_DEVICE_SO_PATH))
    except OSError as exc:
        return None, f"failed to load {_DEVICE_SO_PATH}: {exc}"
    try:
        lib.r1_bridge_anchor4_batch.argtypes = (
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        )
        lib.r1_bridge_anchor4_batch.restype = ctypes.c_int
        lib.r1_bridge_vel4_batch.argtypes = (
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        )
        lib.r1_bridge_vel4_batch.restype = ctypes.c_int
    except AttributeError as exc:
        return None, f"device helper missing exported symbols: {exc}"
    return lib, None


def _device_replay_lib() -> Any | None:
    lib, error = _device_replay_lib_state()
    if lib is None and require_device_replay():
        raise RuntimeError(
            "R1 authority replay requires the device backend "
            f"({REQUIRE_DEVICE_ENV}=1 / --require-device), but it is unavailable: "
            f"{error}. Build with: bash scripts/tools/build_r1_bridge_replay.sh"
        )
    return lib


def active_replay_backend() -> str:
    """Backend that will serve the next ``bridge_anchor4`` / ``bridge_vel4`` call."""
    lib, _error = _device_replay_lib_state()
    if lib is not None:
        return REPLAY_BACKEND_DEVICE
    if require_device_replay():
        # Surface the same failure path before any silent host math.
        _device_replay_lib()
    return REPLAY_BACKEND_HOST_FMA


def replay_backend_provenance() -> dict[str, Any]:
    """Auditable identity of the R0 calculator (source + optional .so + GPU).

    Authority packets must record this block so a byte-identical payload under
    the same tool revision and tolerance can be re-checked without guessing
    whether host FMA or the device helper produced the numbers.
    """
    lib, load_error = _device_replay_lib_state()
    backend = REPLAY_BACKEND_DEVICE if lib is not None else REPLAY_BACKEND_HOST_FMA
    if require_device_replay() and lib is None:
        backend = "unavailable_device_required"
    build_meta = _load_build_meta()
    gpu = _gpu_runtime_info()
    return {
        "replay_backend": backend,
        "require_device": require_device_replay(),
        "consumer_module_path": str(
            Path("src/saccade/perception/eval/consumer_a_bridge_fidelity.py")
        ),
        "consumer_module_sha256": _sha256_file(Path(__file__).resolve()),
        "device_helper_source_path": "src/saccade/perception/eval/_cuda/r1_bridge_replay.cu",
        "device_helper_source_sha256": _sha256_file(_DEVICE_SRC_PATH),
        "device_helper_binary_path": (
            "src/saccade/perception/eval/_cuda/libr1_bridge_replay.so"
            if _DEVICE_SO_PATH.is_file()
            else None
        ),
        "device_helper_binary_sha256": _sha256_file(_DEVICE_SO_PATH),
        "device_helper_loadable": lib is not None,
        "device_helper_load_error": load_error if lib is None else None,
        "device_helper_build_meta_path": (
            "src/saccade/perception/eval/_cuda/libr1_bridge_replay.build.json"
            if _DEVICE_BUILD_META_PATH.is_file()
            else None
        ),
        "nvcc_version": build_meta.get("nvcc_version"),
        "compile_flags": build_meta.get("compile_flags"),
        "cuda_architectures": build_meta.get("cuda_architectures"),
        "gpu_name": gpu["gpu_name"],
        "gpu_compute_capability": gpu["gpu_compute_capability"],
        "cuda_runtime_available": gpu["cuda_runtime_available"],
        "production_tracker_source_path": "src/tracking/tracker_gpu.cu",
        "production_tracker_source_sha256": _sha256_file(_PRODUCTION_TRACKER_PATH),
    }


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
    """CUDA ``bridge_vel4``: v = (3 y3 + y2 − y1 − 3 y0) / 10.

    Prefer the device batch helper when available.  Host float64 evaluation of
    the closed form diverges by up to ~6e-5 on real MOT rings (above the sealed
    R1 1e-5 budget); the libc-FMA binary32 fallback contracts like ``-O3`` nvcc.
    """
    lib = _device_replay_lib()
    if lib is not None:
        buf = (ctypes.c_float * 4)(*(_f32(samples[i]) for i in range(4)))
        out = (ctypes.c_float * 1)()
        err = int(lib.r1_bridge_vel4_batch(buf, out, 1))
        if err != 0:
            raise RuntimeError(f"device bridge_vel4 failed with cuda error {err}")
        return float(out[0])
    y0, y1, y2, y3 = (_f32(samples[i]) for i in range(4))
    # Matches device: fmaf(y3,3,y2) → fmaf(y1,-1,t) → fmaf(y0,-3,t) → /10.
    acc = _fmaf(y3, 3.0, y2)
    acc = _fmaf(y1, -1.0, acc)
    acc = _fmaf(y0, -3.0, acc)
    return _f32(acc / 10.0)


def bridge_linres4(samples: Sequence[float]) -> float:
    """CUDA ``bridge_linres4`` residual sum of squares vs OLS line (float32+FMA).

    Adaptive-anchor weights depend on these residuals; plain binary32 multiply-
    add drifts ``ay``/``cy0`` by up to ~1.2e-4 on real MOT rings, above the
    sealed R1 1e-5 budget. Matching the device FMA contraction restores them.
    """
    y = [_f32(samples[i]) for i in range(4)]
    total = _f32(y[0] + y[1])
    total = _f32(total + y[2])
    total = _f32(total + y[3])
    ybar = _f32(0.25 * total)
    # sxy = Σ c_i (y_i - ybar) with c = (-1.5, -0.5, 0.5, 1.5)
    sxy = 0.0
    for coeff, yi in zip((-1.5, -0.5, 0.5, 1.5), y):
        sxy = _fmaf(_f32(yi - ybar), coeff, sxy)
    slope = _f32(sxy / 5.0)
    res = 0.0
    for i in range(4):
        fit = _fmaf(slope, _f32(_f32(i) - 1.5), ybar)
        d = _f32(y[i] - fit)
        res = _fmaf(d, d, res)
    return res


def bridge_anchor4(
    ring4: Sequence[tuple[float, float, float]],
    *,
    anchor_mode: int,
    rate_gate: float,
    endpoint_idx: int,
) -> tuple[float, float, float, float]:
    """CUDA ``bridge_anchor4`` host replica (device kernel when available).

    ``ring4`` is four chronological ``(cx, cy, h)`` samples consumed by the
    kernel (candidate head-4 or lost last-4).  Returns ``(ax, ay, vx, vy)``.
    """
    if len(ring4) != 4:
        raise ValueError("bridge_anchor4 requires exactly 4 samples")
    lib = _device_replay_lib()
    if lib is not None:
        flat = (ctypes.c_float * 12)(*(_f32(v) for sample in ring4 for v in sample))
        modes = (ctypes.c_int * 1)(int(anchor_mode))
        rates = (ctypes.c_float * 1)(_f32(rate_gate))
        endpoints = (ctypes.c_int * 1)(int(endpoint_idx))
        out = (ctypes.c_float * 4)()
        err = int(lib.r1_bridge_anchor4_batch(flat, modes, rates, endpoints, out, 1))
        if err != 0:
            raise RuntimeError(f"device bridge_anchor4 failed with cuda error {err}")
        return float(out[0]), float(out[1]), float(out[2]), float(out[3])

    # Host fallback: binary32 + FMA.  Exact on center velocity; adaptive
    # residual weights may still drift a few 1e-5 without the device lib.
    cx = [_f32(p[0]) for p in ring4]
    cy = [_f32(p[1]) for p in ring4]
    h = [_f32(p[2]) for p in ring4]
    yt = [_f32(c - _f32(0.5 * hh)) for c, hh in zip(cy, h)]
    yb = [_f32(c + _f32(0.5 * hh)) for c, hh in zip(cy, h)]
    hbar = 0.0
    for hh in h:
        hbar = _f32(hbar + _f32(0.25 * hh))
    vx = bridge_vel4(cx)
    ax = cx[endpoint_idx]
    use_edges = anchor_mode == 2
    rate = _f32(rate_gate)
    if use_edges and rate > 0.0:
        dh = _f32(
            (abs(_f32(h[1] - h[0])) + abs(_f32(h[2] - h[1])) + abs(_f32(h[3] - h[2])))
            / 3.0
        )
        if _f32(dh / _f32(hbar + 1e-3)) <= rate:
            use_edges = False
    if anchor_mode == 1:
        vy = bridge_vel4(yb)
        ay = yb[endpoint_idx]
    elif use_edges:
        hn = _f32(_f32(hbar * hbar) + 1e-3)
        wt = _f32(1.0 / _f32(_f32(bridge_linres4(yt) / hn) + 0.01))
        wb = _f32(1.0 / _f32(_f32(bridge_linres4(yb) / hn) + 0.01))
        ws = _f32(wt + wb)
        vy = _f32(_fmaf(wt, bridge_vel4(yt), _fmaf(wb, bridge_vel4(yb), 0.0)) / ws)
        ay = _f32(_fmaf(wt, yt[endpoint_idx], _fmaf(wb, yb[endpoint_idx], 0.0)) / ws)
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


def _hypot_f32(dx: float, dy: float) -> float:
    """CUDA ``sqrtf(dx*dx + dy*dy)`` host replica."""
    dx_f = _f32(dx)
    dy_f = _f32(dy)
    return _f32(math.sqrt(_f32(_f32(dx_f * dx_f) + _f32(dy_f * dy_f))))


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

    Returns ``(bdist, dist_h, fwd_r, bwd_r, s_lost, w)`` in binary32 algebra.
    """
    href = _f32(max(_f32(h_ref), 1.0))
    la = _f32(horizon)
    lx_f, ly_f = _f32(lx), _f32(ly)
    cx0_f, cy0_f = _f32(cx0), _f32(cy0)
    vxl_f, vyl_f = _f32(vxl), _f32(vyl)
    vxc_f, vyc_f = _f32(vxc), _f32(vyc)
    fwd_x = _f32(lx_f + _f32(vxl_f * la))
    fwd_y = _f32(ly_f + _f32(vyl_f * la))
    bwd_x = _f32(cx0_f - _f32(vxc_f * la))
    bwd_y = _f32(cy0_f - _f32(vyc_f * la))
    fwd_r = _f32(_hypot_f32(fwd_x - cx0_f, fwd_y - cy0_f) / href)
    bwd_r = _f32(_hypot_f32(bwd_x - lx_f, bwd_y - ly_f) / href)
    dist_h = _f32(_hypot_f32(lx_f - cx0_f, ly_f - cy0_f) / href)
    s_lost = _f32(_hypot_f32(vxl_f, vyl_f) / href)
    w = _f32(math.sqrt(min(max(_f32(s_lost / _f32(SPEED_WEIGHT_REF)), 0.0), 1.0)))
    bdist = _f32(
        _f32(_f32(w * 0.5) * _f32(fwd_r + bwd_r)) + _f32(_f32(1.0 - w) * dist_h)
    )
    # Keep this branch line-for-line aligned with relink_bidir_propose_kernel.
    # It is normally dormant in the active preset, but an R1 replay cannot
    # silently change the quantity when a capture records a non-zero bonus.
    bonus = _f32(bridge_dir_bonus)
    if bonus > 0.0:
        sl = _hypot_f32(vxl_f, vyl_f)
        sc = _hypot_f32(vxc_f, vyc_f)
        min_speed = min(sl, sc)
        speed_trust = min(_f32(min_speed / max(_f32(href * 0.005), 1e-3)), 1.0)
        if speed_trust > 0.0:
            cos_sim = _f32(
                _f32(_f32(vxl_f * vxc_f) + _f32(vyl_f * vyc_f))
                / max(_f32(sl * sc), 1e-9)
            )
            if cos_sim > 0.5:
                ux = _f32(_f32(vxl_f / sl) + _f32(vxc_f / sc))
                uy = _f32(_f32(vyl_f / sl) + _f32(vyc_f / sc))
                un = _hypot_f32(ux, uy)
                ux = _f32(ux / un)
                uy = _f32(uy / un)
                px, py = _f32(-uy), ux
                fe_x = _f32(_f32(lx_f + _f32(vxl_f * la)) - cx0_f)
                fe_y = _f32(_f32(ly_f + _f32(vyl_f * la)) - cy0_f)
                fwd_cross = _f32(abs(_f32(_f32(fe_x * px) + _f32(fe_y * py))) / href)
                be_x = _f32(_f32(cx0_f - _f32(vxc_f * la)) - lx_f)
                be_y = _f32(_f32(cy0_f - _f32(vyc_f * la)) - ly_f)
                bwd_cross = _f32(abs(_f32(_f32(be_x * px) + _f32(be_y * py))) / href)
                bdist_dir = _f32(0.5 * _f32(fwd_cross + bwd_cross))
                gap_scale = min(_f32(la / 30.0), 1.0)
                alpha = _f32(bonus * cos_sim * cos_sim * speed_trust * gap_scale)
                alpha = min(alpha, 1.0)
                bdist = _f32(_f32(bdist * _f32(1.0 - alpha)) + _f32(bdist_dir * alpha))
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
        lc_x, lc_y, lh = (_f32(v) for v in lost_ring[-1])
        lx = lc_x
        ly = _f32(lc_y + _f32(0.5 * lh)) if anchor_mode == 1 else lc_y
        vxl, vyl = 0.0, 0.0
    cx0, cy0, vxc, vyc = bridge_anchor4(
        cand_ring[:4],
        anchor_mode=anchor_mode,
        rate_gate=rate_gate,
        endpoint_idx=0,
    )
    h_ref = _f32(max(_f32(0.5 * _f32(_f32(ema_lost) + _f32(ema_cand))), 1.0))
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
