#!/usr/bin/env python3
"""Run the #465 PR-2 head parity gate (TRT head vs PyTorch oracle) and write its packet.

Implements the frozen pre-declaration
``docs/reference/native_runtime_head_parity_declaration.md`` (blob
``FROZEN_DECLARATION_BLOB``) without re-deciding anything in it:

* V1 frozen inputs (§2) — checked before any measurement; any mismatch is
  ``UNRESOLVED``.
* L1 (§4) — per-frame head tensors of C (compiled PyTorch head, primary
  oracle), E (eager PyTorch head) and T (PR-1 TRT engine) on identical TRT
  backbone features, over all 5316 frames; V2 re-runs the first 20 frames of
  every sequence and requires bit-identical outputs.
* L2 (§5) — six unmodified ``scripts/eval/mot17.py`` runs
  ``A_C#1, A_T#1, A_N#1, A_C#2, A_T#2, A_N#2``; V3 requires each arm's two runs
  to be byte-identical; metrics are recomputed from counts without rounding.
* Terminal (§6) — first matching of UNRESOLVED, HEAD_PARITY_GROSS_ERROR,
  HEAD_PARITY_EXACT, HEAD_PARITY_WITHIN_TOLERANCE, HEAD_PARITY_OUT_OF_TOLERANCE.

It reads no time quantity. The formal run must be the direct child of a
``machine-bench`` lease on a clean tree::

    .venv/bin/python tools/resctl.py run machine-bench -- \\
        .venv/bin/python scripts/eval/diagnostics/native_head_parity.py

``--smoke-frames N`` exercises the whole pipeline on the first N frames of two
sequences; its packet is marked ``evidence: false`` and its terminal carries a
``SMOKE:`` prefix, so it can never stand in for the gate.
"""
# status: experiment

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# --- frozen by the declaration (§2, §4, §5); change only through §9 ----------
DECLARATION = "docs/reference/native_runtime_head_parity_declaration.md"
FROZEN_DECLARATION_BLOB = "0941a010bdc4ca30344e580108403d6074ebdc67"
HEAD_ONNX_SHA256 = "6e919dad14af81083a25679225930a3473a8cdd6ebf9828ea07b414a9316b58b"
HEAD_STEM = "models/yolo/mamba_head_s_v14replica_t3_t1_fp32"
HEAD_ENGINE = HEAD_STEM + ".engine"
HEAD_LINEAGE = HEAD_STEM + ".lineage.json"
CKPT = "runs/mamba_gt_v14replica_t3_t1/best.ckpt"
CKPT_SHA256 = "c161c88e50b894d8b51cc614c46c3700370373decf05a15825bdf00ccf0e0876"
BACKBONE = "models/yolo/yolo26s_backbone_640_best.engine"
PRESET_NAME = "mamba_whole_graph"
PRESET = f"configs/presets/{PRESET_NAME}.yaml"
EXPORT_TOOL = "scripts/model/export_headline_mamba_head.py"
DATA_ROOT = "datasets/MOT17"
SPLIT = "train"
SEQUENCES = (
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
)
TOTAL_FRAMES = 5316
LEASE = "machine-bench"

IMG_SIZE = 640
SCORE_FLOOR = 0.05  # base_score_floor = min(conf_threshold, track_thresh)
L1_SCORE_MAX = 0.05
L1_BOX_MAX_PX = 4.0
V2_FRAMES = 20
L1_PAIRS = (("T", "C"), ("T", "E"), ("E", "C"))
L1_DECISION_PAIR = ("T", "C")

ARMS = {
    "A_C": [],
    "A_T": ["--mamba-head-engine", HEAD_ENGINE],
    "A_N": ["--no-compile"],
}
RUN_ORDER = (("A_C", 1), ("A_T", 1), ("A_N", 1), ("A_C", 2), ("A_T", 2), ("A_N", 2))
L2_METRICS = ("IDF1", "HOTA", "MOTA", "IDs")
TOL_FLOOR = {"IDF1": 0.20, "HOTA": 0.20, "MOTA": 0.20, "IDs": 5.0}
TOL_CAP = {"IDF1": 1.00, "HOTA": 1.00, "MOTA": 1.00, "IDs": 30.0}

TERMINALS = (
    "UNRESOLVED",
    "HEAD_PARITY_GROSS_ERROR",
    "HEAD_PARITY_EXACT",
    "HEAD_PARITY_WITHIN_TOLERANCE",
    "HEAD_PARITY_OUT_OF_TOLERANCE",
)

# |Δ| histogram for the report-only 99.9th percentile: a zero bin plus 100
# log-spaced bins per decade over [1e-10, 1e3); the reported percentile is the
# upper edge of the bin that holds it (conservative, never below the truth).
HIST_LO_EXP, HIST_HI_EXP, HIST_PER_DECADE = -10, 3, 100


# --------------------------------------------------------------------------
# pure decision logic (unit-tested without a GPU)
# --------------------------------------------------------------------------
def tolerance(delta_n: float, metric: str) -> float:
    """b_m = min(max(|Δ_N,m|, floor_m), cap_m) (declaration §5)."""
    return min(max(abs(delta_n), TOL_FLOOR[metric]), TOL_CAP[metric])


def l1_verdict(score_maxabs: float, box_maxabs_px: float) -> str:
    """κ_L1 on the (T,C) pair; NaN or inf fails (a non-finite Δ is not a pass)."""
    ok = (
        math.isfinite(score_maxabs)
        and math.isfinite(box_maxabs_px)
        and score_maxabs <= L1_SCORE_MAX
        and box_maxabs_px <= L1_BOX_MAX_PX
    )
    return "L1_PASS" if ok else "L1_GROSS_ERROR"


def l2_verdict(
    metrics_c: dict[str, float],
    metrics_t: dict[str, float],
    metrics_n: dict[str, float],
    exact: bool,
) -> tuple[str, dict[str, dict[str, float | bool]]]:
    """κ_L2: EXACT if the A_T and A_C txt are identical, else WITHIN/OUT (two-sided)."""
    detail: dict[str, dict[str, float | bool]] = {}
    for m in L2_METRICS:
        d_t = metrics_t[m] - metrics_c[m]
        d_n = metrics_n[m] - metrics_c[m]
        b = tolerance(d_n, m)
        detail[m] = {
            "C": metrics_c[m],
            "T": metrics_t[m],
            "N": metrics_n[m],
            "delta_T": d_t,
            "delta_N": d_n,
            "tolerance": b,
            "within": abs(d_t) <= b,
        }
    if exact:
        return "L2_EXACT", detail
    within = all(bool(v["within"]) for v in detail.values())
    return ("L2_WITHIN" if within else "L2_OUT"), detail


def decide_terminal(validity_ok: bool, l1: str | None, l2: str | None) -> str:
    """Declaration §6, evaluated in order; anything missing is UNRESOLVED."""
    if not validity_ok or l1 is None or l2 is None:
        return "UNRESOLVED"
    if l1 == "L1_GROSS_ERROR":
        return "HEAD_PARITY_GROSS_ERROR"
    if l1 != "L1_PASS":
        return "UNRESOLVED"
    return {
        "L2_EXACT": "HEAD_PARITY_EXACT",
        "L2_WITHIN": "HEAD_PARITY_WITHIN_TOLERANCE",
        "L2_OUT": "HEAD_PARITY_OUT_OF_TOLERANCE",
    }.get(l2, "UNRESOLVED")


def metrics_from_counts(
    counts: dict[str, int], hota: dict[str, float]
) -> dict[str, float]:
    """Unrounded percentages from summed motmetrics counts (the formulas of
    ``metrics._format_overall_metrics_from_counts``) plus TrackEval HOTA."""
    idtp, idfp, idfn = counts["idtp"], counts["idfp"], counts["idfn"]
    idf1_den = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / idf1_den) if idf1_den > 0 else 0.0
    fp, fn, ids = (
        counts["num_false_positives"],
        counts["num_misses"],
        counts["num_switches"],
    )
    mota = 1.0 - (fn + fp + ids) / max(counts["num_objects"], 1)
    return {
        "IDF1": idf1 * 100.0,
        "HOTA": hota["HOTA"] * 100.0,
        "MOTA": mota * 100.0,
        "IDs": float(ids),
        "DetA": hota["DetA"] * 100.0,
        "AssA": hota["AssA"] * 100.0,
        "FP": float(fp),
        "FN": float(fn),
    }


def first_divergent_frame(a: bytes, b: bytes) -> int | None:
    """First MOT frame whose ordered output rows differ; None when identical."""
    if a == b:
        return None

    def by_frame(data: bytes) -> dict[int, list[bytes]]:
        rows: dict[int, list[bytes]] = {}
        for line in data.splitlines():
            if line.strip():
                rows.setdefault(int(float(line.split(b",", 1)[0])), []).append(line)
        return rows

    ra, rb = by_frame(a), by_frame(b)
    for frame in sorted(set(ra) | set(rb)):
        if ra.get(frame) != rb.get(frame):
            return frame
    return -1  # same rows per frame, different bytes (e.g. trailing whitespace)


def hist_edges() -> list[float]:
    n = (HIST_HI_EXP - HIST_LO_EXP) * HIST_PER_DECADE
    return [10.0 ** (HIST_LO_EXP + i / HIST_PER_DECADE) for i in range(n + 1)]


def hist_quantile(counts: list[int], q: float) -> float | None:
    """Upper edge of the bin holding quantile q; bin 0 is exact zero, the last
    bin is overflow (returned as inf)."""
    total = sum(counts)
    if total == 0:
        return None
    edges = hist_edges()
    target = math.ceil(q * total)
    running = 0
    for i, c in enumerate(counts):
        running += c
        if running >= target:
            if i == 0:
                return 0.0
            return edges[i - 1] if i - 1 < len(edges) else math.inf
    return math.inf


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _seq_frames(seq: str) -> list[Path]:
    return sorted((project_root / DATA_ROOT / SPLIT / seq / "img1").glob("*.jpg"))


def _saccade_env() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("SACCADE_")}


# --------------------------------------------------------------------------
# V1 — frozen inputs (declaration §2)
# --------------------------------------------------------------------------
def check_v1(smoke: bool, sequences: tuple[str, ...]) -> tuple[bool, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(item: str, ok: bool, observed: Any, expected: Any = None) -> None:
        checks.append(
            {"item": item, "ok": bool(ok), "observed": observed, "expected": expected}
        )

    lineage_path = project_root / HEAD_LINEAGE
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads(lineage_path.read_text())
    except Exception as exc:  # noqa: BLE001 — recorded, fails closed
        check("lineage manifest readable", False, repr(exc), HEAD_LINEAGE)
    get = lambda *keys: _dig(manifest, keys)  # noqa: E731

    onnx = project_root / HEAD_STEM
    onnx_sha = _sha_or_none(onnx.with_suffix(".onnx"))
    check("head ONNX sha256", onnx_sha == HEAD_ONNX_SHA256, onnx_sha, HEAD_ONNX_SHA256)
    check(
        "manifest onnx.sha256",
        get("onnx", "sha256") == HEAD_ONNX_SHA256,
        get("onnx", "sha256"),
        HEAD_ONNX_SHA256,
    )
    engine_sha = _sha_or_none(project_root / HEAD_ENGINE)
    check(
        "head engine sha256 == manifest",
        engine_sha is not None and engine_sha == get("engine", "sha256"),
        engine_sha,
        get("engine", "sha256"),
    )
    export_check = subprocess.run(
        [sys.executable, EXPORT_TOOL, "--check"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    last = export_check.stdout.strip().splitlines()[-1:] or [""]
    check(
        "export_headline_mamba_head.py --check",
        export_check.returncode == 0 and last[0].startswith("OK"),
        {"returncode": export_check.returncode, "last_line": last[0]},
        "exit 0, OK",
    )
    ckpt_sha = _sha_or_none(project_root / CKPT)
    check("checkpoint sha256", ckpt_sha == CKPT_SHA256, ckpt_sha, CKPT_SHA256)
    bb_sha = _sha_or_none(project_root / BACKBONE)
    check(
        "backbone engine sha256 == manifest companions",
        bb_sha is not None and bb_sha == get("companions", "backbone_engine", "sha256"),
        bb_sha,
        get("companions", "backbone_engine", "sha256"),
    )
    preset_sha = _sha_or_none(project_root / PRESET)
    check(
        "preset sha256 == manifest",
        preset_sha is not None and preset_sha == get("preset", "sha256"),
        preset_sha,
        get("preset", "sha256"),
    )

    porcelain = _git("status", "--porcelain")
    check("clean tree", smoke or porcelain == "", porcelain or "(clean)", "(clean)")
    runner_rel = str(Path(__file__).resolve().relative_to(project_root))
    blobs = {
        "head": _git("rev-parse", "HEAD"),
        "runner_blob": _blob_or_none(runner_rel),
        "declaration_blob": _blob_or_none(DECLARATION),
        "runner_worktree_blob": _git("hash-object", runner_rel),
        "declaration_worktree_blob": _git("hash-object", DECLARATION),
    }
    check(
        "declaration blob (HEAD and worktree) == frozen",
        blobs["declaration_blob"] == FROZEN_DECLARATION_BLOB
        and blobs["declaration_worktree_blob"] == FROZEN_DECLARATION_BLOB,
        blobs["declaration_blob"],
        FROZEN_DECLARATION_BLOB,
    )
    check(
        "runner committed (HEAD blob == worktree blob)",
        smoke or blobs["runner_blob"] == blobs["runner_worktree_blob"],
        blobs["runner_blob"],
        blobs["runner_worktree_blob"],
    )

    import tensorrt as trt
    import torch

    env = {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "tensorrt_version": trt.__version__,
        "host": platform.node(),
    }
    expected_env = {
        "cudnn_allow_tf32": get("environment", "cudnn_allow_tf32"),
        "matmul_allow_tf32": get("environment", "matmul_allow_tf32"),
        "tensorrt_version": get("engine", "tensorrt_version"),
        "host": get("environment", "host"),
    }
    for k in env:
        check(f"environment {k}", env[k] == expected_env[k], env[k], expected_env[k])

    # The children inherit this environment; a caller-set hatch or parent-run
    # claim would change every arm without appearing on any command line.
    saccade_env = _saccade_env()
    check("no SACCADE_* set by the caller", smoke or not saccade_env, saccade_env, {})

    frames = {s: len(_seq_frames(s)) for s in sequences}
    gts = {
        s: (project_root / DATA_ROOT / SPLIT / s / "gt" / "gt.txt").exists()
        for s in sequences
    }
    check(
        "data sequences and frame count",
        (smoke or (sequences == SEQUENCES and sum(frames.values()) == TOTAL_FRAMES))
        and all(frames.values())
        and all(gts.values()),
        {"frames": frames, "total": sum(frames.values()), "gt": gts},
        {"sequences": list(SEQUENCES), "total": TOTAL_FRAMES},
    )

    lease = held_lease()
    check(
        f"direct child of a {LEASE} lease",
        lease is not None and (smoke or lease.get("resource") == LEASE),
        lease,
        f"{LEASE} held by parent pid {os.getppid()}",
    )

    ok = all(c["ok"] for c in checks)
    return ok, {"ok": ok, "checks": checks, "git": blobs, "lease": lease}


def _dig(obj: Any, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _sha_or_none(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


def _blob_or_none(rel: str) -> str | None:
    try:
        return _git("rev-parse", f"HEAD:{rel}")
    except subprocess.CalledProcessError:
        return None


def held_lease() -> dict[str, Any] | None:
    """The resctl lease whose owner is this process's parent (the ``resctl run``)."""
    out = subprocess.run(
        [sys.executable, "tools/resctl.py", "status", "--json"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    for row in json.loads(out.stdout).get("leases", []):
        owner = row.get("owner") or {}
        if row.get("state") == "BUSY" and owner.get("pid") == os.getppid():
            return {
                "resource": row.get("resource"),
                "pid": owner.get("pid"),
                "start_time": owner.get("start_time"),
                "head": owner.get("head"),
                "command_str": owner.get("command_str"),
            }
    return None


# --------------------------------------------------------------------------
# L1 worker (runs in its own process; declaration §4)
# --------------------------------------------------------------------------
def run_l1_worker(
    out_dir: Path, sequences: tuple[str, ...], max_frames: int | None
) -> int:
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    import saccade_tracking_ext  # noqa: F401  (before torchvision; see export tool)

    import torch
    import torch.nn.functional as F
    from torchvision.io import ImageReadMode, decode_jpeg, read_file

    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        TRTMambaHead,
        _dfl_decode,
        _dist2bbox_xywh,
        build_mamba_gated_detector,
    )

    export = _load_export_tool()
    inputs = export.resolve_inputs(
        export.DEFAULT_YOLO_WEIGHTS, export.DEFAULT_TEACHER_CKPT
    )
    det = build_mamba_gated_detector(
        yolo_pt_path=str(inputs["yolo_weights"]),
        teacher_ckpt=str(inputs["teacher_ckpt"]),
        mamba_ckpt=str(inputs["ckpt"]),
        img_size=IMG_SIZE,
        device="cuda",
        conf_thr=0.001,
        trt_backbone_engine=str(inputs["backbone"]),
        use_whole_graph=True,
    )
    det.eval()
    backbone = det._trt_backbone
    anchors_t = det._whole_graph_anchors.T.unsqueeze(0)
    astrides = det._whole_graph_anchor_strides.squeeze(-1).unsqueeze(0)
    # E: an independent eager instance of the same state dict, taken before C
    # is compiled so that no instance ever toggles compile.
    head_e = copy.deepcopy(det.mamba_head).eval()
    head_c = det.mamba_head
    head_c.set_head_compile(True)
    head_c.set_block_compile(True)
    head_t = TRTMambaHead(str(project_root / HEAD_ENGINE))

    def run_torch(head: Any, feats: list[Any]) -> list[Any]:
        cls, reg = head._forward_eager(list(feats), return_embeddings=False)
        return [t.clone() for t in (*cls, *reg)]

    def run_trt(feats: list[Any]) -> list[Any]:
        cls, reg = head_t.infer_graph(*feats)
        return [t.clone() for t in (*cls, *reg)]

    runners = {
        "C": lambda f: run_torch(head_c, f),
        "E": lambda f: run_torch(head_e, f),
        "T": run_trt,
    }

    def decode(outs: list[Any], sx: float, sy: float) -> tuple[Any, Any]:
        cls_all = torch.cat([c.flatten(2) for c in outs[:3]], dim=2)[0]
        reg_all = torch.cat([r.flatten(2) for r in outs[3:]], dim=2)
        # the anchor/stride decode of _postprocess_mamba_fixed_eager
        bboxes = _dist2bbox_xywh(_dfl_decode(reg_all), anchors_t, dim=1) * astrides
        xywh = bboxes[0].T  # (N, 4)
        xyxy = torch.cat(
            [xywh[:, :2] - xywh[:, 2:4] / 2, xywh[:, :2] + xywh[:, 2:4] / 2], 1
        )
        scale = torch.tensor([sx, sy, sx, sy], dtype=xyxy.dtype, device=xyxy.device)
        return cls_all.sigmoid(), xyxy * scale

    edges = torch.tensor(hist_edges(), dtype=torch.float64, device="cuda")
    nbins = len(edges) + 1

    def hist(values: Any) -> Any:
        v = values.double().flatten()
        idx = torch.where(v == 0, 0, torch.bucketize(v, edges, right=True) + 1)
        idx = idx.clamp(max=nbins)
        return torch.bincount(idx, minlength=nbins + 1)

    def new_acc() -> dict[str, Any]:
        z = lambda: torch.zeros((), dtype=torch.float64, device="cuda")  # noqa: E731
        return {
            "score_max": z(),
            "box_max": z(),
            "score_hist": torch.zeros(nbins + 1, dtype=torch.int64, device="cuda"),
            "box_hist": torch.zeros(nbins + 1, dtype=torch.int64, device="cuda"),
            "masked_anchors": torch.zeros((), dtype=torch.int64, device="cuda"),
            "crossings": torch.zeros((), dtype=torch.int64, device="cuda"),
            "nonfinite": torch.zeros((), dtype=torch.int64, device="cuda"),
            "frames": 0,
        }

    per_seq: dict[str, dict[str, dict[str, Any]]] = {}
    v2_failures: list[dict[str, Any]] = []
    v2_checked = 0
    rgb = ImageReadMode.RGB
    with torch.inference_mode():
        for seq in sequences:
            accs = {f"{a},{b}": new_acc() for a, b in L1_PAIRS}
            frames = _seq_frames(seq)
            if max_frames is not None:
                frames = frames[:max_frames]
            for i, path in enumerate(frames):
                img = decode_jpeg(read_file(str(path)), device="cuda", mode=rgb)
                h_orig, w_orig = int(img.shape[1]), int(img.shape[2])
                frame = img.float().unsqueeze(0) / 255
                frame_640 = F.interpolate(
                    frame,
                    size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                feats = [p.clone() for p in backbone.infer(frame_640)]
                outs = {k: fn(feats) for k, fn in runners.items()}
                if i < V2_FRAMES:
                    v2_checked += 1
                    for k, fn in runners.items():
                        again = fn(feats)
                        if not all(torch.equal(a, b) for a, b in zip(outs[k], again)):
                            v2_failures.append(
                                {"sequence": seq, "frame_index": i, "head": k}
                            )
                sx, sy = w_orig / IMG_SIZE, h_orig / IMG_SIZE
                dec = {k: decode(v, sx, sy) for k, v in outs.items()}
                for a, b in L1_PAIRS:
                    acc = accs[f"{a},{b}"]
                    (sa, ba), (sb, bb) = dec[a], dec[b]
                    ds = (sa - sb).abs()
                    ma, mb = (
                        sa.max(0).values >= SCORE_FLOOR,
                        sb.max(0).values >= SCORE_FLOOR,
                    )
                    mask = ma | mb
                    db = (ba - bb).abs()[mask]
                    acc["nonfinite"] += (~torch.isfinite(ds)).sum() + (
                        ~torch.isfinite(db)
                    ).sum()
                    acc["score_max"] = torch.maximum(
                        acc["score_max"], ds.max().double()
                    )
                    if db.numel():
                        acc["box_max"] = torch.maximum(
                            acc["box_max"], db.max().double()
                        )
                    acc["score_hist"] += hist(ds)
                    acc["box_hist"] += hist(db)
                    acc["masked_anchors"] += mask.sum()
                    acc["crossings"] += (ma ^ mb).sum()
                    acc["frames"] += 1
            per_seq[seq] = {
                pair: {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in acc.items()
                }
                for pair, acc in accs.items()
            }
            print(f"[L1] {seq}: {len(frames)} frames", flush=True)

    rows = []
    summary: dict[str, dict[str, Any]] = {}
    for a, b in L1_PAIRS:
        pair = f"{a},{b}"
        tot_score_hist = [0] * (nbins + 1)
        tot_box_hist = [0] * (nbins + 1)
        agg = {
            "score_max": 0.0,
            "box_max": 0.0,
            "masked_anchors": 0,
            "crossings": 0,
            "nonfinite": 0,
            "frames": 0,
        }
        for seq in sequences:
            s = per_seq[seq][pair]
            rows.append(_l1_row(pair, seq, s))
            tot_score_hist = [x + y for x, y in zip(tot_score_hist, s["score_hist"])]
            tot_box_hist = [x + y for x, y in zip(tot_box_hist, s["box_hist"])]
            agg["score_max"] = max(agg["score_max"], s["score_max"])
            agg["box_max"] = max(agg["box_max"], s["box_max"])
            for k in ("masked_anchors", "crossings", "nonfinite", "frames"):
                agg[k] += s[k]
        all_row = _l1_row(
            pair, "ALL", {**agg, "score_hist": tot_score_hist, "box_hist": tot_box_hist}
        )
        rows.append(all_row)
        summary[pair] = all_row
    with (out_dir / "l1_pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tc = summary[",".join(L1_DECISION_PAIR)]
    score_max = math.nan if tc["nonfinite"] else tc["score_maxabs"]
    verdict = l1_verdict(score_max, tc["box_maxabs_px"])
    _write_json(
        out_dir / "l1.json",
        {
            "frames": summary[",".join(L1_DECISION_PAIR)]["frames"],
            "decision_pair": ",".join(L1_DECISION_PAIR),
            "thresholds": {
                "score_maxabs": L1_SCORE_MAX,
                "box_maxabs_px": L1_BOX_MAX_PX,
                "score_floor": SCORE_FLOOR,
            },
            "verdict": verdict,
            "summary": summary,
            "v2": {
                "frames_checked": v2_checked,
                "failures": v2_failures,
                "ok": not v2_failures,
            },
            "histogram": {
                "lo_exp": HIST_LO_EXP,
                "hi_exp": HIST_HI_EXP,
                "per_decade": HIST_PER_DECADE,
            },
            "per_sequence_histograms": {
                seq: {
                    pair: {"score_hist": v["score_hist"], "box_hist": v["box_hist"]}
                    for pair, v in d.items()
                }
                for seq, d in per_seq.items()
            },
            "environment": {
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "torch": torch.__version__,
            },
        },
    )
    print(f"[L1] verdict {verdict}; V2 failures {len(v2_failures)}", flush=True)
    return 0


def _l1_row(pair: str, seq: str, s: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair": pair,
        "sequence": seq,
        "frames": s["frames"],
        "score_maxabs": s["score_max"],
        "score_p999_upper": hist_quantile(s["score_hist"], 0.999),
        "box_maxabs_px": s["box_max"],
        "box_p999_upper_px": hist_quantile(s["box_hist"], 0.999),
        "masked_anchors": s["masked_anchors"],
        "floor_crossings": s["crossings"],
        "nonfinite": s["nonfinite"],
    }


def _load_export_tool() -> Any:
    sys.path.insert(0, str(project_root / "scripts" / "model"))
    spec = importlib.util.spec_from_file_location(
        "export_headline_mamba_head", project_root / EXPORT_TOOL
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# L2 (declaration §5)
# --------------------------------------------------------------------------
def run_l2_arm(
    l2_dir: Path, arm: str, rep: int, sequences: tuple[str, ...], max_frames: int | None
) -> dict[str, Any]:
    out = l2_dir / f"{arm}_{rep}"
    out.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable,
        "scripts/eval/mot17.py",
        "--preset",
        PRESET_NAME,
        "--detector",
        "SDP",
        "--double-buffer",
        "--sequences",
        ",".join(sequences),
        "--output",
        str(out.relative_to(project_root)),
        *ARMS[arm],
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    from saccade.perception.eval.assoc_basis import resolved_env_overrides

    log = l2_dir / f"{arm}_{rep}.stdout.log"
    started = _utc()
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT)
    txts = {s: out / f"{s}.txt" for s in sequences}
    return {
        "arm": arm,
        "rep": rep,
        "cmd": cmd,
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": _utc(),
        "stdout": str(log.relative_to(project_root)),
        "output_dir": str(out.relative_to(project_root)),
        "resolved_env_overrides": resolved_env_overrides(),
        "txt_sha256": {
            s: (_sha256(p) if p.exists() else None) for s, p in txts.items()
        },
    }


def arm_manifest_problem(run: dict[str, Any], head: str, smoke: bool) -> str | None:
    """The harness's own run_manifest.json must show this arm's exact command,
    the measured commit and (formal runs) a clean tree."""
    path = project_root / run["output_dir"] / "run_manifest.json"
    if not path.exists():
        return "run_manifest.json missing"
    m = json.loads(path.read_text())
    if m.get("cmdline") != run["cmd"][1:]:
        return f"run_manifest cmdline {m.get('cmdline')} != launched {run['cmd'][1:]}"
    if m.get("commit") != head:
        return f"run_manifest commit {m.get('commit')} != {head}"
    if not smoke and m.get("dirty") is not False:
        return "run_manifest reports a dirty tree"
    return None


def score_arm(out_dir: Path, sequences: tuple[str, ...]) -> dict[str, Any]:
    from saccade.perception.eval import metrics as M

    gt = {
        s: str(project_root / DATA_ROOT / SPLIT / s / "gt" / "gt.txt")
        for s in sequences
    }
    jobs = [(s, gt[s], str(out_dir / f"{s}.txt")) for s in sequences]
    per_counts = {s: M._evaluate_single_sequence(s, g, t) for s, g, t in jobs}
    totals = {
        k: sum(int(c[k]) for c in per_counts.values())
        for k in next(iter(per_counts.values()))
    }
    hota = M._calculate_hota(str(project_root / DATA_ROOT), SPLIT, str(out_dir), jobs)
    if hota is None:
        raise RuntimeError(f"TrackEval HOTA unavailable for {out_dir}")
    per_seq = {}
    for job in jobs:
        h = M._calculate_hota(str(project_root / DATA_ROOT), SPLIT, str(out_dir), [job])
        if h is None:
            raise RuntimeError(f"TrackEval HOTA unavailable for {job[0]}")
        per_seq[job[0]] = {
            "counts": per_counts[job[0]],
            "metrics": metrics_from_counts(per_counts[job[0]], h),
        }
    combined = metrics_from_counts(totals, hota)
    display = M._format_overall_metrics_from_counts(totals)
    display_ok = (
        display["IDF1"] == f"{combined['IDF1']:.1f}%"
        and display["MOTA"] == f"{combined['MOTA']:.1f}%"
        and display["IDs"] == int(combined["IDs"])
    )
    if not display_ok:
        raise RuntimeError(
            f"unrounded metrics disagree with metrics.py display: {display}"
        )
    return {
        "counts": totals,
        "hota_raw": hota,
        "combined": combined,
        "display": {**display, "HOTA": f"{combined['HOTA']:.1f}%"},
        "per_sequence": per_seq,
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet-root", default="results/native_head_parity_465")
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=None,
        help="non-evidence smoke: first N frames of two sequences",
    )
    parser.add_argument(
        "--_l1-worker", dest="l1_worker", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_sequences", dest="seqs", default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    smoke = args.smoke_frames is not None
    if args.l1_worker:
        seqs = tuple(args.seqs.split(",")) if args.seqs else SEQUENCES
        return run_l1_worker(Path(args.l1_worker), seqs, args.smoke_frames)

    sequences = ("MOT17-05-SDP", "MOT17-09-SDP") if smoke else SEQUENCES
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    packet = project_root / args.packet_root / (stamp + ("_smoke" if smoke else ""))
    # ADR 021 AP-2: claim the packet directory before the first result byte.
    # The mot17.py arms claim their own sub-directories: no parent-claim env is
    # passed down, so each arm's run_manifest.json records its own command.
    from scripts.provenance.run_manifest import open_run

    open_run(
        packet,
        produced_by="diagnostic",
        preset=PRESET_NAME,
        detector="SDP",
        dataset=f"{DATA_ROOT} {SPLIT}",
    )
    record: dict[str, Any] = {
        "schema": "saccade.head_parity_packet/v1",
        "issue": "#465 Phase B PR-2 (U1b)",
        "declaration": DECLARATION,
        "evidence": not smoke,
        "mode": "smoke" if smoke else "formal",
        "smoke_frames": args.smoke_frames,
        "sequences": list(sequences),
        "started_utc": _utc(),
        "caller_saccade_env": _saccade_env(),
    }
    print(f"packet: {packet.relative_to(project_root)}", flush=True)

    terminal = "UNRESOLVED"
    reasons: list[str] = []
    l1 = l2 = None
    try:
        v1_ok, v1 = check_v1(smoke, sequences)
        record["v1"] = v1
        _write_json(packet / "v1.json", v1)
        if not v1_ok:
            reasons.append(
                "V1: " + "; ".join(c["item"] for c in v1["checks"] if not c["ok"])
            )
        else:
            l1, l2, v_reasons = _measure(packet, sequences, args.smoke_frames, record)
            reasons += v_reasons
            end_lease = held_lease()
            record["lease_at_end"] = end_lease
            if end_lease != v1["lease"]:
                reasons.append("lease changed during the measurement")
            terminal = decide_terminal(not reasons, l1, l2)
    except Exception as exc:  # noqa: BLE001 — execution-invalid ⇒ UNRESOLVED
        reasons.append(f"runner error: {exc!r}")
        terminal = "UNRESOLVED"
    if reasons:
        terminal = "UNRESOLVED"

    record.update(
        {
            "finished_utc": _utc(),
            "l1_verdict": l1,
            "l2_verdict": l2,
            "unresolved_reasons": reasons,
            "terminal": ("SMOKE:" if smoke else "") + terminal,
        }
    )
    _write_json(packet / "packet.json", record)
    _write_manifest(packet)
    print(f"terminal: {record['terminal']}", flush=True)
    for r in reasons:
        print(f"  reason: {r}", flush=True)
    return 0 if terminal != "UNRESOLVED" else 2


def _measure(
    packet: Path,
    sequences: tuple[str, ...],
    max_frames: int | None,
    record: dict[str, Any],
) -> tuple[str | None, str | None, list[str]]:
    reasons: list[str] = []
    l1_dir = packet / "l1"
    l1_dir.mkdir()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_l1-worker",
        str(l1_dir),
        "--_sequences",
        ",".join(sequences),
    ]
    if max_frames is not None:
        cmd += ["--smoke-frames", str(max_frames)]
    with (l1_dir / "stdout.log").open("w") as f:
        proc = subprocess.run(cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT)
    l1_verdict_value: str | None = None
    if proc.returncode != 0 or not (l1_dir / "l1.json").exists():
        reasons.append(f"L1 worker failed (exit {proc.returncode})")
    else:
        l1 = json.loads((l1_dir / "l1.json").read_text())
        record["l1"] = {
            k: l1[k] for k in ("frames", "verdict", "summary", "v2", "environment")
        }
        l1_verdict_value = l1["verdict"]
        if not l1["v2"]["ok"]:
            reasons.append(f"V2: {len(l1['v2']['failures'])} re-run mismatches")
        expected_frames = (
            TOTAL_FRAMES
            if max_frames is None
            else sum(min(len(_seq_frames(s)), max_frames) for s in sequences)
        )
        if l1["frames"] != expected_frames:
            reasons.append(
                f"L1 covered {l1['frames']} frames, expected {expected_frames}"
            )

    l2_dir = packet / "l2"
    l2_dir.mkdir()
    runs = []
    for arm, rep in RUN_ORDER:
        print(f"[L2] {arm}#{rep}", flush=True)
        runs.append(run_l2_arm(l2_dir, arm, rep, sequences, max_frames))
    record["l2_runs"] = runs
    by = {(r["arm"], r["rep"]): r for r in runs}
    for r in runs:
        if r["returncode"] != 0 or any(v is None for v in r["txt_sha256"].values()):
            reasons.append(f"V3: {r['arm']}#{r['rep']} failed or missing output")
    envs = {json.dumps(r["resolved_env_overrides"], sort_keys=True) for r in runs}
    if len(envs) != 1:
        reasons.append("resolved_env_overrides differ between arms")
    head = record["v1"]["git"]["head"]
    for r in runs:
        problem = arm_manifest_problem(r, head, smoke=max_frames is not None)
        if problem:
            reasons.append(f"{r['arm']}#{r['rep']}: {problem}")
    for arm in ARMS:
        if by[(arm, 1)]["txt_sha256"] != by[(arm, 2)]["txt_sha256"]:
            reasons.append(f"V3: {arm} runs #1 and #2 are not byte-identical")
    if reasons:
        return l1_verdict_value, None, reasons

    scored = {
        arm: score_arm(project_root / by[(arm, 1)]["output_dir"], sequences)
        for arm in ARMS
    }
    _write_json(packet / "l2_metrics.json", scored)
    exact = by[("A_T", 1)]["txt_sha256"] == by[("A_C", 1)]["txt_sha256"]
    verdict, detail = l2_verdict(
        scored["A_C"]["combined"],
        scored["A_T"]["combined"],
        scored["A_N"]["combined"],
        exact,
    )
    divergence = {}
    for other in ("A_T", "A_N"):
        divergence[other] = {
            s: first_divergent_frame(
                (project_root / by[("A_C", 1)]["output_dir"] / f"{s}.txt").read_bytes(),
                (project_root / by[(other, 1)]["output_dir"] / f"{s}.txt").read_bytes(),
            )
            for s in sequences
        }
    record["l2"] = {
        "verdict": verdict,
        "exact_A_T_vs_A_C": exact,
        "A_N_identical_to_A_C": by[("A_N", 1)]["txt_sha256"]
        == by[("A_C", 1)]["txt_sha256"],
        "decision": detail,
        "first_divergent_frame_vs_A_C": divergence,
        "report_only": {
            arm: {k: scored[arm]["combined"][k] for k in ("DetA", "AssA", "FP", "FN")}
            for arm in ARMS
        },
    }
    return l1_verdict_value, verdict, reasons


def _write_manifest(packet: Path) -> None:
    files = sorted(
        p for p in packet.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
    )
    _write_json(
        packet / "MANIFEST.json",
        {
            "generated_utc": _utc(),
            "files": [
                {
                    "path": str(p.relative_to(packet)),
                    "sha256": _sha256(p),
                    "bytes": p.stat().st_size,
                }
                for p in files
            ],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
