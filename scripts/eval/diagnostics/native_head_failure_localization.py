#!/usr/bin/env python3
"""Run the #465 head failure-localization study (anchor-level C/T/E replay) and write its packet.

Implements the frozen pre-declaration
``docs/reference/native_runtime_head_failure_localization_declaration.md``
(blob ``FROZEN_DECLARATION_BLOB``) without re-deciding anything in it. The
study identity -- declaration, engine, lineage manifest, arms, threshold
floors, terminals and freeze tag -- is fixed in this file and is never a
command-line option, so the runner blob recorded in the packet identifies the
study by itself.

* Replay (§3) -- each arm is one unmodified ``scripts/eval/mot17.py`` run
  (``--preset mamba_whole_graph --detector SDP --double-buffer``) whose
  detector ``detect_raw`` is replaced by :class:`ReplayDetect`: one TRT
  backbone pass, then C (compiled head, the oracle), T (PR-1R engine) and E
  (eager copy taken before C is compiled) on the same features; the arm's
  84-dim anchor logits are composed by anchor index over Δ(f) and decoded by
  the oracle's compiled ``_postprocess_mamba_fixed``. Every other detector
  entry point is made to raise, so the replay is the only detection path.
* V1 (§4) -- frozen inputs, clean tree, declaration blob, and the execution
  freeze point: HEAD is a 2-parent commit on ``origin/main``'s first-parent
  chain and equals the peeled commit of the annotated tag ``FREEZE_TAG``,
  locally and on ``origin``. V2 -- each arm's two runs byte-identical. V3 --
  per frame, per hybrid arm, the composed logits equal their sources bit for
  bit and the recomputed membership/class equal the in-Δ source; the first
  failing frame aborts the run. V4 -- ``R_T`` is out of bounds against ``R_C``.
* Decision (§5) -- ``UNRESOLVED`` / ``EAGER_NUMERICS_WITHIN`` /
  ``EAGER_NUMERICS_OUT`` from ``R_E`` alone; the mechanism label
  (``DELTA_ANCHOR_SUFFICIENT`` / ``COMMON_ANCHOR_VALUES_SUFFICIENT`` /
  ``MIXED``) from ``H_M`` and ``H_V``, reported beside it.
* Report only (§6) -- Δ census, first top-k/member divergence, first
  ``tracker_input`` divergence and its Δ/values attribution, and the output
  divergence traced back to ``det_idx``, for all seven sequences. A row is
  attributed to the Δ or values channel only through a full-row match whose
  row→anchor check held and whose score is not tied with a row of the other
  channel; otherwise it is reported as inconsistent / ambiguous / unmapped.
  Each run is validated before the next launches; the first invalid run ends
  the study (UNRESOLVED) with the completed runs kept in the packet.

It reads no time quantity and has no smoke mode (§9: the runner PR reads no
MOT17 frame). The one formal run must be the direct child of a
``machine-bench`` lease, from the freeze commit, on a clean tree::

    .venv/bin/python tools/resctl.py run machine-bench -- \\
        .venv/bin/python scripts/eval/diagnostics/native_head_failure_localization.py
"""
# status: experiment

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# --- frozen by the declaration (§2–§5); change only through §11 ------------
DECLARATION = "docs/reference/native_runtime_head_failure_localization_declaration.md"
FROZEN_DECLARATION_BLOB = "38f15007f35e5918924f7614c2f1900182ae48f2"
# §4 V1 execution freeze point: the runner PR's merge commit, named by an
# annotated tag created after that merge. Compared by peeled commit SHA only.
FREEZE_TAG = "freeze/465-head-localization"
FREEZE_REMOTE = "origin"
FREEZE_BRANCH = "origin/main"
HEAD_ONNX_SHA256 = "6e919dad14af81083a25679225930a3473a8cdd6ebf9828ea07b414a9316b58b"
HEAD_STEM = "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_notf32"
HEAD_ENGINE = HEAD_STEM + ".engine"
HEAD_LINEAGE = HEAD_STEM + ".lineage.json"
HEAD_LINEAGE_SHA256 = "a015bce04884c68df2b5c303dc9197dfd9c4856301e9bea6d2acf7cf251358fd"
HEAD_ENGINE_SHA256_PREFIX = "c77148a8"  # cited by §3; the full sha is the manifest's
EXPECTED_PRECISION = "fp32-no-tf32"
EXPECTED_BUILDER_FLAGS = {"fp16": False, "tf32": False}
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
PACKET_ROOT = "results/native_head_failure_localization_465"

IMG_SIZE = 640
SCORE_FLOOR = 0.05  # headline base_score_floor (§2)
MAX_DET = 300
CONF_THR = 0.001
N_CLS = 80
N_REG = 4  # reg_max = 1, no DFL (§3)
LEVEL_HW = ((80, 80), (40, 40), (20, 20))  # P3/P4/P5 flatten order (§3)
N_ANCHORS = sum(h * w for h, w in LEVEL_HW)  # 8400

HEADS = ("C", "T", "E")
# arm -> (source for a ∈ Δ(f), source for a ∉ Δ(f))  (§3 table)
ARM_SOURCES = {
    "R_C": ("C", "C"),
    "R_T": ("T", "T"),
    "H_M": ("T", "C"),
    "H_V": ("C", "T"),
    "R_E": ("E", "E"),
}
ARMS = tuple(ARM_SOURCES)
HYBRID_ARMS = ("H_M", "H_V")
REFERENCE_ARM = "R_C"
RUN_ORDER = tuple((arm, rep) for rep in (1, 2) for arm in ARMS)
L2_METRICS = ("IDF1", "HOTA", "MOTA", "IDs")
TOL_FLOOR = {"IDF1": 0.20, "HOTA": 0.20, "MOTA": 0.20, "IDs": 5.0}

TERMINALS = ("UNRESOLVED", "EAGER_NUMERICS_WITHIN", "EAGER_NUMERICS_OUT")
MECHANISM_LABELS = (
    "DELTA_ANCHOR_SUFFICIENT",
    "COMMON_ANCHOR_VALUES_SUFFICIENT",
    "MIXED",
)
PROBE_STAGES = ("detector_output", "post_nms", "tracker_input")

# Report-only byte-identity references (§4 last paragraph); never gates.
PRIOR_REFERENCE_TXT = {
    "R_C_vs_PR2_A_C": "results/native_head_parity_465/20260927T133422Z/l2/A_C_1",
    "R_C_vs_PR2R_A_C": "results/native_head_parity_465_tf32_off/20260927T151100Z/l2/A_C_1",
    "R_T_vs_PR2R_A_T": "results/native_head_parity_465_tf32_off/20260927T151100Z/l2/A_T_1",
}


# --------------------------------------------------------------------------
# pure decision logic (unit-tested without a GPU)
# --------------------------------------------------------------------------
def out_of_bounds(
    metrics_x: dict[str, float], metrics_c: dict[str, float]
) -> tuple[bool, dict[str, dict[str, float | bool]]]:
    """§5: arm X is out against R_C iff any |Δ| on the unrounded 7-seq
    combined metrics exceeds its fixed floor (two-sided)."""
    detail: dict[str, dict[str, float | bool]] = {}
    for m in L2_METRICS:
        d = metrics_x[m] - metrics_c[m]
        detail[m] = {
            "R_C": metrics_c[m],
            "X": metrics_x[m],
            "delta": d,
            "floor": TOL_FLOOR[m],
            "out": abs(d) > TOL_FLOOR[m],
        }
    return any(bool(v["out"]) for v in detail.values()), detail


def mechanism_label(d_suff: bool, v_suff: bool) -> str:
    """§5 mechanism label; neither-sufficient is interaction, i.e. MIXED."""
    if d_suff and not v_suff:
        return "DELTA_ANCHOR_SUFFICIENT"
    if v_suff and not d_suff:
        return "COMMON_ANCHOR_VALUES_SUFFICIENT"
    return "MIXED"


def decide(
    validity_ok: bool,
    v4_r_t_out: bool | None,
    e_out: bool | None,
    d_suff: bool | None,
    v_suff: bool | None,
) -> tuple[str, str | None]:
    """§5 decision terminal (from validity and E_out only) plus the mechanism
    label, which is given only when V1–V4 all hold."""
    if not validity_ok or v4_r_t_out is not True or e_out is None:
        return "UNRESOLVED", None
    terminal = "EAGER_NUMERICS_OUT" if e_out else "EAGER_NUMERICS_WITHIN"
    if d_suff is None or v_suff is None:
        return "UNRESOLVED", None
    return terminal, mechanism_label(d_suff, v_suff)


def freeze_problems(
    head: str,
    parents: list[str],
    on_first_parent_chain: bool,
    tag_object_type: str | None,
    tag_local_commit: str | None,
    tag_remote_peeled: str | None,
) -> list[str]:
    """§4 V1 execution freeze point; every comparison is by commit SHA."""
    problems = []
    if len(parents) != 2:
        problems.append(
            f"HEAD has {len(parents)} parent(s); the freeze commit is a merge"
        )
    if not on_first_parent_chain:
        problems.append(f"HEAD is not on the {FREEZE_BRANCH} first-parent chain")
    if tag_object_type != "tag":
        problems.append(f"{FREEZE_TAG} is {tag_object_type!r}, not an annotated tag")
    if tag_local_commit != head:
        problems.append(f"local {FREEZE_TAG}^{{commit}} {tag_local_commit} != HEAD")
    if tag_remote_peeled != head:
        problems.append(
            f"{FREEZE_REMOTE} {FREEZE_TAG}^{{}} {tag_remote_peeled} != HEAD"
        )
    return problems


def parse_ls_remote_peeled(stdout: str) -> str | None:
    """The peeled SHA from ``git ls-remote <remote> refs/tags/<tag>^{}``."""
    want = f"refs/tags/{FREEZE_TAG}^{{}}"
    hits = [
        line.split("\t", 1)[0]
        for line in stdout.splitlines()
        if "\t" in line and line.split("\t", 1)[1] == want
    ]
    return hits[0] if len(hits) == 1 else None


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


# --------------------------------------------------------------------------
# anchor-level composition (§2, §3, §4 V3); torch, CPU-testable
# --------------------------------------------------------------------------
def flatten_logits(cls_preds: list[Any], reg_preds: list[Any]) -> Any:
    """(1,80,H,W)x3 + (1,4,H,W)x3 -> (84, 8400) in P3/P4/P5 anchor order, the
    concat order of ``_postprocess_mamba_fixed_eager``."""
    import torch

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)[0]
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)[0]
    return torch.cat([cls_all, reg_all], dim=0)


def split_logits(logits: Any) -> tuple[list[Any], list[Any]]:
    """Inverse of :func:`flatten_logits` (contiguous per-level tensors)."""
    cls_out, reg_out = [], []
    start = 0
    for h, w in LEVEL_HW:
        n = h * w
        block = logits[:, start : start + n]
        cls_out.append(block[:N_CLS].reshape(1, N_CLS, h, w).contiguous())
        reg_out.append(block[N_CLS:].reshape(1, N_REG, h, w).contiguous())
        start += n
    return cls_out, reg_out


def membership(logits: Any) -> dict[str, Any]:
    """§2: s = max sigmoid over 80 classes, class = argmax, rank = top-300 by s,
    m = [s ≥ 0.05] ∧ [rank ≤ 300]; ``order`` is the top-k index list."""
    import torch

    s, cls = logits[:N_CLS].sigmoid().max(dim=0)
    order = s.topk(MAX_DET).indices
    in_topk = torch.zeros_like(s, dtype=torch.bool)
    in_topk[order] = True
    above = s >= SCORE_FLOOR
    return {"s": s, "cls": cls, "m": above & in_topk, "above": above, "order": order}


def delta_set(a: dict[str, Any], b: dict[str, Any]) -> Any:
    """§2 Δ(f): membership differs, or both members with different class."""
    return (a["m"] != b["m"]) | (a["m"] & b["m"] & (a["cls"] != b["cls"]))


def compose(inside: Any, outside: Any, delta: Any) -> Any:
    """§3: whole 84-dim anchor vectors, ``inside`` on Δ, ``outside`` elsewhere."""
    import torch

    return torch.where(delta.unsqueeze(0), inside, outside)


def _bits(t: Any) -> Any:
    import torch

    return t.contiguous().view(torch.int32)


def v3_problems(
    composed: Any,
    inside: Any,
    outside: Any,
    delta: Any,
    reference: dict[str, Any],
    composed_member: dict[str, Any],
    roundtrip: Any,
) -> list[str]:
    """§4 V3 for one frame of a hybrid arm: (i) composed logits equal their
    sources bit for bit per anchor (and survive the per-level split), (ii) the
    membership recomputed from the composed logits equals the in-Δ source's
    membership, and member classes equal the in-Δ source's classes."""
    import torch

    problems = []
    if not torch.equal(_bits(composed[:, delta]), _bits(inside[:, delta])):
        problems.append("Δ anchors differ from the in-Δ source")
    if not torch.equal(_bits(composed[:, ~delta]), _bits(outside[:, ~delta])):
        problems.append("non-Δ anchors differ from the out-of-Δ source")
    if not torch.equal(_bits(roundtrip), _bits(composed)):
        problems.append("per-level split does not round-trip the composed logits")
    if not torch.equal(composed_member["m"], reference["m"]):
        problems.append("recomputed membership differs from the in-Δ source")
    ref_m = reference["m"]
    if not torch.equal(composed_member["cls"][ref_m], reference["cls"][ref_m]):
        problems.append("recomputed member class differs from the in-Δ source")
    return problems


def pair_census(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """§6.1–6.2 per-frame census of one head pair (0-d tensors / bools)."""
    import torch

    both = a["m"] & b["m"]
    list_a = a["order"][a["m"][a["order"]]]
    list_b = b["order"][b["m"][b["order"]]]
    return {
        "delta": int(delta_set(a, b).sum()),
        "member_diff": int((a["m"] != b["m"]).sum()),
        "floor_cross": int((a["above"] != b["above"]).sum()),
        "class_flip": int((both & (a["cls"] != b["cls"])).sum()),
        "above_a": int(a["above"].sum()),
        "above_b": int(b["above"].sum()),
        "set_differs": bool((a["m"] != b["m"]).any()),
        "class_differs": bool((both & (a["cls"] != b["cls"])).any()),
        "order_differs": not torch.equal(list_a, list_b),
    }


CENSUS_KEYS = (
    "delta",
    "member_diff",
    "floor_cross",
    "class_flip",
    "above_a",
    "above_b",
    "set_differs",
    "class_differs",
    "order_differs",
)


# --------------------------------------------------------------------------
# replay detector (runs inside the arm worker process)
# --------------------------------------------------------------------------
class ReplayDetect:
    """The composed ``detect_raw`` of one arm (§3 steps 1–4).

    Holds the last few frames' tensors so the caching allocator cannot hand a
    block still read by the consumer stream to the next frame's allocation
    (the oracle returns a static CUDA-graph buffer instead).
    """

    KEEP_FRAMES = 8

    def __init__(self, arm: str, det: Any, head_e: Any, head_t: Any) -> None:
        import torch.nn.functional as F

        from saccade.perception.temporal_yolo import mamba_gated_detector as mgd

        self.arm = arm
        self.inside, self.outside = ARM_SOURCES[arm]
        self.det = det
        self.head_c = det.mamba_head
        self.head_e = head_e
        self.head_t = head_t
        self.F = F
        self.mgd = mgd
        self.frames: list[dict[str, Any]] = []
        self._keep: list[Any] = []
        self.calls = 0

    def heads(self, frame: Any) -> dict[str, Any]:
        det = self.det
        frame_640 = self.F.interpolate(
            frame, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        feats = [p.clone() for p in det._trt_backbone.infer_graph(frame_640)]
        out = {}
        for name, head in (("C", self.head_c), ("E", self.head_e)):
            cls, reg = head._forward_eager(list(feats), return_embeddings=False)
            out[name] = flatten_logits(cls, reg).clone()
        cls, reg = self.head_t.infer_graph(*feats)
        out["T"] = flatten_logits(cls, reg).clone()
        self._keep.append((frame_640, feats))
        return out

    def __call__(self, frame: Any) -> Any:
        import torch

        det = self.det
        with torch.no_grad():
            logits = self.heads(frame)
            mem = {k: membership(v) for k, v in logits.items()}
            delta = delta_set(mem["C"], mem["T"])
            delta_e = delta_set(mem["E"], mem["C"])
            if self.inside == self.outside:
                composed = logits[self.inside]
                problems: list[str] = []
                composed_member = mem[self.inside]
            else:
                composed = compose(logits[self.inside], logits[self.outside], delta)
                composed_member = membership(composed)
                roundtrip = flatten_logits(*split_logits(composed))
                problems = v3_problems(
                    composed,
                    logits[self.inside],
                    logits[self.outside],
                    delta,
                    mem[self.inside],
                    composed_member,
                    roundtrip,
                )
            if problems:
                raise RuntimeError(
                    f"V3 failed at replay call {self.calls} ({self.arm}): {problems}"
                )
            cls_preds, reg_preds = split_logits(composed)
            detections = self.mgd._postprocess_mamba_fixed(
                cls_preds,
                reg_preds,
                det.stride,
                det.conf_thr,
                max(det.max_det, det._whole_graph_nms_pad),
                anchors=det._whole_graph_anchors,
                anchor_strides=det._whole_graph_anchor_strides,
                small_p3_max_threshold=det.small_p3_max_threshold,
                box_scale_x=det._whole_graph_sx,
                box_scale_y=det._whole_graph_sy,
            )
            detections[:, :, det._whole_graph_x_idx] *= det._whole_graph_sx
            detections[:, :, det._whole_graph_y_idx] *= det._whole_graph_sy
            order = composed_member["order"]
            rows = detections[0]
            consistent = (_bits(rows[:, 4]) == _bits(composed_member["s"][order])) & (
                rows[:, 5] == composed_member["cls"][order].to(rows.dtype)
            )
            self.frames.append(
                {
                    "v3_checked": self.inside != self.outside,
                    "ct": pair_census(mem["C"], mem["T"]),
                    "ec": pair_census(mem["E"], mem["C"]),
                    "row_anchor": order.to(torch.int32).cpu().numpy(),
                    "row_consistent": consistent.cpu().numpy(),
                    "row_in_delta": delta[order].cpu().numpy(),
                    "row_in_delta_e": delta_e[order].cpu().numpy(),
                    "rows": rows.float().cpu().numpy(),
                }
            )
        self._keep.append((logits, composed, detections))
        self._keep = self._keep[-2 * self.KEEP_FRAMES :]
        self.calls += 1
        return detections


def check_replay_preconditions(det: Any, head_e: Any) -> list[str]:
    """The replay reproduces ``_whole_graph_fn`` only for the headline form."""
    from saccade.perception.temporal_yolo import mamba_gated_detector as mgd

    checks = {
        "use_whole_graph": bool(getattr(det, "use_whole_graph", False)),
        "TRT backbone present": getattr(det, "_trt_backbone", None) is not None,
        "no harness TRT head (C is the PyTorch oracle)": getattr(det, "_trt_head", None)
        is None,
        "no detail fusion": not getattr(det, "use_detail_fusion", False),
        "small-P3 fusion off": float(det.small_p3_max_threshold) == 0.0,
        "max_det == 300": int(det.max_det) == MAX_DET,
        "no NMS pad": int(det._whole_graph_nms_pad) == 0,
        "conf_thr == 0.001": float(det.conf_thr) == CONF_THR,
        "postprocess compile on": bool(mgd._POSTPROCESS_COMPILE_ENABLED),
        "C head compile on": bool(det.mamba_head._head_compile_enabled),
        "C block compile on": bool(det.mamba_head._block_compile_enabled),
        "E head compile off": not head_e._head_compile_enabled,
        "E block compile off": not head_e._block_compile_enabled,
        "E is a separate instance": head_e is not det.mamba_head,
    }
    return [k for k, ok in checks.items() if not ok]


# --------------------------------------------------------------------------
# arm worker (own process; installs the replay and observers, runs mot17.py)
# --------------------------------------------------------------------------
def mot17_argv(
    out_rel: str, sequences: tuple[str, ...], data_root: str | None
) -> list[str]:
    argv = [
        "scripts/eval/mot17.py",
        "--preset",
        PRESET_NAME,
        "--detector",
        "SDP",
        "--double-buffer",
        "--sequences",
        ",".join(sequences),
        "--output",
        out_rel,
    ]
    if data_root is not None:  # structural checks only; never from the CLI
        argv += ["--data-root", data_root]
    return argv


def run_arm_worker(
    arm: str,
    out_dir: Path,
    evidence_dir: Path,
    sequences: tuple[str, ...],
    data_root: str | None = None,
) -> int:
    # Same sys.path and first import as mot17.py's own header (the TRT
    # detector module must load before torchvision; libjpeg conflict).
    build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
    if build_path.exists():
        sys.path.insert(0, str(build_path))
    import saccade.perception.detector_trt  # noqa: F401

    import numpy as np
    import torch

    import saccade.perception.eval.evaluator as evaluator_module
    import saccade.perception.eval.runner as runner_module
    from saccade.perception.eval import stages as stages_module
    from saccade.perception.temporal_yolo import mamba_gated_detector as mgd

    evidence_dir.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {"replay": None, "problems": []}
    probes: dict[str, list[tuple[str, int, str, Any]]] = {"rows": []}
    tracker_rows: list[tuple[str, int, dict[str, Any]]] = []
    seq_frames: dict[str, list[dict[str, Any]]] = {}

    original_build = mgd.build_mamba_gated_detector
    original_run_eval = runner_module.run_eval
    original_stages_emit = stages_module._run_emit
    original_evaluator_emit = evaluator_module._run_emit

    def build(*args: Any, **kwargs: Any) -> Any:
        det = original_build(*args, **kwargs)
        # E is copied here, before mot17.py enables compile on C (§3 step 2).
        state["head_e"] = copy.deepcopy(det.mamba_head).eval()
        state["detector"] = det
        return det

    def refuse(name: str) -> Any:
        def _raise(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError(f"replay: detector path {name} bypasses the replay")

        return _raise

    def to_np(t: Any) -> Any:
        if t is None:
            return None
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def stage_probe(
        seq: str, frame_id: int, stage: str, boxes: Any, scores: Any, classes: Any
    ) -> None:
        if stage not in PROBE_STAGES:
            return
        n = int(scores.shape[0])
        cls_col = (
            torch.full((n, 1), -1.0, device=scores.device)
            if classes is None
            else classes[:n].float().reshape(n, 1)
        )
        rows = torch.cat(
            [
                boxes[:n].float().reshape(n, 4),
                scores[:n].float().reshape(n, 1),
                cls_col,
            ],
            dim=1,
        )
        probes["rows"].append((str(seq), int(frame_id), stage, rows.cpu().numpy()))

    def observing_emit(original: Any) -> Any:
        def wrapped(state_: Any, **kwargs: Any) -> Any:
            tr = kwargs.get("track_results") or {}
            count_raw = tr.get("count", 0)
            count = int(count_raw.item() if hasattr(count_raw, "item") else count_raw)
            det_idx = to_np(tr.get("det_idx"))
            tracker_rows.append(
                (
                    str(state_.seq),
                    int(kwargs["frame_id"]),
                    {
                        "ids": to_np(tr.get("ids"))[:count]
                        if tr.get("ids") is not None
                        else np.zeros((0,), np.int32),
                        "det_idx": det_idx[:count] if det_idx is not None else None,
                        "boxes": to_np(tr.get("boxes"))[:count]
                        if tr.get("boxes") is not None
                        else np.zeros((0, 4), np.float32),
                    },
                )
            )
            return original(state_, **kwargs)

        return wrapped

    def sequence_done(seq: str, lines: Any) -> None:
        replay = state["replay"]
        seq_frames[str(seq)] = replay.frames
        replay.frames = []

    def run_eval(**kwargs: Any) -> Any:
        det = kwargs.get("detector")
        head_e = state.get("head_e")
        if det is None or det is not state.get("detector") or head_e is None:
            raise RuntimeError(
                "replay: harness detector was not built by the patched builder"
            )
        bad = check_replay_preconditions(det, head_e)
        if bad:
            raise RuntimeError(f"replay preconditions failed: {bad}")
        head_t = mgd.TRTMambaHead(str(project_root / HEAD_ENGINE))
        replay = ReplayDetect(arm, det, head_e, head_t)
        state["replay"] = replay
        det.detect_raw = replay
        for name in ("detect_raw_preprocessed", "detect_raw_with_detail", "forward"):
            setattr(det, name, refuse(name))
        for key in ("stage_probe_callback", "sequence_result_callback"):
            if kwargs.get(key) is not None:
                raise RuntimeError(f"replay: harness already set {key}")
        kwargs["stage_probe_callback"] = stage_probe
        kwargs["sequence_result_callback"] = sequence_done
        return original_run_eval(**kwargs)

    mgd.build_mamba_gated_detector = build
    runner_module.run_eval = run_eval
    stages_module._run_emit = observing_emit(original_stages_emit)
    evaluator_module._run_emit = observing_emit(original_evaluator_emit)
    eval_dir = str(project_root / "scripts" / "eval")
    sys.path.insert(0, eval_dir)
    out_arg = (
        str(out_dir.relative_to(project_root))
        if out_dir.is_relative_to(project_root)
        else str(out_dir)  # structural checks write outside the repository
    )
    argv = mot17_argv(out_arg, sequences, data_root)
    sys.argv = list(argv)
    exit_code = 0
    try:
        # runpy sets sys.argv[0] to this path; keep it the relative argv[0] so
        # the harness's run_manifest cmdline is exactly ``argv`` (cwd = root).
        if Path.cwd().resolve() != project_root:
            raise RuntimeError("the arm worker must run with the repository as cwd")
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as exc:
        exit_code = (
            exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        )
    finally:
        mgd.build_mamba_gated_detector = original_build
        runner_module.run_eval = original_run_eval
        stages_module._run_emit = original_stages_emit
        evaluator_module._run_emit = original_evaluator_emit
    replay = state.get("replay")
    if replay is None:
        raise RuntimeError("replay was never installed")
    if replay.frames:
        raise RuntimeError(
            f"{len(replay.frames)} replay frames after the last sequence"
        )
    _write_arm_evidence(evidence_dir, argv, seq_frames, probes["rows"], tracker_rows)
    return exit_code


def _ragged(items: list[tuple[int, Any]], width: int, dtype: Any) -> dict[str, Any]:
    import numpy as np

    frames = np.asarray([f for f, _ in items], dtype=np.int32)
    counts = np.asarray([len(r) for _, r in items], dtype=np.int32)
    body = (
        np.concatenate(
            [np.asarray(r, dtype=dtype).reshape(-1, width) for _, r in items]
        )
        if items
        else np.zeros((0, width), dtype=dtype)
    )
    return {"frames": frames, "counts": counts, "rows": body}


def _write_arm_evidence(
    evidence_dir: Path,
    argv: list[str],
    seq_frames: dict[str, list[dict[str, Any]]],
    probe_rows: list[tuple[str, int, str, Any]],
    tracker_rows: list[tuple[str, int, dict[str, Any]]],
) -> None:
    import numpy as np

    summary: dict[str, Any] = {"mot17_argv": argv, "sequences": {}}
    for seq, frames in seq_frames.items():
        arrays: dict[str, Any] = {}
        for pair in ("ct", "ec"):
            for key in CENSUS_KEYS:
                arrays[f"{pair}_{key}"] = np.asarray(
                    [fr[pair][key] for fr in frames], dtype=np.int64
                )
        for key in (
            "row_anchor",
            "row_consistent",
            "row_in_delta",
            "row_in_delta_e",
            "rows",
        ):
            arrays[key] = (
                np.stack([fr[key] for fr in frames]) if frames else np.zeros((0,))
            )
        arrays["v3_checked"] = np.asarray(
            [fr["v3_checked"] for fr in frames], dtype=bool
        )
        for stage in PROBE_STAGES:
            items = [(f, r) for s, f, st, r in probe_rows if s == seq and st == stage]
            for k, v in _ragged(items, 6, np.float32).items():
                arrays[f"{stage}_{k}"] = v
        trk = [(f, d) for s, f, d in tracker_rows if s == seq]
        arrays["tracker_frames"] = np.asarray([f for f, _ in trk], dtype=np.int32)
        arrays["tracker_counts"] = np.asarray(
            [len(d["ids"]) for _, d in trk], dtype=np.int32
        )
        arrays["tracker_ids"] = (
            np.concatenate([np.asarray(d["ids"], np.int64).reshape(-1) for _, d in trk])
            if trk
            else np.zeros((0,), np.int64)
        )
        arrays["tracker_det_idx"] = (
            np.concatenate(
                [
                    np.asarray(
                        d["det_idx"]
                        if d["det_idx"] is not None
                        else [-2] * len(d["ids"]),
                        np.int64,
                    ).reshape(-1)
                    for _, d in trk
                ]
            )
            if trk
            else np.zeros((0,), np.int64)
        )
        arrays["tracker_boxes"] = (
            np.concatenate(
                [np.asarray(d["boxes"], np.float32).reshape(-1, 4) for _, d in trk]
            )
            if trk
            else np.zeros((0, 4), np.float32)
        )
        det_out = ragged_by_frame(
            arrays["detector_output_frames"],
            arrays["detector_output_counts"],
            arrays["detector_output_rows"],
        )
        aligned = sum(
            1
            for k, fr in enumerate(frames)
            if (k + 1) in det_out
            and det_out[k + 1].shape == fr["rows"].shape
            and det_out[k + 1].tobytes() == fr["rows"].tobytes()
        )
        np.savez_compressed(evidence_dir / f"{seq}.npz", **arrays)
        summary["sequences"][seq] = {
            "replay_frames": len(frames),
            "v3_frames_checked": int(arrays["v3_checked"].sum()),
            "probe_frames": {
                st: int(len(arrays[f"{st}_frames"])) for st in PROBE_STAGES
            },
            "detector_output_equals_replay_frame_k_plus_1": aligned,
            "tracker_frames": int(len(trk)),
            "det_idx_missing_frames": int(
                sum(1 for _, d in trk if d["det_idx"] is None)
            ),
        }
    _write_json(evidence_dir / "worker.json", summary)


# --------------------------------------------------------------------------
# §6 analysis (numpy only; unit-tested on synthetic arrays)
# --------------------------------------------------------------------------
def ragged_by_frame(frames: Any, counts: Any, rows: Any) -> dict[int, Any]:
    out: dict[int, Any] = {}
    start = 0
    for f, n in zip(frames.tolist(), counts.tolist()):
        out[int(f)] = rows[start : start + n]
        start += n
    return out


# Provenance statuses of one row. Only the first two are an attribution; the
# others keep the uncertainty (§6 is report-only: none of this gates).
PROVENANCE = ("delta_anchor", "values", "inconsistent", "ambiguous", "unmapped")
MISSING = None  # a frame whose tracker_input observation is missing


def detector_row_status(rows: Any, in_delta: Any, consistent: Any) -> list[str]:
    """Per replay output row: ``inconsistent`` when the recorded row→anchor
    check failed; ``ambiguous`` when rows with bit-identical scores (a top-k
    tie, where index order is not guaranteed, or duplicate rows) disagree on
    the Δ flag; otherwise the row's Δ channel."""
    scores = [rows[j, 4].tobytes() for j in range(len(rows))]
    flags_by_score: dict[bytes, set[bool]] = {}
    for j, key in enumerate(scores):
        flags_by_score.setdefault(key, set()).add(bool(in_delta[j]))
    out = []
    for j, key in enumerate(scores):
        if not bool(consistent[j]):
            out.append("inconsistent")
        elif len(flags_by_score[key]) > 1:
            out.append("ambiguous")
        else:
            out.append("delta_anchor" if bool(in_delta[j]) else "values")
    return out


def row_provenance(rows: Any, detector_rows: Any, status: list[str]) -> list[str]:
    """Provenance of rows observed downstream: a full-row (box, score, class)
    bit-identical match against the replay rows of the same frame. No match is
    ``unmapped``; several matches are duplicates, so they share a score and
    :func:`detector_row_status` has already folded them into one status."""
    index: dict[bytes, list[int]] = {}
    for j in range(len(detector_rows)):
        index.setdefault(detector_rows[j].tobytes(), []).append(j)
    out = []
    for i in range(len(rows)):
        cands = index.get(rows[i].tobytes(), [])
        states = {status[j] for j in cands}
        if not cands:
            out.append("unmapped")
        elif len(states) == 1:
            out.append(states.pop())
        else:
            out.append("ambiguous")
    return out


def provenance_counts(labels: list[str]) -> dict[str, int]:
    return {k: sum(1 for x in labels if x == k) for k in PROVENANCE}


def attribution_class(counts: dict[str, int]) -> str:
    """``delta_anchor`` if any row is attributed to a Δ anchor; ``values`` only
    if every row is attributed and none to Δ; ``undetermined`` if no row is Δ
    but some are unattributed; ``none`` for no rows."""
    if counts.get("delta_anchor", 0):
        return "delta_anchor"
    unknown = sum(counts.get(k, 0) for k in ("inconsistent", "ambiguous", "unmapped"))
    if unknown:
        return "undetermined"
    return "values" if counts.get("values", 0) else "none"


def multiset_excess(a: Any, b: Any) -> list[int]:
    """Indices of ``a`` rows beyond their multiplicity in ``b`` (by exact bytes)."""
    remaining: dict[bytes, int] = {}
    for j in range(len(b)):
        key = b[j].tobytes()
        remaining[key] = remaining.get(key, 0) + 1
    out = []
    for i in range(len(a)):
        key = a[i].tobytes()
        if remaining.get(key, 0):
            remaining[key] -= 1
        else:
            out.append(i)
    return out


def tracker_input_view(ev: Any) -> dict[int, Any]:
    """tracker_input rows per replayed frame. A frame without the probe is a
    verified empty frame -- compared as zero rows -- only when it has the
    signature of the harness's two ``fused_boxes.numel() == 0`` early returns
    (evaluator: after ``detector_output``, before ``post_nms``): the frame was
    probed at ``detector_output`` and emitted no tracker output. Otherwise it
    is ``MISSING``."""
    import numpy as np

    ti = ragged_by_frame(
        ev["tracker_input_frames"], ev["tracker_input_counts"], ev["tracker_input_rows"]
    )
    det_frames = set(ev["detector_output_frames"].tolist())
    trk_frames = set(ev["tracker_frames"].tolist())
    out: dict[int, Any] = {}
    for frame in range(1, len(ev["rows"]) + 1):
        if frame in ti:
            out[frame] = ti[frame]
        elif frame in det_frames and frame not in trk_frames:
            out[frame] = np.zeros((0, 6), dtype=np.float32)
        else:
            out[frame] = MISSING
    return out


def first_rows_divergence(a: dict[int, Any], b: dict[int, Any]) -> int | None:
    """First frame whose ordered rows differ bit for bit; a frame observed on
    one side only, or ``MISSING`` on exactly one side, differs."""
    for frame in sorted(set(a) | set(b)):
        ra, rb = a.get(frame, MISSING), b.get(frame, MISSING)
        if ra is MISSING and rb is MISSING:
            continue
        if ra is MISSING or rb is MISSING:
            return frame
        if ra.shape != rb.shape or ra.tobytes() != rb.tobytes():
            return frame
    return None


def explain_rows(
    other_rows: Any,
    ref_rows: Any,
    other_status: tuple[Any, list[str]],
    ref_status: tuple[Any, list[str]],
) -> dict[str, Any]:
    """Why two ordered row arrays differ: rows only in one side (additions /
    removals, multiplicity-aware; a changed row appears as one of each) with
    their provenance, or, when the multisets agree, the displaced positions
    of a pure reordering and their provenance."""
    if other_rows is MISSING or ref_rows is MISSING:
        return {
            "kind": "missing_observation",
            "missing_side": "other" if other_rows is MISSING else "R_C",
            "class": "unknown",
        }
    only_other = multiset_excess(other_rows, ref_rows)
    only_ref = multiset_excess(ref_rows, other_rows)
    if not only_other and not only_ref:
        displaced = [
            i
            for i in range(len(other_rows))
            if other_rows[i].tobytes() != ref_rows[i].tobytes()
        ]
        counts = provenance_counts(
            row_provenance(other_rows[displaced], *other_status) if displaced else []
        )
        return {
            "kind": "order_only",
            "displaced_positions": displaced,
            "displaced_rows": counts,
            "class": attribution_class(counts),
        }
    added = provenance_counts(
        row_provenance(other_rows[only_other], *other_status) if only_other else []
    )
    removed = provenance_counts(
        row_provenance(ref_rows[only_ref], *ref_status) if only_ref else []
    )
    total = {k: added[k] + removed[k] for k in PROVENANCE}
    return {
        "kind": "rows_changed",
        "rows_only_in_other": added,
        "rows_only_in_R_C": removed,
        "class": attribution_class(total),
    }


def census_first_frames(ev: Any) -> dict[str, int | None]:
    """§6.2: first replay frame (1-based call order) where C and T differ in
    member set, member class, or member order."""
    out: dict[str, int | None] = {}
    for key in ("set_differs", "class_differs", "order_differs"):
        hits = ev[f"ct_{key}"].nonzero()[0]
        out[key] = int(hits[0]) + 1 if len(hits) else None
    return out


def census_totals(ev: Any, pair: str) -> dict[str, int]:
    return {
        "frames": int(len(ev[f"{pair}_delta"])),
        "delta": int(ev[f"{pair}_delta"].sum()),
        "member_diff": int(ev[f"{pair}_member_diff"].sum()),
        "floor_cross": int(ev[f"{pair}_floor_cross"].sum()),
        "class_flip": int(ev[f"{pair}_class_flip"].sum()),
        "frames_with_delta": int((ev[f"{pair}_delta"] > 0).sum()),
        "frames_top300_truncated_above_floor_a": int(
            (ev[f"{pair}_above_a"] > MAX_DET).sum()
        ),
        "frames_top300_truncated_above_floor_b": int(
            (ev[f"{pair}_above_b"] > MAX_DET).sum()
        ),
    }


def _frame_status(ev: Any, k: int, delta_key: str) -> tuple[Any, list[str]]:
    rows = ev["rows"][k]
    return rows, detector_row_status(rows, ev[delta_key][k], ev["row_consistent"][k])


def delta_flow(ev: Any, delta_key: str = "row_in_delta") -> dict[str, Any]:
    """§6.1 in one run: do Δ-anchor detections reach tracker_input, and are
    they referenced by an output track's det_idx (which indexes the same
    frame's tracker_input rows)? Only attributed rows count as Δ or values;
    the rest are reported by provenance status."""
    view = tracker_input_view(ev)
    trk_idx = ragged_by_frame(
        ev["tracker_frames"], ev["tracker_counts"], ev["tracker_det_idx"].reshape(-1, 1)
    )
    replay_above: list[str] = []
    tracker_input: list[str] = []
    refs: list[str] = []
    missing_det_idx = 0
    missing_frames = 0
    for k in range(len(ev["rows"])):
        rows, status = _frame_status(ev, k, delta_key)
        above = rows[:, 4] >= SCORE_FLOOR
        replay_above += [s for s, a in zip(status, above.tolist()) if a]
        tin = view.get(k + 1, MISSING)
        if tin is MISSING:
            missing_frames += 1
            continue
        prov = row_provenance(tin, rows, status)
        tracker_input += prov
        idx = trk_idx.get(k + 1)
        if idx is None:
            continue
        for d in idx.reshape(-1).tolist():
            if d == -2:
                missing_det_idx += 1
            elif 0 <= d < len(prov):
                refs.append(prov[d])
    return {
        "replay_rows_above_floor": provenance_counts(replay_above),
        "tracker_input_rows": provenance_counts(tracker_input),
        "track_det_idx_refs": provenance_counts(refs),
        "track_det_idx_missing": missing_det_idx,
        "tracker_input_missing_frames": missing_frames,
    }


def compare_runs(ref: Any, other: Any, delta_key: str) -> dict[str, Any]:
    """§6.3/§6.4 for one sequence: the first tracker_input / tracker output
    divergence of ``other`` against R_C and its explanation."""
    import numpy as np

    ti_ref, ti_oth = tracker_input_view(ref), tracker_input_view(other)
    first_ti = first_rows_divergence(ti_ref, ti_oth)
    explanation = None
    if first_ti is not None:
        k = first_ti - 1
        explanation = explain_rows(
            ti_oth.get(first_ti, MISSING),
            ti_ref.get(first_ti, MISSING),
            _frame_status(other, k, delta_key),
            _frame_status(ref, k, delta_key),
        )
    trk = {}
    joined = {}
    for name, ev in (("ref", ref), ("other", other)):
        ids = ragged_by_frame(
            ev["tracker_frames"], ev["tracker_counts"], ev["tracker_ids"].reshape(-1, 1)
        )
        det_idx = ragged_by_frame(
            ev["tracker_frames"],
            ev["tracker_counts"],
            ev["tracker_det_idx"].reshape(-1, 1),
        )
        boxes = ragged_by_frame(
            ev["tracker_frames"], ev["tracker_counts"], ev["tracker_boxes"]
        )
        trk[name] = {"ids": ids, "det_idx": det_idx, "boxes": boxes}
        # no emit on a frame = no tracks on it (the skipped-frame path)
        joined[name] = {
            f: _join_tracker(trk[name], f) if f in ids else np.zeros((0, 6))
            for f in range(1, len(ev["rows"]) + 1)
        }
    return {
        "first_tracker_input_divergence": first_ti,
        "first_tracker_input_explanation": explanation,
        "first_tracker_output_divergence": first_rows_divergence(
            joined["ref"], joined["other"]
        ),
        "_tracker": trk,
    }


def _join_tracker(trk: dict[str, Any], frame: int) -> Any:
    import numpy as np

    return np.concatenate(
        [
            trk["ids"][frame].astype(np.float64).reshape(-1, 1),
            trk["det_idx"][frame].astype(np.float64).reshape(-1, 1),
            trk["boxes"][frame].astype(np.float64).reshape(-1, 4),
        ],
        axis=1,
    )


def trace_output_divergence(
    txt_first: int | None, comparison: dict[str, Any]
) -> dict[str, Any] | None:
    """§6.4: whether the first txt divergence has a tracker_input divergence at
    or before it, and the tracks (id, det_idx) at that frame in both runs."""
    if txt_first is None:
        return None
    ti = comparison["first_tracker_input_divergence"]
    out: dict[str, Any] = {
        "txt_first_divergence": txt_first,
        "tracker_input_divergence_at_or_before": ti is not None
        and 0 <= ti <= txt_first,
        "tracker_output_divergence_at_or_before": (
            comparison["first_tracker_output_divergence"] is not None
            and 0 <= comparison["first_tracker_output_divergence"] <= txt_first
        ),
    }
    for name in ("ref", "other"):
        trk = comparison["_tracker"][name]
        ids = trk["ids"].get(txt_first)
        det_idx = trk["det_idx"].get(txt_first)
        out[f"tracks_{'R_C' if name == 'ref' else 'other'}"] = (
            None
            if ids is None
            else [
                {"id": int(i), "det_idx": int(d)}
                for i, d in zip(ids.reshape(-1).tolist(), det_idx.reshape(-1).tolist())
            ]
        )
    return out


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


def _git_or_none(*args: str) -> str | None:
    try:
        return _git(*args)
    except subprocess.CalledProcessError:
        return None


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _seq_frames(seq: str) -> list[Path]:
    return sorted((project_root / DATA_ROOT / SPLIT / seq / "img1").glob("*.jpg"))


def _saccade_env() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("SACCADE_")}


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


def observed_freeze() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    parents = _git("log", "-1", "--format=%P", "HEAD").split()
    chain = _git_or_none("rev-list", "--first-parent", FREEZE_BRANCH) or ""
    ref = f"refs/tags/{FREEZE_TAG}"
    ls = subprocess.run(
        ["git", "ls-remote", FREEZE_REMOTE, f"{ref}^{{}}"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    return {
        "head": head,
        "parents": parents,
        "on_first_parent_chain": head in chain.split(),
        "tag_object_type": _git_or_none("cat-file", "-t", ref),
        "tag_object_sha": _git_or_none("rev-parse", ref),
        "tag_local_commit": _git_or_none("rev-parse", f"{ref}^{{commit}}"),
        "tag_remote_peeled": parse_ls_remote_peeled(ls.stdout)
        if ls.returncode == 0
        else None,
        "ls_remote_returncode": ls.returncode,
    }


# --------------------------------------------------------------------------
# V1 — frozen inputs (declaration §4)
# --------------------------------------------------------------------------
def check_v1() -> tuple[bool, dict[str, Any]]:
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

    onnx_sha = _sha_or_none((project_root / HEAD_STEM).with_suffix(".onnx"))
    check("head ONNX sha256", onnx_sha == HEAD_ONNX_SHA256, onnx_sha, HEAD_ONNX_SHA256)
    check(
        "manifest onnx.sha256",
        get("onnx", "sha256") == HEAD_ONNX_SHA256,
        get("onnx", "sha256"),
        HEAD_ONNX_SHA256,
    )
    lineage_sha = _sha_or_none(lineage_path)
    check(
        "lineage manifest file sha256",
        lineage_sha == HEAD_LINEAGE_SHA256,
        lineage_sha,
        HEAD_LINEAGE_SHA256,
    )
    check(
        "manifest engine.precision",
        get("engine", "precision") == EXPECTED_PRECISION,
        get("engine", "precision"),
        EXPECTED_PRECISION,
    )
    readback = get("engine", "builder_flag_readback")
    expected_readback = {
        "before_build": EXPECTED_BUILDER_FLAGS,
        "after_build": EXPECTED_BUILDER_FLAGS,
    }
    check(
        "manifest builder flags before and after build",
        readback == expected_readback,
        readback,
        expected_readback,
    )
    engine_sha = _sha_or_none(project_root / HEAD_ENGINE)
    check(
        "head engine sha256 == manifest (and §3 prefix)",
        engine_sha is not None
        and engine_sha == get("engine", "sha256")
        and engine_sha.startswith(HEAD_ENGINE_SHA256_PREFIX),
        engine_sha,
        get("engine", "sha256"),
    )
    export_check = subprocess.run(
        [sys.executable, EXPORT_TOOL, "--precision", EXPECTED_PRECISION, "--check"],
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
    check("clean tree", porcelain == "", porcelain or "(clean)", "(clean)")
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
        blobs["runner_blob"] == blobs["runner_worktree_blob"],
        blobs["runner_blob"],
        blobs["runner_worktree_blob"],
    )
    freeze = observed_freeze()
    problems = freeze_problems(
        freeze["head"],
        freeze["parents"],
        freeze["on_first_parent_chain"],
        freeze["tag_object_type"],
        freeze["tag_local_commit"],
        freeze["tag_remote_peeled"],
    )
    check(
        f"HEAD is the execution freeze commit ({FREEZE_TAG})",
        not problems,
        {**freeze, "problems": problems},
        "2-parent commit on the first-parent chain == peeled tag, local and remote",
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

    saccade_env = _saccade_env()
    check("no SACCADE_* set by the caller", not saccade_env, saccade_env, {})

    frames = {s: len(_seq_frames(s)) for s in SEQUENCES}
    gts = {
        s: (project_root / DATA_ROOT / SPLIT / s / "gt" / "gt.txt").exists()
        for s in SEQUENCES
    }
    check(
        "data sequences and frame count",
        sum(frames.values()) == TOTAL_FRAMES
        and all(frames.values())
        and all(gts.values()),
        {"frames": frames, "total": sum(frames.values()), "gt": gts},
        {"sequences": list(SEQUENCES), "total": TOTAL_FRAMES},
    )

    lease = held_lease()
    check(
        f"direct child of a {LEASE} lease",
        lease is not None and lease.get("resource") == LEASE,
        lease,
        f"{LEASE} held by parent pid {os.getppid()}",
    )

    ok = all(c["ok"] for c in checks)
    return ok, {
        "ok": ok,
        "checks": checks,
        "git": blobs,
        "freeze": freeze,
        "lease": lease,
    }


# --------------------------------------------------------------------------
# arms (declaration §3, §5)
# --------------------------------------------------------------------------
def run_arm(l2_dir: Path, arm: str, rep: int) -> dict[str, Any]:
    out = l2_dir / f"{arm}_{rep}"
    out.mkdir(parents=True, exist_ok=False)
    evidence = l2_dir / f"{arm}_{rep}.evidence"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().relative_to(project_root)),
        "--_arm-worker",
        arm,
        "--_out",
        str(out.relative_to(project_root)),
    ]
    from saccade.perception.eval.assoc_basis import resolved_env_overrides

    log = l2_dir / f"{arm}_{rep}.stdout.log"
    started = _utc()
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT)
    txts = {s: out / f"{s}.txt" for s in SEQUENCES}
    return {
        "arm": arm,
        "rep": rep,
        "cmd": cmd,
        "mot17_argv": mot17_argv(str(out.relative_to(project_root)), SEQUENCES, None),
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": _utc(),
        "stdout": str(log.relative_to(project_root)),
        "output_dir": str(out.relative_to(project_root)),
        "evidence_dir": str(evidence.relative_to(project_root)),
        "resolved_env_overrides": resolved_env_overrides(),
        "txt_sha256": {
            s: (_sha256(p) if p.exists() else None) for s, p in txts.items()
        },
    }


def arm_manifest_problem(run: dict[str, Any], head: str) -> str | None:
    """The harness's own run_manifest.json must show this arm's exact mot17.py
    command line, the measured commit and a clean tree."""
    path = project_root / run["output_dir"] / "run_manifest.json"
    if not path.exists():
        return "run_manifest.json missing"
    m = json.loads(path.read_text())
    if m.get("cmdline") != run["mot17_argv"]:
        return f"run_manifest cmdline {m.get('cmdline')} != {run['mot17_argv']}"
    if m.get("commit") != head:
        return f"run_manifest commit {m.get('commit')} != {head}"
    if m.get("dirty") is not False:
        return "run_manifest reports a dirty tree"
    return None


def run_problems(
    run: dict[str, Any], head: str, runs: list[dict[str, Any]]
) -> list[str]:
    """Everything that invalidates the study as soon as ``run`` finishes."""
    tag = f"{run['arm']}#{run['rep']}"
    if run["returncode"] != 0 or any(v is None for v in run["txt_sha256"].values()):
        return [f"{tag} failed (exit {run['returncode']}) or missing output"]
    problems = []
    problem = arm_manifest_problem(run, head)
    if problem:
        problems.append(f"{tag}: {problem}")
    problems += [f"V3/coverage: {p}" for p in evidence_problems(run)]
    if run["resolved_env_overrides"] != runs[0]["resolved_env_overrides"]:
        problems.append(f"{tag}: resolved_env_overrides differ from {runs[0]['arm']}#1")
    if run["rep"] == 2:
        first = next(r for r in runs if r["arm"] == run["arm"] and r["rep"] == 1)
        if first["txt_sha256"] != run["txt_sha256"]:
            problems.append(f"V2: {run['arm']} runs #1 and #2 are not byte-identical")
    return problems


def evidence_problems(run: dict[str, Any]) -> list[str]:
    """Replay coverage: one replay call per frame, every hybrid frame V3-checked."""
    path = project_root / run["evidence_dir"] / "worker.json"
    if not path.exists():
        return [f"{run['arm']}#{run['rep']}: worker.json missing"]
    summary = json.loads(path.read_text())["sequences"]
    problems = []
    for seq in SEQUENCES:
        s = summary.get(seq)
        n = len(_seq_frames(seq))
        tag = f"{run['arm']}#{run['rep']} {seq}"
        if s is None:
            problems.append(f"{tag}: no replay evidence")
            continue
        if s["replay_frames"] != n:
            problems.append(f"{tag}: {s['replay_frames']} replay calls for {n} frames")
        if run["arm"] in HYBRID_ARMS and s["v3_frames_checked"] != n:
            problems.append(f"{tag}: V3 checked {s['v3_frames_checked']} of {n} frames")
    return problems


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
    parser.add_argument(
        "--_arm-worker", dest="arm_worker", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--_out", dest="out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.arm_worker:
        if args.arm_worker not in ARM_SOURCES or not args.out:
            parser.error("internal worker flags are malformed")
        out = project_root / args.out
        return run_arm_worker(
            args.arm_worker, out, out.with_name(out.name + ".evidence"), SEQUENCES
        )

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    packet = project_root / PACKET_ROOT / stamp
    # ADR 021 AP-2: claim the packet directory before the first result byte.
    # Each arm's mot17.py run claims its own sub-directory (no parent-claim
    # env is passed down), so its run_manifest.json records its own argv.
    from scripts.provenance.run_manifest import open_run

    open_run(
        packet,
        produced_by="diagnostic",
        preset=PRESET_NAME,
        detector="SDP",
        dataset=f"{DATA_ROOT} {SPLIT}",
    )
    record: dict[str, Any] = {
        "schema": "saccade.head_failure_localization_packet/v1",
        "issue": "#465 Phase B redesign: head failure localization",
        "declaration": DECLARATION,
        "frozen_declaration_blob": FROZEN_DECLARATION_BLOB,
        "freeze_tag": FREEZE_TAG,
        "evidence": True,
        "sequences": list(SEQUENCES),
        "run_order": [f"{a}#{r}" for a, r in RUN_ORDER],
        "started_utc": _utc(),
        "caller_saccade_env": _saccade_env(),
    }
    print(f"packet: {packet.relative_to(project_root)}", flush=True)

    terminal, label = "UNRESOLVED", None
    reasons: list[str] = []
    try:
        v1_ok, v1 = check_v1()
        record["v1"] = v1
        _write_json(packet / "v1.json", v1)
        if not v1_ok:
            reasons.append(
                "V1: " + "; ".join(c["item"] for c in v1["checks"] if not c["ok"])
            )
        else:
            decision_inputs, m_reasons = _measure(packet, record)
            reasons += m_reasons
            end_lease = held_lease()
            record["lease_at_end"] = end_lease
            if end_lease != v1["lease"]:
                reasons.append("lease changed during the measurement")
            terminal, label = decide(not reasons, *decision_inputs)
    except Exception as exc:  # noqa: BLE001 — execution-invalid ⇒ UNRESOLVED
        reasons.append(f"runner error: {exc!r}")
    if reasons:
        terminal, label = "UNRESOLVED", None

    record.update(
        {
            "finished_utc": _utc(),
            "unresolved_reasons": reasons,
            "terminal": terminal,
            "mechanism_label": label,
        }
    )
    _write_json(packet / "packet.json", record)
    _write_manifest(packet)
    print(f"terminal: {terminal}; mechanism label: {label}", flush=True)
    for r in reasons:
        print(f"  reason: {r}", flush=True)
    return 0 if terminal != "UNRESOLVED" else 2


def _measure(
    packet: Path, record: dict[str, Any]
) -> tuple[tuple[bool | None, bool | None, bool | None, bool | None], list[str]]:

    none: tuple[bool | None, bool | None, bool | None, bool | None] = (None,) * 4
    reasons: list[str] = []
    l2_dir = packet / "l2"
    l2_dir.mkdir()
    # Each run is validated before the next one launches: a failed worker
    # (V3 raises on its first failing frame), missing output, a manifest
    # mismatch, incomplete coverage, differing env overrides, or a V2 mismatch
    # ends the study at once with the completed runs kept in the packet.
    runs: list[dict[str, Any]] = []
    record["runs"] = runs
    head = record["v1"]["git"]["head"]
    for arm, rep in RUN_ORDER:
        print(f"[arm] {arm}#{rep}", flush=True)
        run = run_arm(l2_dir, arm, rep)
        runs.append(run)
        problems = run_problems(run, head, runs)
        if problems:
            record["aborted_after"] = f"{arm}#{rep}"
            return none, reasons + problems
    by = {(r["arm"], r["rep"]): r for r in runs}

    scored = {
        arm: score_arm(project_root / by[(arm, 1)]["output_dir"], SEQUENCES)
        for arm in ARMS
    }
    _write_json(packet / "metrics.json", scored)
    ref = scored[REFERENCE_ARM]["combined"]
    decision = {}
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        out, detail = out_of_bounds(scored[arm]["combined"], ref)
        decision[arm] = {"out": out, "detail": detail}
    v4 = decision["R_T"]["out"]
    if not v4:
        reasons.append("V4: R_T is not out of bounds against R_C")
    record["decision"] = {
        "V4_R_T_out": v4,
        "E_out": decision["R_E"]["out"],
        "D_suff": decision["H_M"]["out"],
        "V_suff": decision["H_V"]["out"],
        "per_arm": decision,
        "report_only": {
            arm: {k: scored[arm]["combined"][k] for k in ("DetA", "AssA", "FP", "FN")}
            for arm in ARMS
        },
    }
    try:
        record["report"] = _report(by)
        record["report"]["prior_byte_identity"] = _prior_identity(by)
    except Exception as exc:  # noqa: BLE001 — §6 is report-only; never gates
        record["report"] = {"error": repr(exc)}
    return (
        v4,
        decision["R_E"]["out"],
        decision["H_M"]["out"],
        decision["H_V"]["out"],
    ), reasons


def _load_evidence(run: dict[str, Any], seq: str, np: Any) -> Any:
    with np.load(project_root / run["evidence_dir"] / f"{seq}.npz") as z:
        return {k: z[k] for k in z.files}


def _report(
    by: dict[tuple[str, int], dict[str, Any]],
    sequences: tuple[str, ...] = SEQUENCES,
) -> dict[str, Any]:
    """§6 for every sequence, from each arm's run #1 (run #2 is byte-identical
    in output; census identity across all runs in ``by`` is reported, not
    assumed). ``sequences`` differs from ``SEQUENCES`` only in the structural
    check."""
    import numpy as np

    report: dict[str, Any] = {
        "census": {},
        "delta_flow_R_T": {},
        "vs_R_C": {},
        "worker_summaries": {
            f"{a}#{r}": json.loads(
                (project_root / run["evidence_dir"] / "worker.json").read_text()
            )["sequences"]
            for (a, r), run in by.items()
        },
    }
    census_identical = True
    for seq in sequences:
        evs = {key: _load_evidence(run, seq, np) for key, run in by.items()}
        base = evs[("R_C", 1)]
        for key, ev in evs.items():
            for k in base:
                if k.startswith(("ct_", "ec_")) and not np.array_equal(ev[k], base[k]):
                    census_identical = False
        rt = evs[("R_T", 1)]
        report["census"][seq] = {
            "C_vs_T": census_totals(rt, "ct"),
            "E_vs_C": census_totals(rt, "ec"),
            "first_frame_C_vs_T": census_first_frames(rt),
            "row_anchor_consistency": {
                arm: {
                    "rows_above_floor": int(
                        (evs[(arm, 1)]["rows"][:, :, 4] >= SCORE_FLOOR).sum()
                    ),
                    "inconsistent_rows_above_floor": int(
                        (
                            (~evs[(arm, 1)]["row_consistent"])
                            & (evs[(arm, 1)]["rows"][:, :, 4] >= SCORE_FLOOR)
                        ).sum()
                    ),
                }
                for arm in ARMS
            },
        }
        report["delta_flow_R_T"][seq] = delta_flow(rt)
        report.setdefault("delta_e_flow_R_E", {})[seq] = delta_flow(
            evs[("R_E", 1)], "row_in_delta_e"
        )
        ref_txt = (
            project_root / by[("R_C", 1)]["output_dir"] / f"{seq}.txt"
        ).read_bytes()
        per_arm = {}
        for arm in ARMS:
            if arm == REFERENCE_ARM:
                continue
            delta_key = "row_in_delta_e" if arm == "R_E" else "row_in_delta"
            comparison = compare_runs(base, evs[(arm, 1)], delta_key)
            txt = (
                project_root / by[(arm, 1)]["output_dir"] / f"{seq}.txt"
            ).read_bytes()
            txt_first = first_divergent_frame(ref_txt, txt)
            per_arm[arm] = {
                "first_tracker_input_divergence": comparison[
                    "first_tracker_input_divergence"
                ],
                "first_tracker_input_explanation": comparison[
                    "first_tracker_input_explanation"
                ],
                "first_tracker_output_divergence": comparison[
                    "first_tracker_output_divergence"
                ],
                "output_trace": trace_output_divergence(txt_first, comparison),
            }
        report["vs_R_C"][seq] = per_arm
    report["census_identical_across_runs"] = census_identical
    return report


def _prior_identity(by: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    arms = {"R_C_vs_PR2_A_C": "R_C", "R_C_vs_PR2R_A_C": "R_C", "R_T_vs_PR2R_A_T": "R_T"}
    for key, rel in PRIOR_REFERENCE_TXT.items():
        prior = project_root / rel
        if not prior.is_dir():
            out[key] = None
            continue
        mine = project_root / by[(arms[key], 1)]["output_dir"]
        out[key] = all(
            (prior / f"{s}.txt").exists()
            and (prior / f"{s}.txt").read_bytes() == (mine / f"{s}.txt").read_bytes()
            for s in SEQUENCES
        )
    return out


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
