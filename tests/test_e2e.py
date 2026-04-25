"""End-to-end pipeline tests.

Skipped automatically when any of the following are unavailable:
  - CUDA GPU
  - models/yolo/yolo26s_batch4.engine
  - models/embedding/google_siglip2-base-patch16-224.engine
  - datasets/MOT17/train/MOT17-04-SDP/
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

ENGINE = ROOT / "models/yolo/yolo26s_batch4.engine"
REID_ENGINE = ROOT / "models/embedding/google_siglip2-base-patch16-224.engine"
DATA_ROOT = ROOT / "datasets/MOT17"
SEQ = "MOT17-04-SDP"

skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)
skip_no_engines = pytest.mark.skipif(
    not (ENGINE.exists() and REID_ENGINE.exists()), reason="TRT engines not found"
)
skip_no_dataset = pytest.mark.skipif(
    not (DATA_ROOT / "train" / SEQ).exists(), reason="MOT17 dataset not found"
)


def _run_eval(tmp_path, *, max_frames, no_reid, seq=SEQ):
    from perception.detector_trt import TRTYoloDetector  # noqa: F401 — must precede torchvision
    from scripts.eval.mot17 import run_eval

    out = str(tmp_path / "out")
    run_eval(
        engine=str(ENGINE),
        output=out,
        data_root=str(DATA_ROOT),
        split="train",
        sequences=seq,
        max_frames=max_frames,
        conf_threshold=0.25,
        no_reid=no_reid,
        warmup_frames=0,
        track_thresh=0.1,
        high_thresh=0.5,
        match_thresh=0.8,
        profile_stages=False,
    )
    return Path(out) / f"{seq}.txt"


@skip_no_gpu
@skip_no_engines
@skip_no_dataset
def test_e2e_smoke(tmp_path):
    """30 frames: pipeline completes and writes valid MOT-format output."""
    result = _run_eval(tmp_path, max_frames=30, no_reid=True)

    assert result.exists(), "result file not written"
    lines = [ln for ln in result.read_text().splitlines() if ln.strip()]
    assert len(lines) > 0, "no tracking output produced"

    mot_re = re.compile(
        r"^\d+,\d+,[0-9.eE+\-]+,[0-9.eE+\-]+,[0-9.eE+\-]+,[0-9.eE+\-]+,[0-9.eE+\-]+,-1,-1,-1$"
    )
    for line in lines[:20]:
        assert mot_re.match(line), f"malformed MOT line: {line!r}"

    track_ids = {int(ln.split(",")[1]) for ln in lines}
    assert len(track_ids) >= 1, "no track IDs assigned"


@skip_no_gpu
@skip_no_engines
@skip_no_dataset
def test_e2e_tracking_continuity(tmp_path):
    """60 frames, IoU-only: at least one track persists across 10+ consecutive frames."""
    result = _run_eval(tmp_path, max_frames=60, no_reid=True)

    tracks: dict[int, list[int]] = defaultdict(list)
    for line in result.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        tracks[int(parts[1])].append(int(parts[0]))

    max_span = max((max(f) - min(f) for f in tracks.values()), default=0)
    assert max_span >= 9, (
        f"no track spans 10+ frames; longest span = {max_span} frames"
    )


@skip_no_gpu
@skip_no_engines
@skip_no_dataset
def test_e2e_mota_floor(tmp_path):
    """150 frames, full pipeline with ReID: MOTA must be >= 15% on MOT17-04-SDP."""
    mm = pytest.importorskip("motmetrics")
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

    result = _run_eval(tmp_path, max_frames=150, no_reid=False)

    gt_path = DATA_ROOT / "train" / SEQ / "gt" / "gt.txt"
    gt_all = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp = mm.io.loadtxt(str(result), fmt="mot15-2D", min_confidence=-1.0)

    # Compare only on frames we actually evaluated so un-run frames
    # don't inflate FN and artificially tank MOTA.
    if not hyp.empty:
        eval_frames = hyp.index.get_level_values("FrameId").unique()
        gt = gt_all[gt_all.index.get_level_values("FrameId").isin(eval_frames)]
    else:
        gt = gt_all

    acc = mm.utils.compare_to_groundtruth(gt, hyp, "iou", distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota"], name="e2e")
    mota = float(summary["mota"].iloc[0])

    MOTA_FLOOR = 0.15
    assert mota >= MOTA_FLOOR, f"MOTA {mota:.1%} below floor {MOTA_FLOOR:.1%}"
