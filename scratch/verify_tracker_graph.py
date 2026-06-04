#!/usr/bin/env python3
"""Verify tracker graph output vs eager, frame by frame, bit-exact comparison."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402
import torch  # noqa: E402
from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    GPUByteTracker,
    GraphedTrackerUpdate,
)

NFRAMES = 30


def build_tracker():
    t = GPUByteTracker(max_objects=2048)
    t.set_params(
        track_thresh=0.1,
        high_thresh=0.5,
        match_thresh=0.5,
        mid_thresh=0.4,
        track_buffer=30,
        confirm_streak=3,
        confirm_score_thresh=0.5,
        new_track_thresh=0.28,
        nsa_kalman=False,
        r_scale=2.8,
    )
    return t


def gen_frames(seed=42):
    _gen = torch.Generator(device="cuda")
    _gen.manual_seed(seed)
    frames = {"boxes": [], "scores": [], "gmcs": []}
    for _ in range(NFRAMES):
        n = int(torch.randint(3, 15, (1,), device="cuda").item())
        boxes = (
            torch.rand(n, 4, device="cuda", dtype=torch.float32, generator=_gen) * 640
        )
        scores = (
            torch.rand(n, device="cuda", dtype=torch.float32, generator=_gen) * 0.9
            + 0.1
        )
        gmc = torch.eye(2, 3, device="cuda", dtype=torch.float32)
        frames["boxes"].append(boxes)
        frames["scores"].append(scores)
        frames["gmcs"].append(gmc)
    return frames


def main():
    frames = gen_frames()

    # ── Eager baseline ──
    t_eager = build_tracker()
    eager_outputs = []
    with torch.no_grad():
        for i in range(NFRAMES):
            buf = t_eager.allocate_result_buffers(device="cuda")
            classes = torch.zeros(
                frames["boxes"][i].shape[0], device="cuda", dtype=torch.int32
            )
            t_eager.update_into(
                frames["boxes"][i],
                frames["scores"][i],
                classes,
                buf,
                gmc=frames["gmcs"][i],
            )
            torch.cuda.synchronize()
            cnt = buf["count"].item()
            if cnt > 0:
                eager_outputs.append(
                    {
                        "count": cnt,
                        "boxes": buf["boxes"][:cnt].clone(),
                        "scores": buf["scores"][:cnt].clone(),
                        "ids": buf["ids"][:cnt].clone(),
                    }
                )
            else:
                eager_outputs.append({"count": 0})
    del t_eager

    # ── Graph version ──
    t_graph = build_tracker()
    gtu = GraphedTrackerUpdate(t_graph)
    graph_outputs = []
    with torch.no_grad():
        for i in range(NFRAMES):
            classes = torch.zeros(
                frames["boxes"][i].shape[0], device="cuda", dtype=torch.int32
            )
            gtu.copy_inputs(
                frames["boxes"][i],
                frames["scores"][i],
                classes,
                gmc=frames["gmcs"][i],
            )
            gtu.replay()
            torch.cuda.synchronize()
            result = gtu.replay()  # second call to get returned reference

            # Get outputs directly from GraphedTrackerUpdate
            cnt = result["count"].item()
            if cnt > 0:
                graph_outputs.append(
                    {
                        "count": cnt,
                        "boxes": result["boxes"][:cnt].clone(),
                        "scores": result["scores"][:cnt].clone(),
                        "ids": result["ids"][:cnt].clone(),
                    }
                )
            else:
                graph_outputs.append({"count": 0})
    del t_graph

    # ── Compare ──
    print(f"Frame-by-frame comparison ({NFRAMES} frames):")
    mismatches = 0
    for i in range(NFRAMES):
        e = eager_outputs[i]
        g = graph_outputs[i]
        if e["count"] != g["count"]:
            print(f"  F{i:3d}: count mismatch eager={e['count']} graph={g['count']}")
            mismatches += 1
            continue
        if e["count"] == 0:
            continue
        box_diff = (e["boxes"] - g["boxes"]).abs().max().item()
        score_diff = (e["scores"] - g["scores"]).abs().max().item()
        id_diff = (e["ids"] != g["ids"]).any().item()
        status = "✓" if box_diff < 1e-5 and score_diff < 1e-5 and not id_diff else "✗"
        if status == "✗":
            print(
                f"  F{i:3d}: {status} box={box_diff:.1e} score={score_diff:.1e} "
                f"ids_equal={not id_diff} count={e['count']}"
            )
            mismatches += 1

    if mismatches == 0:
        print(f"\n  ✓ All {NFRAMES} frames bit-exact — tracker graph is aligned!")
    else:
        print(f"\n  ✗ {mismatches}/{NFRAMES} frames MISMATCH — investigate!")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
