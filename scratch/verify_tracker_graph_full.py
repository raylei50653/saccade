#!/usr/bin/env python3
"""Verify tracker graph with evaluator's exact init sequence."""

import sys, torch
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "build")
from saccade.perception.detector_trt import TRTYoloDetector
from saccade.perception.tracking.tracker_gpu import GPUByteTracker, GraphedTrackerUpdate


def init_tracker(t):
    t.set_homography(None)
    t.set_reid_params(cos_threshold=0.90, iou_low=0.3, iou_high=0.6, weight=0.4)
    t.set_reid_min_candidates(1)
    t.set_frame_size(1920, 1080)
    t.set_quality_params(enabled=False, w_aspect=0.5, w_center=0.3, w_area=0.2)
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
    t.set_oao_params(tau=0.0)


# Eager
t_eager = GPUByteTracker(max_objects=2048)
init_tracker(t_eager)

# Graph (created after init, like evaluator)
t_graph = GPUByteTracker(max_objects=2048)
init_tracker(t_graph)
gtu = GraphedTrackerUpdate(t_graph)

CPU_GEN = torch.Generator(device="cpu")
CPU_GEN.manual_seed(1)
mismatched = 0
for f in range(40):
    n = int(torch.randint(3, 12, (1,), generator=CPU_GEN).item())
    boxes = torch.randn(n, 4).abs_().clamp_(0, 640).cuda()
    scores = torch.rand(n, generator=CPU_GEN).cuda() * 0.8 + 0.15
    classes = torch.zeros(n, device="cuda", dtype=torch.int32)
    gmc = torch.eye(2, 3, device="cuda")

    with torch.no_grad():
        buf = t_eager.allocate_result_buffers(device="cuda")
        t_eager.update_into(boxes, scores, classes, buf, gmc=gmc)
        torch.cuda.synchronize()
        e_cnt = buf["count"].item()

        gtu.copy_inputs(boxes, scores, classes, gmc=gmc)
        result = gtu.replay()
        torch.cuda.synchronize()
        g_cnt = result["count"].item()

    if e_cnt != g_cnt:
        print(f"F{f:3d}: count eager={e_cnt} graph={g_cnt}")
        mismatched += 1
    elif e_cnt > 0:
        e_ids = set(buf["ids"][:e_cnt].tolist())
        g_ids = set(result["ids"][:g_cnt].tolist())
        if e_ids != g_ids:
            print(f"F{f:3d}: ID mismatch eager={e_ids} graph={g_ids}")
            mismatched += 1

print(f"✓ All {40} match" if mismatched == 0 else f"✗ {mismatched}/40 mismatched")
