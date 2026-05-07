
import sys
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.helpers import budget_reid_candidates as _budget_reid_candidates  # noqa: E402
from saccade.perception.tracking.dynamic_reid import DynamicReIDController  # noqa: E402
from saccade.perception.tracking.tracker_gpu import ReIDTrackObservation  # noqa: E402

def test_budget_reid_candidates_no_budget():
    fused_boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
    fused_scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    
    # budget = 0 means unlimited
    indices = _budget_reid_candidates(fused_boxes, fused_scores, budget=0)
    assert torch.equal(indices, torch.tensor([0, 1]))

def test_budget_reid_candidates_within_budget():
    fused_boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
    fused_scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    
    # num_dets <= budget
    indices = _budget_reid_candidates(fused_boxes, fused_scores, budget=5)
    assert torch.equal(indices, torch.tensor([0, 1]))

def test_budget_reid_candidates_fallback_score():
    fused_boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=torch.float32)
    fused_scores = torch.tensor([0.7, 0.9, 0.8], dtype=torch.float32)
    
    # budget = 2, pick top 2 by score: idx 1 (0.9) and idx 2 (0.8)
    indices = _budget_reid_candidates(fused_boxes, fused_scores, budget=2)
    # Sorted indices: [1, 2]
    assert torch.equal(indices, torch.tensor([1, 2]))

def test_budget_reid_candidates_with_dynamic_priorities():
    # Setup dynamic ReID with some history to have priorities
    dynamic_reid = DynamicReIDController(mode="score_ema")
    
    # Frame 1: observe one track
    # TID 1 at [0, 0, 10, 10]
    dynamic_reid.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 10.0), 0.9)})
    
    # Now in Frame 2, TID 1 should have some priority (e.g. baseline or instability if we could fake it)
    # Let's say we want to prioritize TID 1.
    
    fused_boxes = torch.tensor([
        [40, 40, 50, 50], # Distant, score 0.95
        [1, 1, 11, 11],   # Near TID 1, score 0.70
    ], dtype=torch.float32)
    fused_scores = torch.tensor([0.95, 0.70], dtype=torch.float32)
    
    # Without prioritization, budget=1 would pick index 0 (score 0.95)
    indices_no_priorities = _budget_reid_candidates(fused_boxes, fused_scores, budget=1)
    assert torch.equal(indices_no_priorities, torch.tensor([0]))
    
    # With prioritization, if TID 1 has high priority, index 1 should be picked.
    # We can force priority by making it "unstable" or "lost"
    # Actually, if we just observe Frame 1, TID 1 is "new" in Frame 1.
    # priorities = dynamic_reid.get_priorities()
    # TID 1 priority = weight_new * 0.9 = 1.0 * 0.9 = 0.9
    
    indices_with_priorities = _budget_reid_candidates(fused_boxes, fused_scores, budget=1, dynamic_reid=dynamic_reid)
    # det_priorities:
    # idx 0: 0.95 + 0 (IoU with TID 1 is 0) = 0.95
    # idx 1: 0.70 + 0.9 * IoU([1,1,11,11], [0,0,10,10])
    # IoU is roughly (9*9) / (10*10 + 10*10 - 81) = 81 / 119 ~= 0.68
    # idx 1 priority ~= 0.70 + 0.9 * 0.68 ~= 0.70 + 0.61 ~= 1.31
    # 1.31 > 0.95, so index 1 should be picked.
    
    assert torch.equal(indices_with_priorities, torch.tensor([1]))

def test_budget_reid_candidates_with_gmc():
    dynamic_reid = DynamicReIDController(mode="score_ema")
    dynamic_reid.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 10.0), 0.9)})
    
    # Camera moved right by 50 pixels
    gmc_warp = torch.tensor([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    
    fused_boxes = torch.tensor([
        [0, 0, 10, 10],   # Original position, now empty due to camera move
        [51, 1, 61, 11], # Near warped TID 1 position
    ], dtype=torch.float32)
    fused_scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    
    # Budget = 1
    # Warped TID 1 box: [50, 0, 60, 10]
    # IoU with idx 0: 0
    # IoU with idx 1: high
    
    indices = _budget_reid_candidates(fused_boxes, fused_scores, budget=1, dynamic_reid=dynamic_reid, gmc_warp=gmc_warp)
    assert torch.equal(indices, torch.tensor([1]))

if __name__ == "__main__":
    pytest.main([__file__])
