import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "build"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
saccade_tracking_ext = pytest.importorskip("saccade_tracking_ext", exc_type=ImportError)


def test_auction_real_values():
    # Similar to real training values: high base costs
    # Q:100, GT:20
    # Min L1:0.6, Min GIoU:0.8, Min ScoreCost:-0.1
    # Cost = 5*L1 + 2*GIoU + 1*ScoreCost
    # Min Cost ~ 5*0.6 + 2*0.8 - 0.1 = 3.0 + 1.6 - 0.1 = 4.5

    Q, GT = 100, 20
    rng = np.random.default_rng(0)
    # Random costs around 4.0 - 8.0
    cost = rng.uniform(4.0, 8.0, (Q, GT)).astype(np.float32)

    # Force some low values to simulate potential matches
    cost[0, 0] = 0.5
    cost[1, 1] = 1.0
    cost[2, 2] = 2.0

    row_idx, col_idx = saccade_tracking_ext.auction_solve_cpp(cost, 0.01)
    matches = dict(zip(row_idx, col_idx))
    assert len(row_idx) == GT
    assert matches[0] == 0
    assert matches[1] == 1
    assert matches[2] == 2

    # Testing with values ALL around 4.0
    cost_all_high = rng.uniform(4.0, 5.0, (Q, GT)).astype(np.float32)
    row_idx2, col_idx2 = saccade_tracking_ext.auction_solve_cpp(cost_all_high, 0.01)
    assert len(row_idx2) == GT
    assert len(set(row_idx2)) == GT
    assert len(set(col_idx2)) == GT

    # Negative costs are still costs: the solver should minimize them directly,
    # not reinterpret them as rewards.
    cost_neg = rng.uniform(-8.0, -4.0, (Q, GT)).astype(np.float32)
    cost_neg[0, 0] = -20.0
    cost_neg[1, 1] = -19.0
    cost_neg[2, 2] = -18.0
    row_idx3, col_idx3 = saccade_tracking_ext.auction_solve_cpp(cost_neg, 0.01)
    matches3 = dict(zip(row_idx3, col_idx3))
    assert len(row_idx3) == GT
    assert matches3[0] == 0
    assert matches3[1] == 1
    assert matches3[2] == 2


if __name__ == "__main__":
    test_auction_real_values()
