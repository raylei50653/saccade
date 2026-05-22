import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
saccade_tracking_ext = pytest.importorskip("saccade_tracking_ext")


def test_auction_real_values():
    # Similar to real training values: high base costs
    # Q:100, GT:20
    # Min L1:0.6, Min GIoU:0.8, Min ScoreCost:-0.1
    # Cost = 5*L1 + 2*GIoU + 1*ScoreCost
    # Min Cost ~ 5*0.6 + 2*0.8 - 0.1 = 3.0 + 1.6 - 0.1 = 4.5

    Q, GT = 100, 20
    # Random costs around 4.0 - 8.0
    cost = np.random.uniform(4.0, 8.0, (Q, GT)).astype(np.float32)

    # Force some low values to simulate potential matches
    cost[0, 0] = 0.5
    cost[1, 1] = 1.0
    cost[2, 2] = 2.0

    print(f"Testing with {Q}x{GT} cost matrix, values 4-8, with few low values.")
    row_idx, col_idx = saccade_tracking_ext.auction_solve_cpp(cost, 0.01)
    print(f"Matches: {len(row_idx)}")

    # Testing with values ALL around 4.0
    cost_all_high = np.random.uniform(4.0, 5.0, (Q, GT)).astype(np.float32)
    print(f"Testing with {Q}x{GT} all high (4-5):")
    row_idx2, col_idx2 = saccade_tracking_ext.auction_solve_cpp(cost_all_high, 0.01)
    print(f"Matches: {len(row_idx2)}")

    # Testing with negative values (just in case it's a reward maximizer)
    cost_neg = -cost
    print("Testing with negative costs (all -8 to -4):")
    row_idx3, col_idx3 = saccade_tracking_ext.auction_solve_cpp(cost_neg, 0.01)
    print(f"Matches: {len(row_idx3)}")


if __name__ == "__main__":
    test_auction_real_values()
