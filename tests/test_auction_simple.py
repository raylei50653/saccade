import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
saccade_tracking_ext = pytest.importorskip("saccade_tracking_ext")


def test_auction():
    # 3 bidders, 3 items
    # Diagonal should be the best match if cost is low on diagonal
    cost = np.array(
        [[0.1, 0.5, 0.5], [0.5, 0.1, 0.5], [0.5, 0.5, 0.1]], dtype=np.float32
    )

    print("Cost matrix:")
    print(cost)

    row_idx, col_idx = saccade_tracking_ext.auction_solve_cpp(cost, 0.01)
    print(f"Matches: row={row_idx}, col={col_idx}")

    # 2 bidders, 3 items
    cost2 = np.array([[0.1, 0.5, 0.5], [0.5, 0.1, 0.5]], dtype=np.float32)
    row_idx2, col_idx2 = saccade_tracking_ext.auction_solve_cpp(cost2, 0.01)
    print(f"Matches (2x3): row={row_idx2}, col={col_idx2}")


if __name__ == "__main__":
    test_auction()
