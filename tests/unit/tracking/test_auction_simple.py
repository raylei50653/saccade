"""Unit tests for the temporal-YOLO auction matcher (perception.temporal_yolo.loss.AuctionMatcher)."""

# scope: detection
# function: behavior
# lifecycle: active

import numpy as np
import pytest
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "build"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
saccade_tracking_ext = pytest.importorskip("saccade_tracking_ext", exc_type=ImportError)

from saccade.perception.temporal_yolo.loss import AuctionMatcher  # noqa: E402


def test_auction():
    # 3 bidders, 3 items
    # Diagonal should be the best match if cost is low on diagonal
    cost = np.array(
        [[0.1, 0.5, 0.5], [0.5, 0.1, 0.5], [0.5, 0.5, 0.1]], dtype=np.float32
    )

    row_idx, col_idx = saccade_tracking_ext.auction_solve_cpp(cost, 0.01)
    assert row_idx == [0, 1, 2], f"Expected diagonal match, got row_idx={row_idx}"
    assert col_idx == [0, 1, 2], f"Expected diagonal match, got col_idx={col_idx}"

    # 2 bidders, 3 items
    cost2 = np.array([[0.1, 0.5, 0.5], [0.5, 0.1, 0.5]], dtype=np.float32)
    row_idx2, col_idx2 = saccade_tracking_ext.auction_solve_cpp(cost2, 0.01)
    assert len(row_idx2) == 2
    assert len(col_idx2) == 2


def test_auction_solve_cpp_minimizes_cost():
    cost = np.array([[0.1, 9.0], [8.0, 0.2]], dtype=np.float32)

    row_idx, col_idx = saccade_tracking_ext.auction_solve_cpp(cost, 0.01)

    assert row_idx == [0, 1]
    assert col_idx == [0, 1]


def test_auction_solve_cpp_rectangular_cost_padding():
    cost = np.array([[0.1, 100.0, 100.0], [0.2, 0.3, 100.0]], dtype=np.float32)

    row_idx, col_idx = saccade_tracking_ext.auction_solve_cpp(cost, 0.01)

    assert row_idx == [0, 1]
    assert col_idx == [0, 1]


def test_auction_matcher_passes_cost_without_double_negation():
    cost = torch.tensor([[0.1, 9.0], [8.0, 0.2]], dtype=torch.float32)

    row_idx, col_idx = AuctionMatcher().match_from_cost(cost)

    assert row_idx == [0, 1]
    assert col_idx == [0, 1]


def test_auction_matcher_detaches_cost_tensor():
    cost = torch.tensor(
        [[0.1, 9.0], [8.0, 0.2]], dtype=torch.float32, requires_grad=True
    )

    row_idx, col_idx = AuctionMatcher().match_from_cost(cost)

    assert row_idx == [0, 1]
    assert col_idx == [0, 1]


if __name__ == "__main__":
    test_auction()
