"""
TemporalYOLOHybrid 訓練損失函數。

使用 Auction Matching 在 Track Queries 輸出與 GT boxes 之間做最優二分匹配，
然後計算：
  1. Box Regression Loss   (L1 + GIoU)
  2. Existence Score Loss  (Binary Cross-Entropy)

Auction vs scipy Hungarian：
  - 速度：O(N²) per iteration vs O(N³)；N≤200 的追蹤場景下實測快 2~5×
  - 一致性：與 tracker_gpu.cu 的 parallel_auction_shmem_kernel 同一演算法族
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, Future
import os

import saccade_tracking_ext


def _auction_solve(
    cost_matrix: np.ndarray,  # (N_bidders, N_items)
    epsilon: float = 0.01,
) -> tuple[list[int], list[int]]:
    """
    呼叫 C++ 原生的 AuctionAlgorithm::Solve 進行線性指派匹配。
    """
    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        return [], []

    row_ind, col_ind = saccade_tracking_ext.auction_solve_cpp(
        cost_matrix.astype(np.float32), epsilon
    )
    return row_ind, col_ind


class AuctionMatcher:
    def __init__(
        self,
        w_bbox: float = 5.0,
        w_giou: float = 2.0,
        w_score: float = 1.0,
        epsilon: float = 0.01,
    ):
        self.w_bbox = w_bbox
        self.w_giou = w_giou
        self.w_score = w_score
        self.epsilon = epsilon

    @torch.no_grad()
    def get_cost_matrix(
        self,
        pred_boxes: torch.Tensor,  # (N_q, 4) [cx, cy, w, h] normalized
        pred_scores: torch.Tensor,  # (N_q,)
        gt_boxes: torch.Tensor,  # (N_gt, 4) [x1, y1, x2, y2] absolute px
        img_hw: tuple[int, int],
    ) -> torch.Tensor:
        """計算代價矩陣，保留在原始 device (GPU)。"""
        N_q = pred_boxes.shape[0]
        N_gt = gt_boxes.shape[0]
        if N_gt == 0:
            return pred_boxes.new_zeros((N_q, 0))

        H, W = img_hw
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 / W
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 / H
        gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]) / W
        gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]) / H
        gt_norm = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)  # (N_gt, 4)

        cost_l1 = torch.cdist(pred_boxes.float(), gt_norm.float(), p=1)
        cost_giou = 1.0 - _batch_giou_cxcywh(pred_boxes, gt_norm)
        # 使用 sigmoid 後的分數，確保數值範圍在 0~1，與 L1/GIoU 同階
        cost_score = -torch.sigmoid(pred_scores).unsqueeze(1).expand(N_q, N_gt)

        return (
            self.w_bbox * cost_l1 + self.w_giou * cost_giou + self.w_score * cost_score
        )

    def match_from_cost(self, cost: torch.Tensor) -> tuple[list[int], list[int]]:
        """傳入 CPU 上的 cost matrix 並求解。注意：C++ Solver 是 Reward Maximizer。"""
        # 取負號：將 Minimization 轉為 Maximization
        reward = -cost.numpy()
        return _auction_solve(reward, epsilon=self.epsilon)


_SavedBatch = tuple[
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    tuple[int, int],
]


class TemporalTrackingLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 5.0,
        w_giou: float = 2.0,
        w_bce: float = 1.0,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_giou = w_giou
        self.w_bce = w_bce
        self.matcher = AuctionMatcher(w_bbox=w_l1, w_giou=w_giou, w_score=w_bce)
        # Persistent executor — avoids thread-pool creation overhead per batch.
        # Not saved by torch.save (not a parameter/buffer), which is intentional.
        self._executor = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_cost_matrices(
        self,
        pred_boxes_batch: list[list[torch.Tensor]],
        pred_scores_batch: list[list[torch.Tensor]],
        gt_boxes_batch: list[list[torch.Tensor]],
        img_hw: tuple[int, int],
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        """GPU: compute cost matrices for all (b, t) pairs with non-empty GT."""
        device = pred_boxes_batch[0][0].device
        cost_matrices, matching_keys = [], []
        for b, (pb_seq, ps_seq, gb_seq) in enumerate(
            zip(pred_boxes_batch, pred_scores_batch, gt_boxes_batch)
        ):
            for t, (pb, ps, gb) in enumerate(zip(pb_seq, ps_seq, gb_seq)):
                gb = gb.to(device)
                if gb.shape[0] > 0:
                    cost_matrices.append(
                        self.matcher.get_cost_matrix(pb, ps, gb, img_hw)
                    )
                    matching_keys.append((b, t))
        return cost_matrices, matching_keys

    def _solve_all(
        self,
        cost_matrices_cpu: list[torch.Tensor],
        matching_keys: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[list[int], list[int]]]:
        """CPU: run auction solver for every cost matrix (called in thread pool)."""
        return {
            key: self.matcher.match_from_cost(cost)
            for key, cost in zip(matching_keys, cost_matrices_cpu)
        }

    def _compute_loss(
        self,
        pred_boxes_batch: list[list[torch.Tensor]],
        pred_scores_batch: list[list[torch.Tensor]],
        gt_boxes_batch: list[list[torch.Tensor]],
        img_hw: tuple[int, int],
        all_matches: dict[tuple[int, int], tuple[list[int], list[int]]],
    ) -> dict[str, torch.Tensor]:
        """GPU: accumulate box / score losses using pre-computed matches."""
        H, W = img_hw
        device = pred_boxes_batch[0][0].device
        total_l1 = pred_boxes_batch[0][0].new_zeros(())
        total_giou = pred_boxes_batch[0][0].new_zeros(())
        total_bce = pred_boxes_batch[0][0].new_zeros(())
        n_matched = 0

        for b, (pb_seq, ps_seq, gb_seq) in enumerate(
            zip(pred_boxes_batch, pred_scores_batch, gt_boxes_batch)
        ):
            for t, (pb, ps, gb) in enumerate(zip(pb_seq, ps_seq, gb_seq)):
                score_target = torch.zeros_like(ps)
                match = all_matches.get((b, t))
                if match is not None:
                    q_idx, g_idx = match
                    if q_idx:
                        q_t = torch.tensor(q_idx, device=device)
                        g_t = torch.tensor(g_idx, device=device)
                        gb_dev = gb.to(device)
                        gt_cx = (gb_dev[g_t, 0] + gb_dev[g_t, 2]) / 2 / W
                        gt_cy = (gb_dev[g_t, 1] + gb_dev[g_t, 3]) / 2 / H
                        gt_w = (gb_dev[g_t, 2] - gb_dev[g_t, 0]) / W
                        gt_h = (gb_dev[g_t, 3] - gb_dev[g_t, 1]) / H
                        gt_norm = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)
                        pred_matched = pb[q_t]
                        total_l1 = total_l1 + F.l1_loss(
                            pred_matched, gt_norm, reduction="sum"
                        )
                        total_giou = (
                            total_giou
                            + (1.0 - _batch_giou_cxcywh(pred_matched, gt_norm)).sum()
                        )
                        score_target[q_t] = 1.0
                        n_matched += len(q_idx)
                total_bce = total_bce + F.binary_cross_entropy_with_logits(
                    ps, score_target, reduction="sum"
                )

        denom = max(n_matched, 1)
        loss_l1 = self.w_l1 * total_l1 / denom
        loss_giou = self.w_giou * total_giou / denom
        loss_bce = self.w_bce * total_bce / denom
        return {
            "loss_l1": loss_l1,
            "loss_giou": loss_giou,
            "loss_bce": loss_bce,
            "loss_total": loss_l1 + loss_giou + loss_bce,
            "n_matched": torch.tensor(n_matched, device=device),
        }

    # ------------------------------------------------------------------
    # Async API (for overlapping CPU solving with GPU forward)
    # ------------------------------------------------------------------
    def start_matching(
        self,
        pred_boxes_batch: list[list[torch.Tensor]],
        pred_scores_batch: list[list[torch.Tensor]],
        gt_boxes_batch: list[list[torch.Tensor]],
        img_hw: tuple[int, int],
    ) -> tuple[Future[dict[tuple[int, int], tuple[list[int], list[int]]]], _SavedBatch]:
        """
        GPU: compute cost matrices, then submit CPU auction to background thread.
        Returns (Future, saved_batch) for finish_loss().
        """
        cost_matrices, matching_keys = self._build_cost_matrices(
            pred_boxes_batch, pred_scores_batch, gt_boxes_batch, img_hw
        )
        cost_matrices_cpu = [c.cpu() for c in cost_matrices]
        future = self._executor.submit(
            self._solve_all, cost_matrices_cpu, matching_keys
        )
        saved = (pred_boxes_batch, pred_scores_batch, gt_boxes_batch, img_hw)
        return future, saved

    def finish_loss(
        self,
        future: Future[dict[tuple[int, int], tuple[list[int], list[int]]]],
        saved: _SavedBatch,
    ) -> dict[str, torch.Tensor]:
        """Wait for CPU auction result, compute GPU loss. Pair with start_matching()."""
        pred_boxes_batch, pred_scores_batch, gt_boxes_batch, img_hw = saved
        all_matches = future.result()
        return self._compute_loss(
            pred_boxes_batch, pred_scores_batch, gt_boxes_batch, img_hw, all_matches
        )

    # ------------------------------------------------------------------
    # Fast forward: GPU-only greedy matching (no D2H, no CPU auction)
    # ------------------------------------------------------------------
    def forward_fast(
        self,
        pred_boxes_batch: list[list[torch.Tensor]],
        pred_scores_batch: list[list[torch.Tensor]],
        gt_boxes_batch: list[list[torch.Tensor]],
        img_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        Greedy matching entirely on GPU: for each GT find argmin-L1 prediction.
        No D2H transfer, no CPU solver — ~3-5× faster than auction forward().
        Matching quality is slightly lower than auction but good enough for Phase 1.
        """
        H, W = img_hw
        device = pred_boxes_batch[0][0].device
        total_l1 = pred_boxes_batch[0][0].new_zeros(())
        total_giou = pred_boxes_batch[0][0].new_zeros(())
        total_bce = pred_boxes_batch[0][0].new_zeros(())
        n_matched = 0

        for pb_seq, ps_seq, gb_seq in zip(
            pred_boxes_batch, pred_scores_batch, gt_boxes_batch
        ):
            for pb, ps, gb_abs in zip(pb_seq, ps_seq, gb_seq):
                score_target = torch.zeros_like(ps)
                gb_abs = gb_abs.to(device)

                if gb_abs.shape[0] > 0:
                    # Normalize GT: xyxy absolute → cxcywh normalized
                    gt_cx = (gb_abs[:, 0] + gb_abs[:, 2]) * 0.5 / W
                    gt_cy = (gb_abs[:, 1] + gb_abs[:, 3]) * 0.5 / H
                    gt_w = (gb_abs[:, 2] - gb_abs[:, 0]) / W
                    gt_h = (gb_abs[:, 3] - gb_abs[:, 1]) / H
                    gt_norm = torch.stack(
                        [gt_cx, gt_cy, gt_w, gt_h], dim=1
                    )  # (N_gt, 4)

                    # L1 cost on GPU: (N_q, N_gt) → for each GT pick best pred
                    cost = torch.cdist(pb.float(), gt_norm.float(), p=1)  # (N_q, N_gt)
                    matched_q = cost.argmin(dim=0)  # (N_gt,) — index into predictions

                    pred_matched = pb[matched_q]  # (N_gt, 4)
                    total_l1 = total_l1 + F.l1_loss(
                        pred_matched, gt_norm, reduction="sum"
                    )
                    total_giou = (
                        total_giou
                        + (1.0 - _batch_giou_cxcywh(pred_matched, gt_norm)).sum()
                    )
                    score_target[matched_q] = 1.0
                    n_matched += gb_abs.shape[0]

                total_bce = total_bce + F.binary_cross_entropy_with_logits(
                    ps, score_target, reduction="sum"
                )

        denom = max(n_matched, 1)
        loss_l1 = self.w_l1 * total_l1 / denom
        loss_giou = self.w_giou * total_giou / denom
        loss_bce = self.w_bce * total_bce / denom
        return {
            "loss_l1": loss_l1,
            "loss_giou": loss_giou,
            "loss_bce": loss_bce,
            "loss_total": loss_l1 + loss_giou + loss_bce,
            "n_matched": torch.tensor(n_matched, device=device),
        }

    # ------------------------------------------------------------------
    # Synchronous forward (kept for compatibility / single-batch eval)
    # ------------------------------------------------------------------
    def forward(
        self,
        pred_boxes_batch: list[list[torch.Tensor]],
        pred_scores_batch: list[list[torch.Tensor]],
        gt_boxes_batch: list[list[torch.Tensor]],
        img_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        cost_matrices, matching_keys = self._build_cost_matrices(
            pred_boxes_batch, pred_scores_batch, gt_boxes_batch, img_hw
        )
        cost_matrices_cpu = [c.cpu() for c in cost_matrices]
        all_matches = self._solve_all(cost_matrices_cpu, matching_keys)
        return self._compute_loss(
            pred_boxes_batch, pred_scores_batch, gt_boxes_batch, img_hw, all_matches
        )


# ---------------------------------------------------------------------------
# GIoU helpers
# ---------------------------------------------------------------------------
def _batch_giou_cxcywh(
    pred: torch.Tensor,  # (N, 4) [cx, cy, w, h]
    gt: torch.Tensor,  # (M, 4) [cx, cy, w, h]
) -> torch.Tensor:
    def to_xyxy(b: torch.Tensor) -> torch.Tensor:
        x1 = b[..., 0] - b[..., 2] / 2
        y1 = b[..., 1] - b[..., 3] / 2
        x2 = b[..., 0] + b[..., 2] / 2
        y2 = b[..., 1] + b[..., 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    if pred.shape[0] == gt.shape[0]:
        p = to_xyxy(pred)
        g = to_xyxy(gt)
        return _giou_pairs(p, g)
    else:
        p = to_xyxy(pred).unsqueeze(1)  # (N, 1, 4)
        g = to_xyxy(gt).unsqueeze(0)  # (1, M, 4)
        return _giou_pairs(p, g)


def _giou_pairs(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    ix1 = torch.maximum(p[..., 0], g[..., 0])
    iy1 = torch.maximum(p[..., 1], g[..., 1])
    ix2 = torch.minimum(p[..., 2], g[..., 2])
    iy2 = torch.minimum(p[..., 3], g[..., 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    ap = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union = ap + ag - inter
    iou = inter / union.clamp(1e-6)
    cx1 = torch.minimum(p[..., 0], g[..., 0])
    cy1 = torch.minimum(p[..., 1], g[..., 1])
    cx2 = torch.maximum(p[..., 2], g[..., 2])
    cy2 = torch.maximum(p[..., 3], g[..., 3])
    enclose = (cx2 - cx1).clamp(0) * (cy2 - cy1).clamp(0)
    return iou - (enclose - union) / enclose.clamp(1e-6)
