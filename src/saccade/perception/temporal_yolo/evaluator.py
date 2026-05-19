"""
MOTREvaluator — 使用 TemporalYOLOHybrid 的逐幀追蹤評估器。

負責：
  - 驅動影片序列的幀迴圈
  - 將 updated_queries 跨幀傳遞（核心的時序回饋機制）
  - 將輸出結果寫成 MOT txt 格式
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Iterator
import torch
from saccade.perception.eval.config import EvalConfig
from .model import TemporalYOLOConfig, build_temporal_yolo_model, TemporalYOLOHybrid


def _iter_frames(seq_dir: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    從序列目錄讀取 jpg 影像，依 frame number 排序，
    以 (frame_id, tensor) 逐幀 yield。
    """
    try:
        import torchvision.io as tv_io
    except ImportError:
        raise ImportError("torchvision is required for frame loading")

    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(img_dir.glob("*.png"))

    for fpath in frames:
        frame_id = int(fpath.stem)  # e.g. "000001" -> 1
        img = tv_io.read_image(str(fpath))  # (3, H, W) uint8
        img = img.float() / 255.0  # (3, H, W) float32
        yield frame_id, img.unsqueeze(0)  # (1, 3, H, W)


class MOTREvaluator:
    """
    End-to-End 追蹤評估器，採用 TemporalYOLOHybrid。

    核心設計：
      - self.track_queries 持有跨幀的 Track Queries（(1, N_q, D)）
      - 每幀呼叫 model(frame, prev_queries=self.track_queries)
      - 模型輸出 updated_queries 後更新 self.track_queries
      - 存活 Query（score >= threshold，query_id != -1）輸出為追蹤結果
    """

    def __init__(
        self,
        config: EvalConfig,
        sequence_name: str,
        sequence_dir: Path,
        model_cfg: TemporalYOLOConfig | None = None,
        weights_path: str = "",
    ):
        self.config = config
        self.sequence_name = sequence_name
        self.sequence_dir = sequence_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 建立模型（PoC 階段使用 Stub 架構；weights_path 有值時才載入真實權重）
        self.model: TemporalYOLOHybrid = build_temporal_yolo_model(
            cfg=model_cfg,
            weights_path=weights_path,
            device=self.device,
        )
        self.model.eval()

        # 跨幀 Track Queries 狀態（第一幀前為 None）
        self.track_queries: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self) -> list[str]:
        """
        執行整個序列的 End-to-End 追蹤，回傳 MOT txt 格式的結果行。

        MOT txt 格式：
          frame_id, track_id, x1, y1, w, h, -1, -1, -1, -1
        """
        mot_lines: list[str] = []
        self.model.reset_sequence()
        self.track_queries = None

        output_dir = self.config.output_root / self.sequence_name
        output_dir.mkdir(parents=True, exist_ok=True)

        total_frames = 0
        t_start = time.perf_counter()

        with torch.inference_mode():
            for frame_id, frame_tensor in _iter_frames(self.sequence_dir):
                frame_tensor = frame_tensor.to(self.device)

                # ── 核心的 Temporal Feedback 推理 ──
                outputs = self._time_stage(frame_tensor)

                # ── 解析輸出 ──
                mot_lines.extend(
                    self._to_mot_lines(frame_id, outputs, frame_tensor.shape)
                )
                total_frames += 1

        elapsed = time.perf_counter() - t_start
        fps = total_frames / max(elapsed, 1e-6)
        print(
            f"[MOTREvaluator] {self.sequence_name}: "
            f"{total_frames} frames, {fps:.1f} FPS"
        )

        # 寫入 txt
        mot_file = output_dir / "results.txt"
        mot_file.write_text("\n".join(mot_lines))

        return mot_lines

    # ------------------------------------------------------------------
    # Core feedback loop
    # ------------------------------------------------------------------
    def _time_stage(self, frame: torch.Tensor) -> dict:
        """
        單幀推理。

        流程：
          1. 將 prev_queries (self.track_queries) 傳入模型
          2. YOLO Backbone 提取特徵圖
          3. Transformer Decoder 讓 Track Queries × 特徵圖做 Cross-Attention
          4. 輸出 boxes, scores, updated_queries, query_ids
          5. 將 updated_queries 存回 self.track_queries，供下一幀使用

        這是整個混合架構的核心。
        """
        outputs = self.model(frame, prev_queries=self.track_queries)

        # 更新狀態（關鍵一行：把新的 queries 存起來，下一幀才能「記住」物體）
        self.track_queries = outputs["updated_queries"].detach()

        return outputs

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------
    def _to_mot_lines(
        self,
        frame_id: int,
        outputs: dict,
        frame_shape: tuple,
    ) -> list[str]:
        """
        將模型輸出轉換為 MOT txt 格式：
          frame_id, track_id, x1, y1, w, h, -1, -1, -1, -1

        boxes 格式為 [cx, cy, w, h]（相對座標），需轉換為絕對像素座標。
        """
        _, _, H, W = frame_shape
        boxes = outputs["boxes"][0]  # (N_q, 4)
        scores = outputs["scores"][0]  # (N_q,)
        query_ids: list[int] = outputs["query_ids"]
        threshold = self.model.cfg.score_threshold

        lines: list[str] = []
        for i, (box, score, qid) in enumerate(zip(boxes, scores, query_ids)):
            if score < threshold or qid == -1:
                continue

            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            pw = bw * W
            ph = bh * H

            # MOT txt format
            lines.append(
                f"{frame_id},{qid},{x1:.2f},{y1:.2f},{pw:.2f},{ph:.2f},-1,-1,-1,-1"
            )

        return lines
