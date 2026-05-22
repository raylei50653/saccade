"""
Temporal YOLO Hybrid — YOLO backbone + Transformer Track Queries feedback.

Architecture:
  1. YOLOFeatureExtractor  — CNN backbone 輸出多尺度特徵圖 (Neck output)
  2. TrackQueryDecoder     — Transformer Cross-Attention，讓 Track Queries 在特徵圖上找到對應物體
  3. TrackQueryLifecycle   — 管理 Query 的出生（新物體）與死亡（消失物體）
  4. TemporalYOLOHybrid    — 整體模型，串接以上三個模組

輸入：
  - frame:          (B, 3, H, W)  當前幀的 RGB 影像
  - prev_queries:   (B, N_q, D)   上一幀的 Track Queries（第一幀時可為 None）

輸出 (dict)：
  - 'boxes':           (B, N_q, 4)    Bounding boxes，格式 [cx, cy, w, h]（相對座標 0~1）
  - 'scores':          (B, N_q)       每個 Query 的物體存在機率
  - 'updated_queries': (B, N_q, D)    更新後的 Track Queries，傳給下一幀
  - 'query_ids':       (B, N_q)       每個 Query 的追蹤 ID（-1 代表已消失）
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TemporalYOLOConfig:
    embed_dim: int = 256  # Track Query / attention 向量維度
    num_heads: int = 8  # Multi-head attention 的 head 數量
    num_queries: int = 100  # 最大同時追蹤的物體數（Query 槽位數）
    ffn_dim: int = 1024  # Cross-Attention FFN 的隱藏層維度
    dropout: float = 0.1  # Attention dropout rate
    num_decoder_layers: int = 3  # Transformer Decoder 層數（愈深愈準，但愈慢）
    score_threshold: float = 0.3  # 低於此分數的 Query 視為無效（物體消失）
    yolo_feat_channels: int = 256  # YOLO Neck 輸出特徵圖的 channel 數
    use_self_attn: bool = True  # 是否開啟 Query 之間的 Self-Attention (Option D)


# ---------------------------------------------------------------------------
# 1. YOLO Feature Extractor (Backbone + Neck Stub)
#    實際部署時替換為真實的 YOLO Neck 輸出
# ---------------------------------------------------------------------------
class YOLOFeatureExtractor(nn.Module):
    """
    從 YOLO Backbone/Neck 提取多尺度特徵圖。
    目前為 Stub：輸出一個與真實 YOLO Neck 格式相符的 dummy feature map。
    最終部署時，此模組的 forward() 會接收已預先提取的特徵圖，或直接掛接真實模型。
    """

    def __init__(self, cfg: TemporalYOLOConfig):
        super().__init__()
        # 用三個卷積層模擬 YOLO Neck 的多尺度特徵融合輸出
        # 真實的 YOLO Neck 接在 P3/P4/P5 之後
        self.proj_p3 = nn.Conv2d(256, cfg.yolo_feat_channels, kernel_size=1)
        self.proj_p4 = nn.Conv2d(256, cfg.yolo_feat_channels, kernel_size=1)
        self.proj_p5 = nn.Conv2d(256, cfg.yolo_feat_channels, kernel_size=1)

        # Stub CNN backbone（真實版本替換為 YOLO26s/m 的前向傳播）
        self._stub_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # /8  → P3
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 原始 RGB 影像

        Returns:
            feat: (B, C, H/8, W/8) 最終被壓扁為序列的特徵圖（Transformer 的 Key/Value）
        """
        feat = self._stub_backbone(x)  # (B, 256, H/8, W/8)
        return feat  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 2. Sinusoidal 2D Positional Encoding（給特徵圖的位置編碼）
# ---------------------------------------------------------------------------
class SinusoidalPositionEncoding2D(nn.Module):
    """
    對特徵圖的 H×W 空間位置加上 Sinusoidal 位置編碼。
    讓 Cross-Attention 知道「每個 token 在空間上的位置」。
    """

    def __init__(self, embed_dim: int, temperature: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            pos_enc: (B, C, H, W) 位置編碼，加到特徵圖後再展平
        """
        B, C, H, W = x.shape
        device = x.device
        half = C // 2

        y_pos = torch.arange(H, device=device).float() / H
        x_pos = torch.arange(W, device=device).float() / W

        dim = torch.arange(0, half, device=device).float()
        dim_factor = self.temperature ** (2 * (dim // 2) / half)

        pe_y = torch.zeros(H, half, device=device)
        pe_y[:, 0::2] = torch.sin(y_pos.unsqueeze(1) / dim_factor[0::2])
        pe_y[:, 1::2] = torch.cos(y_pos.unsqueeze(1) / dim_factor[1::2])

        pe_x = torch.zeros(W, half, device=device)
        pe_x[:, 0::2] = torch.sin(x_pos.unsqueeze(1) / dim_factor[0::2])
        pe_x[:, 1::2] = torch.cos(x_pos.unsqueeze(1) / dim_factor[1::2])

        # 組合 (H, W, C)
        pe_y_expand = pe_y.unsqueeze(1).expand(H, W, half)
        pe_x_expand = pe_x.unsqueeze(0).expand(H, W, half)
        pos = torch.cat([pe_y_expand, pe_x_expand], dim=-1)  # (H, W, C)
        pos = pos.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return pos.expand(B, C, H, W)


# ---------------------------------------------------------------------------
# 3. Track Query Cross-Attention Decoder Layer
# ---------------------------------------------------------------------------
class TrackQueryDecoderLayer(nn.Module):
    """
    單層 Transformer Decoder，包含：
    1. Self-Attention：Track Queries 之間互相溝通（避免多個 Query 追同一個人）
    2. Cross-Attention：Track Queries 去特徵圖（Key=Value）上尋找對應物體
    3. FFN：非線性特徵轉換
    """

    def __init__(self, cfg: TemporalYOLOConfig):
        super().__init__()
        D = cfg.embed_dim

        # Self-Attention (Option D)
        self.self_attn: nn.MultiheadAttention | None
        if cfg.use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                D, cfg.num_heads, cfg.dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(D)
            self.drop1 = nn.Dropout(cfg.dropout)
        else:
            self.self_attn = None

        # Cross-Attention（Query 去特徵圖上找對應物體）
        self.cross_attn = nn.MultiheadAttention(
            D, cfg.num_heads, cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(D)
        self.drop2 = nn.Dropout(cfg.dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(D, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, D),
        )
        self.norm3 = nn.LayerNorm(D)
        self.drop3 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        queries: torch.Tensor,  # (B, N_q, D) Track Queries
        feat_seq: torch.Tensor,  # (B, H*W, D) 特徵圖展平為序列
        query_pos: torch.Tensor,  # (B, N_q, D) Query 位置編碼
        feat_pos: torch.Tensor,  # (B, H*W, D) 特徵圖位置編碼
    ) -> torch.Tensor:
        q = queries + query_pos
        k = feat_seq + feat_pos

        # 1. Self-Attention (Option D)
        if hasattr(self, "self_attn") and self.self_attn is not None:
            sa, _ = self.self_attn(q, q, q)
            queries = self.norm1(queries + self.drop1(sa))

        # 2. Cross-Attention (Query 從特徵圖抓資訊)
        ca, _ = self.cross_attn(queries + query_pos, k, feat_seq)
        queries = self.norm2(queries + self.drop2(ca))

        # 3. FFN
        queries = self.norm3(queries + self.drop3(self.ffn(queries)))
        return queries


# ---------------------------------------------------------------------------
# 4. Track Query Decoder（多層 Decoder 堆疊）
# ---------------------------------------------------------------------------
class TrackQueryDecoder(nn.Module):
    """
    把多層 TrackQueryDecoderLayer 堆疊起來，
    並在最後接上 Detection Head（輸出 box + score）。
    """

    def __init__(self, cfg: TemporalYOLOConfig):
        super().__init__()
        # 特徵圖降維到 embed_dim
        self.feat_proj = nn.Linear(cfg.yolo_feat_channels, cfg.embed_dim)
        self.feat_pos_enc = SinusoidalPositionEncoding2D(cfg.embed_dim)

        # Query 位置編碼（可學習，每個 Query slot 都有自己的位置先驗）
        self.query_pos_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)

        self.layers = nn.ModuleList(
            [TrackQueryDecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)]
        )

        # Detection Head
        self.box_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, 4),  # [cx, cy, w, h]
        )
        self.score_head = nn.Linear(cfg.embed_dim, 1)  # 物體存在機率

    def forward(
        self,
        queries: torch.Tensor,  # (B, N_q, D) 前一幀的 Track Queries
        feat: torch.Tensor,  # (B, C, H, W) YOLO Neck 特徵圖
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = feat.shape
        N_q = queries.shape[1]

        # 特徵圖位置編碼 + 展平為序列
        pos_2d = self.feat_pos_enc(feat)  # (B, C, H, W)
        feat_seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        feat_seq = self.feat_proj(feat_seq)  # (B, H*W, D)
        feat_pos = pos_2d.flatten(2).transpose(1, 2)  # (B, H*W, C)
        feat_pos = self.feat_proj(feat_pos)  # (B, H*W, D)

        # Query 位置編碼
        slots = torch.arange(N_q, device=feat.device)
        query_pos = self.query_pos_embed(slots)  # (N_q, D)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)  # (B, N_q, D)

        # 逐層 Decoder
        for layer in self.layers:
            queries = layer(queries, feat_seq, query_pos, feat_pos)

        # Detection Head
        # boxes: sigmoid → 相對座標 [0, 1]
        # scores: raw logits → 由 BCEWithLogitsLoss 處理（推論時再 sigmoid）
        boxes = torch.sigmoid(self.box_head(queries))  # (B, N_q, 4)
        scores = self.score_head(queries).squeeze(-1)  # (B, N_q) raw logits

        return boxes, scores, queries  # 更新後的 queries 傳給下一幀


# ---------------------------------------------------------------------------
# 5. Track Query Lifecycle Manager
# ---------------------------------------------------------------------------
class TrackQueryLifecycle:
    """
    管理 Track Queries 的「出生」與「死亡」，維護 Query ID 對照表。
    - 分數 < threshold 的 Query 視為「物體消失」，該槽位被重設為 newborn embedding。
    - 新物體出現時，空槽位會被填入新的 Query。
    """

    def __init__(self, cfg: TemporalYOLOConfig, newborn_embed: nn.Embedding):
        self.cfg = cfg
        self.newborn_embed = newborn_embed
        self._query_ids: list[list[int]] = []
        self._next_id: list[int] = []

    def reset(self) -> None:
        self._query_ids = []
        self._next_id = []

    def update(
        self,
        queries: torch.Tensor,  # (B, N_q, D)
        scores: torch.Tensor,  # (B, N_q)
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """
        根據分數決定哪些 Query 存活（繼續追蹤），哪些消失（槽位重設）。支援 Batch 處理。
        Returns:
            queries_updated: (B, N_q, D)
            query_ids: list[list[int]] — 每個 Query 的追蹤 ID（-1 代表空槽 / 剛初始化）
        """
        B, N_q, D = queries.shape
        sc = torch.sigmoid(scores)  # (B, N_q)

        alive_mask = sc >= self.cfg.score_threshold  # (B, N_q)

        queries_np = queries.clone()
        if not alive_mask.all():
            slots = torch.arange(N_q, device=queries.device).unsqueeze(0).expand(B, -1)
            fresh = self.newborn_embed(slots)  # (B, N_q, D)
            # 使用 torch.where 取代 advanced indexing (dead_mask)，避免隱式 CUDA sync (計算元素個數)
            queries_np = torch.where(
                alive_mask.unsqueeze(-1), queries_np, fresh.detach()
            )

        # 動態初始化 Batch 狀態（若是第一幀）
        if len(self._query_ids) != B:
            self._query_ids = [[-1] * N_q for _ in range(B)]
            self._next_id = [1] * B

        # 避免在 GPU Tensor 上逐點存取造成巨大的 PCIe 同步延遲 (CUDA Sync)
        alive_mask_cpu = alive_mask.cpu().tolist()

        batch_ids = []
        for b in range(B):
            for i in range(N_q):
                is_alive = alive_mask_cpu[b][i]
                if not is_alive:
                    self._query_ids[b][i] = -1
                elif is_alive and self._query_ids[b][i] == -1:
                    self._query_ids[b][i] = self._next_id[b]
                    self._next_id[b] += 1
            batch_ids.append(list(self._query_ids[b]))

        return queries_np, batch_ids


# ---------------------------------------------------------------------------
# 6. TemporalYOLOHybrid — 主模型
# ---------------------------------------------------------------------------
class TemporalYOLOHybrid(nn.Module):
    """
    YOLO + Track Queries 混合追蹤模型。

    Forward 簽名：
        model(frame, prev_queries=None) -> dict
    """

    def __init__(self, cfg: TemporalYOLOConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = TemporalYOLOConfig()
        self.cfg = cfg

        self.backbone = YOLOFeatureExtractor(cfg)
        self.decoder = TrackQueryDecoder(cfg)

        # Newborn Query embedding（對應空槽位的初始 embedding）
        self.newborn_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)
        nn.init.normal_(self.newborn_embed.weight, std=0.02)

        # Lifecycle manager（非 nn.Module，純 Python 狀態機）
        self.__lifecycle: TrackQueryLifecycle | None = None

    def _init_lifecycle(self) -> None:
        self.__lifecycle = TrackQueryLifecycle(self.cfg, self.newborn_embed)

    def reset_sequence(self) -> None:
        """在新的影片序列開始時呼叫，重置 Query 狀態與 ID 計數器。"""
        if self.__lifecycle is not None:
            self.__lifecycle.reset()
        self.__lifecycle = None  # 強制重建

    def _make_init_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """第一幀時初始化所有 Query 為 newborn embedding。"""
        slots = torch.arange(self.cfg.num_queries, device=device)
        init = self.newborn_embed(slots)  # (N_q, D)
        return init.unsqueeze(0).expand(batch_size, -1, -1).clone()  # type: ignore[no-any-return]

    def forward(
        self,
        frame: torch.Tensor,  # (B, 3, H, W)
        prev_queries: torch.Tensor | None = None,  # (B, N_q, D) or None
    ) -> dict[str, torch.Tensor | list[int]]:
        B, _, H, W = frame.shape

        # 確保 Lifecycle manager 已初始化
        if self.__lifecycle is None:
            self._init_lifecycle()
        lc: TrackQueryLifecycle = self.__lifecycle  # type: ignore[assignment]

        # 第一幀或手動重置後初始化 Queries
        if prev_queries is None:
            prev_queries = self._make_init_queries(B, frame.device)

        # --- YOLO Backbone ---
        feat = self.backbone(frame)  # (B, C, H/8, W/8)

        # --- Transformer Decoder（Track Queries × YOLO 特徵圖）---
        boxes, scores, updated_queries = self.decoder(prev_queries, feat)

        # --- Lifecycle 管理（Query ID 維護）---
        updated_queries, query_ids = lc.update(updated_queries, scores)

        return {
            "boxes": boxes,  # (B, N_q, 4) [cx, cy, w, h]
            "scores": scores,  # (B, N_q)
            "updated_queries": updated_queries,  # (B, N_q, D)
            "query_ids": query_ids,  # list[int], -1=空槽
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_temporal_yolo_model(
    cfg: TemporalYOLOConfig | None = None,
    weights_path: str = "",
    device: torch.device | str = "cuda",
) -> TemporalYOLOHybrid:
    """
    建立並（可選地）載入預訓練權重的 TemporalYOLOHybrid 模型。
    """
    model = TemporalYOLOHybrid(cfg)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.to(device)


# 向下相容舊名稱
def load_temporal_yolo_model(
    model_path: str = "",
    device: torch.device | str = "cuda",
) -> TemporalYOLOHybrid:
    return build_temporal_yolo_model(weights_path=model_path, device=device)
