# Temporal YOLO — 架構對比

> **[結案 2026-05-19 — NO-GO]** Option D 實作完成，gate ablation 確認 gate-on vs gate-off ∆ <0.2pp。
> 保留作設計參考。

## YOLO26s 基礎結構回顧

```
Input (B, 3, 640, 640)
  │
  ├─ model.0~10   Backbone：Conv + C3k2 + SPPF + C2PSA
  │                  → P5_raw (B, 512, 20, 20)  stride/32
  │
  ├─ model.11~16  Neck FPN Top-Down：Upsample + Concat + C3k2
  │                  → P3 (B, 128, 80, 80)  stride/8
  │
  ├─ model.17~19  Neck Bottom-Up：Conv + Concat + C3k2
  │                  → P4 (B, 256, 40, 40)  stride/16
  │
  ├─ model.20~22  Neck Bottom-Up：Conv + Concat + C3k2
  │                  → P5 (B, 512, 20, 20)  stride/32
  │
  └─ model.23     Detect Head → (B, 300, 6)  end2end
```

---

## Option B：凍結 Backbone + Cross-Attention Decoder（已實作）

```
Frame_t ──→ [YOLO26s frozen] ──→ P5 (B, 512, 20, 20)
                                       │
                                  Linear proj → (B, 400, 256)
                                       │
Track Queries_{t-1} (B, 100, 256) ──→ Cross-Attention (3 layers)
                                       │
                              boxes_t + scores_t + Queries_t
```

**優點**：訓練快，decoder 容易收斂。  
**缺點**：backbone 提特徵時完全不知道追蹤狀態；P5 解析度低（20×20），小目標損失大。

---

## Option C：聯合訓練（已實作）

```
Frame_t ──→ [YOLO26s trainable]
              │
              ├─ P3 (B, 128, 80, 80)  ┐
              ├─ P4 (B, 256, 40, 40)  │── FPNSequenceProjection
              └─ P5 (B, 512, 20, 20)  ┘      → (B, N_tokens, 256)
                                                     │
Track Queries_{t-1} (B, 100, 256) ────────→ Cross-Attention (3 layers)
                                                     │
                                            boxes_t + Queries_t
```

**差異**：
- YOLO backbone 全部可訓練（backbone lr=1e-5，decoder lr=1e-4）
- 可選 P5-only（400 tokens）或 P3+P4+P5 多尺度（1200 tokens）
- 梯度回流到 backbone Conv/BN 層

**仍然的限制**：backbone 提特徵時仍是「單幀獨立」的，
追蹤狀態只透過 decoder 的梯度間接影響 backbone 的表示學習。

**Matcher**：`loss.py` 使用 `AuctionMatcher`（移植自 `auction.hpp`，O(N²)，無 scipy）。

---

## Option D：Track-Conditioned YOLO Neck（已完成，NO-GO）

Gate 輸入來自**外部 ByteTrack**，而非 Track Queries——ByteTrack 已是穩定系統，
從 day 1 就能提供乾淨的空間先驗，無 curriculum 問題。

```
Frame_{t-1} ──→ YOLO26s ──→ ByteTrack
                                  │
                     TrackResult (boxes, scores, det_idx)
                     TrackStateSnapshot (Kalman state[4:6] = vx, vy)
                     TrackCandidateSnapshot (tentative tracks)
                                  │
                     TrackerGateInput.from_tracker_results(...)
                                  │
                         ┌────────┴────────┐
                         │  TrackSpatialGate│
                         │  1. vx,vy 預測下一幀位置
                         │  2. Gaussian heatmap per scale
                         │  3. gate = 1 + alpha * heatmap
                         └────────┬────────┘
                                  │
Frame_t ──→ YOLO26s Backbone & Neck
                   │
                   ▼
           P3_raw / P4_raw / P5_raw
                   │
                   ▼
           ┌────────┴────────┐
           │ TrackSpatialGate │ ← gate_input
           └────────┬────────┘
                    │
           P3_gated / P4_gated / P5_gated
                    │
                    ▼
           FPNSequenceProjection
                    │
                    ▼
           Track Query Decoder (Cross-Attn)
                    │
                    ▼
           boxes_t + scores_t + Queries_t
```

**核心機制**：
- `alpha` 初始化為 0 → 訓練初始 gate=1，不破壞 pretrained YOLO
- det_idx=-1（遮擋中）→ sigma×1.5 + 強度×0.5（提示而非強調）
- Velocity prediction：heatmap 中心 = `(cx+vx, cy+vy)`，比歷史位置更準

詳見 [track-conditioned-design.md](track-conditioned-design.md)。

---

## Option E：GatedYOLODetector + Fine-Tune（當前有效 Baseline）

跳過 cross-attention decoder，直接在 gate 架構上使用標準 YOLO Detect head，
並針對 MOT17 行人資料 fine-tune backbone。**目前已驗證有效的方向。**

```
Frame_t → YOLO26s Backbone (layers 0~10)
               │
          TrackSpatialGate (gate_input 可選)
               │
          P3_gated / P4_gated / P5_gated
               │
          標準 YOLO Detect Head  → (B, 300, 6)
               │
          GPUByteTracker (外部，不變)
```

**實驗結果（`runs/gated_det_v1/best.ckpt`，gate-off 推論）：**

| 指標 | 舊 baseline（yolo26s） | Option E | ∆ |
|------|----------------------|----------|---|
| IDF1 | 52.0% | **57.2%** | +5.2pp |
| MOTA | 41.6% | **52.6%** | +11.0pp |
| FP | 14,563 | **3,233** | **-78%** |
| IDs | 475 | 489 | +14 |
| Rcll | 55.0% | 56.1% | +1.1pp |

**關鍵發現：**
- 改善主要來自 **fine-tune**（FP -78%），而非 gate
- Gate 貢獻 ∆ <0.2pp（已確認，default=off）
- Gate 架構保留於程式碼，但 FPN 空間先驗注入的增益仍待進一步驗證

---

## 五個選項比較

| 面向 | 基準（純 YOLO） | Option B | Option C | Option D | **Option E** |
|------|----------------|----------|----------|----------|--------------|
| Backbone 可訓練 | — | 否 | 是 | 是 | **是（fine-tune）** |
| 追蹤狀態影響特徵提取 | — | 否 | 否（間接） | 是（直接注入）| 是（gate，效果待確認） |
| Decoder 架構 | — | Cross-Attn | Cross-Attn | Cross-Attn | **無（標準 head）** |
| 特徵尺度 | P3/P4/P5 | P5 only | P3+P4+P5 | P3+P4+P5 | **P3+P4+P5** |
| 訓練 Matcher | — | Auction | Auction | 標準 det loss | **標準 det loss** |
| 訓練穩定性 | — | 中 | 中 | 高 | **高** |
| 實測 IDF1 | 52.0% | — | — | 31.7%（NO-GO）| **57.2%** |
| 狀態 | baseline | 結案 | 結案 | NO-GO | **✅ 當前 baseline** |

## 關鍵實作決策紀錄

| 決策 | 選擇 | 理由 |
|------|------|------|
| Option D gate 輸入 | ByteTrack output | Track Queries 訓練初期不可靠，會污染梯度 |
| Kalman 速度 vs EMA MotionModel | Kalman `state[4:6]` | Train/inference 一致；EMA 僅在 pure-Python eval 路徑 |
| Assignment solver | AuctionMatcher（`loss.py`）| 移植自 `auction.hpp`，無 scipy，O(N²)，與 tracker 一致 |
| Gate 注入位置 | FPN 輸出後（P3/P4/P5）| 不動 YOLO 結構，從 Option C checkpoint 熱啟動無縫 |
| Alpha 初始化 | 0 | 確保訓練初始行為與無 gate 完全相同 |
