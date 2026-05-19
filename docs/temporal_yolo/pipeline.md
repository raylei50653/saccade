# Temporal YOLO — Pipeline & Dataflow

## 資料流總覽

```
                     ┌───────────────────────────────────┐
                     │          Option C (Joint)          │
                     │                                    │
  Frame_t ──────────►│ YOLOFeaturePyramid                 │
  (B,3,640,640)      │   P3 (B,128,80,80)                 │
                     │   P4 (B,256,40,40)                 │
  Queries_{t-1} ────►│   P5 (B,512,20,20)                 │──► boxes_t  (B,N_q,4)
  (B,N_q,256)        │     → FPNSequenceProjection        │    scores_t (B,N_q)
                     │       → TrackQueryDecoderLayer ×3  │    Queries_t (B,N_q,256)
                     └───────────────────────────────────┘

                     ┌───────────────────────────────────┐
                     │          Option D (Conditioned)    │
                     │                                    │
  GateInput_{t-1} ──►│ TrackSpatialGate                   │
  (TrackerGateInput)  │   gate_p3 (B,1,80,80)             │
                     │   gate_p4 (B,1,40,40)             │
  Frame_t ──────────►│   gate_p5 (B,1,20,20)             │──► boxes_t  (B,300,6)
  (B,3,640,640)      │     ×FPN → gated_p3/p4/p5         │    (標準 YOLO Detect)
                     │       → YOLO Detect Head           │
                     └───────────────────────────────────┘
```

---

## Option C：逐幀推論資料流

```
Frame_t (B, 3, 640, 640)
  │
  ▼
[1] YOLOFeaturePyramid (yolo_joint.py)
  ├─ YOLO26s Backbone  layers 0~10   → P5_raw  (B, 512, 20, 20)
  ├─ YOLO26s Neck FPN  layers 11~16  → P3       (B, 128, 80, 80)
  ├─ YOLO26s Neck PAN  layers 17~19  → P4       (B, 256, 40, 40)
  └─ YOLO26s Neck PAN  layers 20~22  → P5       (B, 512, 20, 20)
  │
  ▼
[2] FPNSequenceProjection (yolo_joint.py)
  ├─ P3: Linear(128→256) + sinusoidal pos-enc → (B, 6400, 256)   [僅 scales 含 p3]
  ├─ P4: Linear(256→256) + sinusoidal pos-enc → (B, 1600, 256)   [僅 scales 含 p4]
  └─ P5: Linear(512→256) + sinusoidal pos-enc → (B,  400, 256)
  Concat → (B, S, 256)   S=400|1200|6400（取決於 scales 設定）
  │
  ▼
[3] TrackQueryDecoderLayer × 3  (model.py)
  Input:  Queries_{t-1} (B, N_q, 256)  ×  tokens (B, S, 256)
  Self-Attention (Queries 內部)
  Cross-Attention (Queries attend to FPN tokens)
  FFN
  → Queries_t (B, N_q, 256)
  │
  ▼
[4] Output Heads
  box_head:   Linear(256→4)   → pred_boxes  (B, N_q, 4)  [cx,cy,w,h] normalized
  score_head: Linear(256→1)   → pred_scores (B, N_q)     logits
  │
  ▼
[5] LifecycleManager（推論專用，yolo_joint.py）
  管理 N_q=100 個 query 的 active/dormant 狀態
  輸出：confirmed tracks（score > threshold）
```

### 跨幀狀態傳遞

```
t=0: Queries_0 = nn.Embedding 初始化  (B, N_q, 256)
t=1: model(frame_1, Queries_0) → Queries_1, boxes_1, scores_1
t=2: model(frame_2, Queries_1) → Queries_2, boxes_2, scores_2
...
推論時：queries 帶梯度（或不帶，視 eval/train）
訓練時：queries.detach() 傳下一幀（避免 TBPTT 無界梯度）
```

---

## Option C：訓練資料流

```
MOT17TemporalClip (dataset.py)
  __getitem__ → {
    'frames':   (T, 3, 640, 640)   # T 幀連續片段
    'gt_boxes': list[Tensor(N_gt_t, 4)]  # [x1,y1,x2,y2] 絕對像素
  }
  │
  DataLoader → batch = {
    'frames':   (B, T, 3, 640, 640)
    'gt_boxes': list[list[Tensor]]   # B × T
  }
  │
  ▼
  ──── for b in range(B): ────────────────────────────────────────
    prev_queries = model.init_queries()   # (1, N_q, 256)

    ──── for t in range(T): ──────────────────────────────────────
      frame_t = frames[b, t]              # (1, 3, 640, 640)
      │
      ▼
    TemporalYOLOJoint.forward(frame_t, prev_queries)
      → pred_boxes_t  (1, N_q, 4)
      → pred_scores_t (1, N_q)
      → Queries_t     (1, N_q, 256)
      │
      ▼
    AuctionMatcher.match(pred_boxes_t.detach(), pred_scores_t.detach(), gt_boxes_t)
      cost = w_bbox * L1 + w_giou * (1-GIoU) + w_score * (-score)
      → (query_indices, gt_indices)   # 最優二分匹配
      │
      ▼
    TemporalTrackingLoss (loss.py)
      loss_l1:  L1(pred_boxes[q_idx], gt_norm[g_idx])
      loss_giou: (1-GIoU)(pred_boxes[q_idx], gt_norm[g_idx])
      loss_bce: BCE(pred_scores, target)  # matched=1, unmatched=0
      │
      prev_queries = Queries_t.detach()   # ← detach：切斷跨幀梯度
    ──────────────────────────────────────────────────────────────
    loss_total.backward()
    optimizer.step()
  ────────────────────────────────────────────────────────────────

參數組（parameter_groups）：
  backbone params  lr = 1e-5
  decoder  params  lr = 1e-4
```

---

## Option D：逐幀推論資料流

```
Frame_{t-1}
  │
  ▼
[0] YOLO26s + ByteTrack（外部，已在主 pipeline 執行）
  tracker.update(detections) →
    TrackResult × N          (boxes, scores, det_idx)
    TrackStateSnapshot × N   (Kalman state[4:6] = vx, vy)
    TrackCandidateSnapshot × M (候選軌跡)
  │
  ▼
[1] TrackerGateInput.from_tracker_results(...)  (yolo_conditioned.py)
  confirmed_boxes:  (N, 4)  [x1,y1,x2,y2] 絕對像素（score > min_score，age ≥ min_age）
  confirmed_scores: (N,)
  velocities:       (N, 2)  [vx,vy] px/frame（來自 Kalman state[4:6]）
  tentative_boxes:  (M, 4)  候選軌跡
  tentative_ratios: (M,)    hit_streak / required_confirm_streak
  img_hw:           (640, 640)

Frame_t (B, 3, 640, 640)
  │
  ▼
[2] YOLO26s Backbone（layers 0~10）
  → P5_raw (B, 512, 20, 20)
  │
  │  TrackerGateInput（來自上方 [1]）
  │
  ▼
[3] TrackSpatialGate（yolo_conditioned.py）
  ┌─ GaussianHeatmapRenderer（per scale）：
  │    cx_pred = cx + vx          ← velocity prediction（dt=1 frame）
  │    cy_pred = cy + vy
  │    sigma = box_size * 0.5     ← 確認軌跡
  │          * 0.5 * 1.5          ← det_idx=-1（遮擋，較寬）
  │    intensity = sigmoid(score) * 1.0    (確認)
  │              * sigmoid(score) * 0.5    (det_idx=-1)
  │              * tentative_ratio * 0.3   (候選)
  │    heatmap = max(all Gaussians)  → (B, 1, H_s, W_s)
  │
  └─ gate_s = 1 + alpha_s * heatmap_s
       alpha_p3, alpha_p4, alpha_p5：per-scale 可學習純量，初始化=0
  │
  ▼  gate_p3 (B,1,80,80)  gate_p4 (B,1,40,40)  gate_p5 (B,1,20,20)
  │
  ▼
[4] YOLO Neck（layers 11~22）接受 gated 特徵
  P3_input = P3_feat × gate_p3
  P4_input = P4_feat × gate_p4
  P5_input = P5_feat × gate_p5
  → gated_p3/p4/p5

[5] 標準 YOLO Detect Head（layer 23）
  → (B, 300, 6)  [x1,y1,x2,y2, score, class_id]  end2end
```

### 跨幀狀態傳遞（Option D）

```
t=0: 無 gate（gate_input=None → gate=1，等效標準 YOLO）
t=1: gate_input = TrackerGateInput from frame_0 tracker output
t=2: gate_input = TrackerGateInput from frame_1 tracker output
...
gate 輸入來自「上一幀的 tracker 輸出」（已知穩定先驗，非自身輸出）
無 TBPTT 問題（gate 輸入與 YOLO 輸出無梯度循環）
```

---

## Option D：訓練資料流

### 前置：快取生成（一次性）

```
[v1 GT Oracle 快取]

MOT17 gt.txt
  │
  ▼
cache_gt_tracks.py
  per-frame: {'boxes': Tensor(N,4), 'scores': ones(N)}
  存至: datasets/mot17_gt_cache/{seq}/frame_{t:06d}.pt
  用途: Phase 1 驗證 gate 有效性（oracle 上界）

[v2 ByteTrack 快取（推薦）]

MOT17 frames + yolo26s.pt
  │
  ▼
cache_tracker_states.py
  per-frame: {
    'boxes':        Tensor(N,4),
    'scores':       Tensor(N,),
    'velocities':   Tensor(N,2),   ← Kalman state[4:6]
    'det_idx':      Tensor(N,),    ← -1 = 遮擋中
    'tentative_boxes':   Tensor(M,4),
    'tentative_ratios':  Tensor(M,),
  }
  存至: datasets/mot17_track_cache/{seq}/frame_{t:06d}.pt
  用途: Phase 2 真實訓練；與推論行為完全一致
```

### 兩階段訓練迴圈

```
MOT17TemporalClip（track_cache_dir 已指定）
  __getitem__ → {
    'frames':      (T, 3, 640, 640)
    'gt_boxes':    list[Tensor(N_gt_t, 4)]
    'gate_inputs': list[TrackerGateInput | None]   # 長度 T，gate_inputs[0]=None（第一幀無先驗）
  }
  │
  ▼
  ──── Phase 1（凍結 YOLO backbone + Detect Head，只訓 gate.alpha）────
  for t in range(1, T):
    gate_input = gate_inputs[t-1]
    pred = model(frame_t, gate_input)    # TemporalYOLOConditioned
      └─ gate = TrackSpatialGate(gate_input, ...)
         backbone frozen → gated feats → Detect Head frozen
    loss = yolo_detection_loss(pred, gt_boxes_t)   # 標準 L_box + L_score
    loss.backward()
    optimizer_gate.step()   # 只更新 alpha_p3, alpha_p4, alpha_p5

  驗收：alpha_p5 > 0，gate heatmap 視覺對齊目標位置

  ──── Phase 2（全部解凍，差分 LR）───────────────────────────────────
  optimizer = AdamW([
    {'params': backbone.params, 'lr': 1e-5},
    {'params': gate.params,     'lr': 5e-5},
    {'params': detect.params,   'lr': 1e-4},
  ])
  切換至 ByteTrack 快取（v2）
  for t in range(1, T):
    gate_input = load_from_cache(track_cache_dir, seq, t-1)
    pred = model(frame_t, gate_input)
    loss = yolo_detection_loss(pred, gt_boxes_t)
    loss.backward()
    optimizer.step()
```

---

## 模組對應一覽

| Stage | 模組 | 檔案 | 輸入 | 輸出 |
|-------|------|------|------|------|
| YOLO Backbone | `YOLOFeaturePyramid` | `yolo_joint.py` | `(B,3,H,W)` | `{p3,p4,p5}` |
| FPN → Tokens | `FPNSequenceProjection` | `yolo_joint.py` | `{p3,p4,p5}` | `(B,S,256)` |
| Cross-Attention | `TrackQueryDecoderLayer×3` | `model.py` | queries + tokens | queries' |
| Tracker Data | `TrackerGateInput` | `yolo_conditioned.py` | TrackResult/Snapshot | dataclass |
| Gaussian Gate | `GaussianHeatmapRenderer` | `yolo_conditioned.py` | TrackerGateInput | `(B,1,H_s,W_s)` |
| Learnable Gate | `TrackSpatialGate` | `yolo_conditioned.py` | heatmap + alpha | gate per scale |
| Detection Loss | YOLO native | `train_conditioned.py` | pred, gt | L_box + L_score |
| Tracking Loss | `TemporalTrackingLoss` | `loss.py` | pred, gt | L1+GIoU+BCE |
| Matching | `AuctionMatcher` | `loss.py` | pred_boxes, gt_boxes | (q_idx, g_idx) |

---

## 關鍵路徑差異對比

| | Option C | Option D |
|--|----------|----------|
| 跨幀狀態 | Track Queries (B,N_q,256) | TrackerGateInput（外部 ByteTrack） |
| FPN 使用方式 | 投影為 tokens → Cross-Attention | 直接 × gate（乘法調制） |
| backbone 訓練 | 是（lr=1e-5） | 是（Phase 2 lr=1e-5） |
| 追蹤狀態影響特徵 | 間接（decoder 梯度） | 直接（gate 乘到 FPN） |
| 損失函數 | AuctionMatcher + L1/GIoU/BCE | 標準 YOLO detection loss |
| 初始化 | Queries 隨機 → 需 warm-up | alpha=0 → 等效純 YOLO → 無 warm-up |
| train/inference 一致性 | 需 curriculum（Queries 初期不穩定） | 天然一致（ByteTrack 快取=推論路徑） |
