# Dual-Resolution Mamba Head：原始解析度細節分支研究計畫

日期：2026-06-11  
狀態：Phase 1 scaffold 已實作 / 尚未完成 oracle 與 production 驗證
基線：Option F v14，640 input，Cross-Scan + N=16，共享 A，純空間 Mamba head  
訓練決策：由 v14 checkpoint warm-start，加入原始解析度 detail branch 後重訓完整 Mamba head

### 目前實作範圍

已完成：

- stride-4 shallow detail encoder。
- P3 footprint `3x3` dense sampling。
- cls/reg 獨立 zero-init residual，初始化輸出等價於 v14。
- `global` B1-L 與 fixed high-resolution bucket B1-H/C1 training mode。
- aspect-preserving detail resize、右下 padding 與 `valid_detail_region` 對齊。
- shape 相符時由 frozen YOLO 第一層 Conv+BN warm-start detail encoder。
- shallow detail 支援 Python eager 與 Python whole-graph detector。
- native-P3 oracle 推論限定 Python eager。
- B2 frozen high-resolution YOLO P3 semantic oracle training path。
- TaskAlignedAssigner conflict/positive-anchor 離線診斷工具。

尚未完成：

- B2 與 matched controls 的完整 MOT17 訓練/評估結果。
- 全 MOT17 TaskAlignedAssigner 容量統計結果。
- detail token distillation pretraining。
- C++ detector、TorchScript export 與 multi-stream batching 的雙輸入介面。
- TensorRT detail encoder 與 production latency 驗證。

---

## 1. 核心問題

目前 v14 的完整視覺路徑從 `640x640` 影像開始：

```text
原始影像
  → resize/stretch 到 640x640
  → YOLO backbone
  → P3/P4/P5
  → Mamba head
```

對 MOT17 六條 `1920x1080` sequence，stretch 到 `640x640` 時：

- 水平方向縮小 3 倍。
- 垂直方向縮小約 1.69 倍。
- 同時引入非等比例形變。

原圖寬度只有 12-20 px 的遠距行人，進入 640 後可能只剩 4-7 px。經過 stride-8 P3 後，其有效訊號小於一個 feature cell。

因此真正的問題不是 Mamba context 不夠，而是：

> 小目標的辨識資訊在進入 backbone 前已被 resize 丟失，後續任何只依賴 640 image 的 P2/P3/P4/P5 都無法恢復不存在的資訊。

令：

```text
I640 = Resize(Iorig)
```

若兩個不同的原始 patch 經過 resize 後變成近似相同的 `I640` patch，任何只使用 `I640` 的模型 `f(I640)` 都無法再區分它們。增加 Mamba blocks、改用 640-P2 或回採樣 640-P3，都只是在處理同一份已受損訊號。

---

## 2. 對前一版方案的修正

### 2.1 640-P2 不是解法

YOLO layer 2 的 stride-4 P2 雖然比 P3 高解析，但它仍然來自 resize 後的 640 image：

```text
Iorig → Resize 640 → P2
```

它可能改善 backbone 內部的下採樣損失，但不能恢復 resize 前已消失的細節。因此：

- 640-P2 保留作 negative/control baseline。
- 不再作為主 source。
- 若 640-P2 有效，代表瓶頸主要在 backbone stride，不是原圖 resize。

### 2.2 原圖單點取樣也不夠

只把每個 P3 cell center 映射回原圖取一個 RGB/feature point，仍有兩個問題：

- 小目標未必落在 cell center。
- 640 context 已看不見目標時，Mamba 無法可靠預測 offset 去找它。

所以主方案不是「原圖單點 sampling」，而是：

> 將每個 640-P3 cell 對應的原圖區域整塊編碼，保留 cell 內的高解析度空間結構，再交給 Mamba head 判斷。

---

## 3. 建議架構

採用雙解析度、單一 detection head：

```text
                         ┌─────────────────────────────┐
原始影像 ── resize 640 ─→│  v14 global stream          │
                         │  P3/P4/P5 → Cross-Scan      │
                         └──────────────┬──────────────┘
                                        │ global context
原始影像 ────────────────────────────────┤
       │                                │
       ▼                                │
lightweight high-res detail encoder     │
       │                                │
       ▼                                │
native-resolution detail feature        │
       │                                │
       ▼                                │
P3-cell footprint mapping + patch token │
       └────────────────┬───────────────┘
                        ▼
             cls/reg-specific fusion
                        │
                        ▼
               原 v14 dense outputs
```

### 3.1 Global stream

完全保留 v14：

- 640 input。
- P3/P4/P5 backbone features。
- Cross-Scan Mamba。
- PixelShuffle upsampling。
- 原 cls/reg head。

Global stream 負責：

- 場景與人體語意。
- 長距空間 context。
- 大中型目標。
- 與現有 checkpoint、tracker、postprocess 的相容性。

### 3.2 Detail stream

Detail stream 必須在破壞性 resize 前讀取影像：

```text
Iorig → shallow high-resolution encoder → Fdetail
```

第一個 production-oriented 候選：

- 輸入：原始 frame，或保留 aspect ratio 的 high-resolution shape bucket。
- encoder：2-3 個輕量卷積/downsample blocks。
- output stride：4。
- output channels：32 或 64。
- 不跑完整 YOLO neck。
- 不在高解析 feature 上跑 Mamba。

Detail stream 只負責保留：

- 細輪廓。
- 人體局部形狀。
- cell 內多個小目標的空間分布。
- resize 後已不可逆消失的高頻訊號。

### 3.3 P3 cell footprint mapping

不能只映射 cell center，應映射完整 cell footprint。

目前 `preprocess: none` 的 Mamba 路徑對原圖做 stretch resize。對 P3 cell `(u, v)`：

```text
x0_640 = u * 8
x1_640 = (u + 1) * 8
y0_640 = v * 8
y1_640 = (v + 1) * 8
```

映射回原圖：

```text
x0_orig = x0_640 * Worig / 640
x1_orig = x1_640 * Worig / 640
y0_orig = y0_640 * Horig / 640
y1_orig = y1_640 * Horig / 640
```

對 `1920x1080` frame，一個 P3 cell 對應約：

```text
24 x 13.5 original pixels
```

這個區域足以包含 resize 到 640 後只剩數個像素的小行人細節。

若後續改用 letterbox，禁止重新推導近似比例；資料流必須顯式傳遞 preprocess transform：

```text
T_orig_to_global
T_global_to_orig
```

同一 transform 必須用於：

- GT box。
- P3 grid footprint。
- detail ROI。
- 最終 box 座標還原。

### 3.4 Patch token，而不是單點 feature

每個 P3 footprint 從 `Fdetail` 抽取 `2x2` 或 `3x3` patch：

```text
ROIAlign(Fdetail, footprint, output_size=3)
  → C x 3 x 3
  → local projection
  → d_detail token
```

P3 的 `80x80` footprints 對固定 input shape 是規則網格。部署時應預先建立 sampling grid，使用一次 batched sampling 產生所有 tokens，不應逐 cell 建立 6400 個動態 ROI。`ROIAlign` 在研究文件中代表取樣語意，不限定生產實作必須使用動態 ROI API。

不能一開始就 average pool 成單一值，否則又會丟失：

- 目標在 cell 內的位置。
- 垂直人體形狀。
- 同一 cell 內多個局部峰值。

建議第一版：

| 參數 | 值 |
|---|---:|
| target level | P3 |
| detail stride | 4 |
| detail channels | 32 |
| footprint output | `3x3` |
| token projection | `32x3x3 → 64` |
| fusion | cls/reg separate residual |
| residual scale | zero-init |

### 3.5 Fusion

前一版只建議進 regression branch，但在目前問題定義下不夠。若 640 已失去辨識能力，classification 必須能看到原圖 detail。

目前 v14 在每個 level 的實際 head interface 是：

```text
x_cat = concat([x_proj, x_up], channel)
cls = cls_head(x_cat)
reg = reg_head(x_cat)
```

因此 detail fusion 不能把 `x_up` 省略或改成另一套 head input。第一版只修改 P3 的 `x_proj` 分量，保留原本 `x_up` 與 channel layout：

```text
d = detail_token
g = mamba_global_context
x_proj = original_P3_projected_feature
x_up = original_P3_mamba_upsampled_feature

gate_cls = sigmoid(MLP_cls(g))
gate_reg = sigmoid(MLP_reg(g))

x_proj_cls = x_proj + gamma_cls * gate_cls * proj_cls(d)
x_proj_reg = x_proj + gamma_reg * gate_reg * proj_reg(d)

x_cls = concat([x_proj_cls, x_up], channel)
x_reg = concat([x_proj_reg, x_up], channel)

cls = cls_head_P3(x_cls)
reg = reg_head_P3(x_reg)
```

設計要求：

- `gamma_cls = gamma_reg = 0` 初始化，未訓練時等價於 v14。
- `proj_cls(d)` 與 `proj_reg(d)` 的輸出 channel 必須等於 P3 `x_proj` 的 `d_model=128`，不能直接使用 64-channel token。
- cls/reg 使用不同 projection，避免定位與辨識互相牽制。
- Mamba 只控制 detail evidence 的權重，不負責從低解析訊號猜測 detail 位置。
- P4/P5 第一輪完全不改。
- zero-init 可放在 projection 最後一層，或使用 staged `gamma`。若只令 scalar `gamma=0`，第一個 step 不會有梯度進入 projection/detail encoder，需在訓練設計中明確接受或避免。

### 3.6 必須保持 dense

不應先由 640 detections 選 top-k，再回原圖 crop：

- 640 完全漏掉的目標不會進入 top-k。
- 此方案只能 refine 已存在候選，不能補 recall。

P3 每個 grid cell 都必須取得 detail token，讓原本低分或不存在的候選能在 classification 前獲得原圖證據。

---

## 4. 第二個瓶頸：輸出網格容量

即使原圖 detail 被保留，P3 仍是 `80x80` dense grid。需要區分：

1. **資訊瓶頸**：一個 P3 cell 看不出是否有人。
2. **容量瓶頸**：同一 P3 cell 內有多個人，但輸出只能表達一個主要候選。

### 描述性統計

對每個 GT center 計算其 P3 cell：

```text
cell_x = floor(cx_640 / 8)
cell_y = floor(cy_640 / 8)
```

統計：

- 每幀有多少 P3 cells 含 2 個以上 GT centers。
- collision 是否集中在 `<50px` 或 `<100px` 目標。
- collision GT 在原圖中是否可分離。

這只能描述幾何密度，不能直接代表目前 head 無法分配正樣本。現有訓練使用 `v8DetectionLoss` 與 `TaskAlignedAssigner`，會同時考慮 P3/P4/P5 anchors、alignment top-k，並在多個 GT 競爭同一 anchor 時依 overlap 解衝突。小於 assigner `stride_val` 的 GT 也會在 candidate selection 階段被放大。因此「GT center 落在同一個 P3 cell」不是實際 output-capacity 的充分條件。

### 實際 assigner 診斷

必須用 baseline v14 predictions 與訓練時相同的 `TaskAlignedAssigner` 跑一次離線統計，至少記錄：

- 每個 GT 在衝突解決前後分配到的正 anchors 數量。
- 完全沒有正 anchor 的 GT 比例，依原圖尺寸與 resize-640 尺寸分層。
- 一個 anchor 同時被多個 GT 選中的比例。
- `select_highest_overlaps` 後被其他 GT 搶走 anchor 的數量與 GT 比例。
- 每個 GT 最終由 P3、P4、P5 各自負責的正 anchor 數量。
- 加入 detail feature 後，GT 已可分類但仍因 anchor conflict 無法保留正樣本的比例。

### 決策

- 若 center collision 高但 assigner 仍能為 GT 保留足夠正 anchors：先做 P3 detail fusion，不能僅因 center collision 新增 P2。
- 若大量 small GT 在 P3/P4/P5 都沒有正 anchor，或正 anchor 在 conflict resolution 後消失：後續需要新增 P2 detection level，不能只靠 P3 feature refinement。
- P2 決策以實際 assigner 統計與 detection error 為主，center collision 只作輔助解釋。

可選 P2 head：

```text
160x160 target grid
  + high-res detail tokens
  + upsampled P3 Mamba context
  → small-object-only cls/reg
```

P2 head 是第二階段架構，不與第一個 detail experiment 同時加入，否則無法區分資訊增益與輸出容量增益。

---

## 5. Source 架構選項

### O1：完整原圖 YOLO P3/P4/P5

```text
Iorig → full YOLO backbone/neck → original-resolution P3/P4/P5
```

用途：**oracle baseline**。

優點：

- 高解析度與 detection semantic 最完整。
- 最直接回答「原圖 feature 是否能補 v14」。

缺點：

- `1920x1080` 像素量是 `640x640` 的約 5.06 倍。
- 再加原本 640 stream，總 backbone compute 約為單一 640 的 6 倍量級。
- 原圖 feature maps 與顯存成本高。

因此不作 production 候選，只作訊號驗證。若 O1 都無法改善 small-object recall，整條雙解析度路線應停止。

完整 YOLO oracle 的輸入必須明確定義 padding 與座標轉換。`1920x1080` 的高度不能直接通過要求 stride-32 對齊的 backbone，第一版使用 aspect-preserving bottom padding：

```text
1920x1080 → pad to 1920x1088 → full YOLO P3/P4/P5
```

資料流必須保存：

```text
T_orig_to_detail
T_detail_to_orig
valid_detail_region
```

sampling grid 不得讀取 padding 區域。若 oracle 包含完整 Mamba head，而不只是 YOLO source backbone，輸入高寬還必須滿足目前 head 的 `÷128` 約束，例如 `1920x1080 → 1920x1152`；這與只取 YOLO P3 的 stride-32 約束不同。

### O2：Native shallow detail encoder

用途：**主要 production 候選**。

```text
Iorig → Conv s2 → lightweight block → Conv s2 → Fdetail(stride 4)
```

優點：

- 保留 resize 前細節。
- 比完整 YOLO 原圖 backbone 便宜。
- 可與 v14 global stream 並行。

缺點：

- shallow feature 語意弱，可能把背景紋理當成人。
- 原圖 shape 可變，CUDA graph 需按 shape 建立。

O2 不應從隨機初始化直接只靠 MOT17 訓練。第一版至少採一種初始化：

- 從 frozen YOLO stem/layer 0-2 複製可對應權重，再縮減 channels。
- 先讓 shallow encoder 蒸餾 O1/B2 native P3 oracle 的 P3-aligned detail tokens，再進入 detection joint training。

隨機初始化版本只作 ablation，不作主要 production 判斷依據。

### O3：Canonical high-resolution detail stream

將原圖保持 aspect ratio resize 到較高解析 shape bucket，例如：

```text
1920x1080 → 1280x720
640x480   → 640x480 或 1024x768
```

用途：**部署折衷候選**。

它不是完整原圖，但相較 640：

- 1920x1080 的水平方向只縮小 1.5 倍，而不是 3 倍。
- 保留 aspect ratio。
- shape 與顯存較容易控制。

需要與 native O2 比較，確認 1280 級 detail 是否已保留足夠辨識能力。

### O4：640-P2

用途：**negative control**。

若 O4 有效而 O2/O3 沒有額外收益，表示真正問題是 backbone stride，而不是 resize information loss。

O4 在實驗矩陣中對應 B1-L，必須與使用相同 frozen encoder/layer 的 B1-H 成對比較。不能直接用 640-P2 對 full native P3 的差值宣稱是 resolution gain。

---

## 6. 現有高解析實驗的解讀

Repo 已存在 1024 warm-start checkpoint 與 eval preset，但目前不能作為有效 accuracy baseline：

- `mamba_gt_vgt_1024_*` 的現有 eval 紀錄曾受座標雙重縮放影響，出現 `IDF1 0.9-4.2%`，不代表 1024 模型真實能力。
- 1024 TRT backbone 與 eager PyTorch feature cosine 僅約 `0.878`，而 640 約為 `1.0`，現有 1024 engine 不可信。
- 1024 Mamba head 必須在 1024 重訓，不能直接套用 640 v14。

因此 Phase 0 必須先建立**可信 1024 oracle**：

- eager PyTorch backbone。
- 正確座標還原。
- detector-only sanity check。
- 確認輸出範圍與 GT 尺度正確。
- 不使用現有有問題的 1024 TRT engine。

1024 full-model baseline 很重要：

- 若正確 1024 已大幅改善且成本可接受，直接部署 1024 可能比雙 stream 簡單。
- 雙 stream 只有在 accuracy/latency Pareto 優於 full 1024 時才有工程價值。

---

## 7. 實驗計畫

### Phase 0：確認是 resize bottleneck

在訓練新模型前完成：

1. 統計 GT box 在原圖與 640 的寬高分布。
2. 統計 `<4px`、`<8px`、`<16px` resized width/height 的 GT 數量。
3. 統計 P3 center collision，僅作描述性指標。
4. 用 baseline predictions 跑實際 `TaskAlignedAssigner`，統計正樣本、跨 level responsibility 與 conflict resolution。
5. 建立可信 eager-1024 v14 warm-start baseline。
6. 分 sequence 比較 1920x1080 與 MOT17-05 640x480：
   - 若收益只出現在 1920x1080 sequences，支持 resize bottleneck。
   - 若 MOT17-05 同樣改善，需考慮 backbone/grid bottleneck。

建議補一個原圖 vs 640 crop separability probe：

- 使用相同 GT/背景位置。
- 比較原圖 patch 與 resize-640 patch 的簡單 classifier 或 frozen feature separability。
- 目的不是建立 production classifier，而是直接驗證「資訊是否在 resize 時消失」。

### Phase 1：高解析 source oracle

固定 v14 global stream，先用 matched controls 隔離解析度、source layer 與 semantic depth。所有 detail 組必須使用相同 token extractor、fusion 位置、訓練 budget 與 augmentation：

| ID | Global | Detail source | Source weights/layer | Detail token | 目的 |
|---|---|---|---|---|---|
| B0 | v14 640 | 無 | - | 無 | 原始基線 |
| B0-R | v14 640 | 無 | - | 無 | 相同 budget 重訓控制 |
| B1-L | v14 640 | 640 view | frozen YOLO layers 0-2 / P2 | `3x3` | matched low-resolution control |
| B1-H | v14 640 | padded native/high-res view | 與 B1-L 相同 frozen YOLO layers 0-2 / P2 | `3x3` | 隔離 pre-resize resolution gain |
| B2 | v14 640 | padded native/high-res view | full frozen YOLO / P3 | `3x3` | high-res semantic oracle |
| B3 | full 1024 | 無 | - | 無 | single-stream high-res baseline |

判讀：

- B1-H > B1-L：在相同 encoder/layer 下，確認 resize 前資訊有價值。
- B1-H 與 B1-L 相同：不支持 resize information-loss 假說；B2 即使有效，也可能只是更深 semantic feature 的收益。
- B2 > B1-H：較深 high-resolution semantic feature 另有增益，不可把全部差值歸因於 resolution。
- B1-H 與 B2 都不優於各自控制：停止雙解析度方向。
- B3 與 B2 接近且 latency 更低：優先 full 1024。
- B2 明顯優於 B3：native detail 或雙 stream 有繼續研究價值。

目前可執行的 matched control：

```bash
# B1-L：相同 detail encoder，但只讀 640 global view
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/train/temporal_yolo/train_mamba_gt.py \
  --data-root datasets/MOT17 \
  --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt \
  --detail-source global \
  --run-dir runs/mamba_detail_b1_low

# B1-H/C1 scaffold：保比例 high-resolution detail bucket
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/train/temporal_yolo/train_mamba_gt.py \
  --data-root datasets/MOT17 \
  --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt \
  --detail-source high \
  --detail-height 768 \
  --detail-width 1280 \
  --run-dir runs/mamba_detail_b1_high

# B2：同一 high-resolution view，改用 frozen YOLO P3 semantic oracle
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/train/temporal_yolo/train_mamba_gt.py \
  --data-root datasets/MOT17 \
  --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt \
  --detail-source native-p3 \
  --detail-height 768 \
  --detail-width 1280 \
  --run-dir runs/mamba_detail_b2_native_p3
```

三組除 detail source/resolution 外，batch、epoch、LR、augmentation、cache 與 seed 必須一致。

### Assigner 容量診斷

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/tools/mamba_assigner_diagnostics.py \
  --data-root datasets/MOT17 \
  --mamba-ckpt runs/mamba_gt_vgt_mamba_v14/best.ckpt \
  --output report_data/mamba_assigner_diagnostics.json
```

工具直接呼叫訓練使用的 `TaskAlignedAssigner.get_pos_mask()` 與
`select_highest_overlaps()`，輸出：

- conflict 前後每個 GT 的平均 positive-anchor count。
- zero-positive GT rate。
- 因 conflict resolution 失去全部 positives 的 GT rate。
- conflict-anchor rate。
- P3/P4/P5 positive responsibility。
- resize-640 與原圖尺寸 bins。

### B2 初步結果：MOT17-05-SDP

Checkpoint：

```text
runs/mamba_detail_b2_native_p3/best.ckpt
epoch=30
best training loss=0.5976578762
detail source=native-p3
detail bucket=1280x768
```

完整 MOT17-05-SDP train sequence（837 frames），tracker 設定相同：

| Metric | v14 | B2 native-P3 | Delta |
|---|---:|---:|---:|
| IDF1 | 68.7 | 69.2 | +0.5 |
| MOTA | 65.4 | 66.6 | +1.2 |
| HOTA | 58.2 | 58.6 | +0.4 |
| DetA | 60.0 | 61.4 | +1.4 |
| Recall | 71.8 | 73.7 | +1.9 |
| Precision | 93.0 | 92.3 | -0.7 |
| FN | 1949 | 1820 | -129 |
| FP | 375 | 424 | +49 |
| IDs | 70 | 67 | -3 |

此結果只證明 B2 path 有 detection signal，不能直接證明 1920→640 resize
是主因。MOT17-05 原始解析度為 640x480，B2 在此 sequence 仍改善，表示收益
可能包含 high-resolution semantic oracle、upscale 後的 feature sampling、額外參數，
或相同 budget 重訓效果。必須完成 B0-R、B1-L、B1-H 才能歸因。

隔離的 100-frame latency smoke：

```text
v14: 8.40 ms / 119.1 FPS
B2: 16.19 ms / 61.8 FPS
```

完整 sequence 的兩組評估曾同時執行，因此該次 latency 受 GPU contention 影響，
不納入正式 runtime 結論。

### B2 初步結果：MOT17-02-SDP

完整 MOT17-02-SDP train sequence（600 frames、1920x1080），兩組分開執行：

| Metric | v14 | B2 native-P3 | Delta |
|---|---:|---:|---:|
| IDF1 | 52.1 | 55.1 | +3.0 |
| MOTA | 57.1 | 57.5 | +0.4 |
| HOTA | 43.7 | 47.7 | +4.0 |
| DetA | 50.6 | 52.3 | +1.7 |
| AssA | 37.8 | 43.5 | +5.7 |
| Recall | 59.3 | 59.6 | +0.3 |
| Precision | 97.2 | 97.3 | +0.1 |
| FN | 7559 | 7501 | -58 |
| FP | 319 | 309 | -10 |
| IDs | 92 | 93 | +1 |
| Mean latency | 8.68 ms | 17.00 ms | +8.32 ms |

判讀：

- B2 在 1920x1080 sequence 有整體收益，證明 high-resolution semantic source
  不是無效訊號。
- Recall 僅提升 `+0.3pp`，小於 MOT17-05 的 `+1.9pp`，目前不支持
  「pre-resize 小目標資訊恢復是主要收益來源」。
- 最大增益出現在 AssA/IDF1/HOTA，更像 box/score 穩定性、semantic context，
  或額外重訓造成的 tracking association 改善。
- 必須取得 detector-only、依 box size 分層的 recall，以及 B0-R/B1-L/B1-H，
  才能判斷小目標 feature 是否真的被恢復。

### B0-R 相同 budget 重訓控制

Checkpoint：

```text
runs/mamba_detail_b0_retrain/best.ckpt
epoch=30
best training loss=0.5971813508
detail source=none
```

B0-R 與 B2 使用相同 30 epoch、batch 1、accumulation 4、LR `1e-4`，
只移除 detail branch。完整 sequence 結果：

| Sequence | Metric | v14 | B0-R | B2 | B2 - B0-R |
|---|---|---:|---:|---:|---:|
| MOT17-02 | IDF1 | 52.1 | 54.2 | 55.1 | +0.9 |
| MOT17-02 | MOTA | 57.1 | 57.8 | 57.5 | -0.3 |
| MOT17-02 | HOTA | 43.7 | 46.8 | 47.7 | +0.9 |
| MOT17-02 | DetA | 50.6 | 52.4 | 52.3 | -0.1 |
| MOT17-02 | AssA | 37.8 | 41.9 | 43.5 | +1.6 |
| MOT17-02 | Recall | 59.3 | 60.0 | 59.6 | -0.4 |
| MOT17-05 | IDF1 | 68.7 | 68.9 | 69.2 | +0.3 |
| MOT17-05 | MOTA | 65.4 | 66.0 | 66.6 | +0.6 |
| MOT17-05 | HOTA | 58.2 | 58.2 | 58.6 | +0.4 |
| MOT17-05 | DetA | 60.0 | 61.3 | 61.4 | +0.1 |
| MOT17-05 | AssA | 56.8 | 55.5 | 56.2 | +0.7 |
| MOT17-05 | Recall | 71.8 | 73.0 | 73.7 | +0.7 |

判讀：

- B0-R 在 MOT17-02 已取得 `IDF1 +2.1`、`HOTA +3.1`、Recall `+0.7`，
  證明原先 B2 對 v14 的大部分差值包含相同 budget 重訓效應。
- B2 相對 B0-R 在 1920x1080 的 MOT17-02 沒有改善 Recall 或 DetA；
  剩餘收益集中在 AssA/IDF1，仍不支持小目標資訊恢復是主要機制。
- B2 在 640x480 的 MOT17-05 反而保留 Recall `+0.7`，同樣不能歸因於
  pre-resize 高解析資訊。
- 下一個必要實驗是 matched B1-L/B1-H；在此之前不擴大 B2 全序列評估，
  也不進入 production optimization。

### B1-L 低解析 detail control

Checkpoint：

```text
runs/mamba_detail_b1_low/best.ckpt
epoch=30
best training loss=2.4227338687
detail source=global
batch=4
accumulation=1
```

| Sequence | Metric | B0-R | B1-L | B1-L - B0-R |
|---|---|---:|---:|---:|
| MOT17-02 | IDF1 | 54.2 | 54.0 | -0.2 |
| MOT17-02 | MOTA | 57.8 | 58.2 | +0.4 |
| MOT17-02 | HOTA | 46.8 | 46.5 | -0.3 |
| MOT17-02 | DetA | 52.4 | 52.7 | +0.3 |
| MOT17-02 | AssA | 41.9 | 41.1 | -0.8 |
| MOT17-02 | Recall | 60.0 | 60.0 | 0.0 |
| MOT17-05 | IDF1 | 68.9 | 67.9 | -1.0 |
| MOT17-05 | MOTA | 66.0 | 65.8 | -0.2 |
| MOT17-05 | HOTA | 58.2 | 57.4 | -0.8 |
| MOT17-05 | DetA | 61.3 | 60.7 | -0.6 |
| MOT17-05 | AssA | 55.5 | 54.7 | -0.8 |
| MOT17-05 | Recall | 73.0 | 72.5 | -0.5 |

B1-L 沒有優於 B0-R，表示增加同一個 640 view 的 shallow detail branch 本身
沒有帶來有效訊號。這使 B1-H 與 B1-L 的 matched comparison 更重要。

原本位於 `runs/mamba_detail_b1_high/best.ckpt` 的候選 checkpoint 經檢查為
`detail_source=none`、`use_detail_fusion=false`，且沒有任何 `detail_fusion.*`
權重，因此不是有效 B1-H，不得納入比較。B1-H 必須以 `--detail-source high`
重新訓練並使用新的 run directory。

### B1-H matched high-resolution control

有效 checkpoint：

```text
runs/mamba_detail_b1_high_v2/best.ckpt
epoch=29
best training loss=2.4134444469
detail source=high
detail bucket=1280x768
batch=4
accumulation=1
```

| Sequence | Metric | B1-L | B1-H | B1-H - B1-L |
|---|---|---:|---:|---:|
| MOT17-02 | IDF1 | 54.0 | 54.2 | +0.2 |
| MOT17-02 | MOTA | 58.2 | 58.4 | +0.2 |
| MOT17-02 | HOTA | 46.5 | 47.0 | +0.5 |
| MOT17-02 | DetA | 52.7 | 52.4 | -0.3 |
| MOT17-02 | AssA | 41.1 | 42.2 | +1.1 |
| MOT17-02 | Recall | 60.0 | 60.0 | 0.0 |
| MOT17-02 | FP | 234 | 212 | -22 |
| MOT17-02 | FN | 7441 | 7430 | -11 |
| MOT17-05 | IDF1 | 67.9 | 70.1 | +2.2 |
| MOT17-05 | MOTA | 65.8 | 65.8 | 0.0 |
| MOT17-05 | HOTA | 57.4 | 58.8 | +1.4 |
| MOT17-05 | DetA | 60.7 | 60.5 | -0.2 |
| MOT17-05 | AssA | 54.7 | 57.4 | +2.7 |
| MOT17-05 | Recall | 72.5 | 72.0 | -0.5 |
| MOT17-05 | FP | 384 | 356 | -28 |
| MOT17-05 | FN | 1903 | 1940 | +37 |

隔離 runtime：

```text
MOT17-02: B1-L 8.92 ms, B1-H 10.66 ms
MOT17-05: B1-L 9.24 ms, B1-H 10.62 ms
```

判讀：

- B1-H 相對 B1-L 的收益集中在 AssA、IDF1、HOTA 與較少 FP，表示
  high-resolution shallow feature 可改善預測穩定性或 association quality。
- MOT17-02 Recall 完全相同且 DetA 略退；MOT17-05 Recall 反而下降。
  因此 matched control 不支持「pre-resize 小目標資訊恢復」是主要收益來源。
- 最大 IDF1/AssA 收益出現在原始解析度只有 640x480 的 MOT17-05，而不是
  1920x1080 的 MOT17-02，進一步削弱 resize bottleneck 假說。
- 目前不應以 small-object recovery 為理由擴充 native semantic branch。
  若繼續此方向，目標應改為低成本 box/score stability 或 association feature，
  並先做 detector-only size-binned recall，確認 tracking 指標沒有掩蓋 detector
  regression。

### Detector-only size-binned recall

評估工具：

```text
scripts/eval/mamba_size_binned_recall.py
```

設定：

- tracker、GMC、birth gate、interpolation 全部停用。
- native-640 stretch global input；detail 組額外讀取原圖。
- detector confidence floor `0.001`、person class only、NMS IoU `0.5`。
- GT matching IoU `0.5`，prediction 依 score 由高至低一對一 matching。
- GT 過濾與訓練一致：`mark=1`、pedestrian、visibility `>=0.1`。
- size bin 使用 resize-640 後的 GT min-side，直接量測進入 global stream
  後剩餘的像素尺度。

MOT17-02-SDP 有 10,993 個有效 GT，其中 2,259 個落在 resize-640
min-side `4-8 px`；MOT17-05-SDP 沒有 `4-8 px` GT，`8-16 px` 也只有 73 個，
因此 resize bottleneck 的主要判讀以 MOT17-02 為準。

#### MOT17-02-SDP recall

| Score | Size bin | B0-R | B1-L | B1-H | B2 |
|---:|---|---:|---:|---:|---:|
| 0.001 | all | 96.30 | 96.32 | 96.43 | 96.42 |
| 0.001 | 4-8 px | 90.62 | 90.57 | 90.84 | 90.79 |
| 0.001 | 8-16 px | 97.35 | 97.41 | 97.56 | 97.54 |
| 0.25 | all | 94.12 | 94.21 | 94.43 | 94.26 |
| 0.25 | 4-8 px | 88.40 | 88.31 | 88.76 | 88.71 |
| 0.25 | 8-16 px | 94.58 | 94.75 | 95.08 | 94.75 |

Matched-control delta：

| Comparison | Score | all | 4-8 px | 8-16 px |
|---|---:|---:|---:|---:|
| B1-H - B1-L | 0.001 | +0.11 | +0.27 | +0.15 |
| B1-H - B1-L | 0.25 | +0.22 | +0.45 | +0.33 |
| B2 - B0-R | 0.001 | +0.12 | +0.17 | +0.19 |
| B2 - B0-R | 0.25 | +0.14 | +0.31 | +0.17 |

MOT17-05 在 score `0.25` 時，B1-H 相對 B1-L 的整體 recall 為
`-0.09pp`，8-16 px bin 為 `-4.11pp`；但該 bin 只有 73 個 GT。
B2 相對 B0-R 的整體 recall只有 `+0.02pp`，8-16 px 完全相同。

判讀：

- 原本 `+3pp` architecture GO 門檻是在約 50% recall 的早期觀察下制定，
  不適用目前 `88-97%` 的 detector-only recall 區間；高 baseline 下應同時看
  absolute recall delta 與相對 FN reduction。
- B1-H 在 MOT17-02 有弱正向 detector signal。相對 B1-L，4-8 px 的 FN
  reduction 為 `0.4-3.8%`，8-16 px 為 `5.1-6.3%`，依 score threshold
  而異。這不足以直接宣告 GO，但也不能依舊門檻判為 NO-GO。
- B2 semantic oracle 沒有超過 B1-H，也沒有顯著超過 B0-R，表示增加
  high-resolution semantic depth 沒有展現額外優勢。
- detector-only 結果確認先前 HOTA/IDF1 增益主要來自 FP、box/score stability
  或 association quality；同時存在小幅 detector FN reduction，但不是大幅
  recovery。
- Aggregate 結果先視為 `weak positive`；以下用 paired moving-block
  bootstrap 檢查改善是否只是固定場景或少數 frame 所造成。

#### B1-H vs B1-L paired moving-block bootstrap

使用相同 600 個 MOT17-02 frame 做 paired resampling。為保留相鄰 frame 的
時序相關性，主結果使用 16-frame circular moving blocks、20,000 samples；
另以 8/32/64-frame blocks 做 sensitivity check。

16-frame block 主結果：

| Score | Size bin | Recall delta | 95% CI | FN reduction | 95% CI |
|---:|---|---:|---:|---:|---:|
| 0.001 | all | +0.118pp | [0.000, +0.274] | +3.21% | [0.00, +7.29] |
| 0.001 | 4-8 px | +0.266pp | [0.000, +0.606] | +2.82% | [0.00, +6.73] |
| 0.001 | 8-16 px | +0.144pp | [-0.040, +0.380] | +5.56% | [-1.56, +13.91] |
| 0.10 | 4-8 px | +0.044pp | [-0.225, +0.344] | +0.43% | [-2.46, +3.35] |
| 0.10 | 8-16 px | +0.205pp | [-0.021, +0.459] | +5.10% | [-0.62, +10.69] |
| 0.25 | all | +0.227pp | [+0.076, +0.409] | +3.92% | [+1.33, +6.97] |
| 0.25 | 4-8 px | +0.443pp | [+0.052, +0.834] | +3.79% | [+0.43, +8.12] |
| 0.25 | 8-16 px | +0.328pp | [+0.061, +0.638] | +6.25% | [+1.24, +11.47] |

在 score `0.25`，4-8 px 與 8-16 px 的 recall delta CI 在
8/16/32/64-frame block lengths 下皆排除 0。這個 threshold 接近實際
new-track confidence 區間，因此有 operational relevance。低 threshold
`0.001/0.10` 的結果則不穩定，表示 B1-H 主要把部分已存在的候選提高到可用
confidence，而不是創造大量原本完全不存在的 detections。

更新判讀：

- matched high-resolution branch 存在 statistically supported、但幅度小的
  detector signal，應由 `INCONCLUSIVE` 升為 `provisional signal GO`。
- 這只證明單一 sequence、單一訓練 seed 下的 frame/block 穩定性；不能替代
  跨 sequence 與跨 seed 驗證，也未校正多 threshold/bin comparisons。
- 架構下一步應優先確認此 confidence-lift 是否能以更低成本保留，而不是增加
  full semantic depth；B2 目前未顯示超過 shallow B1-H 的必要性。

#### U1 native-1080 resolution upper bound

固定 B1-H 的 32-channel shallow encoder 與 `3x3` token，只把 detail bucket
由 1280x768 提高至 1920x1080。MOT17-02 可直接使用原生像素，不需縮小：

```text
runs/mamba_detail_b1_native_1080/best.ckpt
epoch=30
detail source=high
detail bucket=1920x1080
batch=2
accumulation=2
```

Detector-only paired bootstrap，以 1280x768 B1-H 為 baseline：

| Score | Size bin | U1 - B1-H | 95% CI |
|---:|---|---:|---:|
| 0.001 | all | 0.000pp | [-0.084, +0.087] |
| 0.001 | 4-8 px | 0.000pp | [-0.281, +0.309] |
| 0.001 | 8-16 px | 0.000pp | [-0.129, +0.134] |
| 0.10 | 4-8 px | +0.089pp | [-0.121, +0.347] |
| 0.10 | 8-16 px | +0.123pp | [-0.064, +0.315] |
| 0.25 | 4-8 px | -0.044pp | [-0.392, +0.312] |
| 0.25 | 8-16 px | -0.123pp | [-0.291, +0.044] |

所有 95% CI 都包含 0，沒有解析度增益證據。

完整 MOT17-02 tracking：

| Metric | B1-H 1280x768 | U1 1920x1080 | Delta |
|---|---:|---:|---:|
| IDF1 | 54.2 | 52.6 | -1.6 |
| MOTA | 58.4 | 56.7 | -1.7 |
| HOTA | 47.0 | 45.8 | -1.2 |
| DetA | 52.4 | 51.2 | -1.2 |
| AssA | 42.2 | 41.1 | -1.1 |
| Recall | 60.0 | 59.3 | -0.7 |
| FP | 212 | 389 | +177 |
| FN | 7430 | 7558 | +128 |
| Mean latency | 10.66 ms | 11.73 ms | +1.07 ms |

判讀：

- 1280x768 已接近目前 shallow encoder/fusion 的解析度 sweet spot；增加到
  native 1920x1080 不提升 detector recall，且 tracking calibration 明顯退步。
- 不再提高 detail input resolution。下一個 upper-bound 實驗固定 1280x768，
  改測 `5x5` patch token，判斷限制是否來自 detail receptive field。
- 若 `5x5` 仍無增益，再測 encoder channels/depth；一次只改一個變因。

#### U2 5x5 patch receptive-field upper bound

固定 1280x768 detail bucket 與 32-channel stride-4 encoder，只把 patch token
由 `3x3` 提高至 `5x5`：

```text
runs/mamba_detail_b1_high_patch5/best.ckpt
epoch=29
best training loss=2.3866543841
detail patch=5x5
```

Detector-only paired bootstrap，以 `3x3` B1-H 為 baseline：

| Score | Size bin | U2 - B1-H | 95% CI |
|---:|---|---:|---:|
| 0.001 | all | -0.055pp | [-0.151, +0.037] |
| 0.001 | 4-8 px | -0.089pp | [-0.380, +0.178] |
| 0.001 | 8-16 px | -0.041pp | [-0.176, +0.081] |
| 0.10 | 4-8 px | 0.000pp | [-0.230, +0.241] |
| 0.10 | 8-16 px | -0.021pp | [-0.232, +0.180] |
| 0.25 | 4-8 px | -0.044pp | [-0.400, +0.315] |
| 0.25 | 8-16 px | -0.144pp | [-0.379, +0.075] |

所有 CI 都包含 0，且 point estimate 多數偏負。

完整 MOT17-02 tracking：

| Metric | B1-H 3x3 | U2 5x5 | Delta |
|---|---:|---:|---:|
| IDF1 | 54.2 | 52.6 | -1.6 |
| MOTA | 58.4 | 57.6 | -0.8 |
| HOTA | 47.0 | 45.3 | -1.7 |
| DetA | 52.4 | 52.2 | -0.2 |
| AssA | 42.2 | 39.4 | -2.8 |
| Recall | 60.0 | 59.5 | -0.5 |
| FP | 212 | 259 | +47 |
| FN | 7430 | 7519 | +89 |
| Mean latency | 10.66 ms | 11.77 ms | +1.11 ms |

判讀：

- 增大 local patch receptive field 無法提升 detector recall，並破壞 tracking
  association/calibration；`3x3` 保持為最佳 patch。
- 目前上限不在 detail input resolution 或 patch footprint。下一個單變因是
  encoder capacity：固定 1280x768、`3x3`，把 channels 從 32 提高至 64。

#### U3 64-channel encoder capacity upper bound

固定 1280x768 detail bucket 與 `3x3` token，把第二個 stride-2 stage 從
32 channels 擴至 64。第一個 stage 維持 32 channels，保留 YOLO Conv+BN
warm-start：

```text
runs/mamba_detail_b1_high_ch64/best.ckpt
epoch=29
best training loss=2.4081854884
encoder=3->32->64
```

Detector-only paired bootstrap，以 32-channel B1-H 為 baseline：

| Score | Size bin | U3 - B1-H | 95% CI |
|---:|---|---:|---:|
| 0.001 | all | -0.036pp | [-0.123, +0.056] |
| 0.001 | 4-8 px | -0.044pp | [-0.325, +0.215] |
| 0.001 | 8-16 px | +0.021pp | [-0.104, +0.166] |
| 0.10 | 4-8 px | +0.266pp | [+0.042, +0.555] |
| 0.10 | 8-16 px | -0.144pp | [-0.377, +0.085] |
| 0.25 | 4-8 px | +0.089pp | [-0.299, +0.491] |
| 0.25 | 8-16 px | -0.062pp | [-0.254, +0.124] |

只有 score `0.10` 的 4-8 px bin 顯著正向；同 threshold 的 8-16 px、
整體 recall，以及 operational score `0.25` 都沒有一致收益。

完整 MOT17-02 tracking：

| Metric | B1-H 32ch | U3 64ch | Delta |
|---|---:|---:|---:|
| IDF1 | 54.2 | 52.4 | -1.8 |
| MOTA | 58.4 | 56.9 | -1.5 |
| HOTA | 47.0 | 45.5 | -1.5 |
| DetA | 52.4 | 51.8 | -0.6 |
| AssA | 42.2 | 40.1 | -2.1 |
| Recall | 60.0 | 59.7 | -0.3 |
| FP | 212 | 411 | +199 |
| FN | 7430 | 7491 | +61 |
| Mean latency | 10.66 ms | 11.25 ms | +0.59 ms |

判讀：

- 增加 encoder channels 只在單一低 confidence/bin cell 有局部收益，沒有轉化
  成 operational threshold 或 tracking 改善，且造成大量 FP。
- 32-channel shallow encoder 保持為最佳 dense detail architecture。
- 在現有 v14 warm-start 基底下，解析度、patch footprint、encoder width
  三個容量軸均已達平台。但 v14 training audit 發現 resume LR、cache gate、
  frame coverage 與 checkpoint selection 問題，因此這是 conditional ceiling，
  不是乾淨訓練下的架構絕對上限。
- 下一步先建立 controlled v14-R baseline，再以相同 split/seed 重訓 32-channel
  B1-H；之後才決定是否投入 sparse original-resolution ROI path。
- controlled v14-R pipeline 已加入 seed、`clip-stride=4`、cache/gate 防誤用、
  exact/reset resume 語意、單一 warmup+cosine scheduler、holdout 排除與
  external recall/HOTA candidate selection。Stage-1 起點固定為
  `runs/mamba_distill_cs_n16/best.ckpt`。
- 但 `gated_det_v1`、`trt_feat_cache_v2`、`mamba_distill_cs_n16` 均已使用
  全部 MOT17-SDP，因此目前只能稱 controlled re-finetune；strict clean
  必須從 gated teacher 開始以相同 split 重建完整 lineage。

### Phase 2：輕量 detail encoder

只在 B1-H 證明 pre-resize resolution 有訊號，且 B2 顯示 high-resolution semantic source 可形成有效 oracle 後執行：

| ID | Detail resolution | Encoder | Channels | Token |
|---|---|---|---:|---|
| C1 | native | 2-stage shallow，YOLO stem init 或 B2 token distillation | 32 | `3x3` |
| C2 | native | YOLO stem layers 0-2 | 128 | `3x3` |
| C3 | max-side 1280 | 與 C1 相同初始化的 2-stage shallow | 32 | `3x3` |
| C4 | max-side 1024 | 與 C1 相同初始化的 2-stage shallow | 32 | `3x3` |
| C0-R | native | random-init 2-stage shallow | 32 | `3x3` |

判讀：

- C1/C3 能保留 B2 大部分收益：production candidate。
- 只有 C2 有效：需要較強 semantic encoder，成本需重新評估。
- 只有 native 有效：固定高解析 resize 仍丟失關鍵資訊。
- C0-R 只用來量測 pretraining 的必要性，不用其失敗否定 shallow detail encoder。

### Phase 3：完整 v14 warm-start 重訓

#### T0：Identity migration

- 載入 v14 所有可對應權重。
- 新 detail fusion residual zero-init。
- 未訓練輸出需與 v14 一致。

#### T1：Detail branch warm-up

- 凍結 global YOLO backbone。
- 暫時凍結 P4/P5 path。
- 主要訓練 detail encoder、patch token projection、cls/reg fusion。
- P3 Mamba/head 使用低 LR。

#### T2：完整 Mamba head joint retraining

解凍完整 `MambaDetectionHead`：

- P3/P4/P5 input projection。
- spatial reduction。
- Cross-Scan Mamba blocks。
- PixelShuffle upsampler。
- cls/reg heads。
- detail fusion modules。

Global YOLO backbone 維持凍結；detail encoder 可訓練。

建議 LR group：

| Parameter group | 相對 LR |
|---|---:|
| global YOLO backbone | `0` |
| v14 P4/P5 path | `0.1x` |
| v14 P3 path | `0.25x` |
| detail encoder/fusion | `1.0x` |

#### Distillation 注意事項

不能對所有位置強制模仿 v14。v14 正是會漏掉 resize 後小目標的 teacher，若在這些位置做強 preservation loss，會壓制新 detail branch。

建議：

- large/medium GT 與背景區域可使用 v14 preservation。
- small GT 的 distillation exclusion mask 由實際 assigned positive anchors 建立，並擴張一圈 candidate halo；不能只排除 GT center 所在 cell。
- 特別是 v14 false-negative GT，其 P3/P4/P5 positive anchors 與鄰近候選區域都不使用 cls distillation。
- 若 assigner 為極小 GT 使用 `stride_val` 擴張 candidate box，exclusion mask 必須使用相同幾何規則。
- distillation weight 隨訓練衰減。
- 新增候選必須由 GT supervision 決定，不由 v14 teacher score 決定。

### Phase 4：P2 output head

只有 Phase 0 的實際 assigner 統計顯示 small GT 缺少正 anchors/在 conflict resolution 後失去正 anchors，或 detail branch 已能辨識但仍無法分開密集小目標時執行。P3 center collision 本身不足以啟動此 phase。

必須獨立比較：

- P3 detail fusion。
- P2 output only。
- P3 detail + P2 output。

避免把 source information gain 與 grid capacity gain 混在同一實驗。

### Phase 5：部署

精度 GO 後才處理：

1. 選定 native 或 canonical detail shape。
2. export detail encoder。
3. 建立 global/detail 雙 input 或共享原始 frame 的 TensorRT graph。
4. 訓練與推論都使用有限 shape buckets；batch 內 pad 到 bucket shape 並攜帶 `valid_detail_region` mask。
5. whole graph capture key 至少包含 detail bucket shape、batch/stream layout 與固定 global `640x640` shape。
6. 每個 bucket 預建 `T_orig_to_detail` 對應的 sampling grid，禁止 replay 時動態改 shape。
7. 驗證 coordinate transform、padding mask、dtype 與 ROIAlign convention。
8. 量測 1/4/8 stream throughput。

---

## 8. 訓練資料流

Dataset 必須同時提供：

```text
Iorig / Ihigh
I640
GT_orig
GT_640
T_orig_to_640
T_640_to_orig
T_orig_to_detail
T_detail_to_orig
valid_detail_region
```

幾何 augmentation 必須先作用於原圖與 GT，再由同一結果產生 high/global views。禁止兩個 stream 各自做 random crop、flip 或 resize。

建議順序：

```text
原圖 + GT
  → shared geometric augmentation
  → high-resolution detail view
  → 640 global view + transform metadata
```

顏色 augmentation 可共享參數，或只對 global/detail 做小幅獨立擾動；第一輪建議完全共享，減少分布差異。

對 variable-size native frames，training dataloader 應先依 detail shape bucket 分組，再 pad 到 bucket 邊界。loss 與 token sampling 必須套用 `valid_detail_region`，避免 padding 被學成穩定背景特徵。

---

## 9. 成本分析

### 9.1 Full native backbone

對 `1920x1080`：

```text
1920 * 1080 / (640 * 640) = 5.06x pixels
```

再加原本 640 global stream，完整 native YOLO source 約為單一 640 backbone 的 6 倍量級，不適合作預設部署。

### 9.2 Shallow detail feature memory

以 FP16 計算：

| Detail feature | 大小 |
|---|---:|
| native `32x480x270`，stride 4 | 約 8.3 MB |
| native `64x480x270`，stride 4 | 約 16.6 MB |
| 1280x720 source，`32x320x180` | 約 3.7 MB |
| P3 detail tokens，`64x80x80` | 約 0.82 MB |

因此優先考慮：

- 32-channel shallow detail。
- 盡早轉成 P3-aligned tokens。
- 不把 raw native feature 長時間留在 pipeline。

### 9.3 與 full 1024 比較

full 1024 pixel count 是 full 640 的 2.56 倍。雙 stream 若無法在低於 full 1024 的 latency 下達到相同或更好精度，就不值得承擔額外架構複雜度。

---

## 10. 評估指標

### Detector-level

- AP50、AP75、AP small。
- Recall，依原圖 box width/height 分層。
- Recall，依 resize-640 後 box width/height 分層。
- small-visible recall。
- v14 FN cell 的 recovery rate。
- matched box IoU 與 center error。
- P3 center collision recovery，僅作描述性指標。
- 每個 GT 在 assigner conflict 前後的 positive-anchor count。
- 無 positive anchor 的 GT 比例與 P3/P4/P5 responsibility。
- FP 數量與 score distribution。

### Tracking-level

- IDF1、MOTA、HOTA、DetA、AssA。
- FP、FN、IDs。
- 1920x1080 sequences 與 MOT17-05 分開報告。
- 每 sequence 結果，避免 aggregate gain 被單一場景主導。

### Runtime

- global backbone latency。
- detail encoder latency。
- patch-token extraction latency。
- Mamba head latency。
- peak VRAM。
- whole-detect FPS。
- 1/4/8 stream throughput。

---

## 11. GO / NO-GO

### Resize bottleneck confirmed

至少滿足兩項：

- 正確 1024 baseline 明顯改善 small-object detector recall。
- matched high-resolution source B1-H 明顯優於相同 encoder/layer 的 B1-L。
- gain 主要出現在 1920x1080 sequences，而非 MOT17-05。
- 原圖 crop separability 明顯高於 640 crop。

### Architecture signal GO

- 不使用固定 `+3pp` 門檻；同時報告 absolute recall delta、相對 FN reduction
  與 paired bootstrap confidence interval。
- B1-H 相對 B1-L 在目標 size bins 的改善方向跨合理 score thresholds 一致，
  且 paired 95% confidence interval 排除 0。
- B2 相對 B1-H 的額外收益需獨立報告；若無額外收益，優先 shallow branch。
- AP75 或 matched IoU 不退。
- FP 增幅不超過 2%。

### Production GO

- 輕量 detail candidate 保留至少 80% 的 B2 oracle gain。
- full MOT17-SDP IDF1 或 HOTA 至少 `+0.5pp`。
- MOTA 不退超過 `0.2pp`。
- 至少 5/7 sequences 不退。
- 相對 640 v14 latency 增加不超過 20%，或在相同精度下明顯快於 full 1024。
- 固定 shape/shape-bucket CUDA graph 可 capture/replay。

### NO-GO

- matched B1-H 無法優於 B1-L，且 crop separability/可信 1024 baseline 也不支持 resize bottleneck。
- matched B1-H 的 target-bin FN reduction 在 paired bootstrap 下不穩定或
  confidence interval 包含具實務意義的負向效果。
- 只有完整 native YOLO backbone 有效，lightweight detail encoder 無法保留收益且成本不可接受。
- valid full 1024 在 accuracy 與 latency 上都優於雙 stream。
- gain 來自降低 threshold 或 tracker retuning，而非 detector evidence。
- detail branch 增加大量背景 FP。
- 實際 assigner 診斷確認正樣本容量是主要限制，但不接受新增 P2 output head。

---

## 12. 決策摘要

| 問題 | 結論 |
|---|---|
| 640-P2 能否恢復 resize 前細節？ | 不能，只能減少 backbone 內部下採樣損失 |
| 是否用原圖單點取樣？ | 不足，應編碼完整 P3 cell footprint |
| 主架構 | 640 v14 global stream + pre-resize high-resolution detail stream |
| detail 表徵 | stride-4 shallow feature + `3x3` patch token |
| detail 注入位置 | P3 cls/reg 分支，使用獨立 zero-init residual |
| 是否依賴 640 proposals？ | 否，必須 dense，否則無法補完全漏檢 |
| 原圖完整 YOLO P3/P4/P5 | 只作 oracle，成本過高 |
| production source | native 或 max-side 1280 lightweight detail encoder |
| v14 訓練方式 | warm-start，穩定期後重訓完整 Mamba head |
| 重要控制組 | matched B1-L/B1-H、相同 budget v14、可信 full 1024 |
| 額外決策 | 依實際 assigner 正樣本/衝突診斷決定是否新增 P2 output head |

整體判斷：**問題若確實發生在 640 resize，修正點必須位於 resize 之前。** Mamba 應繼續負責全域判斷，但小目標的局部證據必須由原始或近原始解析度分支提供。前一版只使用 640-P2/P3 的方案不足以解決目前描述的資訊瓶頸。
