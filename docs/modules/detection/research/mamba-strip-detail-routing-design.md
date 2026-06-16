# Mamba Strip Detail Routing — 設計文件

> 日期：2026-06-13
> 狀態：**⏸ PARKED — ROI NO-GO（registry #36，2026-06-13）**。實作完成（`P3StripDetailFusion`
> / `P3AttentionStripDetailFusion` / `_gather_strip_pixels` / `build_strip_oracle_positions`，
> 6 routing tests pass，uint8 VRAM 修復，`--detail-source strip-oracle`），但在跑出 oracle
> recall 天花板**之前**依成本判定停止：增益天花板 <0.5pp IDF1（min_4to8 2259 GT@0.826 →
> 上限 ~390 框、dense B1-H 實測 +0.3pp）vs 高解析度部署成本（TRT 重編 / +1.4ms / 兩階段訓練）。
> **復活條件**：其他方向耗盡時重啟，先跑 Phase 1 取 oracle 天花板再定生死（命令見
> `scripts/train/temporal_yolo/run_v14replica_strip_oracle.sh`，B=2/accum=2 已調好 6.1GB headroom）。
> 代碼 default off 保留。未取 oracle 數字 → 屬成本判定，**不可當結構天花板引用**。
> 前身：[mamba-dual-resolution-original-detail-plan.md](mamba-dual-resolution-original-detail-plan.md)
> 動機：B1-H dense detail fusion 已證明 shallow high-resolution branch 有弱但統計顯著的
> detection signal，但 dense 方案對所有 P3 cell 皆運算，成本過高（+1.4ms）。
> Strip routing 是**稀疏化方向**的延伸 — 只對需要原圖解析度的 P3 cell 讀取 pixel strip。

---

## 1. 核心假說

640 resize 對 small objects（原圖 <24×24 px）是不可逆的資訊損失：

```
原圖 24px → resize 640 (~8px) → backbone stride8 (P3: ~1px) → downsample stride4 (~0.25 token)
```

無論 downsample 怎麼調、Mamba 多強，輸入本身就沒訊號。1024 統一只改善比例，不解決根因。
**唯一資訊完整的路徑是從原圖直接讀取像素。**

Dense detail encoder（B1-H）用 stride-4 conv 在整個原圖上編碼，然後 grid_sample 對齊 P3 grid，
對所有 6400 個 P3 cell 都算。Strip routing 把成本限制在需要原圖解析度的極少數 cell：

```
routed cell:       原圖 → integer index strip → Conv1d stem → Mamba → +Δ
non-routed cell:   原 640 pipeline（unchanged）
```

## 2. 架構

### 2.1 Routing mask

P3 grid (80×80, 6400 cells)。Binary mask 決定哪些 cell 開啟 detail strip：

| Phase | Mask 來源 | 訓練/推論一致？ |
|:---:|---|:---:|
| Warm-up | GT box min-side < threshold → 覆蓋 cell = 1 | ❌ 推論無 GT |
| Inference | Lightweight predictor: Conv3→SiLU→Conv1→1, BCE loss（target = GT mask）, on x_proj | ✅ |

Phase 1 用 GT-driven mask 讓 strip 路徑在乾淨訊號下收斂，Phase 2 冷凍 strip encoder + Mamba，
只訓練 routing predictor。predictor 成本 ~可忽略（小 conv on x_proj at P3）。

MOT17-02 (1920×1080)，8px threshold 下 routed cell 約 200–400（3–6%）。
若 predictor 挑 10%，strip 成本僅 dense B1-H 的 ~3%。

### 2.2 1D strip 讀取

對每個 routed cell (i, j) in P3 grid，映射回原圖座標，沿四方向讀 raw pixel strip：

```
P3 cell (i,j) stride=8 on 640 input:
  center_px = (i * 8 + 4, j * 8 + 4)   # P3 center in 640 space
  center_orig = center_px * orig_scale   # 映射回原圖

  strip_half_w = W  # 像素半寬

  dir0 (水平左→右): image[:, center_y, center_x-W : center_x+W]  → (3, 2W)
  dir1 (水平右→左): flip(dir0)
  dir2 (垂直上→下): image[:, center_y-H : center_y+H, center_x]  → (3, 2H)
  dir3 (垂直下→上): flip(dir2)
```

關鍵設計：
- **Integer indexing**，不插值，不 crop，不 resize — 純記憶體定址，~零開銷
- **不跑 encoder CNN** — strip 是 raw RGB pixels，送入一個極小 Conv1d stem
- 四方向結構和 Mamba cross-scan **同構** — 都是四向 1D sequence

### 2.3 Conv1d stem

```
raw strip (3, L_strip) → Conv1d(3→stem_ch, k=5, s=2, pad=2) → (stem_ch, L_strip/2)
                        → SiLU
                        → Conv1d(stem_ch→d_model, k=3, s=1, pad=1) → (d_model, L_strip/2)
```

stem_ch = 32，d_model = 128。第一層 Conv1d 可用 YOLO backbone layer0 的 3×3 conv
權重做 1D 近似 warm-start（取中間一列再平均兩側）。兩層共 ~2K params。

### 2.4 Detail Mamba block

```
(d_model, L_strip/2) per direction × 4 → 獨立 MambaBlock *1（或 *2，共用同一組 block）
                                         → aggregate（四個方向的 last hidden 取 mean）
                                         → (d_model,)  vector
                                         → 加到對應 cell 的 x_proj 上
```

參數量：
- MambaBlock *1: d_model=128, d_state=16 → ~197K params（可和 spatial Mamba 共用 block，但獨立 block 歸因更清楚）
- Conv1d stem: ~2K
- Routing predictor: ~5K
- **總計 ~204K params**，僅 B1-H detail encoder（~30K + fusion ~d_model²）的 ~1.5×，但只在 <10% cell 執行

### 2.5 銜接點（插入位置）

在 `MambaDetectionHead._forward_eager` 中，detail strip 插入在 downsample + spatial Mamba 之後、
upsample 之前，對 routed cell 的 x_up 做 residual addition：

```python
# 現有 cross-scan
x_up = _cross_scan_mamba(x_small, self.mamba_blocks[i], Hs, Ws)

# 插入 detail strip（僅 P3，僅 routed cell）
if i == 0 and routing_mask is not None:
    for b in range(B):
        routed_positions = routing_mask[b].nonzero()  # (N_routed, 2) with (y, x) indices
        if routed_positions.numel() == 0:
            continue
        strips = self._read_strips(original_frame[b], routed_positions)  # → 4 × (N, d_model, L)
        detail_vec = self.strip_mamba(strips)                            # → (N, d_model)
        # scatter into x_up at routed positions
        x_up[b, :, routed_positions[:, 0], routed_positions[:, 1]] += detail_vec
```

`original_frame` 是未 resized 的原圖 uint8 tensor。資料集在 `load_images=True`
+ `detail_size` 時已自動載入原圖。

### 2.6 訓練流程

```
Phase 1 (30 epoch, GT-driven mask):
  routing_mask = GT_boxes min-side < 8px → covered cells = 1
  detail strips + Mamba 在 GT cell 上訓練
  spatial Mamba + prediction heads 正常更新
  → strip 路徑在最乾淨訊號下收斂

Phase 2 (10-15 epoch, learned predictor):
  凍結 strip encoder + detail Mamba + spatial Mamba
  加 routing predictor (Conv3→SiLU→Conv1→1, sigmoid)
  BCE loss vs Phase 1 的 GT mask
  → predictor 學到從 x_proj 預測哪些 cell 需要 detail

推論:
  routing_mask = predictor(x_proj) > 0.5
  只對 mask=1 的 cell 執行 strip read + Mamba
```

## 3. 和既有方案的比較

| | B1-H dense | 1024 unified | Strip routing |
|---|---|---|---|
| 新模組 | detail encoder + fusion (~30K) | 無 | stem + Mamba + predictor (~204K) |
| 每幀 overhead | +1.4ms (all cells) | +1.5-2× backbone | +0.02-0.2ms（<10% cells） |
| 小目標像素 | 原圖 → stride-4 feat | backbone 2× scale | 原圖 raw pixels |
| 語意品質 | learned shallow | COCO pretrained deep | raw RGB + learned strip |
| deploy 複雜度 | 加一條 encoder | 重編 TRT engine | 加 ~200K params |
| 實驗成本 | 高（全 pipeline 改動） | 低（config change） | 中（新模組 + 兩階段訓練） |

## 4. 未解決設計問題

1. **strip 長度**：沿原圖取多少像素？W=16 約對應 640 上的 ~4px（原圖 ~12px at 1080p），
   過大會讓 Mamba 序列太長，過小沒上下文。需實測 sweep [8, 16, 24, 32]。

2. **stem warm-start**：YOLO layer0 conv 是 2D 3×3 → 1D conv5 如何映射？取中心列+
   左右平均是近似，可能不如 random init。需 A/B 測試。

3. **獨立 vs 共用 Mamba block**：獨立 block 保證 detail 路徑的梯度不干擾 spatial 路徑
   （歸因清楚），共用 block 則讓兩路徑學到同一組 SSM dynamics（參數效率高、GPT-style routing）。
   先測獨立。

4. **四方向 aggregate**：last hidden mean 最簡單。也可用跨方向 attention 或 weighted sum。
   先測 mean。

5. **routing predictor 的 false positive cost**：預測器可能在不該開的位置開 strip，
   多算成本但 feature 可能有害（zero-init 的 residual 加回 x_proj）。
   可加 inference-time threshold sweep 找到最優 FP/TP trade-off。

## 5. 實驗判準

Phase 1（GT-driven）：

```
scripts/eval/mamba_size_binned_recall.py  on MOT17-02-SDP
  baseline: v14 replica GT2 (no detail)
  strip-routing: GT-masked P3 cells, Phase 1 30ep
  paired bootstrap, <8px bin, score 0.001/0.10/0.25
```

Phase 2（learned predictor）：

```
same recall + 完整 tracking (mot17.py --preset mamba_whole_graph)
  predictor mask vs GT mask: IoU, precision, recall per cell
  end-to-end IDF1/HOTA vs baseline
```

成功條件：
- Phase 1: routed cell recall > Phase 1 no-detail baseline by paired CI 排除 0
- Phase 2: predictor mask IoU > 0.7, end-to-end IDF1 ≥ Phase 1（不退化）
- Inference latency ≤ baseline + 0.5ms（routed cell <10%）

## 6. 不實作的原因（Phase 1 之前）

- 先證實 1024 unified 的方向（config change，零程式）是否已達收益上限。
  若 1024 的收益已超過 B1-H 的 +0.3pp，strip routing 的成本效益變弱。
- 若 1024 負向或中性，再投入 strip routing（程式改動 ~500 lines，兩階段訓練）。
- 順位：1024 unified → P2/P3/P4 架構驗證 → strip routing。
