# Saccade Pipeline Reference

這份文件定義兩件事：

1. 目前的高層 reference Dataflow
2. 各主模組貢獻的 sequential delta ledger 記帳規則

除非特別標註，所有模組 delta 都必須沿同一條 cumulative path 記錄。
每一列只能相對前一列新增一個主模組，最後一列必須回到該測試路徑的 baseline。

最後更新：2026-05-13

> **Baseline 更新（2026-05-10）**：P3 sweep 確認 `match=0.66, new_track=0.28` 為 yolo26s/m 共用最優參數（`per_frame_detection_cap` bug 修正後）。`--preset speed`（yolo26s）為推薦 baseline：**IDF1 51.2% / MOTA 40.8% / IDs 541 / Rcll 53.0%**。`--preset baseline`（yolo26m）適合 recall 優先場景：**IDF1 50.3% / MOTA 42.0% / Rcll 56.9%**。

---

## High-Level Dataflow

> 本圖反映 `evaluator.py` 中 `time_stage()` 的實際串行順序。
> ReID 分支（bank_sync → budget → crop → extract）**完全在 GMC 之前**執行，
> 不是舊版圖示的平行分支。

```text
[ Ingest ]                    [ON]
   DALI / NVDEC
   raw frame -> CHW float32
        |
        v
[ Preprocess ]                 [ON]
   letterbox / gamma / contrast
        |
        v
[ Detection ]                  [ON]
   yolo26s native_960          (--preset speed)
   yolo26m native_960          (--preset baseline)
   └── pose sidecar           [OFF, --pose-engine]
        |
        v
[ Postprocess ]                [ON]
   filter -> NMS -> cross-tile merge
   ├─ detection quality scaling
   ├─ FP hard filter           [OFF, --fp-hard-filter]
   ├─ per-frame det cap        [OFF, --per-frame-detection-cap]
   ├─ stage2 quality gate      [OFF, --stage2-quality-gate]
   ├─ consecutive birth gate   [OFF, --consecutive-birth-gate]
   ├─ birth quality gate       [OFF, --birth-quality-gate]
   └─ multi-birth manager      [OFF, --multi-birth]
        |
        v
[ ReID Bank Sync ]             [OFF]
   appearance bank → tracker    └─ --appearance-bank
        |
        v
[ ReID Budget ]                [OFF]
   budget selection             └─ --reid-mode
        |
        v
[ ReID Crop ]                  [OFF]
   ROI crop (Python only)       └─ --reid-mode
        |
        v
[ ReID Extract ]               [OFF]
   siglip2 / other backbones    └─ --reid-mode
        |
        v
[ Lazy ReID ]                  [OFF, profiling only]
   self-sim profiling           └─ --profile-lazy-reid
        |
        v
[ GMC ]                        [ON*]
   gpu / cpu                    └── --gmc
   └── --gmc-fg-mask            [OFF]
        |
        v
[ Tracker Update ]             [ON]
   association + Kalman         └─ (always ON)
        |
        v
[ Materialize ]                [ON]
   GPU -> host view             └─ (always ON)
        |
        v
[ BG Relink Wait ]             [OFF]
   wait for bg relink           └─ --pipeline-relink
        |
        v
[ Identity Resolve ]           [部分 ON]
   ├─ semantic relink           [OFF, --reid-mode semantic]
   ├─ appearance bank           [OFF, --appearance-bank]
   └─ lifecycle merge           [OFF, --lifecycle-merge]
        |
        v
[ Output ]                     [ON]
   metrics / MOT txt / debug    └─ (always ON)
```

### 模組 ON/OFF 狀態說明

| 模組 | CLI 預設 | 覆蓋開關 | Baseline 角色 |
|:-----|:--------:|:---------|:--------------|
| Ingest | ON | N/A | 無可選 |
| Preprocess | ON | N/A | 無可選 |
| Detection | ON | N/A | 無可選 |
| Postprocess | ON | N/A | 無可選 |
| **GMC** | **ON** | `--gmc` / `--no-gmc` | **推薦 baseline 核心** |
| ReID Bank Sync | **OFF** | `--appearance-bank` | 非 baseline |
| ReID Budget | **OFF** | `--reid-mode off/semantic/hybrid` | 非 baseline |
| ReID Crop | **OFF** | (via ReID) | 非 baseline |
| ReID Extract | **OFF** | `--reid-mode off/semantic/hybrid` | 非 baseline |
| Lazy ReID | **OFF** | `--profile-lazy-reid` | profiling only |
| GPU Tracker | ON | N/A | 無可選 |
| Materialize | ON | N/A | 無可選 |
| BG Relink Wait | **OFF** | `--pipeline-relink` | 非 baseline |
| Semantic Relink | **OFF** | `--reid-mode semantic` | 非 baseline（冗余） |
| Appearance Bank | **OFF** | `--appearance-bank` / `--no-appearance-bank` | 非 baseline |
| Lifecycle Merge | OFF | `--lifecycle-merge` / `--no-lifecycle-merge` | 非 baseline |
| FP Hard Filter | **OFF** | `--fp-hard-filter` | 非 baseline |
| Per-frame Det Cap | **OFF** | `--per-frame-detection-cap 0` | default 0（NO-GO） |
| Stage2 Quality Gate | **OFF** | `--stage2-quality-gate` | 非 baseline |
| Consecutive Birth Gate | **OFF** | `--consecutive-birth-gate` | 非 baseline |
| Birth Quality Gate | **OFF** | `--birth-quality-gate` | 非 baseline |
| Multi-birth Manager | **OFF** | `--multi-birth` | 非 baseline |

> \* GMC 在 CLI 默認為 `--gmc`（ON），但推薦 baseline 明確使用 `--gmc` 標誌。

### Current Default Path

> **CLI defaults 已更新為新 baseline（2026-05-10）。**

```text
# speed preset（yolo26s, IDF1 優先）：
uv run scripts/eval/mot17.py --preset speed --detector SDP

# speed preset, 640 profile（lower FP / higher FPS, lower recall）：
uv run scripts/eval/mot17.py --preset speed --detector SDP --tiling native_640

# baseline preset（yolo26m, Recall 優先）：
uv run scripts/eval/mot17.py --preset baseline --detector SDP

# 完整 pipeline（手動開啟 ReID）：
uv run scripts/eval/mot17.py --preset speed --detector SDP --reid-mode semantic --appearance-bank
```

---

## Engine 比較（同一 params: m=0.66, ntt=0.28, cap=0）

| Engine | Preset | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn | FPS |
|:-------|:-------|:-----|:-----|:----|:---|:---|:-----|:-----|:----|
| **yolo26s** | `speed` | **51.2%** | 40.8% | **541** | 13139 | 52753 | 53.0% | **81.9%** | **~110** |
| **yolo26m** | `baseline` | 50.3% | **42.0%** | 589 | 16112 | **48377** | **56.9%** | 79.9% | ~100 |

**選用原則**：
- **yolo26s**：優先 IDF1、IDs、FP、FPS（identity tracking 精度導向）
- **yolo26m**：優先 MOTA、Rcll、FN（漏偵最小化導向）

### Resolution Profile Note

`--tiling native_640` 是一個明確的保守 profile：

- `FP` 顯著下降
- `FPS` 顯著上升
- `Recall / IDF1 / MOTA` 明顯下降

因此它適合：

- 做 `FP` 問題診斷
- 當低-FP / 高-FPS 的可選 profile

不適合：

- 取代 `speed preset` 成為新的 default

---

## Full-Length Baseline

### P3 確定 Baseline（2026-05-10）

| Profile | Engine | match | ntt | IDF1 | MOTA | IDs | FP | FN | Rcll | FPS |
|:--------|:-------|:------|:----|:-----|:-----|:----|:---|:---|:-----|:----|
| `--preset speed` | yolo26s | 0.66 | 0.28 | `51.2` | `40.8` | `541` | `13139` | `52753` | `53.0%` | `110` |
| `--preset baseline` | yolo26m | 0.66 | 0.28 | `50.3` | `42.0` | `589` | `16112` | `48377` | `56.9%` | `100` |

### P3 Sweep 結果（yolo26s, cap=0）

| match | ntt | IDF1 | MOTA | IDs | FP | FN | Rcll |
|:------|:----|:-----|:-----|:----|:---|:---|:-----|
| 0.72 | 0.35 | 48.8% | 40.6% | 572 | 11044 | 55076 | 51.0% |
| 0.70 | 0.30 | 50.3% | 40.3% | 537 | 13432 | 53122 | 52.7% |
| 0.68 | 0.30 | 50.8% | 40.4% | 528 | 13324 | 53123 | 52.7% |
| 0.66 | 0.30 | 51.0% | 40.6% | 526 | 12965 | 53197 | 52.6% |
| **0.66** | **0.28** | **51.2%** | **40.8%** | 541 | 13139 | 52753 | **53.0%** |
| 0.66 | 0.25 | 50.9% | 40.0% | 571 | 14547 | 52300 | 53.4% |
| 0.64 | 0.30 | 50.9% | 41.0% | 505 | 11942 | 53762 | 52.1% |

**觀察**：
- `match↓`：IDs 大幅下降（association 更穩定），FP 幾乎不變
- `ntt↓`：Rcll 上升，FP 上升（新 track birth 增加）
- `m=0.66, ntt=0.28` 為 IDF1、MOTA、Rcll 三者平衡的 Pareto 最優點

---

## Main Module Delta Ledger

逐步累積的模組貢獻記帳。每個 step 相對前一 step 只新增一個模組，`Δprev` 即該模組的單獨裸增益。

重跑命令：

```bash
uv run python scripts/eval/pipeline_contribution.py --detector SDP
```

> 僅使用 2 序列（MOT17-04-SDP + MOT17-10-SDP）。絕對數值與 7 序列平均不可直接比較。

### Sequential Ledger（P3 baseline, m=0.66, ntt=0.28, fsw=0.4, yolo26s，2026-05-11）

| Step | Module | Profile | IDF1 | MOTA | IDs | FP | FN | FPS | Δprev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | *(bare tracker)* | `tracker_core` | `51.6` | `43.1` | `279` | `6726` | `27348` | `106.14` | start |
| 1 | **GPU GMC** | `tracker_core_gmc` | `54.4` | `44.7` | `146` | `6272` | `26998` | `99.29` | **IDF1 +2.8pp, IDs −133, FPS −6.9** |
| 2 | semantic relink | `semantic_core` | `54.8` | `44.7` | `146` | `6448` | `26826` | `97.05` | IDF1 +0.4pp, IDs ±0, FPS −2.2 |
| 3 | appearance bank + inject | `semantic_bank` | `54.8` | `44.4` | `145` | `6434` | `26989` | `79.71` | IDF1 ±0.0pp, IDs −1, **FPS −17.3** |
| 4 | async_reid + pipeline_relink + thr=0.93 | `full_default` | `54.8` | `44.6` | `145` | `6345` | `26991` | `76.08` | IDF1 ±0.0pp, IDs ±0, FPS −3.6 |

**關鍵結論（P3 baseline）**：

- **GMC 是 pipeline 中唯一顯著貢獻的模組**：IDF1 +2.8pp、IDs −133（比舊 baseline +1.8pp 更大）
- **semantic_core**：+0.4pp IDF1、FPS −2.2。邊際正益，cost 低。
- **semantic_bank**：IDF1 ±0、FPS **−17.3**。零增益高代價，不應設為 default ON。
- **full_default（async + relink）**：FPS 部分回復（+3.6 FPS），IDF1 不變。
- 完整 ReID stack（tracker_core_gmc → full_default）：IDF1 +0.4pp，FPS **−23.2**。代價遠高於收益。

### 舊 baseline 歷史對照（m=0.72, ntt=0.35, yolo26s）

| Step | Module | Profile | IDF1 | MOTA | IDs | FPS | Δprev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | *(bare tracker)* | `tracker_core` | `47.0` | `40.0` | `724` | `115.3` | start |
| 1 | GPU GMC | `tracker_core_gmc` | `48.8` | `40.6` | `570` | `112.0` | IDF1 +1.8pp, IDs −154, FPS −3.3 |
| 2 | semantic relink | `semantic_core` | `48.9` | `40.6` | `574` | `107.0` | IDF1 +0.1pp, FPS −5.0 |
| 3 | appearance bank | `semantic_bank` | `48.8` | `40.6` | `571` | `105.8` | IDF1 −0.1pp, FPS −1.2 |
| 4 | async + relink | `full_default` | `48.8` | `40.6` | `564` | `107.4` | IDF1 ±0, FPS +1.6 |

---

## Semantic Relink 診斷報告 (2026-05-08)

> 對 GMC baseline 執行 7-seq semantic relink 參數 sweep，結論：**semantic relink 在 GMC 開啟下基本是冗余的**。

| Config | IDF1 | MOTA | IDs | 總 accepts | 結論 |
|--------|------|------|-----|-----------|------|
| GMC baseline (`reid-mode off`) | 48.8% | 40.7% | 574 | 0 | 純 GMC，無 relink |
| `--semantic-threshold 0.91` | 48.9% | 40.6% | 576 | 5 | IDs +2（可能是 noise） |
| `--semantic-threshold 0.85` | 48.8% | 40.6% | 586 | 13 | **IDs +12，false relinks 更嚴重** |
| `--semantic-threshold 0.95` | 48.8% | 40.6% | 569 | 1 | 更保守，IDs 微降 |
| `0.85 + ttl=60` | 48.7% | 40.6% | 585 | 16 | **最差**，IDs +11 |

**Reject 分布（9,262 total attempts）**：

| Reject 原因 | 次數 | 佔比 | 說明 |
|:-----------|-----:|-----:|:-----|
| `reject_age` | 8,040 | **86.8%** | GMC 下 lost track 只有兩種命運：(<2 frames 被 tracker 收回，或 >45 frames 超 TTL) |
| `reject_spatial` | 977 | 10.5% | center_norm > 0.20 OR iou < 0.20 |
| `reject_assigned` | 3,574 | 38.6%* | 已被其他 detection 匹配 |
| `reject_similarity` | 141 | 1.5% | cosine < threshold |

**結論**：GMC 的 motion compensation 消除了 semantic relink 的存在意義。降低 threshold 不會帶來淨收益。未來應專注於 **GMC 之外** 的改進方向。

---

## Latency Profile（2026-05-11，MOT17-04-SDP）

### 測量方法

```bash
uv run python scripts/eval/mot17.py \
  --profile-stages --latency-only \
  --sequences MOT17-04-SDP \
  --max-frames 150 --warmup-frames 50 \
  --output runs/latency_<tag>
```

事後比較：

```bash
python scripts/eval/latency_report.py runs/latency_m/ --compare runs/latency_s/
```

> MOT17-04-SDP 是擁擠長廊場景，raw_boxes 每帧固定打滿 topk=300 上限，為最重壓力。全 7 序列平均 FPS 會更高。

### Stage Breakdown（Reid mode=off 條件）

> 以下為 `reid_mode=off` 下的 baseline 測量（ReID 階段皆為 0ms）。
> 若開啟 ReID，需額外加上 `reid_bank_sync` + `reid_budget` + `reid_crop` + `reid_extract`。

| Stage | yolo26m | yolo26s | Delta | 占比（m/s） |
|:------|--------:|--------:|------:|------------:|
| **detect** | 6.73ms | 5.06ms | −1.67ms (−24.8%) | 48.5% / 41.8% |
| **postprocess** | 3.46ms | 3.16ms | −0.30ms (−8.8%) | 25.0% / 26.1% |
| ingest_preprocess | 0.94ms | 0.82ms | −0.12ms | 6.8% |
| gmc | 0.87ms | 0.83ms | −0.04ms | 6.3% / 6.9% |
| track | 0.78ms | 0.78ms | ±0ms | 5.6% / 6.5% |
| relink_write | 0.55ms | 0.87ms* | +0.32ms | 3.9% / 7.2% |
| fetch | 0.48ms | 0.40ms | −0.08ms | 3.5% |
| materialize | 0.38ms | 0.37ms | −0.01ms | 2.7% |
| **frame_total** | **13.88ms** | **12.10ms** | **−1.77ms (−12.8%)** | — |
| **FPS** | **72.07** | **82.62** | — | — |

\* yolo26s `relink_write` std=4.18ms（mean > p95=0.62ms）：pipeline_relink 背景線程偶發 burst，p95 以下正常。

### GMC Sub-Stage（兩引擎相近）

| Sub-Stage | mean | std | p95 |
|:----------|-----:|----:|----:|
| gmc_gray_downscale | 0.08ms | — | 0.12ms |
| gmc_phase_corr | 0.22ms | — | 0.38ms |
| gmc_handoff | 0.10ms | — | 0.31ms |
| gmc_fg_mask | 0ms | — | — |

`gmc_fg_mask` 恆為 0ms（未啟用前景遮罩路徑）。

### P95 Jitter

| Stage | yolo26m p95 | yolo26s p95 |
|:------|------------:|------------:|
| detect | 8.40ms | 6.24ms |
| postprocess | 5.15ms | 4.00ms |
| gmc | 1.31ms | 1.19ms |
| frame_total | 16.33ms | 14.00ms |

### 瓶頸小結

- `detect` + `postprocess` 合計佔 **73.5%（26m）/ 67.9%（26s）**，是唯一可觀優化空間
- `postprocess` 3.16ms 主因是 raw_boxes=300 的全量 NMS；一般序列（raw_boxes<200）時間會下降
- `track`、`gmc`、`materialize` 皆與引擎無關，且 track 與引擎完全一致（0.78ms）
- ReID 全 0ms：測量條件為 `reid_mode=off`，ReID 階段（bank_sync/budget/crop/extract）不執行

---

## Known Issues / No-Go 結論

| 項目 | 結論 | 詳細 |
|:-----|:-----|:-----|
| `per_frame_detection_cap` | **NO-GO**（default 0） | dense scene（n>50）下 adaptive cap 壓至 ~21，破壞 recall |
| `adaptive_detection_cap` | **NO-GO** | 「密集 = FP 多」的假設在 MOT17 中相反（密集 = 真實的人多） |
| P5-2 Stage2 QualityGate | NO-GO | IDF1/MOTA 統計中性，code 已實作，default OFF |
| P5-3 ConsecutiveBirthGate | NO-GO | 統計中性，有速度門控。code 已實作，default OFF |
| Tiled detection | NO-GO | FP ~8000（native_960 的 2 倍），truncation 與 score 污染是根本問題 |
| Semantic relink（GMC ON）| NO-GO | GMC 消除了 relink 需處理的場景 |
| NSA-Kalman | NO-GO | 無效果 |
| PostMerge | NO-GO | 確認有害 |
| match ≥ 0.73 slow path | 未解明 | 需 CUDA profiler（nsys/cupti）調查 |
