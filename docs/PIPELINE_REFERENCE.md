# Saccade Pipeline Reference

這份文件定義兩件事：

1. 目前的高層 reference Dataflow
2. 各主模組貢獻的 sequential delta ledger 記帳規則

除非特別標註，所有模組 delta 都必須沿同一條 cumulative path 記錄。
每一列只能相對前一列新增一個主模組，最後一列必須回到該測試路徑的 baseline。

最後更新：2026-05-10

> **Baseline 更新（2026-05-10）**：P3 sweep 確認 `match=0.66, new_track=0.28` 為 yolo26s/m 共用最優參數（`per_frame_detection_cap` bug 修正後）。`--preset speed`（yolo26s）為推薦 baseline：**IDF1 51.2% / MOTA 40.8% / IDs 541 / Rcll 53.0%**。`--preset baseline`（yolo26m）適合 recall 優先場景：**IDF1 50.3% / MOTA 42.0% / Rcll 56.9%**。

---

## High-Level Dataflow

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
   filter -> NMS -> cross-tile merge -> quality scaling
        |
        +-----------------------------+
        |                             |
        v                             v
[ GMC ]                        [ON*]
   gpu / cpu                    ReID Trigger      [OFF]
   └── --gmc                    budget / policy
        |                             |
        +-------------+---------------+
                      |
                      v
                [ ReID Extract ]       [OFF]
                siglip2 / other backbones
                └── --reid-mode off
                      |
                      v
                [ GPU Tracker ]          [ON]
                association + Kalman update
                      |
                      v
                [ Materialize ]          [ON]
                GPU -> host view
                      |
                      v
                [ Identity Resolve ]     [部分 ON]
                ├─ semantic relink       [OFF]
                ├─ appearance bank       [OFF]
                └─ lifecycle merge      [OFF]
                      |
                      v
                [ Output ]               [ON]
                metrics / MOT txt / debug artifacts
```

### 模組 ON/OFF 狀態說明

| 模組 | CLI 預設 | 覆蓋開關 | Baseline 角色 |
|:-----|:--------:|:---------|:--------------|
| Ingest | ON | N/A | 無可選 |
| Preprocess | ON | N/A | 無可選 |
| Detection | ON | N/A | 無可選 |
| Postprocess | ON | N/A | 無可選 |
| **GMC** | **ON** | `--gmc` / `--no-gmc` | **推薦 baseline 核心** |
| ReID Extract | **OFF** | `--reid-mode off/semantic/hybrid` | 非 baseline |
| GPU Tracker | ON | N/A | 無可選 |
| Materialize | ON | N/A | 無可選 |
| Semantic Relink | **OFF** | `--reid-mode semantic` | 非 baseline（冗余） |
| Appearance Bank | **OFF** | `--appearance-bank` / `--no-appearance-bank` | 非 baseline |
| Lifecycle Merge | OFF | `--lifecycle-merge` / `--no-lifecycle-merge` | 非 baseline |
| Per-frame Det Cap | **OFF** | `--per-frame-detection-cap 0` | default 0（dense scene で recall を破壊するため） |

> \* GMC 在 CLI 默認為 `--gmc`（ON），但推薦 baseline 明確使用 `--gmc` 標誌。

### Current Default Path

> **CLI defaults 已更新為新 baseline（2026-05-10）。**

```text
# speed preset（yolo26s, IDF1 優先）：
uv run scripts/eval/mot17.py --preset speed --detector SDP

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

**使い分け**：
- **yolo26s**：IDF1・IDs・FP・FPS 優先（identity tracking 精度重視）
- **yolo26m**：MOTA・Rcll・FN 優先（見逃し最小化重視）

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

**観察**：
- `match↓`：IDs 激減（association が安定）、FP ほぼ不変
- `ntt↓`：Rcll ↑、FP ↑（新トラック birth 増加）
- `m=0.66, ntt=0.28` が IDF1・MOTA・Rcll のバランス Pareto 最優

---

## Main Module Delta Ledger

これはモジュール追加による cumulative delta の記帳。P3 baseline（m=0.66, ntt=0.28, yolo26s）を Step 0 として更新が必要。

> 現在の ledger は旧 baseline（m=0.72, ntt=0.35）基準。P3 baseline での再計測は未実施。

重跑命令：

```bash
uv run python scripts/eval/pipeline_contribution.py --detector SDP
```

### Sequential Ledger（旧 baseline 参照, m=0.72, ntt=0.35, yolo26s）

| Step | Newly Enabled Module | Cumulative Profile | IDF1 | MOTA | IDs | FP | FN | FPS | Δprev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | *(baseline)* | `tracker_core_gmc` | `48.8` | `40.6` | `570` | `11019` | `55100` | `112.0` | base |
| 1 | semantic relink core | `semantic_core` | `48.9` | `40.6` | `574` | `11050` | `55093` | `107.0` | `IDF1 +0.1pp`, `IDs +4`, `FPS -5.0` |
| 2 | appearance bank + bank inject | `semantic_bank` | `48.8` | `40.6` | `571` | `11047` | `55075` | `105.8` | `IDF1 -0.1pp`, `IDs -3`, `FPS -1.2` |
| 3 | async_reid + pipeline_relink + `semantic-threshold 0.93` | `full_default` | `48.8` | `40.6` | `564` | `11072` | `55074` | `107.4` | `IDF1 +0.0pp`, `IDs -7`, `FPS +1.6` |

**關鍵結論**：GMC が pipeline 中唯一の顕著な貢献モジュール（IDF1 +1.8pp, IDs -154）。後続の semantic / bank / async の総和は ~0.1pp IDF1 のみ。

### 舊 baseline 歷史證據

| Step | Newly Enabled Module | Cumulative Profile | IDF1 | MOTA | IDs | FP | FN | FPS | Δprev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | *(bare baseline)* | `tracker_core` | `47.0` | `40.0` | `724` | `11097` | `55606` | `115.3` | start |
| 1 | GPU GMC | `tracker_core_gmc` | `48.8` | `40.6` | `570` | `11019` | `55100` | `112.0` | `IDF1 +1.8pp`, `IDs -154`, `FPS -3.3` |

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

**結論**：GMC の motion compensation が semantic relink の存在意義を消している。降低 threshold 不會帶來淨收益。未來應專注於 **GMC 之外** 的改進方向。

---

## Known Issues / No-Go 結論

| 項目 | 結論 | 詳細 |
|:-----|:-----|:-----|
| `per_frame_detection_cap` | **NO-GO**（default 0） | dense scene（n>50）で adaptive cap が ~21 まで下降、recall 破壊 |
| `adaptive_detection_cap` | **NO-GO** | 「密集 = FP 多い」の仮定が MOT17 では逆（密集 = 本物の人が多い） |
| P5-2 Stage2 QualityGate | NO-GO | IDF1/MOTA 統計中性、code は実装済み・default OFF |
| P5-3 ConsecutiveBirthGate | NO-GO | 統計中性、速度門控あり。code 実装済み・default OFF |
| Tiled detection | NO-GO | FP ~8000（native_960 の 2×）、truncation・score 汚染が根本問題 |
| Semantic relink（GMC 開）| NO-GO | GMC が relink の基礎シーンを消している |
| NSA-Kalman | NO-GO | 効果なし |
| PostMerge | NO-GO | 有害確認 |
| match ≥ 0.73 slow path | 未解明 | CUDA profiler（nsys/cupti）要調査 |
