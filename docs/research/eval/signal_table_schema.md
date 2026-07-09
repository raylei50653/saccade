<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->

# Signal Table Schema（U × M × D × Pipeline）

**Purpose:** 規範訊號層分析的 **class / 表 / 資料來源 / pipeline 層級先後**，讓多處理對比、分層 AUC、集合交集可重現、可收斂。  
**Code owner:** [`src/saccade/perception/eval/signal_tables.py`](../../../src/saccade/perception/eval/signal_tables.py)  
**Runtime path map:** [../pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md](../pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md)  
**Not this doc:** 端到端 IDF1 調參協議、tracker decision contract（見 tracker-decision）。

---

## 0. 原則

| 規則 | 說明 |
|:--|:--|
| **一宇宙一表** | `U_det` / `U_gt` / `U_cand` / **`U_relink_pair`** / `U_err` 不混在同一 parquet |
| **禁止混宇宙** | `U_cand` = 幀內 (track,det)；`U_relink_pair` = offline (lost→cand)；**AUC 不可橫比** |
| **y 只寫一次** | 標籤由 `StudyMeta` 定義並固化進表；分析腳本不得各自 greedy 出不同 TP |
| **維度存連續值** | `height`/`vis` 等原始量；bin 在分析時 `cut` |
| **處理用 bool 或 method 列** | 寬表 `accept_*` 算交集；長表 `method` 做 groupby |
| **method 綁定 pipeline 終端節點** | 每個 `MethodId` 有 `order` 與 parents；累積研究必須按序 |
| **relink AUC 必分池** | 含 `U_relink_pair` 時 meta 必填 `hard_pool_rule`；**全池+難池 AUC + base rate** 才可引用 |
| **meta 必伴隨** | 每個 study 目錄必有 `meta.json`（見 §3）；含 `cut_design` |

分析語言對照：

- **Pipeline 節點** → 層級 / 先後 / 依賴（本節 §P）  
- **處理 M** → 在某節點**切斷**的觀測；集合交集 / 工作點 P·R  
- **維度 D** → 分層子宇宙 / 條件表  
- **AUC** → 固定 `(universe, y, score_field)`；relink 必報 full+hard + n_pos/n_neg  

### 0.1 雙問題線 recipe（A / B1 / B2）

| Study | 問題 | Substrate | 主表 | 權威腳本 | 必報 |
|:--|:--|:--|:--|:--|:--|
| **A Recall** | 誰進系統？ | det / score | `U_det` / `U_gt` | `scripts/eval/analyze_score_distribution.py` | thr 曲線 P/R；height bins |
| **B1 IDs-signal** | 特徵能否排序真 relink？ | **relink-off + interp-off** MOT | `U_relink_pair` | `build_relink_candidates.py` → **`summarize_relink_pairs.py`** | **full+hard AUC + base rate + thr**（見輸出契約） |
| **B2 IDs-state** | 斷–接是否保住 pred id？ | online 整輪（可 ablate bridge） | events / reconnect | `scripts/eval/diagnostics/reconnect_rate.py` | success@gap；IDs/AssA |

- B1 **不**取代 B2；B1 AUC **不是** e2e IDF1。  
- B1 方法敘事以 [offline_relink_candidate_analysis.md](../../modules/semantic/research/offline_relink_candidate_analysis.md) 為準（**s 歷史數字 as-of 該文日期**；m 主線數字只進 study 目錄）。  
- 預設 hard pool = `bridge_dist<=1.0`。Code：`STUDY_SCRIPT_MAP` / `B1_OUTPUT_FILES`。

### 0.2 B1 輸出契約（分布背景 + 腳本產物）

**目的：** 數字會過時；**當次分布背景**與 **AUC/thr** 必須落在 study 目錄，narrative 只連 path，不嵌 master 表。

```text
out/signal_study/<study_id>/
  meta.json           # StudyMeta（可選但建議；含 hard_pool_rule）
  context.json        # 當次分布背景（必）
  metrics_auc.json    # full + hard AUC（必）
  metrics_thr.csv     # 工作點 thr 表（必）
  pairs.csv           # 可選：builder 輸出副本
```

常數：`CONTEXT_FILENAME` / `METRICS_AUC_FILENAME` / `METRICS_THR_FILENAME` / `DEFAULT_RELINK_THR_GRID`。

**`context.json` 必填區塊：**

| Block | 內容 |
|:--|:--|
| identity | `study_id`, `created_utc`, `commit`, `preset`, `detector` |
| substrate | `mot_dir`, relink/interp flags, `double_buffer`, notes |
| input | `pairs_csv`, `n_rows_raw`, `score_field` |
| pool.full / pool.hard | `n`, `n_pos`, `n_neg`, `base_rate`；hard 含 `hard_pool_rule` |
| score_dist | pos/neg：`median`, `p05`, `p50`, `p95` |
| gap_bins | 1-10 / 11-30 / 31-60 / 61-150 / 151-300 的 n（與 pos n） |
| e2e（可選） | substrate OVERALL 一行 |

**`metrics_thr.csv` 欄位：** `pool,threshold,tp,fp,fn,precision,recall,f1`  
預設 thr：`0.15, 0.30, 0.50, 1.00`（與 offline_relink §3 對齊）。

**腳本：**

```bash
uv run python scripts/tools/build_relink_candidates.py \
  --mot-dir <substrate> --out out/signal_study/<id>/pairs
uv run python scripts/tools/summarize_relink_pairs.py \
  --pairs out/signal_study/<id>/pairs.csv \
  --study-dir out/signal_study/<id> \
  --hard-dist 1.0
```

**過時策略：** 重測 → **新 `study_id`/stamp**；勿改歷史 markdown 內嵌表。s 文方法仍可引用；m 分數以最新 study_dir 為準。

---

## P. Pipeline 結構（層級 · 先後 · 關聯）

### P.1 邏輯圖（frame N 的資料依賴）

Double-buffer 只重疊 **wall-clock**（detect N+1 ‖ track N），**不**改變 frame N 的因果序。

```text
fetch → ingest_preprocess → detect → post_filter → post_nms → [post_merge?]
  │                                              ↓
  └──────────────────────→ gmc ────────→ track → bridge_relink → materialize
                                                    ↓
                                            interpolate → metrics

  (ReID branch off on headline m: post_nms ↛ reid)
```

Code：`PIPELINE_NODES`、`pipeline_ascii_art()`、`pipeline_summary_rows()`。

### P.2 `PipelineLayer`（分析用粗層，≠ CUDA graph L1/L2/L3）

| Layer | 含義 | 主宇宙 | 主問題 |
|:--|:--|:--|:--|
| `L0_ingest` | decode / resize | — | 域是否一致 |
| `L1_detect` | raw boxes+scores | `U_det` / `U_gt` | 偵測訊號、score 分佈 |
| `L2_post` | filter / NMS / merge | `U_det` | 誰被殺、TP/FP 分佈 |
| `L3_motion` | GMC | （進 track） | 相機動殘差 |
| `L4_assoc` | Kalman + auction + birth | `U_cand` | 關聯可分性 |
| `L5_identity` | bridge（ReID off） | `U_err` | 斷鏈修復 vs 假 merge |
| `L6_emit` | materialize | MOT rows | 幀輸出 |
| `L7_postseq` | interpolate | `U_err` / metrics | 填洞 FP |
| `L8_metrics` | motmetrics / HOTA | `method_metrics` | 聚合分數 |

### P.3 節點表（`PipelineNode`）

| order | node_id | layer | parents | MethodId（終端） | headline m |
|------:|:--|:--|:--|:--|:--|
| 10 | `fetch` | L0 | — | — | ON |
| 20 | `ingest_preprocess` | L0 | fetch | — | ON |
| 30 | `detect` | L1 | ingest | `raw` | ON |
| 40 | `post_filter` | L2 | detect | `post_filter` | ON |
| 50 | `post_nms` | L2 | post_filter | `post_nms` | ON |
| 55 | `post_merge` | L2 | post_nms | `post_merge` | optional/off |
| 60 | `reid` | L5 | post_nms | — | **OFF** |
| 70 | `gmc` | L3 | fetch | （並入 bare_gmc） | ON |
| 80 | `track` | L4 | post_nms,**gmc** | `bare_track` / `bare_gmc` | ON |
| 90 | `bridge_relink` | L5 | track | `bare_bridge` | ON |
| 100 | `materialize` | L6 | track, bridge | `full_no_interp` | ON |
| 110 | `interpolate` | L7 | materialize | `full_preset` | ON |
| 120 | `metrics` | L8 | interpolate | — | ON |

**GMC 是 sibling join，不是 NMS 的 child。**  
`bare_track` = track 且 **關 GMC**；`bare_gmc` = track 且 **開 GMC**（同終端節點 `track`，語義不同，meta.notes / method_id 區分）。

### P.4 MethodId → 終端節點

| MethodId | terminal node | 含義（切斷點） |
|:--|:--|:--|
| `raw` | detect | 偵測原料 |
| `post_filter` | post_filter | +分數過濾 |
| `post_nms` | post_nms | +NMS |
| `post_merge` | post_merge | +merge（可選） |
| `bare_track` | track | 關聯，GMC off |
| `bare_gmc` | track | 關聯，GMC on |
| `bare_bridge` | bridge_relink | +bridge |
| `full_no_interp` | materialize | 全鏈無插值 |
| `full_preset` | interpolate | 正式 preset 輸出 |

`METHOD_TERMINAL_NODE` / `method_terminal_node()` / `pipeline_layer_of_method()`。

### P.5 `CutDesign`：多 method 怎麼收

| cut_design | 規則 | Δ 怎麼讀 |
|:--|:--|:--|
| `single` | 只有一個 method | 無 Δ |
| `cumulative` | method 按 pipeline order 遞增 | Δprev = 多開的那一節點 raw uplift |
| `single_on_base` | 皆 = 同一 base + 單模組 | Δbase = 該模組單獨貢獻（可負） |
| `orthogonal` | 獨立對照，**禁止**當因果加總 | 只報各 absolute |
| `custom` | 自訂；notes 必填 | 自訂 |

累積時 `validate_method_order(method_ids)` 強制 order 不回退。

**預設 spine（headline m）：**

```text
detect → post_filter → post_nms → track(bare_gmc) → bridge_relink → interpolate → metrics
```

`DEFAULT_CUMULATIVE_SPINE`。建議 method 序列：

```text
raw → post_nms → bare_gmc → bare_bridge → full_no_interp → full_preset
```

（`post_filter` 可插入；L0 若已用 score_dist 可跳過 raw 細節。）

### P.6 報告怎麼「收」

固定欄位順序，避免層級亂：

1. **Layer 表**：每層一列 — primary universe、claimed_signal、是否量完  
2. **Cumulative metrics**：`metrics_by_method.csv` 按 `order` 排，報 ΔFP/ΔFN/ΔIDs  
3. **集合差**：僅相鄰 cumulative 對（或 base vs single_on_base）  
4. **分層 AUC**：只在該層 primary universe 上做  
5. **結論句模板**：`L{k} 的訊號是 X；傳到 L{k+1} 時被 Y 稀釋/放大`

禁止：把 `bare_track` 與 `full_preset` 的 AUC 直接比大小當「哪個模組更好」而不寫 cut_design。

### P.7 Profile stage 對齊

Evaluator 計時名見 [mot17_default_config §4](../../reference/mot17_default_config.md)；本圖 `profile_stages` 已掛上。`frame_total` 是總計，不是因果節點。

### P.8 連續掃參 / 區間統計（Sweep）

**問題：** 實務幾乎都是掃一個區間（score 門檻、`match_thresh`、`kalman_r_scale`、bridge px…），不是單點 MethodId。  
**解法：** `cut_design=sweep` + `SweepAxis` + `metrics_by_run.csv`（一點一列）。

| 模式 `sweep_mode` | 含義 | 典型 |
|:--|:--|:--|
| `offline` | **同一 dump 上**改 gate/threshold，不重跑 tracker | score 門檻、IoU gate、cost cap |
| `online` | 每個格點完整/半鏈 re-run | `kalman_r_scale`、bridge 幾何 |
| `mixed` | 軸各自 offline/online | meta.notes 寫清 |

**`SweepAxis` 欄位：** `name`, `node_id`（作用在哪個 pipeline 節點）, `kind`∈{manual,linspace,arange,logspace}, `values` 或 `lo/hi/num/step`, `offline` + `offline_universe`/`offline_score_col`。

**表：**

```text
metrics_by_run.csv   # run_id, method, param_<axis>, idf1/mota/... 或 precision/recall/f1
sweep_curve_summary.csv  # 可選：每軸 best / endpoints / y_span
```

**操作（code）：**

| 函式 | 用途 |
|:--|:--|
| `expand_sweep_grid(axes)` | 笛卡爾積 → 參數點列表 |
| `make_run_id(method, params)` | 穩定 run_id |
| `offline_threshold_curve(scores, y, thresholds)` | 凍結標籤上掃門檻 → TP/FP/P/R/F1 |
| `summarize_sweep_metric(rows, axis, metric)` | 曲線統計：best、端點、span、Δ |

**約束：**

1. 一次 sweep **固定** `sweep_base_method`（pipeline 切斷點不變，只動軸）  
2. 每軸 ≥2 點才算「區間」  
3. offline 曲線**不是** e2e IDF1；要 IDF1 必須 online re-run  
4. 多軸時先 1D 主軸 + 其餘固定；全笛卡爾只在點數可控時用  
5. 報告：曲線圖 + `x_best`/`y_best` + 與 baseline 點對照；Δ ≲ noise 當平區  

**與 cumulative 分工：** cumulative 換的是**節點**；sweep 換的是**同節點上的連續旋鈕**。不要混在同一 `method_ids` 序裡裝成 cumulative。

---

## 1. Study 目錄佈局

```text
out/<study_name>/
  meta.json                 # StudyMeta（含 cut_design / method_ids / sweep_axes / hard_pool_rule）
  u_gt.parquet              # optional: one row per GT
  u_det.parquet             # one row per detection (or per stage-det)
  u_cand.parquet            # frame-level (track, det) only
  u_relink_pair.parquet     # offline lost→cand pairs (B1)
  u_err.parquet             # one row per error event
  method_accept_wide.parquet  # optional: same keys + accept_* columns
  metrics_by_method.csv     # 離散 method 切斷（無軸）
  metrics_by_run.csv        # 掃參：一格點一列（param_* + metrics）
  sweep_curve_summary.csv   # 可選：區間彙總
  figures/
```

預設 root 建議：`out/signal_study/` 或 `results/signal_study/<stamp>/`（研究產物，不進 headline preset）。

---

## 2. Class 總覽

| Class / Enum | 角色 |
|:--|:--|
| `UniverseId` | 樣本宇宙（含 **`U_relink_pair`**） |
| `MethodId` | 處理切斷點（綁 terminal node） |
| `PipelineLayer` | 粗層 L0–L8 |
| `PipelineNode` | 節點：order、parents、signal、universe |
| `CutDesign` | 多 method 實驗設計（含 `sweep`） |
| `SweepAxis` / `SweepMode` / `SweepGridKind` | 連續/網格掃參軸 |
| `DataSourceId` | 原始資料來源 |
| `StudyMeta` | study 契約（pipeline + sweep + **hard_pool_rule**） |
| `GtRow` / `DetRow` / `CandRow` / **`RelinkPairRow`** / `ErrRow` | 列型別 |
| `MethodMetricsRow` | 端到端 count（無軸） |
| `RunMetricsRow` | 掃參格點 metrics |
| `SetCompareResult` | 兩 method 集合對比 |
| `auc_full_and_hard_pool` | B1 合規 AUC helper |

Python：`from saccade.perception.eval.signal_tables import ...`

---

## 3. `StudyMeta`（`meta.json`）

| 欄位 | 型別 | 必填 | 說明 |
|:--|:--|:--|:--|
| `study_id` | str | ✓ | 目錄名或 stamp |
| `created_utc` | str ISO | ✓ | |
| `commit` | str | ✓ | `git rev-parse --short HEAD` |
| `preset` | str | ✓ | 如 `mamba_whole_graph_m` |
| `detector` | str | ✓ | 通常 `SDP` |
| `double_buffer` | bool | ✓ | |
| `host` | str | | 可選 |
| `iou_match` | float | ✓ | 預設 **0.5**（與 motmetrics/TrackEval 一致） |
| `universes` | list[str] | ✓ | 本 study 含哪些表 |
| `y_definitions` | dict | ✓ | 各宇宙標籤定義字串 |
| `score_fields` | dict | | 各宇宙預設 AUC 分數欄 |
| `method_ids` | list[str] | | 本 study 出現的 MethodId（cumulative 時須 pipeline 序） |
| `cut_design` | str | | `single` / `cumulative` / `single_on_base` / `orthogonal` / `sweep` / `custom` |
| `pipeline_profile` | str | | 預設 `headline_m_whole_graph` |
| `method_terminal_nodes` | dict | | 覆寫 Method→node；`custom` 建議填 |
| `cumulative_spine` | list[str] | | 累積節點序；空則用 `DEFAULT_CUMULATIVE_SPINE` |
| `sweep_mode` | str | sweep 時 | `offline` / `online` / `mixed` |
| `sweep_base_method` | str | sweep 時 | 固定切斷點 MethodId |
| `sweep_axes` | list[dict] | sweep 時 | `SweepAxis` JSON 列表 |
| `hard_pool_rule` | str | **含 U_relink_pair 時必填** | 如 `bridge_dist<=1.0`；禁止只報全池 AUC |
| `report_base_rate` | bool | B1 建議 true | n_pos/n_neg 與 AUC 同報 |
| `study_line` | str | | `A` / `B1` / `B2`（可選標記） |
| `notes` | str | | |

`y_definitions` 範例：

```json
{
  "U_gt": "matched if some det IoU>=iou_match under score-greedy claim",
  "U_det": "is_tp if claimed a GT with IoU>=iou_match; else FP",
  "U_cand": "is_correct if det is the GT-matched target of this track at frame",
  "U_relink_pair": "gt_match==1 if lost and cand map to the same GT id under builder rules",
  "U_err": "event_type in {fn_miss, id_switch, fp_birth, fp_interp, fragment}"
}
```

---

## 4. 宇宙表 schema

### 4.1 `U_gt` — `GtRow`（一列一個 GT 框）

| 欄位 | 型別 | 來源 | 說明 |
|:--|:--|:--|:--|
| `seq` | str | GT path / eval | 如 `MOT17-04-SDP` |
| `frame` | int | gt.txt col0 | 1-based MOT frame |
| `gt_id` | int | gt.txt col1 | |
| `x,y,w,h` | float | gt.txt | xywh |
| `vis` | float | gt.txt col8 | 可見度 |
| `cls` | int | gt.txt | person=1 |
| `height` | float | 派生 `h` | 尺度維度 |
| `neighbors` | float/int | 派生 | 局部擁擠（可選） |
| `max_overlap` | float | 派生 | 與其他 GT max IoU（可選） |
| `frame_gt` | int | 派生 | 該幀 GT 數（可選） |
| `matched` | bool | **label** | 是否被 match |
| `match_score` | float | det | 認領 det 的 score；未匹配 NaN |
| `match_iou` | float | 派生 | 未匹配 NaN |
| `match_det_key` | str | 派生 | 可 join 的 det 鍵（可選） |

**主 AUC 分數：** `match_score`（僅 `matched==True` 的分佈分析；召回用 `matched` rate）。

### 4.2 `U_det` — `DetRow`（一列一個 detection）

| 欄位 | 型別 | 來源 | 說明 |
|:--|:--|:--|:--|
| `seq` | str | dump | |
| `frame` | int | dump | |
| `stage` | str | dump | `MethodId` 或 stage 名（見 §5） |
| `det_idx` | int | dump | 幀內序 |
| `det_key` | str | 派生 | `{seq}:{frame}:{stage}:{det_idx}` |
| `x1,y1,x2,y2` | float | dump | |
| `w,h` | float | dump / 派生 | |
| `score` | float | dump | **AUC 預設分數** |
| `cls` | int | dump | |
| `is_tp` | bool | **label** | IoU≥`iou_match` greedy 對 GT |
| `gt_id` | int | label | TP 時；FP 為 -1 |
| `match_iou` | float | label | FP 可為 best IoU 或 NaN |
| `height` | float | 派生 | det 或 matched GT h（meta 註明） |
| `vis` | float | 可選 | 來自 matched GT |

**集合對比：** 同一 `det_key` 空間上，不同 `stage`/`MethodId` 的存在性；或 wide 表 `accept_*`。

對齊現有 dump 欄位（`append_stage_dump_rows`）：

`seq, frame, stage, det_idx, x1, y1, x2, y2, w, h, score, cls`

### 4.3 `U_cand` — `CandRow`（一列一個 **幀內** (track, det) 候選）

**不是** relink (lost→cand) 對。勿把 offline relink 列寫進此表。

| 欄位 | 型別 | 來源 | 說明 |
|:--|:--|:--|:--|
| `seq` | str | dump / probe | |
| `frame` | int | | |
| `track_id` | int | tracker | local 或 global（meta 註明） |
| `det_idx` | int | | |
| `cand_key` | str | 派生 | |
| `iou` | float | 計算 | |
| `maha2` | float | 可選 | |
| `cost` | float | tracker | 最終 c_ij；**AUC 常用 −cost 或 1−cost** |
| `affinity` | float | 可選 | A_ij |
| `penalty` | float | 可選 | Π |
| `score_det` | float | det | |
| `height_trk` / `height_det` | float | | 尺度 |
| `speed` | float | 可選 | px/frame |
| `occ_iou` | float | 可選 | |
| `is_correct` | bool | **label** | 正確 association |
| `accepted` | bool | 處理 | 該 method 是否指派此對 |

**主 AUC：** `score_field` 預設建議 `iou` 或 `-cost`（寫進 meta）。

### 4.3b `U_relink_pair` — `RelinkPairRow`（一列一個 offline lost→cand）

**來源權威：** `scripts/tools/build_relink_candidates.py`（COLS 對齊 `RELINK_PAIR_BUILDER_COLS`）。  
**Substrate：** MOT dump with **relink-off + interpolate-off**（見 offline_relink 文 §1）。  
**y：** `gt_match`（CSV 可能是 0/1 int → `RelinkPairRow.gt_match_as_bool`）。

| 欄位（required） | 說明 |
|:--|:--|
| `seq`, `lost_id`, `cand_id` | 對的身份 |
| `gt_match`, `gt_valid` | 標籤 |
| `bridge_dist`, `gap` | 主分數 / 時間隙 |
| `lost_last_frame`, `cand_first_frame` | 時間邊界 |

可選但 builder 常給：`dist_h`, `fwd_resid`, `bwd_resid`, `dir_cos`, `speed_h`, `accepted`, `gt_lost`, `gt_cand`, …

**AUC 合規（強制）：**

1. Full-pool AUC（所有 `gt_valid` 對）  
2. Hard-pool AUC（`hard_pool_rule`，預設 `bridge_dist<=1.0`）  
3. `n_pos` / `n_neg` / base rate 兩池都報  
4. 工作點 thr 表（prec/recall）至少一張  

`bridge_dist` **lower-is-better** → ranking 用 `-bridge_dist`（`auc_full_and_hard_pool(..., lower_is_better=True)`）。  
**禁止：** 只引用全池 ~0.89 當「幾何夠強」而不報難池。

### 4.4 `U_err` — `ErrRow`（一列一個錯誤事件）

| 欄位 | 型別 | 來源 | 說明 |
|:--|:--|:--|:--|
| `seq` | str | | |
| `frame` | int | | 事件代表幀 |
| `event_id` | str | 派生 | 穩定 ID，做集合差 |
| `event_type` | str | 分類 | `fn_miss` / `id_switch` / `fp_birth` / `fp_interp` / `fragment` |
| `gt_id` | int | 可選 | |
| `track_id` | int | 可選 | |
| `method` | str | MethodId | 在哪個處理下觀測到 |
| `height` | float | 可選 | 分層 |
| `notes` | str | | |

**交集：** \(E_A \setminus E_B\) = method A 有、B 修掉的錯誤（用 `event_id` 對齊）。

### 4.5 `MethodMetricsRow`（端到端小表，非框級）

| 欄位 | 說明 |
|:--|:--|
| `method` | MethodId |
| `idf1, mota, hota, deta, assa` | 可選；HOTA 族可缺 |
| `ids, fp, fn, rcll, prcn` | count / rate |
| `n_seq` | |
| `output_dir` | 產出路徑 |

---

## 5. `MethodId`（處理方式）

| ID | 含義 | 典型怎麼得到 |
|:--|:--|:--|
| `raw` | 低 conf det（pre-filter） | `--debug-dump-csv` stage `raw` |
| `post_filter` | 分數/類別過濾後 | dump stage |
| `post_nms` | NMS 後 | dump stage |
| `post_merge` | post-merge 後（若開） | dump stage |
| `bare_track` | track 輸出；GMC/bridge/interp off | L2 CLI profile |
| `bare_gmc` | bare + GMC | L3 |
| `bare_bridge` | bare(+GMC?) + bridge | L3；meta 寫清是否含 GMC |
| `full_preset` | 完整 preset（如 m） | 正式 mot17 |
| `full_no_interp` | full 但 interpolate off | 對照填洞 FP |
| `custom` | 其他；`notes` 必填 | |

**累積序列**與**單開到 bare** 必須在 `StudyMeta.notes` 或 `metrics_by_method` 旁註明實驗設計，避免把 Δ 當可加總因果。

---

## 6. `DataSourceId`（資料來源）

| ID | 產物 | 路徑 / API | 寫入宇宙 |
|:--|:--|:--|:--|
| `mot17_gt` | GT 框 | `datasets/MOT17/train/*/gt/gt.txt` | `U_gt` 幾何 + vis |
| `stage_dump_csv` | 分 stage det | `append_stage_dump_rows` ← `--debug-dump-csv` | `U_det` 幾何/score/stage |
| `score_dist_tool` | GT 匹配分 | `scripts/eval/analyze_score_distribution.py`（及 detector 版） | `U_gt` match_* + 協變量 |
| `fp_by_height_tool` | FP 高度分析 | `scripts/eval/analyze_detection_fp_by_height.py` | 彙總；可回填 `U_det` |
| `mot_result` | 追蹤輸出 | `out|results/**/MOT17-*-SDP.txt` | emit 集合；可衍生 `U_err` |
| `motmetrics_eval` | OVERALL counts | `metrics.run_motmetrics_evaluation` | `MethodMetricsRow` |
| `trackeval_hota` | HOTA/DetA/AssA | vendored TrackEval via metrics.py | `MethodMetricsRow` |
| `neutral_nogo_tool` | 訊號 AUC 先例 | `scripts/tools/analyze_neutral_nogo_signals.py` | 參考；可對齊 `U_cand` |
| `near_miss_stage` | stage 歸因 | `scripts/eval/diagnostics/analyze_near_miss_stage_attribution.py` | 輔助 join |
| `pipeline_contribution` | 累積 cutoff | `scripts/eval/pipeline_contribution.py` | `MethodMetricsRow`（舊 preset 注意） |
| `manual_label` | 人工/腳本標 y | study 內 script | 任何 y 欄 |
| `derived` | 純派生 | join / IoU / set diff | 各表 |

**來源優先級（標 y 時）：**

1. 同一 study 內已固化的 label 欄  
2. `mot17_gt` + 明確 greedy IoU 規則（寫進 meta）  
3. 禁止：分析時臨時改 IoU 卻不改 `meta.iou_match`

---

## 7. 交集與 AUC 在表上的操作契約

### 7.1 集合交集（處理 × 處理）

**寬表**（推薦算交集）：

```text
det_key, is_tp, height, score,
accept_post_nms, accept_bare_track, accept_bare_bridge
```

```text
TP_M     = accept_M & is_tp
FP_M     = accept_M & ~is_tp
unique_A = TP_A & ~TP_B
jaccard  = |TP_A ∧ TP_B| / |TP_A ∨ TP_B|
```

**長表：** `method` + `accepted`，先 pivot 再算。

### 7.2 分層 AUC（維度）

```text
固定: universe, y_col, score_col
分層: pd.cut(height) 或 vis bins
每層: n, n_frac, auc, med_score|y=1, med_score|y=0
n 過小（建議 <50 或正負類缺一）→ 不報 AUC
```

### 7.3 錯誤集合差（流水線）

```text
E(method) = { event_id | method 下存在 }
fixed_by_B = E(A) \ E(B)
regressed  = E(B) \ E(A)
```

---

## 8. 與現有工具的映射（實作入口）

| 需求 | 先用 | 落到 schema |
|:--|:--|:--|
| L0 det 框 | `--debug-dump-csv` + stages | `U_det` |
| L0 score×height | `analyze_score_distribution.py` | `U_gt` |
| L2/L3 count | `mot17.py` + method profile | `MethodMetricsRow` |
| 訊號 AUC 先例 | `neutral_nogo` 文 + tool | 對齊 `U_cand` / 分層表 |
| Stage 誰殺了 GT | `analyze_near_miss_stage_attribution.py` | join `U_gt`↔stage dump |

Code 常數與 dataclass：`signal_tables.py`（`UNIVERSE_COLUMNS`, `load_study_meta`, …）。

---

## 9. 演進規則

- 加欄：可選欄可加；**改 y 語義必須新 `study_id`**，不得覆寫舊 parquet 的 label 含義。  
- 新 MethodId：先加 Enum + `METHOD_TERMINAL_NODE` + 節點表一行 + meta.notes。  
- 新 pipeline 節點：更新 `PIPELINE_NODES` order/parents，並同步本檔 §P。  
- 升格可引用數字：走 evidence_ledger，不把 parquet 當 fact-owner。  
- Headline YAML / ACTIVE knobs：本 schema **不**授權修改 production。

---

## 10. 最小合規 checklist

- [ ] `meta.json` 含 commit / preset / iou_match / y_definitions  
- [ ] `cut_design` + `method_ids`（cumulative 通過 `validate_method_order`）  
- [ ] 每張表只含一個 UniverseId；**relink 用 `U_relink_pair` 不用 `U_cand`**  
- [ ] 報告 AUC 時寫明 `(universe, y, score)` 與 **所在 layer/node**  
- [ ] **B1：** `hard_pool_rule` 已填；full + hard AUC + base rate + thr 表  
- [ ] **B1：** substrate = relink-off + interp-off；腳本 = `build_relink_candidates.py`  
- [ ] **B2：** 狀態用 `reconnect_rate.py`（或等價）；不與 B1 AUC 混稱  
- [ ] 交集用同一 `*_key` 空間；流水線 Δ 只對相鄰 cumulative 或 base 對  
- [ ] 分層表帶 n%  
- [ ] 結論按 L0→L8 收，不跳層把 e2e 分數當模組勝敗  

## 11. Related (script / research bind)

| 線 | Doc / tool |
|:--|:--|
| B1 hub | [offline_relink_candidate_analysis.md](../../modules/semantic/research/offline_relink_candidate_analysis.md) |
| B1 builder | `scripts/tools/build_relink_candidates.py` |
| B2 reconnect | `scripts/eval/diagnostics/reconnect_rate.py` |
| A score dist | `scripts/eval/analyze_score_distribution.py` |
| Scripts index | [association_recovery_scripts_index_20260709.md](../../modules/semantic/research/association_recovery_scripts_index_20260709.md) |

---

## Related

- [neutral_nogo_signal_attribution_20260612.md](neutral_nogo_signal_attribution_20260612.md) — 訊號層 AUC 方法先例  
- [mamba-score-distribution](../../modules/detection/research/mamba-score-distribution-20260613.md) — U_gt 協變量先例  
- [tracker-decision/scoring_semantics.md](../tracker-decision/scoring_semantics.md) — cost 語義（非本表）  
- [evidence_ledger.md](../evidence_ledger.md) — 可引用 e2e 數字  
