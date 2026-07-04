# Sparse Key-Embedding Bank — Cheb-GR 稀疏等價結案報告（2026-07-04）

> 對應 registry [#58](../../../reference/no_go_registry.md)、TODO「Sparse key-embedding bank」條目。
> 一句話結論：**每 track 維護「最近 20 個 visclean clean 樣本」的 FIFO 即可與 dense-50 bank 等價（甚至略優）；embedding 不能平均、複製出的重複樣本不進 graph、品質訊號選樣無效；抽取頻率可稀疏到每 3 個 clean 幀一次。**

## 0. 問題

mnv4 ReID 破牆後（IDF1 80.3 via offline Cheb-GR handover），ReID 的使用被 #57 限制在 offline/async：不得阻塞 tracker critical path。要讓 Cheb-GR 走向線上可維護（async sidecar）或更便宜的 post-hoc，核心問題是：

1. 哪些訊號能挑出「關鍵」embedding？（每 ID 只維護少量 key embeddings 是否成立）
2. bank 可以稀疏/原型化到什麼程度而 handover 決策不變？
3. 「box 用當幀、embedding 從最近抽取複製」的稀疏抽取排程等不等價？

## 1. 方法

所有 A/B 在 **frozen no-handover substrate** 上做（`results/diag_{m,s}_no_reid_current_20260704`，m=`mamba_whole_graph_m` 79.51、s=`mamba_whole_graph` 78.34 substrate）：tracker 輸出凍結後 handover 是純確定性後處理，**零 eval 不確定性**（繞開 GPU-decode race），且 m/s 雙 backbone 提供 cross-condition 檢驗。操作點固定為現行 GO（decide_n 5 / max_cost 0.45 / min_head 2 / margin 0.05 / max_gap 60）。

工具（皆新增於本輪）：
- `scripts/eval/run_offline_handover_ablation.py` — substrate 上重放 handover 變體（`--variant name key=val`），共用抽取、motmetrics 評分。
- `scripts/eval/diagnostics/probe_sparse_bank_equivalence.py` — bank 壓縮策略 × 等價指標（accept-set jaccard、|Δcost| 中位數、IDF1）× per-sample 訊號相關，一次跑完。

## 2. 決策層 veto ablation（全部不值；flags 保留 default off）

applicability map（[[chebgr-handover-applicability-map]]）給的 stable-veto / stable-high-pollution 區間，直接當硬 gate 全部失敗（m substrate，control handover 80.22）：

| 變體 | dIDF1 | 判定 |
|---|---|---|
| `center_dist_veto=2.0` | **+0.06** | 唯一不負，noise 級（僅 10 vetoes；05 +0.39/11 +0.52/10 −0.34） |
| `pollution_veto=0.5`（head_tail_iou 硬砍） | **−0.46** | 高汙染區仍有 1/3~2/3 正確 handover——pollution 是 context 不是 identity gate |
| `neighbor_iou_max=0.5`（crop 過濾） | **−0.26** | 濾髒 crop 同時把靠那些幀成立的正確 handover 打回 min_head |

Flags 已接好（`--cheb-gr-offline-center-dist-veto/-pollution-veto/-neighbor-iou-max`），default 0=off。

## 3. 訊號重要性：visclean 之後沒有品質訊號

Per-sample「cosine 到 track 原型」對便宜訊號的 spearman（m/s 各 ~20k bank samples）：

| 訊號 | m | s |
|---|---|---|
| det score | +0.133 | +0.160 |
| box 高度 | +0.070 | +0.101 |
| **neighbor_iou** | **−0.004** | **−0.025** |
| 時間位置 | −0.039 | −0.033 |

**visclean front-occlusion gate 已把可判別的髒 crop 濾完，殘餘品質訊號不存在**——這是 §2 全負的根因，也否決了「用品質訊號挑 key embedding」的路線（`topscore-5`/`lowpol-5` 都輸同預算的時間取樣）。

## 4. 稀疏 bank 等價（核心結果）

Bank 壓縮策略 vs dense-50 reference（head 證據不動；dIDF1 為 m / s）：

| 策略 | m dIDF1 | s dIDF1 | jaccard m/s | 判定 |
|---|---|---|---|---|
| **recent-20（clean-FIFO）** | **+0.07** | **+0.05** | **0.85 / 0.72** | ✅ 最佳：等價偏優、保真度全場最高 |
| spread-10（時間分散） | +0.01 | +0.01 | 0.80 / 0.67 | ✅ 等價 |
| stridefifo-3-20（每 3 clean 幀抽 1 + FIFO-20） | −0.05 | −0.03 | 0.67 / 0.61 | ✅ 等價，**抽取成本 ÷3** |
| spread-5 | 0.00 | −0.12 | 0.64 / 0.57 | ⚠️ 邊際 |
| recent-10 / recent-5 | −0.20 / −0.48 | −0.03 / −0.05 | — | ❌ N<20 不安全 |
| stridefifo-5/10/15-20 | −0.06 / +0.02 / −0.17 | **−0.21 / −0.25 / −0.49** | 單調掉至 0.33 | ❌ K≥5 條件敏感 |
| stride-15 / stride-30（無 FIFO 上限） | −0.17 / −0.69 | −0.49 / −0.60 | 0.33 / 0.28 | ❌ bank 大小異質化，cost 尺度壞 |
| dupfill-5（5 樣本複製填滿 50） | +0.01 | −0.06 | 0.61 / 0.48 | ⚠️ 不毒但保守化（accepts 51→31、IDs 微升） |
| segmean-3 / segmean-5（分段平均原型） | −0.30 / −0.11 | −0.41 / −0.10 | — | ❌ 同預算輸原始樣本 |
| **mean1（單一平均原型）** | **−2.61** | **−1.14** | 0.33 / 0.29 | ❌ 災難：距離塌縮（\|Δcost\|med 0.066），handover 爆量 51→108 / 67→135，操作點失效 |

### 為什麼 recency 贏

handover 的 query 是死亡後緊接的 newborn head——**死前樣本與比對時刻時間最近、外觀漂移最小**；且 FIFO 存的是最近 20 個 **clean** 樣本，death 若發生在遮擋裡，visclean gate 讓 FIFO 自然往回跳過髒的死亡期。

### 三條硬約束

1. **embedding 只能以原始樣本存在**：任何平均（mean1/segmean）都塌縮 intra-track 變異、扭曲 Cheb-GR k-reciprocal 的距離尺度。
2. **複製出的重複 embedding 不進 graph**：dupfill 不毒但重複樣本互為最近鄰、擠掉 k2 鄰域，機制保守化。「box 用當幀、embedding 前向複製」的 copy 只可用於 graph 外用途；bank 存 unique 抽取。
3. **抽取稀疏化安全點 = 每 3 個 clean 幀**：K≥5 起 jaccard 單調掉、handover 數膨脹（cost 尺度漂移），s 上 IDF1 實跌。

## 5. 落地與部署 spec

**已落地（Python 層）**：`--cheb-gr-offline-bank-mode {spread,recent}` + `--cheb-gr-offline-bank-n`（default spread=現行 GO 不動）；一鍵 config `configs/modules/cheb_gr_offline_mnv4_fifo20.yaml`。wired 路徑已在 m/s substrate 端到端複驗與 probe 一致（m：11 +0.52/13 +0.26/05 +0.10/10 −0.10；s：05 +0.46/10 +0.23；IDs 持平）。附帶收益：post-hoc bank crop 抽取量 ~2.5×↓。

### 5.1 Direct key-bank query 診斷（2026-07-04 追加）

為了驗證「小型 key bank 不只可餵 Cheb-GR，也可作 candidate-level confirm/veto evidence」，`CleanFifoBank` 追加 graph-external query API：

- `metadata()`：保存每個 key embedding 的 `frame_id / quality / source_type`。
- `query()` / `query_all()`：回傳 `best_sim`、`mean_topk_sim`、support count、best source，以及 `best_sim - best_other_track_sim` hard-negative margin。

同時 `causal_handover_lines(..., decision_log=...)` 追加欄位：`key_best_sim`、`key_mean_topk_sim`、`key_best_other_id`、`key_best_other_sim`、`key_margin`、`key_support`、`key_other_support`；`cheb_gr_offline_handover_report.py` 會把 `key_best_sim / key_margin` 納入 feature registry、bucket map、gate search。

7-seq frozen substrate 重跑命令：

```bash
uv run python scripts/eval/run_offline_handover_ablation.py \
  --substrate results/diag_m_no_reid_current_20260704

uv run python scripts/eval/diagnostics/cheb_gr_offline_handover_report.py \
  --handover-log results/diag_m_no_reid_current_20260704_ho_control/_cheb_gr_offline_handover.csv \
  --baseline-dir results/diag_m_no_reid_current_20260704 \
  --gt-root datasets/MOT17/train \
  --out-csv results/diag_m_no_reid_current_20260704_ho_control/_cheb_gr_offline_handover_labeled_key.csv \
  --registry-md results/diag_m_no_reid_current_20260704_ho_control/parameter_registry_key.md \
  --summary-json results/diag_m_no_reid_current_20260704_ho_control/parameter_summary_key.json
```

結果（m substrate，control 80.22 IDF1 / 309 IDs；candidate rows 311、accepted known precision 27/47=0.574）：

| 訊號 | AUC | 方向 | 觀察 |
|---|---:|---|---|
| `key_best_sim` | **0.928** | high | raw key-bank similarity 幾乎追上 `best_cost`，低區是穩定 danger：`<0.3` = 0/104、`0.3~0.5` = 5/126；`0.75~0.85` = 10/11，但樣本仍薄 |
| `key_margin` | 0.788 | high | 很適合 ambiguity veto/abstain：`<=0.03` = 1/87 same-GT；高 margin 支援但不保證 accept（`>=0.25` = 19/56） |
| `best_cost` | 0.915 | low | 仍是主決策量；`<0.25` = 12/13 |

在 accepted-known rows 的 two-feature gate search 裡，最好的 key 組合是：

```text
key_best_sim >= 0.54 && center_dist_norm <= 0.756
  selected=23, correct=22, wrong=1
  precision=0.957, correct_recall=0.815, wrong_keep=0.05
```

正式管線狀態（2026-07-04）：

- 已接入 `causal_handover_lines` 決策 gate：`key_sim_min` 與 `key_margin_min`。
- 已接入 CLI / config / evaluator / ablation runner；預設皆為 `0.0`，等於關閉，不改現行 GO。
- CLI flags：`--cheb-gr-offline-key-sim-min`、`--cheb-gr-offline-key-sim-cost-floor`、`--cheb-gr-offline-key-margin-min`（同時保留 `--cheb-gr-online-*` alias）。
- ablation runner variant keys：`key_sim_min=...`、`key_sim_cost_floor=...`、`key_margin_min=...`，可與既有 `center_dist_veto=...`、`margin=...` 合併重放。
- decision log 會同步記錄 `key_sim_min` / `key_margin_min` 與實際 `key_*` evidence，方便回放分析。

正式 gate 實測（m/s 7-seq frozen substrate）：

| policy | m IDF1 | s IDF1 | 判定 |
|---|---:|---:|---|
| control | 80.22 | 78.83 | 現行 GO |
| `key_sim_min=0.54` | 80.24 | 78.84 | 小幅正向；m 擋 13 個 control accepts（11 wrong / 1 correct / 1 unknown），s 擋 17 個（10 wrong / 3 correct / 4 unknown） |
| `key_margin_min=0.03` | 80.21 | 78.82 | 幾乎無效；低 margin 是 ambiguity 訊號，但 policy 收益不足 |
| `key_sim_min=0.54 + center_dist_veto=0.756` | 80.01 | 78.77 | 不可用；MOT17-10 被打壞（m −3.01、s −0.81） |
| `key_sim_min=0.54 + key_sim_cost_floor=0.25/0.30` | 80.24 | 78.84 | 最合理候選；只在 Cheb-GR cost 不夠強時套低-sim veto，避免誤殺強 graph match |

研究結論：

1. **低 `key_best_sim` 是有效 veto evidence，但收益很小**。最佳區間仍是 `key_sim_min≈0.54`；再高會大量誤殺 correct，再低吞吐不足。
2. **key-sim veto 應該服從 Cheb-GR 主訊號**。control accepted rows 上，`key_best_sim < 0.54 && best_cost >= 0.25/0.30` 的 wrong/known 比率約 0.95/0.96（m+s combined），比裸 `key_best_sim < 0.54` 更乾淨。
3. **center distance 不適合用 tight threshold 硬砍**。`center_dist_norm <= 0.756` 來自 accepted-known 搜尋，但實測會殺掉 MOT17-10 的正確 handover；只能作報告 feature 或寬鬆 veto，不可作 default policy。
4. **confirm 方向暫時不成立**。被 `best_cost > 0.45` 擋掉的正確候選，其 `key_best_sim` 最高約 0.62，沒有高-sim rescue 區；key bank 目前只適合 veto/abstain，不適合放寬 accept。
5. **實驗瓶頸是 CPU，不是 ReID/GPU**。回放期間 Python runner 單核 100%、GPU utilization 0%；後續 threshold search 應先用 decision log 做 CSV counterfactual sweep，只把少量候選 policy 丟進完整 MOT scoring。

判定：

1. **direct key-bank query 是有效診斷訊號**，可用於 confirm/veto analysis；`key_best_sim` 對 same-GT 分離清楚。
2. **`key_margin` 更像 abstain/veto 訊號，不是 accept 訊號**；低 margin 幾乎全錯，但高 margin 仍有 shared-appearance / no-hard-negative 風險。
3. **正式管線已接，但 default off**。目前只在 m substrate 重跑；需與 s backbone / confirm-score 0.30 regime 做 applicability map 後，才可把 `key_best_sim + geometry` 提升成預設 policy。

**C++ 線上層（刻意未做）**：per-track clean-FIFO-20 ring buffer + stride-3 排程 + **event-aware birth 窗**（decide 窗 5 幀須密集抽，min_head=2 才有料——#56 的教訓是 budget 排程餓死 head）。啟動條件=async sidecar 立項（#57 禁 sync 抽取，目前 C++ bank 沒有消費者）。

**保留條款**：等價只驗證於 offline handover 操作點（max_gap 60）。長 gap relink 可能需要更久遠的錨點樣本，屆時 FIFO-20 + 老樣本錨點另測。

## 6. Artifacts

- probe JSON：`results/probe_sparse_bank_m_20260704.json`、`probe_sparse_bank_recency_{m,s}_20260704.json`、`probe_sparse_bank_stridefifo_{m,s}_20260704.json`
- 變體輸出：`results/diag_{m,s}_no_reid_current_20260704_ho_*/`（含 decision log CSV）、`..._sparse_*/`
- direct key-bank query summary：`results/diag_m_no_reid_current_20260704_ho_control/parameter_summary_key.json`、`parameter_registry_key.md`
- 單元測試：`tests/unit/eval/test_clean_fifo_bank.py`、`tests/unit/reid/test_cheb_gr_online.py`、`tests/unit/eval/test_cheb_gr_offline_handover_report.py`
