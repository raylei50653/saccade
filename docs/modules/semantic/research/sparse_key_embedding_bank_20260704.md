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

**C++ 線上層（刻意未做）**：per-track clean-FIFO-20 ring buffer + stride-3 排程 + **event-aware birth 窗**（decide 窗 5 幀須密集抽，min_head=2 才有料——#56 的教訓是 budget 排程餓死 head）。啟動條件=async sidecar 立項（#57 禁 sync 抽取，目前 C++ bank 沒有消費者）。

**保留條款**：等價只驗證於 offline handover 操作點（max_gap 60）。長 gap relink 可能需要更久遠的錨點樣本，屆時 FIFO-20 + 老樣本錨點另測。

## 6. Artifacts

- probe JSON：`results/probe_sparse_bank_m_20260704.json`、`probe_sparse_bank_recency_{m,s}_20260704.json`、`probe_sparse_bank_stridefifo_{m,s}_20260704.json`
- 變體輸出：`results/diag_{m,s}_no_reid_current_20260704_ho_*/`（含 decision log CSV）、`..._sparse_*/`
- 單元測試：`tests/unit/reid/test_cheb_gr_online.py`（veto ×2、crop 過濾 ×2、recent bank mode ×1）
