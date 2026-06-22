# 018 — 專案主線收斂與雙線開發方向

> 狀態:Accepted(2026-06-18)·Superseded headline numbers(2026-06-21)
> 性質:專案級方向決策。定調後續所有工作的合併準則與停損條件。
> 相關:[no_go_registry](../reference/no_go_registry.md)、[mot17_default_config](../reference/mot17_default_config.md)、[PROJECT_SHOWCASE](../PROJECT_SHOWCASE.md)、`configs/presets/mamba_whole_graph.yaml`

> **⚠️ Headline 版本(2026-06-21):** 本文件原以 `frozen_v1`(2026-06-18, IDF1 77.6)為準。
> **canonical headline 已更新為 `frozen_v2`(2026-06-21, IDF1 78.2 / HOTA 70.2 / 269.47 FPS)**,
> 全 repo 數字以此為單一來源,權威表見 [mot17_default_config](../reference/mot17_default_config.md)
> 與 [PROJECT_SHOWCASE](../PROJECT_SHOWCASE.md)。下文方向決策(雙線策略 / 停損 / release 結構)
> 不變;§4–§7 的具體數字已就地更新為 frozen_v2,`frozen_v1` 僅保留為 06-18 執行當下的 dated 紀錄。
> **B 線(衝 80 IDF1 支線)已停損結案**(ReID/長 gap appearance 牆撞到底,見 NO-GO registry),
> `frozen_v2` 78.2 即 final headline。

---

## 0. 一句話主線

> **Real-time-first ReID-free MOT system with AUC-guided conservative association recovery**
> 即時優先、無 ReID 依賴、由可分性(AUC)分析引導的保守式多目標關聯恢復系統。

關鍵字:`real-time-first` · `ReID-free` · `geometry-aware` · `conservative association recovery` · `CUDA tracker runtime` · `AUC-guided NO-GO attribution`

定位敘述:本專題從即時監控串流需求出發,建立低延遲多目標追蹤系統。系統不依賴額外 ReID 模型,而是透過低成本幾何、時序與遮擋訊號,在 CUDA tracker hot path 中進行保守式 association recovery。在維持單幀 ~6–7 ms E2E(269 FPS double-buffer throughput / 144 FPS 單幀延遲模式)的同時,達到接近(內部設定)SOTA 等級的 MOT17 train 表現。

---

## 1. 成果定位(待凍結重現,見 §4)

### 系統速度(`frozen_v2`, 2026-06-21;兩個操作點,協定見 [PROJECT_SHOWCASE](../PROJECT_SHOWCASE.md) 附錄)
```
double-buffer (throughput-optimized): 269 FPS   | mean 7.42 ms/frame | p99 ~12.5 ms
single-frame  (latency-optimized)   : 144 FPS   | mean 6.34 ms/frame | p99 ~7.8 ms
  主導 stage : model(mamba head)launch-bound;tracker / association sub-ms,非瓶頸
```
重點主張:**tracker / association 有明顯分數貢獻,但幾乎不是延遲瓶頸** → 支撐 real-time-first 主張。
⚠️ FPS 是 throughput、mean_ms 是單幀 latency,**兩者不可互推**:double-buffer 把 detect(N)‖tracker(N−1)
排到相鄰 frame → throughput 近 2×,但單幀 latency 反而較高(重疊代價)。報告兩數字須各自標協定。

### 整體成績(MOT17 train / SDP, 7-seq,`frozen_v2`)
```
IDF1 78.2 | MOTA 78.4 | HOTA 70.2 | DetA 70.9 | AssA 69.7
IDs 413   | Prcn 97.2 | Rcll 81.0
```
表述:內部 MOT17 train/SDP 設定下達接近公開 SOTA tracker 量級,且具極低 E2E 延遲。**不喊 official SOTA**(未送 MOTChallenge server)。

### Showcase:MOT17-04
```
IDF1 91.2 | MOTA 91.4 | HOTA 84.2 | DetA 85.0 | AssA 83.6
IDs 26    | Prcn 99.8 | Rcll 91.6 | GT IDs 141
```
表述:高密度、強遮擋、相似外觀、複雜透視下只產生 26 次 ID switch → 不是靠碎裂 ID 或低 precision 補分,而是保守策略下維持穩定 identity assignment。

---

## 2. 三條貢獻(報告章節骨架)

| # | 貢獻 | 主張 | 內容 |
|---|------|------|------|
| C1 | Real-time-first GPU MOT system | 完整 detect+track+assoc 下單幀 ~6–7 ms E2E(269 FPS double-buffer throughput) | Mamba detector/head runtime · FPN 自訂算子 · CUDA/C++ tracker hot path · whole-graph pipeline · RTSP/MediaMTX live |
| C2 | Mamba-head / FPN custom operator | 針對低延遲推論設計算子,非直接套現成 detector | head/FPN 為何慢 · kernel fragmentation · latency 省多少 · 精度維持 · 與 whole-graph preset 配合 |
| C3 | AUC-guided Conservative Association Recovery | ReID 在 crowd/occlusion/crop 污染下失敗 → 幾何/時序訊號**先做 AUC 歸因再設保守 guard**,非把幾何當 identity oracle | bidir bridge relink · scale gate · margin ambiguity guard · OAO duration-ramp · crossing/occlusion guard · interpolation/lifecycle safety |

> **C2 與 C3 不可混章**:C2 是 detector/runtime 貢獻,C3 是算法貢獻。

C3 核心鏈條:
```
ReID 特徵混亂
  → 分析 geometry/motion/scale/duration 訊號可分性
  → full-pool AUC 與 hard-pool AUC 區分
  → 只保留低傷害、高 precision 的條件
  → 形成 conservative recovery system
```

---

## 3. 雙線開發策略

主線已成形,**主線不再被新研究拉動;新研究只有通過 ablation + 一致性檢查才合併回主線**。

### A 線 — 收主線(保底成果,現在做)
```
freeze preset
→ 重現 headline(單一 frozen preset 同源)
→ per-sequence 表
→ without-04 結果
→ ablation 表
→ 確認 TrackEval / split / no leakage
→ README / 報告主線
```

### B 線 — 80% IDF1 衝刺支線(不污染主線)

> **🛑 B 線已停損結案(2026-06-21):** 衝 80 IDF1 的三類嘗試已撞 appearance/長 gap
> ReID 牆(見 NO-GO registry #3/#6/#23/#33 與 fpn-relink #47:raw backbone 特徵無個體
> 區分力、revive 越多越低)。`frozen_v2` 78.2 即 final headline,不再開新研究;以下攻擊序
> 與 occ-gated velocity 候選保留為「若未來有新模態(真 appearance 訊號源)再重啟」的紀錄。

目標 `IDF1 ≥ 80`,只做三類:
1. **per-sequence error budget** — 找非-04 的最大 headroom 序列。
2. **detection-side safe recall calibration** — 安全區補 recall,非全域降門檻。
3. **association NO-GO revival** — 只用現有 tracker state、低成本、可並行、不傷 latency(例:occ-gated velocity,見 registry #7 周邊與 occ_cost_weight 掃描結論)。

**攻擊序(2026-06-21 frozen_v2 per-seq 實測,按 AssA 由低到高;B 線已結案,僅留作歷史):**
```
02  AssA 47.0  ← 最低(HOTA 49.1 全場最低,crowded/static)
09  AssA 49.7
10  AssA 49.8
13  AssA 60.5
05  AssA 61.7
11  AssA 67.3  ← 已近天花板,別碰
(04  AssA 83.6 ← showcase,撐分主力,不在攻擊範圍)
```
**MOT17-02 與 10 是唯一該攻的兩條**:association 退化最重,AssA 比 04 低約 35 分,且正是 occ_state 記憶中 crossing-swap / ≤5f 原地震盪重災區 → B.3 occ-gated velocity 直接對準,不是隨機選序列。**任何在 02/10 沒動、只靠 04/11 變好的方法,觸發下方「單序列撐分」停損即回退。**

**暫停(不做):** 新 ReID · 新大模型 · 新 3D/大型 GMC 架構 · 新大型 relink framework · 任何 >0.5 ms 的方法。

**停損條件(B 線出現任一即回退):**
- 只靠單一 sequence 撐分
- ex-04 退步
- HOTA / AssA 退步
- latency 破壞
- 無 AUC / blocker 歸因

---

## 4. 起手式(A∩B 交會,最高槓桿)

A 線「per-seq 表」、B 線「per-seq error budget」、報告「重現 headline」三者是**同一次 frozen-preset full eval**,一次跑、三個 payoff。

```bash
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --output out/frozen_v1
.venv/bin/python scripts/eval/_perseq_extract.py out/frozen_v1
```

| payoff | 產出 |
|--------|------|
| 驗 headline | 確認 IDF1/HOTA/IDs/FPS 全出自**同一次** frozen run(報告數字唯一能站住的前提) |
| 報告骨架表 | per-seq HOTA/AssA/IDF1 = Experiments 章 per-sequence delta 表骨架 + ex-04(6-seq)真實成色 |
| 餵 B 線 | per-seq 表暴露非-04 最大 headroom 序列 = error-budget 第一攻擊目標 |

凍結狀態:`configs/presets/mamba_whole_graph.yaml` 已 commit 在 HEAD(`oao_tau 0.50 / oao_ramp_frames 25 / relink_bridge_px 0.25 / fuse_score_weight 0.0`),已可作為基準凍結點。

**✅ canonical(2026-06-21,`frozen_v2`):** headline 同源重現 — IDF1 **78.2** / MOTA **78.4** / HOTA **70.2** / DetA **70.9** / AssA **69.7** / IDs **413** / Rcll **81.0** / Prcn **97.2** / **269.47 Eval FPS(double-buffer throughput)/ 7.42 ms mean latency**。聚合與 per-seq 由 `scratch/ab_runs/frozen_v2_repro` 同一次 run 同源產出(COMBINED 70.2/69.7/78.2,bit-exact 重現)。per-seq 表見 §6。
>
> _歷史紀錄:_ frozen_v1(2026-06-18, `out/frozen_v1/`)為 IDF1 77.6 / HOTA 69.9 / IDs 430 / 221.59 FPS;已被 frozen_v2 取代為 canonical,僅留作 06-18 執行佐證。注意 FPS 與單幀 latency 是不同量測,不可互推(double-buffer 提升 throughput 但單幀 latency 反而較高);latency 協定見 PROJECT_SHOWCASE 附錄。

---

## 5. 數字紀律(報告誠信前提)

1. **同源重現**:headline 必須來自同一 frozen preset 的同一次 run,否則一問即破。**✅ 2026-06-21 已驗(§4)**:`frozen_v2` 一次 run 同時產出 IDF1 78.2 / HOTA 70.2 / IDs 413 / 269.47 FPS,全部同源。**FPS/latency 協定**:269.47 是 `--double-buffer` 的 throughput,7.42 ms 是同設定的單幀 mean latency;兩者是不同量測不可互推。`docs/reference/benchmarks/latency_log.md` 記 40–71 FPS 為另一量測脈絡(stream 數 / warm-up / 是否含 decode+track / GPU 不同),報告須各自註明協定,不可與 269 / 7.42 並列而不解釋。
2. **headline 雙軌**:7-seq 與 ex-04 必須並列。**2026-06-21 frozen_v2 實測**:
   - 7-seq(GT 加權):IDF1 **78.2** / HOTA 70.2 / AssA 69.7
   - ex-04(6-seq,GT 加權):IDF1 **67.8** / HOTA 58.4 / AssA 55.8
   - **78.2 與 67.8 的 ~10 分落差全來自 MOT17-04 一條**(IDF1 91.2,占 ~42% GT 權重)。04 是 **showcase 非 headline**;雙軌並列是必要誠實,不是保守。
3. **不喊 official SOTA**:未送 MOTChallenge server 前只說「內部設定下接近 SOTA 量級」。

---

## 6. 報告大綱(對應章節)

```
1. Introduction            — 即時 MOT 需低延遲+穩定尾延遲;ReID 在遮擋下污染且貴 → ReID-free 動機
2. System Overview         — 總架構圖 + latency breakdown(§1)
3. Mamba-head & Runtime    — C2,細節入附錄
4. CUDA Tracker Runtime    — 一萬多行但 runtime sub-ms;parallel assoc/lifecycle/bridge kernel/guards
5. AUC-guided Association Recovery — C3 主章
     5.1 ReID failure & appearance ceiling
     5.2 Geometry/motion signal attribution(full-pool vs hard-pool AUC)
     5.3 Conservative guard composition
6. Experiments             — 整體 metrics / MOT17-04 showcase / latency breakdown / ablation
7. Negative Results & NO-GO Attribution — 主文精簡 5 項,完整入附錄
     (Appearance/ReID ceiling · always-on velocity NO-GO · bridge hard-pool ceiling
      · OAO duration-ramp revival · affine GMC / horizon prior NO-GO)
8. Deployment              — RTSP/MediaMTX/edge;association 非瓶頸,部署以目標平台 profiling 為準
9. Conclusion
```

必要表格:`baseline vs ours` · `relink off / bridge / bridge+guards` · `OAO plain vs duration-ramp` · `with / without 04` · `per-sequence delta`。

**Per-sequence 實測(2026-06-21,`frozen_v2`,MOT17 train/SDP):**

| Seq | HOTA | AssA | IDF1 | 備註 |
|------|------|------|------|------|
| 02 | 49.1 | 47.0 | 58.8 | 最低 AssA,crowded/static,B 線第一目標 |
| 04 | 84.2 | 83.6 | 91.2 | showcase,撐分主力 |
| 05 | 61.1 | 61.7 | 73.8 | |
| 09 | 57.6 | 49.7 | 66.8 | |
| 10 | 55.8 | 49.8 | 63.4 | 次低 AssA,B 線第二目標 |
| 11 | 71.2 | 67.3 | 79.7 | 最佳非-04,近天花板 |
| 13 | 61.3 | 60.5 | 72.4 | |
| **7-seq (GT-w)** | **70.2** | **69.7** | **78.2** | headline |
| **ex-04 (GT-w)** | **58.4** | **55.8** | **67.8** | 真實可轉移成色 |

---

## 7. Release 策略(兩層)

- **主研究倉**:保留全部(`src/ scripts/ tests/ docs/ configs/ archive/` + `no_go_registry.md`)= research workspace + extended NO-GO archive。**不硬清**,NO-GO 本身有價值。
- **release / demo 倉**:乾淨展示(`README` + `configs/` + `src/` + `docs/{overview,results,architecture,association_recovery}.md` + appendix docs + `scripts/eval/mot17.py` + `examples/rtsp_demo/`)。首頁 headline(`frozen_v2`):`269 FPS double-buffer throughput · 7.42 ms mean latency · IDF1 78.2 / HOTA 70.2 · MOT17-04 IDF1 91.2`。FPS 與單幀 latency 須分別標明量測協定,不可互推或與 leaderboard 並列。
