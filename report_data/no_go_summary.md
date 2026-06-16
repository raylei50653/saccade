# Saccade NO-GO 全局登記表 (Global NO-GO Registry)

> **用途**：跨模組「已結案/已踩雷方向」總覽，避免重複探索。每列只記結論一行，數據細節以對應模組的 `research/` 或 `decisions/` ADR 為準。
> 彙整自 `decisions/`、`modules/*/README.md`、`archive/`、`reference/PIPELINE_REFERENCE.md`、`TODO.md`（路徑相對 `docs/`）。
> 最後更新：2026-06-12

**死因分類**（中性結案前必須回答：訊號不存在，還是被遮蔽？）：

| 類型 | 意義 | 復活可能 |
|------|------|----------|
| 🔻 **有害** | 開啟後指標明確退步 | 無，除非前提改變 |
| ❌ **結構天花板** | 訊號本身無區分力（AUC≈隨機、物理瓶頸） | 無 |
| ⚪ **中性（被遮蔽）** | 增益在雜訊內，但已識別 blocker（上游門/低 base rate/記憶長度） | **有** — blocker 移除後可組合復活，見[復活前例](#中性-no-go-的復活前例) |

> **分類門檻**：分類必須附**歸因實驗 + 統計資料**（AUC、候選攔截率、ablation 對照組之類），不可從結果倒推。無歸因實驗的條目**不分類**，只記結果（下表多數歷史條目即屬此類 — 結果論記錄，死因未歸因）。日後若補做歸因實驗，再回填分類。

**GO 判定門檻**（2026-06-12 修訂）：

- 固定 `+3pp` GO 門檻**已廢止** — 它定於 baseline ~50% IDF1 的時代；現行 baseline ~75%，剩餘空間只剩 25pp，固定幅度門檻會系統性誤殺。先例：[mamba 雙解析度研究](../modules/detection/research/mamba-dual-resolution-original-detail-plan.md)已在 detection recall 場域廢止同一門檻。
- 現行原則：**任何超出同日 paired 噪聲（±0.3pp）的變化即有討論價值**。GO 判定看（1）方向跨 seq / 跨 score threshold 一致、（2）paired 95% CI 排除 0、（3）無其他指標嚴格退步 — 不看固定幅度。
- 高 baseline 下同時報告 absolute delta 與**相對剩餘空間的比例**（例：75% 時 +1pp = 吃掉 4% 剩餘 headroom，等價於 50% 時代的 +2pp）。

---

## 核心 NO-GO（有完整 ablation 證據）

| # | 項目 | 時間 | 結論 | 關鍵數據 | 代碼狀態 |
|---|------|------|------|----------|----------|
| 1 | **Option D** Track-Conditioned YOLO | 2026-05-19 | ❌ NO-GO | IDF1 31.7% vs baseline 52.0%，gate ∆ <0.2pp | 歸檔 `docs/archive/option-d/` |
| 2 | **Appearance ReID Bank** (GMC ON) | 2026-05-13 | ❌ NO-GO | IDF1 ±0.0pp、FPS **−17.3**（零增益高代價） | default OFF |
| 3 | **Semantic Relink** (GMC ON) | 2026-05-13 | ❌ NO-GO | GMC + GPU bridge relink 取代；86.8% 候選被 age gate 拒絕 | default OFF |
| 4 | **Appearance 能力上限** | 2026-06-03 | ❌ **結案** | 5 模型 × 4 機制 × SR × 域訓練全撞同一天花板；清晰 200+px 框 rank-1 僅 57%，intra-inter gap ~0.03 | 模組保留 default OFF |
| 5 | **Tiled Detection** (960p 2×2/3×2) | ~2026-05 | ❌ NO-GO | FP ~8000（native_960 的 2 倍），truncation + score 污染 | code 保留，非 default |

## 輔助 NO-GO（邊際/中性/代價過高）

| # | 項目 | 時間 | 結論 | 關鍵數據 |
|---|------|------|------|----------|
| 6 | Motion-based Relinking | 2026-05-17 | ❌ NO-GO | 89% 候選被 age gate 攔截，增益 ≈ 雜訊 |
| 7 | OA-SORT OAO | 2026-05-20 / **06-12 歸因** | ⚪ 被遮蔽 | 前提成立（occ→錯配率 0.06→0.47，AUC 0.727）但懲罰形式錯誤（整列加 cost 不改排序）；新 baseline 端到端 −1.1pp。[歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md) |
| 8 | NSA-Kalman (Noise Scale Adaptive) | 2026-04-27 / **06-12 歸因** | ⚪ 被遮蔽 | 前提成立（Spearman −0.52，低分框殘差 5×）但與 `kalman_r_scale=2.8` 雙重補償；新 baseline 端到端 −3.2pp。[歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md) |
| 9 | PostMerge | 2026-04-27 / **06-12 歸因** | ⚪ 被遮蔽 | combined AUC 0.868 但 base rate 2.4% → 作用點 precision ~20%（FP +1308）；default 路徑因強制 appearance gate + ReID off **失能**；direction 分量 AUC 0.487 純噪聲。[歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md) |
| 10 | Per-frame Detection Cap / Adaptive Cap | — | ❌ NO-GO | 密集場景 adaptive cap 壓至 ~21，破壞 recall |
| 11 | P5-2 Stage2 QualityGate | ~2026-05 | ❌ NO-GO | IDF1/MOTA 統計中性 |
| 12 | P5-3 ConsecutiveBirthGate | ~2026-05 | ❌ NO-GO | 統計中性 |
| 13 | P5-4 Scene-Adaptive | 2026-05-11 | ❌ NO-GO | — |
| 14 | P5-5 Proximity Birth Gate | 2026-05-18 | ❌ NO-GO | prox=0.3 → FN +1038 / Rcll -5.6pp |
| 15 | LaSt-ViT pre-hoc embedding quality | 2026-05-02 | ❌ NO-GO | +0.09pp，SigLIP2 未訓練無區分力 |
| 16 | ROI FPN ReID | 2026-05-19 | ❌ NO-GO | cos_thr 全設定 IDs↑、IDF1 持平 |
| 17 | Horizontal-flip TTA | 2026-05-18 | ❌ NO-GO | 精度在雜訊內 |
| 18 | MOT20 混訓 | 2026-06-01 | ❌ NO-GO | domain shift 退步 |
| 19 | Pose box expansion | 2026-05-10~11 | ❌ NO-GO | 靜態 FP 無法靠 spatial 區分 |
| 20 | GMC FG Mask | 2026-05 | ❌ NO-GO | 背景紋理主導 PCR peak |
| 21 | Vel_dir gate | 2026-06-01 / **06-12 歸因** | ⚪ 被遮蔽 | 訊號分層真實（fast >3px/f AUC 0.751、slow <1 AUC 0.526≈隨機、慢速樣本占 46%）；無速度條件化 → 新 baseline 端到端 −4.6pp。[歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md) |
| 22 | Cheb-GR offline tracklet merge | 2026-06-03 | ❌ NO-GO | AssA 0.0pp |
| 23 | Birth-time lost-bank relink (GPU) | 2026-06-03 | ❌ NO-GO | 無 λ 能降 IDs；長 gap rank-1 僅 13–33% |
| 24 | YOLO non-end2end (cxcywh output) | 2026-05-08 | ❌ NO-GO | 整體退步，不升格 default |
| 25 | Cascade Filter (CrowdHuman→MOT17) | 2026-05-14 | ❌ NO-GO | MOT17 FP score 與 TP 重疊嚴重 (P≈4%)，rule 僅砍 13.3% FP |
| 26 | Pose Bio gate (Biometric relinker) | 2026-05-10 | ❌ NO-GO | Gate 僅 3 veto / 7-seq，FPS -47% |
| 27 | Narrow person score bonus | 2026-05-11 | ❌ NO-GO | 全局 IDF1 -0.3pp，FP +378 |
| 28 | Mamba temporal block (SSM, v15/17) | 2026-05-31 | ❌ NO-GO | R1→R2 grad 崩潰無法收斂 |
| 29 | Per-channel SSM A + MOT20 mix | 2026-06-01 | ❌ NO-GO | DetA 退化 -1.8pp（domain shift） |
| 30 | Cheb-GR standalone (Market-1501) | 2026-06-03 | ❌ 方法成立但不優於 fixed-k | +8.76pp vs classic +10.03pp |
| 31 | Relink bridge **scale gate** (speed 方向) | 2026-06-11 | ❌ NO-GO | MOT17-SDP 小幅正向但速度方向全線死；P0 L_med 復核不重現 |
| 32 | **Appearance relink gate**（顏色直方圖 + OSNet hard pool） | 2026-06-11 | ❌ **結案** | 全 gate AUC≈0.50、短 gap 反向 0.33；外觀方向結案 |
| 33 | **occ_cover live relink**（gap-path 占用門） | 2026-06-11 | ❌ NO-GO | live accepts 全 gap≤1；長 gap 族群被 track_buffer=30 結構性消滅；tb90 解鎖反 −0.8 IDF1 |

## NO-GO 的結構性根因

1. **Appearance 天花板**：MOT17 身份在 embedding 空間本質難分 — 5 個模型 + 4 種機制 + SR + 域訓練全撞同一個上限。這是物理瓶頸，非演算法缺陷。

2. **GMC 壓倒性主導**：GMC ON 後 IDF1 +2.8pp、IDs −133，是唯一顯著貢獻模組。其他模組在 GMC 開啟後基本冗余（∆ <0.4pp）。

3. **「密集 = FP 多」假設錯誤**：MOT17 中高密度場景是真實人多，非 FP。以 density 為信號的 filtering 策略必然傷 recall。

4. **時序資訊難進特徵層**：Mamba temporal block（v15/v17）、per-channel SSM A 全部退步。R1→R2 grad 崩潰無法收斂。

5. **Relink gate 的訊號天花板**：幾何/運動殘差對「真 vs 假橋接」AUC≈0.55（近隨機），外觀 gate AUC≈0.50；scale/occ/appearance gate 一律死在門作用區或被 `track_buffer=30` 結構性消滅。長 gap（80+）目前無單一可靠訊號 — 唯一已驗證正向是 GPU 雙向橋接本身（見 GO 表）。

---

## 中性 NO-GO 的復活前例

**Relink 系列證明：中性 ≠ 訊號不存在。** 軌跡如下：

1. Motion relink（#6）、Semantic relink（#3）單獨測試中性 → 根因是 **age gate 攔掉 86–89% 候選**，運動訊號根本沒機會作用。
2. 雙向中點橋接改變候選生成（farewell archive + 中點外推，繞過 age gate 結構）→ 同樣的運動殘差訊號變成 **IDF1 +2.1 / AssA +2.8 全指標勝利**（見 GO 表）。
3. Scale gate（#31）單獨無速度增益 → 組合進 bridge（px=0.25 + scale gate）成為 preset default 的一部分。

**結論**：中性結果登記時必須附 blocker 歸因；blocker 被後續工作移除時，回來重測。

### 已識別 blocker 的中性項目（⚪ 候補復活名單）

> 僅收錄有歸因實驗數據者；blocker 欄的數字即出處實驗的統計量。

| # | 項目 | Blocker（實測） | 復活條件 |
|---|------|---------|----------|
| 3 | Semantic Relink | age gate 拒 86.8% 候選 | 候選生成結構改變（已部分由 bidir bridge 實現） |
| 6 | Motion-based Relinking | age gate 攔 89% 候選 | 同上（已由 bidir bridge 復活） |
| 33 | occ_cover live relink | `track_buffer=30` 結構性消滅長 gap 族群；base rate 1.3% | bridge 專用過期檔案庫（不靠延長全局記憶 — tb90 已證 −0.8 IDF1） |
| 23 | Birth-time lost-bank relink | 長 gap rank-1 僅 13–33%（接近結構性，但混雜短 gap 易池） | 若有可靠長 gap 訊號源（外觀方向已結案，需新模態) |
| 8 | NSA-Kalman | 前提真（ρ=−0.52）但與 r_scale=2.8 雙重補償 | 以 score-bin 殘差曲線重校準 R(score)，整合而非疊加 r_scale |
| 21 | vel_dir gate | fast AUC 0.751 被 46% 慢速噪聲樣本（AUC 0.526）淹沒 | speed-conditioned：僅 \|v\|>3px/f 啟用 |
| 7 | OA-SORT OAO | 前提真（AUC 0.727）但整列加 cost 不改變該 track 的 det 排序 | occ-conditioned 機制重設計（嚴格 gate / 出生延遲），非 cost 偏移 |
| 9 | PostMerge | combined AUC 0.868 vs base rate 2.4%（precision ~20%）；direction 分量純噪聲 | direction_weight 歸零 + 需正交訊號拉 precision |

> 2026-06-12 歸因方法與完整數據：[訊號層歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md)。
> 共同模式：**前提成立 + 機制形式錯誤/失準** 是「中性→有害」的主要死法，而非訊號不存在。

---

## 對照：目前 GO / 穩定好用的模組

| 模組 | 狀態 | 貢獻 |
|------|------|------|
| **GPU GMC** (phase correlation) | ✅ default ON | IDF1 +2.8pp, IDs −133 |
| **Option F** (Mamba Gated Detector) | ✅ production preset | IDF1 71.2%, MOTA 76.3%, Rcll 82.3% |
| **GPUByteTracker + Sinkhorn-Auction** | ✅ default ON | 關聯延遲 0.67ms (10x 提升) |
| **Async ReID** | ✅ default ON | +2.6% FPS, 零精度損失 |
| **Pipeline Relink** | ✅ default ON | +2.5% FPS |
| **GPU 雙向橋接 Relink** (px=0.25 + scale gate) | ✅ preset default ON | IDF1 +2.1, AssA +2.8, IDs −13.6%, FP −14%（06-11 全指標嚴格優勢） |
| **FP Hard Filter** (area=40000) | ✅ default ON | FP 移除 9021, TP 移除僅 153 |
| **Kalman R Scale** (0.75) | ✅ default ON | — |
| **Detection Quality Scaling** | ✅ default ON | geometry-aware score boost |
| **Interpolation** (max_gap=35) | ✅ default ON | — |
