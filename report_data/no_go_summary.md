# Saccade NO-GO 全局登記表 (Global NO-GO Registry)

> **用途**：跨模組「已結案/已踩雷方向」總覽，避免重複探索。每列只記結論一行，數據細節以對應模組的 `research/` 或 `decisions/` ADR 為準。
> 彙整自 `decisions/`、`modules/*/README.md`、`archive/`、`reference/PIPELINE_REFERENCE.md`、`TODO.md`（路徑相對 `docs/`）。
> 最後更新：2026-06-12

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
| 7 | OA-SORT OAO | 2026-05-20 | ❌ NO-GO | MOT17-SDP ±0.3pp（雜訊內） |
| 8 | NSA-Kalman (Noise Scale Adaptive) | — | ❌ NO-GO | 無效果 |
| 9 | PostMerge | — | ❌ NO-GO | 確認有害 |
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
| 21 | Vel_dir gate | 2026-06-01 | ❌ NO-GO | — |
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
