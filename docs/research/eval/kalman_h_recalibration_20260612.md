# Kalman h-conditioned 噪聲重校準（g(h) + f(score) v2 + Q 速度項）

**日期**：2026-06-12（同日完成擬合、實作、paired ablation）
**Worktree**：`exp/nogo-neutral-rescreen`（main@85a4564a）
**前置**：[中性 NO-GO 訊號層歸因](neutral_nogo_signal_attribution_20260612.md)（NSA 前提成立但實作反向，−3.2pp）

## TL;DR

| 臂 | 假說 | 結果 |
|---|---|---|
| B `g(h)`：R 形狀改 affine | σ∝h 線性假設錯 → 修正應有益 | **−0.4 IDF1**，反證：legacy 高估大框噪聲是有益平滑先驗 |
| D `f(score)` v2：只放大、s0 錨定 | NSA 死因=形式非訊號 | **+1.5 IDF1 / +0.86 AssA**（6/7 seq 正）— 證實；但 DetA −1.44 → **default 不開** |
| D vs B | 「NSA≈h 代理」 | **反證**：score 是主訊號，h 修正反而有害（C 76.0 < D 76.6） |
| E `Q` 速度項 | |Δv| mis-shaped | 跳過（前置條件 B 非正）；擬合本身被相機運動混淆 |
| r_scale sweep | g/f 改 R 水位後 2.8 還最優? | 2.2/3.4 全劣於 2.8（IDF1 76.3/75.8 vs 76.6） |

## 訊號層背景（n=92,980 matched dets，rescreen dumps + GT）

- **NSA 成立區間**：score ∈ [0.3, 0.94] 單調，≥0.94 飽和。42.4% dets ≥0.94；
  legacy floor 0.05 在 s>0.776 即觸發 → 傷害集中在無訊號區。
- **score-bin 殘差 var 比**（錨 score≥0.94）：0.9–0.94→1.65×、0.85–0.9→3.0×、
  0.82→5.1×、0.74→7.9×、0.66→11.2×、0.56→15.8×、0.42→22.7×。
- **σ∝h 線性假設不符**：score≥0.94 子集 σ_px 次線性
  （h~117: 0.98px、h~270: 1.63、h~657: 3.57 vs legacy h/20 = 5.8/13.5/32.8）。
- **h 獨立訊號**：控 score 後 partial ρ=−0.175（7 序列 6 個成立；09 最弱 −0.035）。
- **遮擋不用進 R**：det_occ/gt_occ/gt_vis 全被 score 中介（partial≈0）。

## 設計

新參數（全部 default off = bit-exact legacy，已驗 MOT17-04 輸出全同）：

| flag | 形式 | 啟用 |
|---|---|---|
| `--kalman-h-sigma-a/b` | R `pos_std = a + b·h`（legacy `h/20`） | a ≥ 0 |
| `--kalman-nsa-s0` | `mult = clamp(((1−s)/(1−s0))², 1, 30)`，只放大 | s0 > 0 |
| `--kalman-q-vel-a/b` | Q `vel_std = a + b·h`（legacy `h/160`） | a ≥ 0 |

f(score) v2 與既有 `--nsa-kalman` flag 獨立（保留歸因可重現性）。
`compute_S_inv`（gating 路徑）不動。Plumbing 全鏈：`geometry.py` →
`eval/config.py` → `evaluator.py` → `tracker_gpu.py` → pybind →
`tracker_gpu.cu`（`inline_kalman_update_kernel` / `predict_gmc_sinv_fused_kernel`）→
`kalman_gpu.cuh`（`get_R`/`get_Q`/`update`/`predict`）。新參數 append 末尾
（既有測試 positional）。CUDA graph 相容（kernel scalar args，capture 前 set_params）。

## Phase 0 — 擬合（`scripts/tools/fit_kalman_h_sigma.py`）

- **g(h)**（score≥0.94，pooled x/y 殘差，1.4826·MAD，10 quantile bins，WLS）：
  raw `σ_px = 0.2005 + 0.005101·h`；錨定正規化（n-加權中位 h*=210 處等於 h/20）
  → `pos_std = 1.6561 + 0.042119·h`。per-bin fit 吻合（最大偏差 ~0.16px）。
- **f(score)**：g(h) 校正後殘差（reference σ=1.0009 ✓），grid-search
  **s0 = 0.9305**。中高分段模型/實測吻合（0.88: 2.98/2.99、0.92: 1.19/1.65）；
  低分段模型偏高（0.42: 30/22.7）但 n 小。
  （前session 手動擬合 0.9409；本次以腳本化擬合 0.9305 入測。）
- **Q |Δv|**（GT 加速度噪聲，trimmed-std − 量化變異 0.625）：
  **被序列身份混淆** — 靜態 02/04 σ≈0.1（插值 GT 量化地板）、移動相機
  05/13 σ=2.3–3.7（相機加速度，tracker 在 GMC 後不會全看到）。
  控序列後 h 形狀不存在。pooled `vel_std = 0.2327 + 0.004865·h` 僅作記錄。

## Phase 2 — Paired E2E（MOT17-SDP train 7 seq，preset mamba_whole_graph）

A 臂同日重跑 = 75.1 IDF1 與晨間 baseline bit-same（單機重跑確定性成立，
重複 run 無意義，CI 以 per-seq 變異估）。

| 臂 | IDF1 | MOTA | HOTA | DetA | AssA | IDs | FP | FN |
|---|---|---|---|---|---|---|---|---|
| A baseline | 75.1 | 77.7 | 68.19 | 70.04 | 66.65 | 482 | 3514 | 21082 |
| B g(h) | 74.7 | 77.7 | 67.87 | 70.09 | 66.02 | 477 | 3648 | 20906 |
| C g(h)+f(s) | 76.0 | 77.0 | 67.49 | 68.42 | 66.91 | 514 | 3962 | 21333 |
| **D f(s) only** | **76.6** | 77.3 | 67.89 | 68.60 | **67.51** | 498 | 3813 | 21183 |
| D + r2.2 | 76.3 | 77.4 | 67.35 | 68.83 | 66.19 | 497 | 3628 | 21241 |
| D + r3.4 | 75.8 | 77.0 | 66.90 | 67.96 | 66.20 | 517 | 3919 | 21378 |

D per-seq IDF1 Δ：02 +1.4 / 04 +1.6 / **05 −2.0** / 09 +4.0 / 10 +3.6 /
11 +0.7 / 13 +0.6。IDs Δ：04 −8 / 09 −5 / 10 −22 vs **05 +24 / 13 +33**。
MOTA Δ：**13 −3.4**、11 −1.6、05 −1.6。

## 判決：全臂 default 不開（2026-06-12 判準）

1. **B（主臂）NO-GO**：IDF1 −0.4 / AssA −0.63，per-seq 範圍 +1.5～−3.0
   不一致。資料上「更正確」的 R 形狀端到端更差 — legacy h-linear 對大框的
   噪聲高估在功能上是平滑先驗（大框多近景/低速），非 bug。
2. **D（f(score) v2）最有價值但 NO-GO**：IDF1 +1.5 / AssA +0.86 在瓶頸軸
   （AssA）是 relink 之後最大單 flag 增益，且端到端證實 06-12 歸因鏈
   （同一訊號換形式：−3.2pp → +1.5pp）。但：
   - 嚴格退步存在：DetA −1.44 / MOTA −0.4 / FP +299 / HOTA −0.30；
   - 退步集中移動相機 05/13（13: MOTA −3.4 / IDs +33）；
   - per-seq paired t=1.86 < t₀.₀₅(6)=2.45，CI [−0.5, +3.3] 含 0。
3. **死因（有歸因）**：輸出箱 = Kalman 濾波態。低分量測 R 膨脹 → 狀態滯後
   量測 → IoU 對齊下降，detection-level 指標全域小幅退步，GMC 殘差大的
   移動相機序列放大此效應。AssA 增益與 DetA 損失同源不可在現架構下分離。

## 補充：σ(h,s) 乘法分解檢驗（組合形式結案）

C < D 是否因組合形式（乘法）錯誤?以 raw g(h) 正規化殘差 u，
按 score×h 格點算 var-ratio（vs 同 h 欄 score≥0.94 reference）：

| score bin | h[0–150] | h[150–220] | h[220–320] | h[320+] |
|---|---|---|---|---|
| [0.300, 0.600) | 12.68 (7k) | 21.10 (<1k) | – | – |
| [0.600, 0.776) | 6.25 (11k) | 7.63 (1k) | 10.61 (<1k) | 9.78 (<1k) |
| [0.776, 0.900) | **3.02** (21k) | **3.01** (11k) | **2.90** (2k) | **3.08** (1k) |
| [0.900, 0.940) | **1.59** (12k) | **1.47** (19k) | **1.72** (5k) | **1.45** (3k) |

score ≥ 0.776（~88% dets、R 膨脹主作用區）行內橫跨 h 完全平坦 ⇒
**σ(h,s) = g(h)·f(s) 分解成立，score 倍率不依賴 h**。低分區（<0.776）
疑似交互（小框 12.7× vs 大框 21×/10×）但 n<1k、非單調，且該區被
clamp 30 蓋住。結論：C < D 不是組合方式問題，是 g(h) 成分本身有害
（B 單獨 −0.4）且近似可加帶入；B 在 05 的 +1.5 也未在 C 兌現
（C 的 05 仍 −2.0）— 05/13 是相機運動 regime 問題，非 h。
**組合形式（加法/joint 2D fit/h-conditioned s0）方向結案，不再花 eval 預算。**

## 復活條件

- **解耦**：score-conditioned 膨脹只進 association/gating 的 S（或
  `compute_S_inv`），狀態更新/輸出用原 R 或量測箱 — 保留 +AssA、消 −DetA。
- 或 **條件化啟用**：GMC 殘差/相機運動量低時才啟用（05/13 是唯二退步源）。
- 過擬風險註記：常數擬合與 eval 同源（MOT17 train）；若解耦後 GO，需
  cross-seq holdout（fit 6 驗 1 輪換）。

## 原料與重現

- 擬合：`scripts/tools/fit_kalman_h_sigma.py` →
  `rescreen_logs/analysis/kalman_fit_constants.txt`
- Ablation：`rescreen_logs/run_kalman_ablation.sh`（main / arm_e / rscale）；
  比較：`scripts/tools/compare_kalman_ablation.py`；輸出 `results/kalman_ablation/<arm>/`
- 測試：`tests/unit/tracking/test_kalman_h_noise.py`（disabled=legacy bit-exact、
  legacy positional 相容、enabled smoke）
- 訊號層原始分析：`scripts/tools/analyze_kalman_h_signal.py`（前 session）
