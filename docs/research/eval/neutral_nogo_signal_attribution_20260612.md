# 中性 NO-GO 訊號層歸因分析（2026-06-12）

> **方法論**：拋棄所有後續處理（gate/關聯/合併），裸抽各機制依賴的底層訊號，
> 對 GT 標籤做分布 + 鑑別度（AUC / 條件錯誤率）分析。與
> [NO-GO 登記表](../../reference/no_go_registry.md)的「分類門檻」配套 —
> 分類必須有歸因實驗，不可從端到端結果倒推。

## 實驗環境

- main @ `85a4564a`（PR #17 merge 後），獨立 git worktree，PYTHONPATH 隔離 editable install
- preset `mamba_whole_graph` + SDP，baseline IDF1 75.1% / MOTA 77.7% / IDs 482
- 訊號原料：7 seq per-frame box dump（`--debug-dump-csv`，取 per-frame pipeline 最末 stage）
  + baseline MOT 輸出軌跡 + MOT17 GT（cls=1, consider=1）
- 工具：`scripts/tools/analyze_neutral_nogo_signals.py`（直接 import
  `post_merge.py` 私有 helper 保證公式忠實）；det/track→GT 以 IoU≥0.5 greedy 配對

## 第一部分：端到端 re-screen（結果論對照組）

同日 paired，單 flag 翻轉疊在 baseline 上：

| Run | IDF1 | MOTA | IDs | FP | FN | Δ IDF1 |
|---|---|---|---|---|---|---|
| baseline | 75.1 | 77.7 | 482 | 3514 | 21082 | — |
| `--nsa-kalman` | 71.9 | 77.0 | 564 | 3825 | 21419 | **−3.2** |
| `--vel-dir-weight 0.25` | 70.5 | 75.9 | 769 | 3650 | 22648 | **−4.6** |
| `--oao-tau 0.2` | 74.0 | 77.0 | 537 | 3570 | 21707 | **−1.1** |
| post_merge（A5 hard gate） | 75.1 | 77.7 | 482 | 3514 | 21082 | 0.0（**失能**） |
| post_merge（soft, w=0.25, max_cost=0.8） | 74.4 | 77.4 | 453 | 4822 | 20123 | −0.7（FP +1308） |

- post_merge 預設路徑失能根因（實測）：`eval/config.py` 在 `appearance_weight≤0`
  時強制 `appearance_gate=True`，而 preset `reid_mode=off` 無 bank →
  全部候選 `reject_app_missing`（per-seq 73~1593 對，accepted=0）。
- 三個舊「中性」判決在新 baseline（bidir bridge ON + `kalman_r_scale=2.8`）下
  變成明確有害 — 但這只是結果，死因見第二部分。

## 第二部分：訊號層鑑別度

### A. NSA-Kalman 前提：det score 預測量測噪聲 → **前提成立**

n=92,980 matched dets（IoU≥0.5）：

| 指標 | Spearman ρ | score 0.3–0.5 | 0.5–0.7 | 0.7–0.85 | 0.85+ |
|---|---|---|---|---|---|
| center_err/h（median） | **−0.523** | 0.0470 | 0.0347 | 0.0216 | 0.0088 |
| 1−IoU（median） | **−0.563** | 0.226 | 0.173 | 0.116 | 0.053 |

低分框定位殘差是高分框的 **5×** — score→噪聲關係強且單調。
**前提真、端到端 −3.2pp** ⇒ 死因不在訊號，在實作疊加：全局 `kalman_r_scale=2.8`
已重新校準 R，NSA 的 `(1−score)` 縮放疊上去造成雙重補償/失準。

### B. vel_dir：cos(速度方向, track→det 位移) → **訊號真實但被慢速樣本淹沒**

n=940k (track,det) 候選對：

| 速度層 | AUC | median cos 真/假 | 樣本占比 |
|---|---|---|---|
| 全體 | 0.653 | — | 100% |
| 慢 <1 px/f | **0.526（≈隨機）** | +0.14 / +0.00 | **46%** |
| 中 1–3 | 0.688 | +0.74 / −0.16 | 30% |
| 快 >3 | **0.751** | **+0.92 / −0.68** | 24% |

快速軌跡的方向訊號分離極強，慢速軌跡方向無意義（速度估計 = 噪聲）。
舊 gate 無速度條件化、46% 噪聲樣本拖垮全體 ⇒ 端到端 −4.6pp。

### C. OA-SORT OAO 前提：遮擋 IoU 預測錯配 → **前提成立**

n=91,086 track-frame，top-1 IoU 配對正確性：

| occ IoU bin | P(top-1 錯配) |
|---|---|
| 0.0–0.1 | 0.061 |
| 0.1–0.3 | 0.106 |
| 0.3–0.5 | 0.261 |
| 0.5+ | **0.467** |

AUC(occ→wrong) = 0.727。遮擋確實讓錯配率升 7.6×。
**前提真、端到端 −1.1pp** ⇒ 死因在懲罰形式：`cost += tau·occIoU` 對該 track
整列均勻加成，不改變該 track 對各 det 的相對排序，只把遮擋 track 在全局
指派中降權 — 等於懲罰受害者而非修正選擇。

### D. post_merge cost 分量：真假 tracklet 對 → **可分但 base rate 殺死作用點**

n=3,697 對（gap 1–60），真對 90，**base rate 2.4%**：

| 分量 | AUC | median 真/假 |
|---|---|---|
| spatial | 0.864 | 0.96 / 3.62 |
| motion | 0.745 | 0.88 / 1.00 |
| time | 0.555 | — |
| **direction** | **0.487（純噪聲）** | 0.00 / 0.00 |
| combined | 0.868 | 0.82 / 1.84 |

- combined AUC 0.868 看似可用，但 2.4% base rate 下作用點 precision 只剩
  ~20%（端到端實測：accepted 195、FP +1308、IDF1 −0.7）— 與
  relink candidate dataset 的結論（base rate 1.3%）同構。
- `direction_weight=0.25` 預設權重在加純噪聲（AUC 0.487），與 bidir 殘差
  ROC 分析（幾何特徵 AUC≈0.55）一致。
- gap 30–61 AUC 0.901：長 gap 對在輸出層面可分 — 與 gate 作用區 AUC≈0.65
  不矛盾（此處是 easy+hard 全池，且 bridge/interp 已吃掉易例後的殘餘）。

## 判決與登記表回填

| 機制 | 端到端 | 訊號層 | 分類 | 復活條件 |
|---|---|---|---|---|
| NSA-Kalman | −3.2pp | 前提成立（ρ=−0.52） | ⚪ 被遮蔽 | 以 score-bin 殘差曲線重新校準 R(score)，取代 `(1−score)` 與 r_scale 的疊加 |
| vel_dir | −4.6pp | fast AUC 0.751 / slow 0.526 | ⚪ 被遮蔽 | speed-conditioned 應用（僅 \|v\|>3px/f 啟用） |
| OAO | −1.1pp | 前提成立（AUC 0.727） | ⚪ 被遮蔽 | 改懲罰形式：occ-conditioned 嚴格化（gate/出生延遲），非整列加 cost |
| post_merge | −0.7pp / 失能 | combined AUC 0.868、base rate 2.4% | ⚪ 被遮蔽 | 需正交訊號拉 precision（外觀已結案）；先把 direction_weight 歸零 |

共同模式：**「前提成立 + 機制形式錯誤/失準」是中性→有害的主要死法**，
而非訊號不存在。固定 +3pp 時代把它們記成「中性」掩蓋了這個結構。

## 原料與重現

- 工具：`scripts/tools/analyze_neutral_nogo_signals.py`（exp/nogo-neutral-rescreen）
- 原始 CSV：worktree `rescreen_logs/analysis/{nsa,vel_dir,oao,post_merge}.csv`
- dump 收集：`rescreen_logs/run_dumps.sh`；re-screen：`rescreen_logs/run_rescreen.sh`

---

## 後續：Kalman h-conditioned 噪聲重校準實驗（2026-06-12 同日）

> 完整設計、擬合數據與重現步驟：[kalman_h_recalibration_20260612.md](kalman_h_recalibration_20260612.md)

依上表 NSA 復活條件執行：以資料擬合 affine 修正 R 的 h 形狀（g(h)）、疊加
score 重校準（f(score) v2）、修正 Q 速度項。擬合 + 實作 + paired ablation
全在 worktree `exp/nogo-neutral-rescreen`（main@85a4564a）。

### Phase 0 — 離線擬合（`scripts/tools/fit_kalman_h_sigma.py`）

- **g(h)**（score≥0.94 子集，n=39,409 dets，pooled x/y 殘差）：
  `σ_px = 0.20 + 0.0051·h` — 證實 σ 次線性（h~117 實測 0.98px vs legacy 5.8；
  h~657 實測 3.6 vs legacy 32.8）。錨定正規化（h*=210 處 = h/20）後
  `pos_std = 1.6561 + 0.042119·h`。
- **f(score)**：g(h) 校正後殘差 var-ratio 擬合 s0=**0.9305**
  （bin 實測 22.7/15.8/11.2/7.9/5.1/3.0/1.65/1.0，模型形狀吻合中高分段）。
- **Q |Δv|**：GT 加速度噪聲被序列身份混淆（靜態 02/04 σ≈0.1 量化地板、
  移動 05/13 σ=2.3–3.7 為相機運動非 h 效應）；pooled affine
  `vel_std = 0.2327 + 0.0049·h` 僅作記錄，擬合品質差。

### Phase 2 — Paired E2E（MOT17-SDP, mamba_whole_graph；A 同日重跑 = 75.1 bit-same）

| 臂 | IDF1 | MOTA | HOTA | DetA | AssA | IDs | FP | FN |
|---|---|---|---|---|---|---|---|---|
| A baseline | 75.1 | 77.7 | 68.19 | 70.04 | 66.65 | 482 | 3514 | 21082 |
| B g(h) | 74.7 | 77.7 | 67.87 | 70.09 | 66.02 | 477 | 3648 | 20906 |
| C g(h)+f(s) | 76.0 | 77.0 | 67.49 | 68.42 | 66.91 | 514 | 3962 | 21333 |
| **D f(s) only** | **76.6** | 77.3 | 67.89 | 68.60 | **67.51** | 498 | 3813 | 21183 |
| D + r_scale 2.2 | 76.3 | 77.4 | 67.35 | 68.83 | 66.19 | 497 | 3628 | 21241 |
| D + r_scale 3.4 | 75.8 | 77.0 | 66.90 | 67.96 | 66.20 | 517 | 3919 | 21378 |

（E 臂 g(h)+Q 依預載條件跳過：B 非正。r_scale=2.8 仍最優。）

### 判決

1. **「NSA≈h 代理」假說反證**（D vs B）：score 是主訊號（單獨 +1.5），
   h 形狀修正單獨 **−0.4** 且疊加後拖累（C 76.0 < D 76.6）。資料擬合的
   R 形狀「更正確」端到端卻更差 — legacy h-linear 高估大框噪聲在功能上
   是有益的平滑先驗，非 bug。
2. **f(score) v2 證實 06-12 歸因的死因鏈**：同一 score 訊號，換成
   「只放大、s0 錨定、無 floor 壓平」的形式即從 −3.2pp 變 **+1.5pp IDF1 /
   +0.86 AssA**（6/7 seq IDF1 正：02 +1.4 / 04 +1.6 / 09 +4.0 / 10 +3.6 /
   11 +0.7 / 13 +0.6；05 −2.0）。
3. **但按 06-12 判準 NO-GO（default 不開）**：DetA −1.44 / MOTA −0.4 /
   FP +299 / HOTA −0.30 屬嚴格退步；退步集中移動相機序列
   （13: MOTA −3.4 / IDs +33；05: IDF1 −2.0 / IDs +24）；per-seq paired
   t=1.86 < t₀.₀₅(6)=2.45，CI [−0.5, +3.3] 含 0。
4. **死因（有歸因）**：輸出箱 = 濾波態。低分量測 R 放大 → 狀態滯後量測 →
   IoU 對齊下降（DetA/FP 全域性退步），在 GMC 殘差大的移動相機序列被放大。
   association 增益（AssA +0.86，瓶頸軸）與 localization 損失同源。

### 復活條件（更新）

score-conditioned R 與輸出解耦：膨脹只進 association/gating 的 S，
輸出箱用量測（或後驗用原 R）；或以 GMC 殘差/速度條件化啟用。
flags `--kalman-h-sigma-a/b --kalman-nsa-s0 --kalman-q-vel-a/b` 已全鏈
plumbed（default off = bit-exact，MOT17-04 已驗證）。

### 原料

- 擬合：`scripts/tools/fit_kalman_h_sigma.py`；常數
  `rescreen_logs/analysis/kalman_fit_constants.txt`
- ablation：`rescreen_logs/run_kalman_ablation.sh`、
  `scripts/tools/compare_kalman_ablation.py`；輸出 `results/kalman_ablation/`
- 測試：`tests/unit/tracking/test_kalman_h_noise.py`（4 passed，
  disabled=legacy bit-exact）
