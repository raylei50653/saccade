# Mamba Detector 分數分佈：框高、可見度、人群密度的歸因（2026-06-13）

> 起點是 user 對 tracker 分數門檻（`track_thresh` 0.05 / `mid_thresh` 0.10 /
> `new_track_thresh` 0.28 / `high_thresh` 0.5）的提問：**這些門檻坐落在偵測器
> 輸出分數分佈的哪裡？分數是否受框高、人群密度影響？** 本文用 95,872 個
> MOT17 GT 框的逐框匹配分數做受控歸因，把「框高」與「擁擠/遮蔽」的 confound
> 拆開。
>
> ckpt：`runs/mamba_gt_v14replica_t3_t1/best.ckpt`（current production 候選，
> 見 [T3→T1 curriculum](mamba-t3t1-curriculum-20260613.md)）。

## TL;DR

1. **分佈極度左偏飽和**：pooled matched-score median **0.932**、p05 0.515，
   95% 匹配框 ≥0.50，僅 **~2% <0.28**。→ tracker 的四個門檻**全部坐落在薄左
   尾**，對絕大多數可見行人不起作用，只裁決那 2–5% 的尾巴。
2. **框高 = 最強驅動**（spearman score~height **+0.51** pooled，per-seq 達
   +0.60）。小框（<64px）median 壓在 0.61–0.74、**緊貼門檻帶**；大框（≥128px）
   飽和到 0.94–0.95。`new_track_thresh=0.28` 的真實作用 ≈ **小目標過濾器**，
   非雜訊過濾器。呼應 [小目標高解析度 NO-GO](#) registry #36。
3. **人群密度走「漏檢」不走「壓分」**（控制框高後）：GT-GT overlap≥0.4 讓
   det% 從 ~98% 崩到 **77–83%**（製造漏檢），但**有偵到的框 median 幾乎不動**
   （128–256px 帶 0.945→0.943）。控制框高後 score~overlap 僅 **−0.058**、
   ~neighbors +0.09。
4. **序列級「人群數」是純 confound**：最擁擠的 MOT17-04（45k GT）反而 median
   最高（0.95），只因它的人都是大框（高架靜態鏡頭）。

---

## 1. 方法

工具：`scripts/eval/analyze_score_distribution.py`。對每條序列每幀：

1. 跑 `detect_single_patch_640`（conf_floor 0.001、NMS IoU 0.5、native 640），
   取 person 類別框。
2. 對每個 GT 框做**貪婪匹配**（按偵測分數降序，IoU≥0.5 認領），記錄被認領
   偵測框的分數；未匹配 GT 記 `score=NaN`（計入 detect_rate，不計入分數分佈）。
3. 記錄每個 GT 的協變量：

| 協變量 | 定義 | 代理什麼 |
|---|---|---|
| `height` | GT 框原圖像素高 | 目標尺度 |
| `visibility` | `gt.txt` 第 8 欄 | MOT17 標註的遮蔽程度 |
| `neighbors` | 中心落在本框內的其他 GT 數 | 局部擁擠 |
| `max_overlap` | 與任一其他 GT 的最大 IoU | 直接遮蔽 |
| `frame_gt` | 該幀有效 GT 總數 | 全域擁擠 |

**為何要兩種密度代理**：`frame_gt`（全域）與 `neighbors`/`max_overlap`（局部）
分離，是因為「整幀人多」不等於「這個人被擋住」。後者才是偵測器真正面對的
困難；前者極易被序列身分（鏡頭高度→框尺度）confound。

資料：MOT17 train 7 條 SDP（02/04/05/09/10/11/13），95,872 GT。
輸出：`report_data/score_dist_full.json`（per-seq + pooled binned summary +
spearman + 95k 筆 flat records，`--save-records` 時附）。

> **匹配是 score-greedy 的**：高分偵測框優先認領 GT，因此「matched score」
> 略偏樂觀（同一 GT 若有多框，記到最高分那個）。這對分佈尾部影響小（尾部
> 本就只有一個低分框），但解讀「小框 median」時要記得這是 best-match。

## 2. 分佈形狀：飽和 + 長左尾

```
pooled matched-score: mean 0.873  median 0.932  p05 0.515  p95 0.986
                      frac<.10 0.01   frac<.28 0.02   frac>=.50 0.95
```

分佈不是均勻、也不是單峰高斯，而是**一個堆在 0.9+ 的飽和峰 + 一條拖到 0.1
的左尾**。tracker 的 cascade 分界（`tracker_gpu.cu:556–629`）：

- S0/S1 HiConf：score ∈ [0.5, 1.1) ← **95% 的匹配框在這**
- S1b MidConf：[0.10, 0.5)
- S1c Tentative：[0.10, 1.1)
- S2 LoConf：[0.05, 0.10) ← 只有 ~1%

→ 門檻不是「切在分佈中間做取捨」，而是「在薄尾裡撈回困難目標」。這證成了
detection floor 0.001 與低分回收帶的存在意義。

## 3. 框高：最強驅動

| GT 高(px) | GT 數 | det% | median | frac<.28 |
|---|---|---|---|---|
| <32 | 133 | 82% | 0.61 | 0.14 |
| 32–64 | 7,997 | 92% | 0.74 | 0.09 |
| 64–128 | 17,190 | 96% | 0.84 | 0.06 |
| 128–256 | 54,075 | 96% | **0.94** | 0.01 |
| ≥256 | 16,477 | 99% | **0.95** | 0.01 |

spearman score~height = **+0.51** pooled（per-seq +0.18 ～ +0.60）。小框的
中位數就壓在門檻帶上緣，大框直接飽和。**這是門檻真正在裁決的軸**：抬高
`new_track_thresh` 主要砍掉小框/遠處目標，而非雜訊。

## 4. 人群密度：傷害是漏檢，不是壓分（核心拆解）

把 GT-GT overlap **在每個框高帶內**分層，隔離遮蔽 vs 尺度：

| 框高 | overlap | det% | median_s | frac<.28 |
|---|---|---|---|---|
| 128–256 | ~0 | **99%** | 0.945 | 0.00 |
| 128–256 | ≥0.4 | **77%** | 0.943 | 0.03 |
| 64–128 | ~0 | 98% | 0.873 | 0.03 |
| 64–128 | ≥0.4 | **83%** | 0.796 | **0.17** |
| ≥256 | ≥0.4 | 89% | 0.926 | 0.08 |
| <64 | ≥0.4 | 66% | 0.731 | 0.15 |

三個事實：

1. **重度遮蔽（IoU≥0.4）讓 det% 從 ~98% 崩到 66–83%** —— 遮蔽製造漏檢。
2. **有偵到的框 median 幾乎不動**（0.945→0.943）—— 偵到就還是自信的。偵測器
   沒有「因為被擋住所以給低分」的行為。
3. **唯一的壓分發生在「小框 × 遮蔽」交叉格**：frac<.28 衝到 0.15–0.17。
   這正是低分回收帶賺到錢的地方。

控制框高後 score~overlap 只剩 **−0.058**，score~neighbors **+0.09**。先前
未控制時看到的 neighbors 正相關，是「擁擠前景=大框=高分」的尺度 confound。

**序列級 confound 範例**：`by_frame_gt` 的 frame_15to30 帶 median 0.77 < 
frame_30to50 帶 0.93，看似「越擠分越高」，純粹因為 frame_30to50 由大框的
MOT17-04 主導。**全域人群數不可單獨解讀。**

### per-sequence

| seq | GT | det% | median | sp:height | sp:vis | sp:overlap |
|---|---|---|---|---|---|---|
| MOT17-02 | 10,993 | 95.9 | 0.915 | +0.47 | +0.42 | −0.15 |
| MOT17-04 | 45,270 | 95.0 | 0.950 | +0.18 | +0.09 | −0.05 |
| MOT17-05 | 5,271 | 99.0 | 0.933 | +0.59 | +0.17 | −0.10 |
| MOT17-09 | 4,067 | 97.5 | 0.934 | +0.32 | +0.25 | −0.20 |
| MOT17-10 | 11,117 | 96.8 | 0.838 | +0.49 | +0.30 | −0.25 |
| MOT17-11 | 8,085 | 99.2 | 0.949 | +0.60 | +0.25 | +0.05 |
| MOT17-13 | 11,069 | 95.3 | 0.805 | +0.38 | +0.33 | −0.19 |

移動鏡頭 + 小目標的序列（10/13）median 最低、框高依賴最強；高架靜態大框
（04）分數最飽和、依賴最弱。

## 5. 對門檻設計的意涵

1. **detection floor 0.001 + 低分帶合理**：只服務小框/遮蔽尾巴，正是 BYTE
   低分回收的設計意圖，不該砍。
2. **`new_track_thresh=0.28` = 小目標過濾器**：動它就是直接在小目標/遠處
   recall 上做 trade-off，不是在過濾雜訊。
3. **分佈非 mismatch，而是飽和 + 長尾**：門檻在尾巴裡有槓桿效應，要針對
   「小框 × 遮蔽」那一格調，而非全局平移。
4. **救擁擠場景該攻 detector 漏檢**（ov≥0.4 時 det% 崩塌），**不是調 score
   門檻** —— 門檻救不回根本沒輸出的框。

## 6. 如何分析分佈（重現 + 擴展）

### 重現

```bash
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"
.venv/bin/python scripts/eval/analyze_score_distribution.py \
  --sequences MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP \
  --save-records \
  --output report_data/score_dist_full.json
```

flat records（`--save-records`）每筆含 `score / height / visibility /
neighbors / max_overlap / frame_gt`，可直接餵 pandas/numpy 做：

- **分層交叉表**：固定一個協變量分箱，看另一個的 score/det_rate（§4 的做法，
  用來破 confound）。
- **partial correlation**：對 height rank 殘差化 score 後再與密度求相關。
- **2D heatmap**：score median over (height bin × overlap bin)。

### 已驗證的擴展方向（2026-06-13 同日結案）

| 方向 | 結果 | 狀態 |
|---|---|---|
| (a) 視覺化 | （未做，低優先） | — |
| (b) 門檻 sweep | §7：全域 `new_track_thresh` 淺最優 ~0.20，已落地 preset | ✅ GO（marginal） |
| (c) 框高條件化門檻 | §8：大框升/小框降雙向 oracle 上限 ≤0 | ❌ NO-GO（registry #38） |

> **方法論註記**：分數分佈問題天生充滿 confound（尺度 ↔ 可見度 ↔ 擁擠彼此
> 相關）。任何「X 影響分數」的結論都必須**控制框高後**再下，否則就是在量
> MOT17-04 的鏡頭高度。本文 §4 的分層是最低門檻，partial correlation 是更
> 嚴格的版本。

---

## 7. 門檻策略 sweep（end-to-end，確定性）

7-seq MOT17 train SDP 全域門檻 sweep（`run_threshold_strategies.sh`）。eval 是
**確定性的**——`track_thresh` 0.05→0.02 與 baseline **8 指標位元相同**，證明這些
Δ 是真實可重現的門檻效應，非雜訊。

| 策略 | IDF1 | HOTA | AssA | MOTA | IDs | Rcll | Prcn |
|---|---|---|---|---|---|---|---|
| baseline (ntt 0.28) | 75.4 | 67.7 | 66.0 | 77.6 | 496 | 81.0 | 96.5 |
| **ntt 0.20** | **75.5** | 67.7 | 66.1 | 77.6 | **473** | 80.7 | **96.8** |
| ntt 0.15 | 75.2 | 67.6 | 65.7 | 77.7 | 474 | 80.8 | 96.8 |
| ntt 0.35 | 75.0 | 67.3 | 65.3 | 77.9 | 491 | 81.3 | 96.4 |
| track_thresh 0.02 | 75.4 | 67.7 | 66.0 | 77.6 | 496 | 81.0 | 96.5 |
| mid_thresh 0.05 | 75.4 | 67.6 | 65.8 | 77.6 | 484 | 80.9 | 96.6 |

結論：
1. **`track_thresh` 0.05→0.02 = 確定性 no-op**（位元相同）→ 驗證 [0.02,0.05] 帶
   無 GT-matched 偵測，低分回收地板未 binding。
2. **`new_track_thresh` 聚合淺最優 ~0.20，但逐序列過擬合 → 撤回**：0.28→0.20
   7-seq 聚合 IDF1 +0.1 / IDs −4.6%，**但逐序列 2/7 正 3/7 負（02 +0/04 +0/
   05 −0.4/09 +2.9/10 −0.4/11 −0.9/13 +0.6），聚合全靠 MOT17-09 +2.9 撐起**。
   違反 GO「跨 seq 一致」準則 → 場景過擬合，**preset 維持 0.28**。
3. 天花板已見頂且不泛化（門檻坐薄左尾，平移只動 2-5% 尾巴，效應場景相依）。

> **過擬合教訓**：確定性 eval 的「聚合 Δ 為正」**不等於** GO。單一場景（09）
> 的強效應可把 6 條平手/負的聚合拉成淺正。MOT17 **test GT 在 MOTChallenge
> server 保留、本地不可評**，故**逐序列方向一致性是唯一本地過擬代理**——必須
> 與聚合 Δ 一起看。

## 8. 框高條件化出生門檻：雙向 NO-GO（oracle 上限，未寫 C++）

機制假說：score precision 隨框高遞減（detection 層大框低分 79% FP），故「大框升
門檻殺 FP、小框降門檻保 recall」。兩個方向都用便宜 oracle 否證。

### 8.1 大框升門檻 — 精度側 oracle（`oracle_height_birth_ceiling.py`）

對 baseline MOT 輸出做 **post-filter**（刪 height≥H & score<S 的框，所有 frame），
是 birth-only ramp 的**嚴格超集 / 上界**，用真 motmetrics 重算：

| cut | IDF1 | MOTA | Rcll | Prcn | cut 內 FP/total |
|---|---|---|---|---|---|
| baseline | 75.4 | 77.6 | 81.0 | 96.5 | — |
| h≥128 s<0.28（最 FP-pure） | 75.4 | 77.3 | 80.6 | 96.7 | 409/608 (67%) |
| h≥128 s<0.40 | 75.1 | 76.7 | 79.8 | 96.8 | 768/1589 (48%) |
| h≥256 s<0.50 | 75.0 | 76.5 | 79.7 | 96.6 | 487/1589 (31%) |

**上界 ≤0 IDF1**：最佳 cut 持平 75.4（精度 +0.2 換 recall −0.4）。死因：存活到
輸出的「大框×低分」框**只 33% 是 FP**，confirm gate 早濾掉純 ghost，**detection 層
precision 梯度不轉移到輸出空間**（同 GMC 的「GT 上限不轉移 innovation 空間」）。

### 8.2 小框降門檻 — recall 側 oracle（`oracle_small_birth_ceiling.py`）

對每個 FN GT 查它在偵測流的最佳分數。降門檻能救的 = FN 且偵測落 [thr_low, 0.28)：

| FN 類別 | 數量 | 門檻能救？ |
|---|---|---|
| 已達門檻仍漏（det≥0.28） | **3788** | ❌ 關聯/confirm 失敗 |
| 偵測器全盲（det<floor） | 974 | ❌ detector 問題 |
| 門檻可救（小框 det[.15,.28)） | 298（0.31% GT，**高估**） | ⚠️ 實測 ntt0.15 Rcll 反降 |

**recall 上界 0.31% GT（高估），真瓶頸是關聯失敗的 3788 FN（門檻可救量的 12×）。**
→ 對上 project AssA 瓶頸結論：recall headroom 在**關聯/confirmation**，不在出生門檻。

### 8.3 共同死因

門檻的可作用量是 **GT 的 0.3% 量級薄尾**；recall 的真實洞在**已偵到卻漏追的
3788 GT（關聯）**。框高條件化只是在這薄尾裡換方向，天花板雙向 ≤0。**沒寫一行
C++ 就用 oracle 結案**——詳見 registry #38。

---

## 9. Confirm-gate Pareto（同日 sweep，per-seq 不一致未落地）

`analyze_assoc_fn.py` 拆解 3788 關聯 FN → **57% 建立/確認延遲**（`confirm_score_thresh=0.50`
使偵測落 [0.28,0.50] 的軌長期 tentative）。

| 策略 | IDF1 | MOTA | HOTA | AssA | IDs | Rcll | Prcn |
|---|---|---|---|---|---|---|---|
| baseline (cst 0.50) | 75.4 | 77.6 | 67.7 | 66.0 | 496 | 81.0 | 96.5 |
| cst 0.40 | 75.4 | **78.8** | **68.0** | 65.6 | 545 | **83.2** | 95.5 |

cst0.40 救回建立延遲（MOTA +1.2/HOTA +0.3/Rcll +2.2），但 Prcn −1.0/AssA −0.4/IDs +49，
non-negative-impact-free。Per-seq 不一致（3/7 正 4/7 負、MOT17-02 −1.3~−2.2）。六種幾何
條件化手段（框高/密度/同幀近鄰/上一幀已確認 IoU/去重/3-cluster）全部無法排除負面。
唯一存活：cst0.40 為可選 **recall/MOTA Pareto 點**（非 default）。

## 10. Occlusion crossing-swap = 22% IDs —— 撞既有 appearance ceiling

3788 關聯失敗 FN 的逐幀身分追蹤 → **109 個互遮蔽交會 id-swap（佔 baseline IDs 的 22%）**。
特徵：兩 confirmed 軌互遮蔽（IoU≥0.5）1-2 幀後分開，98% 併到遮擋者軌上 → id 互換。

實作 occlusion-gated 速度鎖（C++：凍結遮蔽前速度 + temporal occl-memory + §6c
speed-weighting，12 插管點），全權重區間 **measured 單調有害**（AssA −0.4~−5.3）。
死因已由 [`offline_relink_candidate_analysis.md`](../../semantic/research/offline_relink_candidate_analysis.md)
**§4/§5/§6c** 預示：MOT17 foot speed 在 box-jitter 地板以下（median 0.01 h/f），速度方向
對慢速主體是噪聲；即使用正確的 height-normalized speed-weight 也無法在 live 交會關聯的
hard operating region（AUC ~0.65）打出淨值 → **同一 geometry/motion ceiling 經
live-association 門再確認**，剩餘 headroom 只在 appearance。該 C++ 改動已 revert。

詳見 `offline_relink_candidate_analysis.md` **§8**。
