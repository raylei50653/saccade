# Bridge relink gate — 穩定邊界與參數耦合（2026-08-08）

> **本文回答的問題**:能不能用數學在 bridge relink gate 上找出**參數關聯與穩定邊界**(而非極值)?
> 附帶結果是發現 `mamba_whole_graph_m` 的高度比 gate 站在一道不連續的錯誤側。
>
> 全部數字出自 2026-08-08 同一批 run:`mamba_whole_graph_m` preset、MOT17 **train** / SDP 七序列、
> RTX 5070 Ti Laptop GPU 12 GB、main `f1dfc616`、`--double-buffer --no-gpu-decode`。
> IDF1 由存下的 MOT 輸出**重算至三位小數**(eval 只印一位,不足以分辨平台與懸崖)。
>
> ⚠️ **leakage:in-sample(training-set)絕對值**,與 [frozen_v2_ablation](frozen_v2_ablation.md) 同一限制。
> §3 的 LOSO 只縮小「單一序列驅動」這一種**選擇洩漏**,**不處理** detector leakage,
> 也不排除 grid / 候選 family / 分析方法本身看過這七條序列。
>
> 姊妹文件:[reid_handover_ablation_20260808](reid_handover_ablation_20260808.md)(同一批 run 的 ReID / handover 部分)。

---

## 0. 為什麼現在做得動:`--no-gpu-decode` 下品質指標 bit-exact

三 arm × 三重複,每個品質指標 stdev **±0.00**。先前這條線的研究把
`no evaluator rerun` 列為 validation gate,正是因為當年重跑不可信;
確定性成立後 **N=1 per cell 即可**,一個 5×5 sweep 才在成本內。

座標選擇沿用已接受的 global safe axes `{dist_h, log_h_ratio}`,它們在 production 就是:

| 研究軸 | production 旋鈕 |
|:--|:--|
| `dist_h` = \(\|a_{\text{lost}}-a_{\text{cand}}\|/h_{\text{ref}}\) | `relink_bridge_px` |
| `log_h_ratio` = \(\log(h_{\text{lost}}/h_{\text{cand}})\) | `relink_bridge_h_lo` / `_h_hi` 的區間 |

**直接量 production gate 本身**,而不是 offline proxy —— 這是刻意繞過
`S0_UNDECIDABLE`(offline 座標扛不動 guarantee),不是解掉它。

---

## 1. `(relink_bridge_px, w)` 曲面

`w` = log 高度比窗的**半寬**,`h_lo = e^{-w}`、`h_hi = e^{+w}`。
用半寬的理由:**兩個出貨 preset 在 log 空間本來就近乎對稱**
(s 0.75/1.33 → w=0.286;m 0.6/1.7 → w=0.521)—— 這本身就是 `log_h_ratio` 是自然座標的證據。

```
    w \ px    0.250    0.325    0.400    0.475    0.550
      0.29   80.742   80.825  [81.089]  80.978   80.867   <- s 的窗
      0.40   80.587   80.686   80.933   80.716   80.575
      0.52   80.543   80.718   80.887   80.393   80.202   <- m 的窗（gridded）
      0.64   80.540   80.713   80.501   79.936   79.674
      0.75   80.539   80.686   80.382   79.661   79.412
              ^s 的 px        ^m 的 px
```

二次擬合(R² = 0.935):cross term \(x\!\cdot\!y = -0.198\)(≠0 ⇒ 不可分離);
Hessian 主軸 **25.7° / 115.7°**,非 0/90°;主曲率 −0.412(沿 25.7°)、+0.095。

> ⚠️ **這個擬合後來被 §4 取代。** 曲面實測為**分段常數 + 跳躍**(平台上 bit-identical、
> 跨界瞬間掉),不是平滑脊。二次擬合只是對「跳躍界線之排列」的平滑描述,
> 不可讀成平滑曲率。§4 用離散、可計數的方式重做同一個問題。

**跨序列脆性**:七條序列有 **6 條偏好最緊的 `w=0.29`**(w 軸有一致最優),
但 `px` 完全不一致 —— 0.25(05)、0.325(04/13)、0.475(10)、0.55(02/11)各自贏過。
且 spread 極不均:MOT17-04 只有 0.212(幾乎不在乎)卻占 GT 權重約 42%,
而真正在乎的 11(9.41)/05(7.80)/09(7.06)彼此不同意。
⇒ **`w` 有明顯的跨序列方向一致性;`px` 的逐序列 optimum 高度異質**,
沒有 per-sequence-consensus 的理由支持移動現行的 0.4。
**但這不等於「全域選 `px` 必然過擬合」** —— §3 的 pooled LOSO selector
(連同時調 `px` 的貪婪版)**7/7 fold 都選回 `px=0.4`**。
兩件事同時成立:逐序列偏好異質 ≠ pooled 選擇不穩。

---

## 2. 出貨值站在懸崖上,而且完全是上緣造成的

控制實驗:把 flags 設成 preset 值 → IDF1 **80.447**,與完全不傳 flags **相同** ⇒ sweep 可信。
於是以下 1% 級的差異是真的:

| `h_lo` / `h_hi` | pooled IDF1 | MOT17-11 |
|:--|--:|--:|
| 0.5945 / 1.6820 | 80.887 | 84.922 |
| **0.6000** / 1.6820 | 80.887 | 84.922 |
| 0.5945 / **1.7000** | 80.447 | 79.752 |
| 0.6000 / 1.7000 ← **出貨 m** | 80.447 | 79.752 |

**`h_lo` 在此完全無影響;`h_hi` 從 1.682 → 1.700(1.07%)= MOT17-11 −5.17、pooled −0.44。**
出貨的 m 正好落在 `h_hi = 1.7` 這道不連續的**錯誤那一側**。

局部剖面(`px=0.4`,對稱窗)顯示候選在**寬平台**上、出貨值是離群:

| `w` | 0.220 | 0.255 | 0.287 | 0.290 | 0.325 | 0.360 | 0.400 | 0.520 | 出貨* |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 81.178 | 81.148 | **81.161** | 81.089 | 81.072 | 80.966 | 80.933 | 80.887 | **80.447** |
| MOT17-11 | 85.980 | 85.980 | 85.980 | 85.980 | 85.980 | 84.530 | 84.530 | 84.922 | **79.752** |

`w` 在 0.220–0.325(48% 範圍)內 pooled 只動 0.11,**MOT17-11 更是 bit-identical 全程不動**。
\*出貨 = `h 0.6/1.7`,在 log 空間略不對稱(−0.511 / +0.531),故單獨列出。

---

## 3. Held-out 驗證:選擇穩定,增益集中

真正的風險是**選擇洩漏** —— `w` 是在同一批七條序列上挑的。
Leave-one-sequence-out(以另六條的**彙總計數**選,IDF1 是比值不可取逐序列平均):

**7/7 fold 都選最緊的 `w=0.29`**;連「同時調 `px`」的貪婪版也每一 fold 都選 `px=0.4, w=0.29`。
⇒ 選擇不依賴任何單一序列;**LOSO 未觀察到 single-sequence-driven selection instability**。
LOSO **不能**排除整個 grid、候選 family 與分析方法本身都看過這七條序列 —— 它縮小的是
單一序列驅動這一種選擇洩漏,不是全部。

逐序列(實際 config,非 grid 近似):

| 序列 | 出貨 0.6/1.7 | 候選 0.75/1.33 | Δ |
|:--|--:|--:|--:|
| MOT17-11 | 79.752 | 85.980 | **+6.229** |
| MOT17-05 | 71.348 | 73.636 | **+2.289** |
| MOT17-10 | 71.249 | 71.635 | +0.386 |
| MOT17-02 | 59.352 | 59.352 | +0.000 |
| MOT17-04 | 93.127 | 93.127 | +0.000 |
| MOT17-09 | 67.735 | 67.714 | −0.021 |
| MOT17-13 | 78.094 | 78.076 | −0.018 |
| **POOLED** | **80.447** | **81.161** | **+0.714** |

**未見實質退化**(量測最差 −0.021;兩條 bit-identical)。
注意 §0 已建立 `--no-gpu-decode` 下 bit-exact ⇒ `−0.021` / `−0.018` **不是 run-to-run noise,
而是可重現但極小的負 delta**,不以「噪聲」替其開脫。但**增益高度集中**:
all-7 **+0.714** → 去掉 11 剩 **+0.200** → 再去掉 05 只剩 **+0.052**。

⇒ 誠實的賣點不是「+0.714 accuracy」,而是**從懸崖移到平台**這件事。
**即使不把 +0.714 視為可泛化的 accuracy gain,「這批資料上的局部參數穩定性(cliff → plateau)」
仍獨立成立。** 但該穩定性也僅在 MOT17-train 這批資料上建立,不是跨資料集或跨 detector 的 robustness。

---

## 4. Threshold-configuration 平面:斜度證實,但零厚度另有主因

高度 gate 就是 `h_lo < ratio < h_hi` —— 字面上兩個 threshold 的 **AND**。
所以 `(h_lo, h_hi)` 平面正是舊 Q4.5 atlas 那種 threshold-configuration 空間,
而**一個 G2 AND term 在此平面上恰好是一個軸對齊矩形**。
於是舊 atlas 的度量可以**直接複製**,不必擬合。`px` 固定於出貨的 0.4。

```
 h_lo\h_hi   1.20   1.30   1.40   1.50   1.60   1.65   1.69   1.75
      0.40  81.03  80.90  80.85  80.83  80.82  80.85  80.84  80.40
      0.52  81.06  81.00  80.95  80.93  80.92  80.95  80.94  80.50
      0.64  81.01  80.96  80.90  80.89  80.87  80.91  80.90  80.46
      0.76  81.20  81.15  81.09  81.08  81.07  80.69  80.68  80.68   <- 最佳列
      0.88  81.08  81.09  80.75  80.74  80.73  80.49  80.49  80.48
      0.96  80.53  80.64  80.50  80.50  80.47  80.23  80.23  80.22
```

### 4.1 斜度證實(非參數)

**跳躍位置隨 `h_lo` 移動**:

| `h_lo` | 懸崖位置(`h_hi`) |
|:--|:--|
| 0.40 / 0.52 / 0.64 | **1.69 → 1.75** |
| 0.76 | **1.60 → 1.65** |
| 0.88 | **1.30 → 1.40** ＋ 1.60 → 1.65 |
| 0.96 | 1.60 → 1.65 |

「上緣何時開始有害」**取決於下緣在哪** ⇒ 兩 threshold **不獨立**。
AND term 說的是「\(h_{lo}>a \wedge h_{hi}<b\)」,\(a,b\) 各為常數,**表達不出 \(b\) 隨 \(a\) 變**。
覆蓋 good set 所需的軸對齊矩形數:tol 0.20 → **3 個**、tol 0.30 → **5 個**。

### 4.2 但「0 厚度」的主因是判準嚴格度,不是文法

| tol (IDF1) | \|good\| | full-nbhd | 單一矩形? | 最少矩形 |
|--:|--:|--:|:--|--:|
| 0.05 | 2 | **0** | True | 1 |
| 0.10 | 2 | **0** | True | 1 |
| 0.20 | 10 | **0** | False | 3 |
| 0.30 | 19 | 1 | False | 5 |
| 0.50 | 31 | **12** | False | 2 |

`full-nbhd` = 四鄰全為 good 的格數,即舊 atlas 的 `full_neighborhood_safe_radius >= 1`(舊值 **0/154**)。

**嚴格 tol 下舊 atlas 的「0 full-neighborhood」在真實座標下完全重現。**
厚度要放寬到約 0.5 IDF1 才出現(12 個內部格)。
⇒ 舊 `isolated_safe_points_only` 的主要驅動力更可能是那個 conservative dual-margin policy 的**嚴格度**,
文法是次要因素(但確有貢獻:界線是斜的)。**換座標不會讓它變厚。**

---

## 5. 這份文件建立與未建立什麼

**建立**:上述曲面在 main `f1dfc616` 上可重現;`h_hi` 在 1.7 附近存在真實不連續;
出貨 m 落在其錯誤側;`h_lo≈0.76` 列存在寬平台;`w` 的選擇在 LOSO 7/7 fold 下穩定;
**`px` 的逐序列 optimum 異質,但 pooled LOSO 7/7 fold 均選回現行的 0.4**;
`(h_lo, h_hi)` 的跳躍界線非軸對齊。

**未建立**:

- **不是 preset 變更。** production 與所有 preset **未改動**;§6 是候選不是決定。
- **未跨資料集驗證。** LOSO 只縮小「單一序列驅動」這一種選擇洩漏 —— 整個 grid、候選 family
  與分析方法本身都看過這七條序列;detector 也在這七條上訓練過。
  MOT20 / DanceTrack / SportsMOT / PersonPath22 都在 `datasets/` 下但**未使用**。
- **不是舊 safe-region 研究線的續作。** 本文量的是 **IDF1 曲面的穩定性**,
  舊線量的是 **GT-retention 安全準則** —— 不同物件,勿當同一件事登記。
  `S0_UNDECIDABLE` / `A1_ACCEPTED_WITH_LIMITS` 等既有終點**不受本文影響**。
- **不是 NO-GO 登記。**
- §1 的 26° 二次擬合**不得**單獨引用;它已被 §4 的非參數結果取代。
- 偶發 DALI `cudaErrorStreamCaptureUnsupported`(CUDA graph capture 期間 DALI 動作)
  會弄掉整格;本批 73 格中 3 格中獎,重跑即通過。

---

## 6. 候選變更(**未套用**;**已於 cross-dataset 驗證中被否決**)

> ⛔ **本節候選已被 [bridge_gate_cross_dataset_20260808](bridge_gate_cross_dataset_20260808.md) 否決**:
> 在 MOT20 上中性(−0.013)、在 DanceTrack 上有害(−0.753),且本文 §2/§4 的 cliff/plateau
> 結構在該二資料集上**均不重現**。以下內容保留為當時的候選與其推導,**不得作為變更依據**。

`configs/presets/mamba_whole_graph_m.yaml`:

```yaml
# 現行
relink_bridge_h_lo: 0.6
relink_bridge_h_hi: 1.7
# 候選（平台中央，非極值）
relink_bridge_h_lo: 0.76      # h_hi 取 1.2–1.6 任意值皆在同一平台內
relink_bridge_h_hi: 1.4
```

`relink_bridge_px` **維持 0.4**:逐序列 optimum 雖異質,但 §3 的 pooled LOSO
**7/7 fold 均選回 0.4**,因此本文**沒有 evidence 支持移動它**。

量到的效果(七序列 pooled):IDF1 80.447 → **81.161**、HOTA 74.4 → **75.0**、
AssA 73.5 → **74.7**、IDs 344 → **339**、MOTA 81.6 持平、FP 1992 → 1922。

**建議以平台區間而非單點記載**,避免後人誤讀為調出來的極值。
[math_model.md §1.1](../math_model.md) 記載 m 放寬高度 gate 的意圖是「小框 recovery」;
本文的量測顯示該 delta 為負收益。套用前需 held-out 驗證。

---

## 7. 重生指令

```bash
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
BASE="scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
      --double-buffer --no-gpu-decode"

# 出貨 m 基準
.venv/bin/python $BASE --output out/gate_shipped

# 候選（平台中央）
.venv/bin/python $BASE --relink-bridge-px 0.4 \
    --relink-bridge-h-lo 0.76 --relink-bridge-h-hi 1.4 --output out/gate_candidate

# 懸崖:只動上緣 1.07%
.venv/bin/python $BASE --relink-bridge-px 0.4 \
    --relink-bridge-h-lo 0.6 --relink-bridge-h-hi 1.682 --output out/gate_below_cliff
.venv/bin/python $BASE --relink-bridge-px 0.4 \
    --relink-bridge-h-lo 0.6 --relink-bridge-h-hi 1.700 --output out/gate_above_cliff
```

> IDF1 必須從輸出重算至三位小數 —— eval 只印一位,分不出 81.07 與 81.20 這種平台結構。
> 重算方式:對每序列呼叫 `saccade.perception.eval.metrics._evaluate_single_sequence`
> 取 `idtp/idfp/idfn` 後彙總,`IDF1 = 2·idtp / (2·idtp + idfp + idfn)`(比值不可逐序列平均)。

---

## 相關

- [bridge_gate_cross_dataset_20260808.md](bridge_gate_cross_dataset_20260808.md) — **後續:本文候選的 cross-dataset 否決**
- [reid_handover_ablation_20260808.md](reid_handover_ablation_20260808.md) — 同批 run 的 ReID / handover 部分
- [frozen_v2_ablation.md](frozen_v2_ablation.md) — 現行 headline 累積消融
- [math_model.md](../math_model.md) — §1.1 s/m preset delta 與 §10 bridge relink 模型
- [mot17_default_config.md](../mot17_default_config.md) — baseline preset
