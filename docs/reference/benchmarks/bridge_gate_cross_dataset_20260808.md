# Bridge gate 候選變更 — cross-dataset 驗證(2026-08-08)

> **結論先講:候選變更被否決。** [bridge_gate_stability_20260808](bridge_gate_stability_20260808.md)
> §6 的候選(`h_lo=0.76 / h_hi=1.4`)在 MOT17 上值 **+0.642 IDF1**,但在 MOT20 上
> **近乎中性(−0.013)**、在 DanceTrack 上**明顯有害(−0.753)**;而 MOT17 那個
> **cliff-plus-row-plateau 幾何在兩個外部資料集上都未重現**。
>
> 全部數字出自 2026-08-08,`mamba_whole_graph_m` preset、RTX 5070 Ti Laptop GPU 12 GB、
> `--double-buffer --no-gpu-decode`、`relink_bridge_px = 0.4` 固定。
> IDF1 由存下的 MOT 輸出重算至三位小數。
>
> **前作**:[bridge_gate_stability_20260808](bridge_gate_stability_20260808.md)(MOT17 上的原始發現)。

---

## 0. 凍結協定:外部資料上不做任何選擇

候選在 MOT17 上選定後**完全凍結**:`px=0.4`、`h_lo=0.76`、`h_hi=1.4`。
外部資料集上**沒有任何參數是被挑出來的**。

六個 arm 在三個資料集上完全相同。其中四個非候選 arm 的作用是**量測結構**
(懸崖是否存在、平台是否平),不是搜尋更好的點:

| arm | `h_lo` | `h_hi` | 角色 |
|:--|:--|:--|:--|
| `shipped` | 0.60 | 1.70 | 現行出貨值 |
| `cliff_below` | 0.60 | 1.682 | MOT17 懸崖的下側 |
| `cliff_far` | 0.60 | 1.60 | 出貨下緣 ＋ 更緊上緣 |
| `cand_lo` | 0.76 | 1.20 | 已宣告平台區間的低端 |
| **`candidate`** | **0.76** | **1.40** | ★ MOT17 凍結候選 |
| `cand_hi` | 0.76 | 1.60 | 平台區間的高端 |

### 0.1 資料集選擇與 provenance

出貨 detector 為 `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt`。
其 checkpoint 目錄**只有 `best.ckpt` / `latest.ckpt`,不含任何 provenance metadata**;
訓練語料是從 [holdout plan](../../modules/detection/research/holdout_generalization_plan.md) §0
反推的 ——「detection head + gated teacher + feature cache **全吃 MOT17 7 條**」。
⇒ MOT20 與 DanceTrack **都在其 training provenance 之外**。

> ⚠️ **這個 provenance 是從一份 plan 文件反推,不是 checkpoint 自帶紀錄。** 補一份
> checkpoint provenance record 是獨立且未完成的工作。

**PersonPath22 刻意排除。** 它是本專案**指定的未來訓練語料**(holdout plan §1:teacher + cache
+ distill + T3→T1 全鏈只吃 PP22)。在其上收的證據會被未來的 PP22-trained detector **回溯汙染**。
排除理由是 **provenance,不是資料品質** —— 它的域其實最接近 MOT17。

**措辭紀律**:cross-dataset 直接處理的是 external generalization / MOT17 in-sample dependence;
本文能主張「解除 detector leakage」**僅因為**已確認這兩個資料集不在 detector 的 training provenance 裡。

---

## 1. 三個資料集的完整結果

| arm | `h_lo`/`h_hi` | MOT17 (7 seq) | MOT20 (4 seq) | DanceTrack (40 seq) |
|:--|:--|--:|--:|--:|
| `shipped` | 0.60 / 1.70 | 80.447 | **39.164** | **34.474** |
| `cliff_below` | 0.60 / 1.682 | 80.887 | 39.138 | 34.455 |
| `cliff_far` | 0.60 / 1.60 | 80.860 | 39.137 | 34.472 |
| `cand_lo` | 0.76 / 1.20 | **81.198** | 39.095 | 32.395 |
| **`candidate`** | **0.76 / 1.40** | 81.089 | 39.151 | 33.722 |
| `cand_hi` | 0.76 / 1.60 | 81.067 | 39.150 | 33.767 |
| | **候選 − 出貨** | **+0.642** | **−0.013** | **−0.753** |
| | **六 arm 總 spread** | 0.751 | **0.069** | 2.079 |

(候選在 MOT17 為 **+0.642** 而非前作的 +0.714,因為此處用的是 §6 的 `(0.76, 1.40)`,
前作的 +0.714 量的是 `(0.75, 1.33)`。兩者同屬前作宣告的平台區間。)

> **delta 由 full-precision 分數計算,各 arm 的顯示值才 round 至三位小數。**
> 故表內相減可能與 delta 差 0.001(例:DanceTrack 顯示 33.722 − 34.474 = −0.752,
> full-precision delta 為 **−0.753**)。

---

## 2. Q1 — 外部表現:MOT20 近乎中性,DanceTrack 明顯有害

> ⚠️ **本文未事先註冊 non-inferiority margin,因此以下不是 non-inferiority 檢定。**
> 沒有預設界限時,**MOT20 不能形式上稱為「通過」,DanceTrack 也不能形式上稱為「失敗」**。
> 以下一律採描述性陳述(near-neutral / materially harmful)—— 候選的否決**不依賴**任何 margin。

| | pooled Δ | IDs | 逐序列 | 最差序列 | 描述 |
|:--|--:|--:|:--|:--|:--|
| **MOT20** | **−0.013** | 3819 → 3800 | 1 好 / 2 壞 / 1 同 | MOT20-02 −0.071 | near-neutral |
| **DanceTrack** | **−0.753** | 4702 → 4875 | **14 好 / 23 壞 / 3 同** | dancetrack0024 **−9.221** | materially harmful |

DanceTrack 最好的序列是 dancetrack0062 +4.386 —— 逐序列離散度極大,方向卻是淨負。

---

## 3. Q2 — 結構:MOT17 的 cliff-plus-row-plateau 幾何兩域皆未重現

MOT17 的兩個結構特徵:`h_lo=0.60` 時 `h_hi` 在 1.682→1.700 有懸崖;`h_lo=0.76` 那列平坦。

| | 懸崖 1.682→1.700 | `h_lo=0.76` 列的 spread |
|:--|--:|--:|
| **MOT17** | **−0.440** | **0.131**(平台) |
| MOT20 | +0.026 | 0.056 |
| DanceTrack | +0.019 | **1.372**(斜坡) |

- **懸崖在兩個外部資料集都消失**(+0.026 / +0.019,方向還相反)。
- **MOT20:整個曲面近乎平坦**(六 arm spread 0.069)。`h_lo=0.76` 那列的 spread 0.056
  **比 MOT17 的 0.131 還平** —— 但正因為整個曲面都平,**那列不再是一個有辨識性的結構**,
  只是普遍平坦的一部分。
- **DanceTrack:該列變成斜坡**(spread 1.372,10 倍)。

⇒ 沒有轉移的是 **MOT17 那個具辨識性的組合:`h_hi≈1.7` 的懸崖 ＋ 在非平坦曲面上突出的 row-specific 平台**。
兩個外部資料集各以不同方式缺少它 —— MOT20 是無處不平(故無突出可言),DanceTrack 是該列反成斜坡。
**⇒ 該 cliff-plus-row-plateau 幾何不是 domain-general 的。**

> ⚠️ 精確地說:**不是**「兩個外部資料集都沒有平坦的列」—— MOT20 的那列比 MOT17 更平。
> 沒有重現的是**該列相對於其所在曲面的突出性**,以及與之配對的懸崖。

---

## 4. MOT20 的近乎中性不是空的 —— mechanism engagement 檢查

MOT20 六 arm 總 spread 僅 0.069,足以懷疑 gate 在該資料集根本不作用 ——
若真如此,**任何**設定都會顯得中性,該結果即為空洞。故先查觸發量:

| dataset | bridge attempts | accepts | accept rate | Δ accept count | ΔIDF1 |
|:--|--:|--:|--:|--:|--:|
| MOT17 | 881 | 194 | 22.0% | — | +0.642 |
| MOT20 | **10,715** | **2,427** | 22.7% | **−72** | −0.013 |
| DanceTrack | 7,643 | 3,931 | **51.4%** | −268 | −0.753 |

**MOT20 的 bridge 觸發量是 MOT17 的 12 倍**,accept rate 幾乎相同,且候選使**總 accept count 減少 72**。
⇒ **gate 有在動,是 MOT20 的 outcome 對這些改變不敏感** —— 該近乎中性的結果為實質而非空洞。

> ⚠️ **`Δ accept count` 是 aggregate 計數差,不是 event-level 的翻轉數。** 來源僅為 stdout 的
> `bridge_attempts` / `bridge_accepts` 聚合計數。它足以證明兩組決策**不可能完全相同**
> (故 mechanism 非 inert),但**不能**推出「恰好 72 個事件被翻轉」——
> 本文**未**做 event-level set diff。

> 未驗證的觀察:DanceTrack 的 accept rate(51.4%)是另兩者(~22%)的 2.3 倍。
> appearance 均勻 ＋ 密集運動可能產生大量高度比可信的匹配,或許與它對 gate 變動特別敏感有關。
> **本文未驗證此因果。**

---

## 5. 判讀:混合結果,但足以否決候選

本輪採用的判讀規則(近域 MOT20 用來區分「結構不轉移」與「DanceTrack 太遠」):

- **(A) 結構非 domain-general — 成立。** 近域 MOT20 與遠域 DanceTrack **都沒有重現
  「懸崖 ＋ 在非平坦曲面上突出的 row-specific 平台」這個組合**(§3;MOT20 是無處不平,
  DanceTrack 是該列反成斜坡)⇒ **0/2**。
- **(B) DanceTrack 太遠 — 只解釋傷害幅度,救不了結構。** 外部表現在近域近乎中性、
  在遠域明顯有害,故域距離決定「傷多重」;但它**不能解釋為何近域也沒有該結構**。
- 實際落在第三種:**外部表現與結構是兩個獨立命題**,兩者結果不同向 ——
  外部表現在近域近乎中性、在遠域明顯有害;結構則兩域皆未重現。

> ⚠️ **上述判讀規則於 run 前採用,但其 provenance 未獨立封存。** repo 內沒有早於本輪結果、
> 明確記錄這三種判讀的 dated artifact(Issue / plan / commit)。因此讀者應把它視為
> **本文自述的分析協定**,而非可獨立審計的 pre-registration。

**⇒ 候選變更否決,`mamba_whole_graph_m` 的 `relink_bridge_h_lo/h_hi` 維持 0.6 / 1.7。**

成立的陳述是:**出貨值在兩個 external dataset 上均為六 arm 最佳;MOT17 的 in-sample gain 沒有跨域存活。**

> ⚠️ **出貨值在 MOT17 上是被打敗的**(shipped 80.447 < candidate 81.089 < cand_lo 81.198)——
> 這正是本文要處理的問題本身。任何「出貨值從未被打敗」的說法都與 §1 的表格直接矛盾。

**Interpretation(非本文建立的機制)**:MOT17 的 +0.642 集中在 7 條中的 2 條(MOT17-11、-05,
見前作 §3),加上外部 0/2 不重現 —— 最簡約的讀法是那是**序列特異的身份組態剛好落在 `h_hi` 門檻附近**。
本文**未**建立此機制,僅陳述它與觀察一致。

---

## 6. 這份文件建立與未建立什麼

**建立**:凍結候選在 MOT20 上近乎中性(−0.013)、在 DanceTrack 上明顯有害(−0.753);
MOT17 的 cliff-plus-row-plateau 幾何在此二資料集上未重現;MOT20 的 gate 觸發量充足,
故其近乎中性的結果為實質而非空洞;候選變更**不應套用**。

**未建立**:

- **不是「MOT17 的結構是假象」。** 三個資料集各自 bit-exact,都是真實量測,只是**不共享結構**。
  正確陳述是**非 domain-general**。
- **未事先註冊 non-inferiority margin ⇒ 本文不做 non-inferiority 檢定**(§2)。
  MOT20 不稱「通過」、DanceTrack 不稱「失敗」;僅用 near-neutral / materially harmful 描述。
  候選的否決不依賴任何 margin。
- **MOT20 僅 4 條序列**,逐序列統計(1 好 / 2 壞 / 1 同)極弱;pooled 值才是可引用的。
- **detector 對兩個外部資料集都有 domain shift**,絕對值(MOT20 39.2 / DanceTrack 34.5
  vs MOT17 80.4)因此不可與 MOT17 直接比較。**但六個 arm 共用同一組 detection,
  故 arm 間的相對比較不受此影響** —— 本文所有結論都建立在 arm 間比較上。
- **未測 SportsMOT**(本地僅 `val/`);**PersonPath22 依 §0.1 刻意排除**。
- **不是 NO-GO 登記**;是否登記由 owner 決定。
- detector 的 MOT17-only provenance 係由 plan 文件反推,非 checkpoint 自帶(§0.1)。

---

## 7. 方法上的產物(對後續工作有用)

1. **報告「外部表現中性」之前必須先驗證 mechanism 有在動。** 若機制未觸發、或觸發後不改變
   任何決策,任何設定都會顯得中性而毫無資訊。§4 的觸發量檢查應成為此類驗證的標準步驟。
   (同型陷阱:前作姊妹文件 [reid_handover](reid_handover_ablation_20260808.md) §5 用
   MOT 輸出 byte-diff 判斷 re-query 是否觸發。)
2. **在單一資料集上做 sweep 找到的 gate 結構,不可假設可轉移** —— 即使該結構在該資料集上
   bit-exact、通過 LOSO、且有明確的機制隔離。本文即為一個完整的反例。
3. **前作的 wording 紀律被資料驗證。** 前作將 robustness 限定為
   「僅在 MOT17-train 這批資料上建立,不是跨資料集或跨 detector 的 robustness」——
   若當時採用更強的說法,本文的結果會直接推翻一份已發布的 claim。

---

## 8. 重生指令

```bash
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# MOT20（4 seq）
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m \
    --data-root datasets/MOT20/MOT20 --split train \
    --double-buffer --no-gpu-decode \
    --relink-bridge-px 0.4 --relink-bridge-h-lo 0.76 --relink-bridge-h-hi 1.4 \
    --output out/xval_mot20_candidate

# DanceTrack（40 seq）
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m \
    --data-root datasets/DanceTrack --split train \
    --double-buffer --no-gpu-decode \
    --relink-bridge-px 0.4 --relink-bridge-h-lo 0.76 --relink-bridge-h-hi 1.4 \
    --output out/xval_dancetrack_candidate

# 出貨對照：把 --relink-bridge-h-lo/h-hi 換成 0.6 / 1.7
```

兩者 layout 均為 `<root>/<split>/<seq>/{img1,gt/gt.txt,seqinfo.ini}`,與 `build_mot17_dataloader` 相容。
**不要傳 `--detector`** —— 這兩個資料集的序列名沒有 detector 後綴。

bridge 觸發量取自 stdout 的 `🔗 Relink debug <seq>: ... bridge_attempts=N bridge_accepts=M`。
IDF1 需從輸出重算至三位小數(方法見前作 §7)。

---

## 相關

- [bridge_gate_stability_20260808.md](bridge_gate_stability_20260808.md) — 前作:MOT17 上的原始發現與候選來源
- [reid_handover_ablation_20260808.md](reid_handover_ablation_20260808.md) — 同批 run 的 ReID / handover 部分
- [holdout_generalization_plan.md](../../modules/detection/research/holdout_generalization_plan.md) — detector 訓練語料與 held-out 設計
- [math_model.md](../math_model.md) — §1.1 s/m preset delta
