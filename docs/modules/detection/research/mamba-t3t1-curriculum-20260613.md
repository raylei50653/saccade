# T3→T1 時序塑形 Curriculum：發現、驗證與邊界（2026-06-13）

> 單日研究紀錄。起點是 user 假說「時序可能增強空間一致性」；終點是一個
> 超越 legacy v14 的 production 候選 checkpoint、三個乾淨結案的邊界實驗、
> 一條 curriculum 設計規則，以及「兩種特徵 regime 如何配合」的開放方向。
>
> 前置脈絡：[v14 復刻協議](../mamba-v14-replication-protocol.md)（frozen-SSM
> regime、replica lineage）。本文所有實驗基於 replica lineage
> （teacher `gated_det_v14replica/epoch_0012`、cache `mamba_teacher_cache_v14replica`）。

## TL;DR

| Checkpoint | IDF1 | MOTA | HOTA | DetA | AssA | 定位 |
|---|---|---|---|---|---|---|
| legacy v14 | 75.1 | 77.7 | 68.2 | — | — | 舊 production |
| replica plain GT2 | 73.4 | 77.3 | 65.3 | ~68.4 | ~62.8 | 復刻 baseline |
| 隱式 all-frames（matched 3-seed） | 73.7 | 77.5 | 65.8 | 69.1 | 62.7 | 同部署的隱式 baseline；顯式 matched **+1.8/+3.0**（§6） |
| **T3→T1**（`mamba_gt_v14replica_t3_t1`） | **75.4** | 77.6 | **67.7** | 69.7 | **66.0** | **IDF1/HOTA 北極星下的 production 候選** |
| T3T1→SSM-ft（錯序疊加） | 73.8 | **79.4** | 66.7 | **71.0** | 62.8 | MOTA/recall 角的 Pareto 點 |
| SSM-ft→T3T1（反向順序） | 74.3 | 78.8 | 66.9 | 70.7 | 63.6 | 中間點，無協同 |

核心發現：
1. **T3→T1 curriculum 首次超越 legacy v14**，3-seed 配對 +2.1/+2.3/+0.4，
   困難序列（02/05/10/13）12/12 正向。
2. **增益機制 = 塑形壓力，不是時序推理**：T=3 訓練迫使 spatial path 產出
   跨幀一致特徵；temporal blocks 的推理貢獻僅 +0.7（已結案）。
3. **一致性 ≠ 身分判別力**：特徵作 relink embedding AUC 0.438（NO-GO
   #35），AssA 增益全部經 box/score 穩定性（IoU 路徑）傳導。
4. **AssA 崩壞 = 單幀續訓遺忘（主因）+ SSM 解凍（次要）**（§4.1 2×2 結構
   歸因）：凍結也照樣續訓 T=1 → −2.0 AssA / +1.2 DetA，故權衡沿**單幀檢測
   目標**非 SSM 自由度；解凍只多 −1.2 AssA、DetA 無回報。**跨幀(T=3)壓力
   是保護項，排序可消除大半拮抗**（cw0 救回 65.4）。〔修正：原寫「沿 SSM
   自由度本質權衡、排序無法消除」〕
5. **顯式 staging > 隱式 all-frames（matched 結案，§6）**：shared-GT1 同 seed
   配對（42/43/44），顯式於 3/3 seed 一致 +1.8 IDF1 / +3.0 AssA、分布不重疊
   （p<0.001）。先前「與 v14 隱式同端點、Occam 偏隱式」的判讀**已被推翻**
   （死因：legacy 75.1 不可復現 + 舊 null seed 為 GT1 confound）。

---

## 1. 假說與配方

假說（user，2026-06-12）：v14 前曾有 uncommitted T=3/T=1 dual-loss
session（`34cabc95`，05-29~05-31）；時序訓練可能增強空間一致性，
即使最終部署是 T=1 單幀。

配方（`run_v14replica_t3t1.sh` / `run_v14replica_t3t1_seed.sh`）：

```text
replica GT1 best（live-teacher 30ep, stop-grad）
  → Phase A：--add-temporal，clip_len=3 stride=6，15ep，cache，stop-grad
              （39 個 temporal keys 隨機初始化加入）
  → Phase B：clip_len=1 stride=2，15ep，cache，stop-grad
              （T=1 時 temporal blocks 自動 bypass，純空間再適應）
  → eval：mamba_whole_graph preset（單幀 forward，temporal 永遠 bypass）
```

與 plain GT2（30ep T=1）同起點、同總預算、同 seed —— 嚴格配對。

## 2. 主結果與驗證

### 2.1 三 seed 配對

| Seed | plain GT2 | T3→T1 | 配對差 |
|---|---|---|---|
| 42 | 73.4 | 75.4 | +2.1 |
| 20260613 | 73.2 | 75.4 | +2.3 |
| 20260614 | 73.0 | 73.4 | +0.4 |

Plain GT2 的 3-seed 噪聲帶寬僅 0.4pp（噪聲集中 FP 側 ±23%；02 recall
跨 seed 穩定 0.958-0.959），前兩個配對差是帶寬的 5 倍以上。

### 2.2 Per-sequence 歸因（正規 motmetrics，與官方 overall 吻合）

| Seq | s42 Δ | s13 Δ | s14 Δ | 判讀 |
|---|---|---|---|---|
| MOT17-02（crowd） | +3.3 | +2.4 | +2.4 | 3/3 正向 |
| MOT17-05 | +3.4 | +2.8 | +2.6 | 3/3 正向 |
| MOT17-10（night） | +6.8 | +4.2 | +1.2 | 3/3 正向 |
| MOT17-13（moving cam） | +2.2 | +6.9 | +4.8 | 3/3 正向 |
| MOT17-04 | +0.6 | +0.6 | −0.8 | 噪聲（但權重最大） |
| MOT17-09 | −3.7 | −0.4 | +0.9 | 小序列，噪聲 |
| MOT17-11 | +2.6 | +2.9 | **−5.3** | s14 的稀釋來源 |

**s14 +0.4 的歸因**：不是塑形失敗 —— 其 plain baseline 在 MOT17-11 抽到
幸運高點（80.3，比其他 seed 的 plain 高 5pp），t3t1 拉回 75.0（與其他
seed 的 t3t1 77.0/78.1 一致），單序列計 −5.3；加上最大權重序列 04 的
−0.8。困難序列 s14 仍 4/4 正向。

**判定：增益真實且結構穩定（困難序列 12/12），overall 幅度受單序列
baseline 抽樣調制（+0.4~+2.3，平均 +1.6）。**

### 2.3 s14 異常排查（收斂與 A/B 界定均排除）

- **收斂**：s13/s14 的 Phase A/B loss 軌跡逐 epoch 重合（A 收 4.12/4.13、
  B 都收 2.85），best 均取 epoch 15。
- **A/B 界定**：Phase A best 均為 epoch 15，Phase B 正確 warm-start。
- **Eval 語義（發現一個潛在地雷並已加聲明）**：`mot17.py` 對 temporal
  checkpoint 預設 `temporal_T=3`，但 `mamba_whole_graph` 的
  `_whole_graph_fn` 只餵當前單幀 → temporal blocks 實際 bypass，所有
  t3t1 數字都是純 T=1。**非 whole-graph** eval 會啟用 streaming temporal
  （train/eval mismatch），須傳 `--no-temporal`。builder
  （`build_mamba_gated_detector`）現在印三態聲明：
  ⚠️ STREAMING ACTIVE / 🧊 whole-graph BYPASSED / 🧊 T=0 BYPASSED。
- 三 run train loss 幾乎相同（2.851/2.851/2.863）→ **train loss 非部署
  品質 selector**（本日第三次驗證；後續 SSM-ft loss 1.95 全場最低而
  IDF1 反跌為第四次）。

## 3. 機制歸因

### 3.1 增益指紋：association，不是 detection

T3→T1 vs replica（seed 42）：AssA +3.2、IDs −14%、FP −17%，
02 recall@0.001 持平（0.958→0.959）。一致性塑形改善的是 box/score
跨幀穩定性，不是檢測能力。

### 3.2 特徵不攜帶身分：relink embedding 探針 NO-GO（registry #35）

`scripts/tools/mamba_relink_features.py`：21k relink 候選對（與 OSNet/
顏色探針同取樣、同分層），P3 cls_head 輸入（256ch）box ROI mean-pool，
cosine 相似度。

| variant | AUC full | AUC hard | gap 1-10 | gap 80+ |
|---|---|---|---|---|
| bridge_dist（幾何 baseline） | 0.898 | 0.625 | — | — |
| T3→T1 特徵 | 0.506 | **0.438** | 0.340 | 0.528 |
| replica 特徵（對照） | 0.507 | **0.438** | 0.333 | 0.524 |

- **配對差 ~0.001 → curriculum 對身分判別力貢獻為零。**
- 短 gap 反向 0.33-0.34 與外觀探針（#32）完全同構 → 端點污染是候選池
  結構性質（軌跡死因即遮擋）。
- **consistency ≠ discriminability**：一致性讓同一目標跨幀穩定，但
  detection 特徵編碼「人＋局部幾何」，個體間無對比分離（v8DetectionLoss
  無 contrastive 項）。
- 三角結案：ROI FPN ReID（#16）、顏色+OSNet（#32）、Mamba head 特徵
  （#35）—— MOT17 relink gate 的特徵/外觀方向全面關閉。要身分必須顯式
  contrastive 訓練 + 解決端點污染。

### 3.3 T=3 streaming 推理：塑形壓力才是價值（結案）

Phase A checkpoint（T=3 訓練的）以原生 T=3 評測。控制污染源：
flow_gate_conv 訓練時從未收到 flow（零梯度、停留隨機初始化），eval 端
GMC flow 會觸發隨機 gate 注入噪聲 → 使用 flow_gate 歸零副本
（`best_flowgate0.ckpt`）；新 preset `mamba_eager_temporal_probe`
（= whole_graph 配置去掉全部 graph，streaming buffer 才會生效）。

| | T=3 streaming | T=1 對照 | Δ |
|---|---|---|---|
| IDF1 | 69.8 | 69.1 | +0.7 |
| MOTA | 71.6 | 72.0 | −0.4 |
| FP | 8393 | 6819 | +1574 |
| FN | 22848 | 23997 | −1149 |

- temporal-at-eval 邊際貢獻 +0.7，代價 FP +23%/IDs +75 —— 不值得配套
  （whole-graph 不支援 + eager latency）。
- **Phase A ckpt 本身 69.8，比 T3→T1 product 低 5.6pp** —— T=3 訓練的
  價值在塑形壓力，紅利須由 Phase B 的 T=1 再適應兌現（69.8→75.4）。
  「訓練時的結構約束、推理時的免費增益」閉環。

## 4. 邊界：與 SSM-ft 的配合問題

前置：SSM 解凍微調（ssmft）單獨自 replica final 收 +0.8 MOTA/HOTA
（見復刻協議「單變因延伸」節）。兩 regime 改進同一輸出的不同性質：
ssmft → 少漏檢（FN↓）；T3→T1 → 跨幀穩定（AssA↑）。

### 4.1 順序疊加實驗

| 順序 | IDF1 | MOTA | DetA | AssA | 判讀 |
|---|---|---|---|---|---|
| T3→T1 單獨 | **75.4** | 77.6 | 69.7 | **66.0** | 塑形完整 |
| T3T1 → ssmft 30ep | 73.8 | **79.4** | **71.0** | 62.8 | **塑形被抹掉**（AssA 退回 plain 水平），檢測全場最高 |
| ssmft → T3T1 | 74.3 | 78.8 | 70.7 | 63.6 | 部分救回（vs 錯序 +0.5/+0.8）但 AssA 回不到 66.0 |

**結構歸因（2×2 discriminator，2026-06-13 補；修正下方原結論）**：原先把
拮抗歸因為「SSM 自由度的本質權衡、排序無法消除」。補做凍結×T=1 對照格
（`run_frozenT1_probe.sh`，從 t3_t1 起續訓 30ep）+ cw0（解凍 T=3→T1）後，
2×2 完整，AssA 分離為兩個可加成分：

| 從 t3_t1（AssA 66.0 / DetA 69.7）續訓 30ep | T1-only | T3→T1 |
|---|---|---|
| **解凍 sg=N** | 62.8（T3T1→ssmft） | 65.4（cw0） |
| **凍結 sg=Y** | 64.0（本 probe） | 66.0（t3_t1 本身） |

- **C3 — 通用 T=1 遺忘（崩壞主體 ≈ −2.0 / −3.2）**：SSM **凍結**下光續訓
  T=1 就 ΔAssA **−2.0** / ΔDetA **+1.2** → DetA↔AssA 權衡是**單幀檢測目標
  本身**的性質，與 SSM 自由度無關。
- **C1 — SSM 解凍 DOF（次要 ≈ −1.2）**：解凍只多掉 −1.2 AssA、DetA 幾乎
  沒多賺（+0.1）→ 在 t3_t1 起點解凍是**淨有害**（此增量偏小、單 seed 邊緣）。
- **跨幀壓力才是保護項**：插一個 T=3 相（cw0）把 AssA 救回 65.4（解凍下）/
  t3_t1 本身 66.0（凍結下）→ **排序確實能消除大半拮抗**。

→ **修正原結論（兩處皆錯）**：(1) 權衡沿**單幀檢測目標**、非 SSM 自由度
（凍結也照樣 +1.2/−2.0）；(2) 排序**可**消除——缺的不是「把塑形放最後」
而是「續訓期間是否保有 T=3 跨幀壓力」。原 §4「與 SSM-ft 的配合」框架因此
鬆動：ssmft 在 t3_t1 起點上 DetA 回報幾乎為零（+0.1），其 +1.2 DetA 與
凍結續訓無異 → ssmft 不是獨立增益源，而是 C3 遺忘的同義詞。

> 下方為原始（已被上述推翻）判讀，保留作軌跡：
> ~~錯序拮抗 = 解凍 SSM 自由度下純單幀檢測最優解拉離一致構型；DetA↔AssA
> 沿 SSM 自由度本質權衡，排序無法消除。~~

### 4.2 權重插值掃描（免訓練配合探針）

兩 checkpoint 共享 69 個 spatial 鍵（相對 L2 距離 12%），線性插值
α ∈ {0.25, 0.5, 0.75}（α=0 為 ssmft、α=1 為 t3t1）：

| α | IDF1 | MOTA | HOTA | DetA | AssA |
|---|---|---|---|---|---|
| 0（ssmft） | 73.3 | 78.1 | 66.1 | — | — |
| 0.25 | 73.8 | 78.5 | 66.7 | 70.1 | 63.7 |
| 0.50 | 74.1 | 78.4 | 66.7 | 70.1 | 63.7 |
| 0.75 | 73.9 | 78.1 | 66.2 | 69.9 | 63.0 |
| 1（t3t1） | 75.4 | 77.6 | 67.7 | 69.7 | 66.0 |

判讀：無協同峰，且 **α=0.75 處有淺谷**（IDF1 73.9 < α=0.50 的 74.1，
AssA 63.0 還低於 0.25/0.50 的 63.7）—— t3t1 的一致性構型集中在 α=1
附近的窄區，靠近就先掉再升。**免費合併排除**；一致性塑形在權重空間
是局部、脆弱的特性，這與「後續 full-grad 訓練輕易抹掉它」（§4.1）
互相印證。配合只能走訓練路線（路線 2/3）。

## 5. 設計規則與未來方向

### 確立的規則

1. **最後一段 curriculum 的目標決定特徵最終構型。** 想保留的塑形必須
   放在最後，或以顯式 loss 項保護。（正例：T3→T1；反例：T3T1→ssmft。）
2. **Train loss 不是部署品質 selector**（本日四次獨立驗證）。
3. **塑形壓力 vs 推理貢獻要分開歸因** —— 結構約束的價值可能完全在
   訓練側（T=3 推理僅 +0.7）。

### 開放方向：兩 regime 的「配合」

關鍵觀察（user）：**特徵不同還互有提升，未來方向是配合而非二選一。**
按成本排序：

| 路線 | 成本 | 狀態 |
|---|---|---|
| 1. 權重插值（model soup） | 免訓練 | 本日已測：無協同峰，排除捷徑 |
| 2. Joint loss（`--t1-weight`） | 1 次訓練 | 未做（註：`--t1-weight` 仍是 argparse 死樁，未接 loss） |
| 3. **顯式一致性保護項**（相鄰幀同 track cls logit L2） | 小代碼+訓練 | **本日已測 = NO-GO（registry #37）**：項機械生效（raw cons 2.0→0.15-0.37）但 tracking 單調退步（weight↑→AssA↓）；硬 L2 壓掉合法逐幀 score 動態。weight=0 control（T=3 full-grad unfreeze）副產物 IDF1 75.7 但單 seed、已 park |
| 4. 雙頭 NMS ensemble | 推理 2× | 僅上限探測用 |

**✅ Production 已切換（2026-06-13）**：`mamba_whole_graph` + `mamba_optimal`
的 `mamba_ckpt` 改指 `runs/mamba_gt_v14replica_t3_t1/best.ckpt`。驗證：preset
default（不帶 `--mamba-ckpt`）跑出 75.4/77.6/67.7/69.7/66.0 = T3→T1。consistency
之所以乾淨：backbone = raw yolo26s engine（lineage-agnostic），gt-ratio 0 →
gate bypass（`yolo_gated_detector.py:103` gate_input None → return output）→
train/deploy 同看 raw FPN 特徵，無 teacher/lineage 不匹配。`mamba_eager_temporal_probe`
為研究探針，維持 pinned 不動。

其他未動工項：02 小框增強（detection 側唯一明確缺口，§6.4 證 recall 已飽和、
h<32 為 #36 成本 NO-GO）、T=5/alternating 掃描。**「配合」分支全結案**（route 1
無協同峰、route 2 joint loss 塌回 implicit、route 3 NO-GO #37）；再要 IDF1 應
轉 tracker-association 軸（bidir bridge）。

## 6. 隱式 vs 顯式：matched 配對結案（翻盤「平手」）

> 起點：本研究一度判讀「顯式 staging 與 v14 隱式 all-frames loss 同端點、
> 增益落在 seed 噪聲內、Occam 偏隱式（顯式多 1.24M deploy-bypass 參數、兩
> 階段、staging 脆弱性，準度卻打平 → 不值）」。matched 配對實驗推翻此判讀。

### 6.1 兩個污染源（為何先前誤判平手）

1. **legacy v14 75.1 當隱式標竿不可復現**：其配方（clip_len=4 all-frames +
   Cross-Scan N=16，git `77fcc262`）在 replica lineage 的 3-seed 復刻穩在
   73.0–73.7，legacy 高出 ~2pp = ~5σ（band 0.0–0.36pp）→ 非 seed noise 可
   解釋（frozen-SSM lucky init / leakage 產物）。乾淨隱式的真實水平是 **73.7**，
   不是 75。
2. **舊顯式 s14「塌到 73.4」的 null seed = GT1 起點 confound**：該 seed 用
   `mamba_gt_v14replica_s14_stage1`（seed-specific GT1）暖啟，非 curriculum
   不穩。先前「增益與 seed 噪聲同量級」正是被這顆 null 帶歪。

### 6.2 乾淨 matched 設計

全部從 **shared GT1**（`mamba_gt_v14replica_stage1`，= 原 seed-42 T3→T1 起點）
暖啟、同 seed（42/43/44）配對，30ep 同預算、scan-stop-grad 同 SSM regime，
**只差**「隱式 all-frames T=4（10.13M，無 temporal block）vs 顯式 T3→T1
staging（11.37M）」。

| seed | 隱式 IDF1 | 顯式 IDF1 | Δ IDF1 | 隱式 AssA | 顯式 AssA | Δ AssA |
|---|---|---|---|---|---|---|
| 42 | 73.7 | 75.45 | +1.75 | 62.6 | 65.96 | +3.36 |
| 43 | 73.7 | 75.7 | +2.0 | 62.5 | 66.0 | +3.5 |
| 44 | 73.7 | 75.3 | +1.6 | 63.1 | 65.1 | +2.0 |
| **mean** | **73.7** | **75.48** | **+1.78** | **62.7** | **65.7** | **+2.95** |

- **3/3 seed 顯式全勝；兩分布完全不重疊**（最差顯式 75.3 > 最佳隱式 73.7）。
- Δ 極穩（IDF1 sd≈0.2）→ 配對 t≈15、p<0.001（n=3 也壓得死，因效應一致）。
- shared-GT1 下顯式 s44 = 75.3 **沒塌** → 反證舊 null 是 GT1 confound。
- 隱式 3-seed band≈0（全 73.7）：all-frames T=4 的隱式一致性本身極穩定。

### 6.3 結論（反向 paper 意涵）

先前「顯式 buys nothing、把貢獻押在簡單隱式、顯式當對照組」的故事**不成立**。
誠實結論反向：

> **顯式 T3→T1 staging 在同部署架構/成本（temporal block 推理 bypass）下，於
> 3 顆配對 seed 一致超越隱式 all-frames baseline +1.8 IDF1 / +3.0 AssA（分布
> 不重疊、p<0.001）。staging 提取了 all-frames loss 隱式抓不到的跨幀
> association 品質。**

Occam 在準度軸不再平手 → 問題從「值不值得多做」變成「**+1.8/+3.0 真增益值不值
1.24M deploy-bypass 參數 + 兩階段訓練**」（對顯式有利得多）。

**Caveat**：dev 數字含 intentional leakage（絕對值偏高），但相對配對兩邊同
leakage、同 lineage、同 seed，故**相對結論有效**；這是 MOT17 半監督 dev 設定
下的結論。

### 6.4 Robustness：關掉 interp/relink + recall 各 size 持平 → 增益純 association

§6.2 用 `mamba_whole_graph` preset（`interpolate_tracklets:true` +
`relink_bridge_enabled:true`）。為排除「增益是後處理拐杖造成」，把兩者關掉
（`--no-interpolate-tracklets --no-relink-bridge-enabled`）重評，並加 detector-only
size-binned recall（本就無 interp/relink）。

**Tracking（interp + relink 全關，全 7 SDP 序列）：**

| | IDF1 | DetA | AssA | Rcll |
|---|---|---|---|---|
| 隱式 42/43/44 | 72.6 / 72.3 / 72.9 | 68.0 | 61.6 / 61.3 / 62.1 | 78.0 |
| 顯式 42/43/44 | 74.1 / 73.8 / 74.0 | 68.4 | 64.6 / 64.2 / 63.4 | 77.5 |
| **paired Δ** | **+1.37** | +0.34 | **+2.40** | −0.5 |

- 顯式仍 **3/3 全勝、分布不重疊**（最差顯式 73.8 > 最佳隱式 72.9）。增益從
  with-post 的 +1.8/+3.0 小幅壓到 **+1.37 / +2.40**；interp/relink 對兩邊各加
  ~1.4 IDF1（method-neutral，不偏袒顯式）。**增益非後處理產物。**
- **DetA 僅 +0.34、Rcll 顯式還略低 −0.5 → 增益幾乎全在 AssA**：顯式不是檢測
  更多，是關聯更好。坐實 §3.1「增益指紋 = association 不是 detection」。

**Size-binned recall（detector-only，MOT17-02-SDP @0.001）：**

| | all | h≥128 | h64-128 | h32-64 | h<32 | min≥16 | min8-16 | min4-8 |
|---|---|---|---|---|---|---|---|---|
| 隱式（3 seed） | 0.957 | 0.981 | 0.969 | 0.896 | 0.23 | 0.980 | 0.973 | 0.884 |
| 顯式（3 seed） | 0.958 | 0.982 | 0.970 | 0.895 | **0.40** | 0.981 | 0.973 | 0.887 |

- 各 size bin 幾乎一模一樣（all 0.957 vs 0.958、所有箱差 ≤0.003）→ **recall 非
  增益來源**。
- 唯一明顯差的是 **h<32**（顯式 ~0.40 vs 隱式 ~0.23），但**該箱僅 44 GT**、顯式
  自身 seed 跨 0.318–0.455 變異極大 → 噪聲，非 driver（~7-8 框，對 overall 無感）。
  有意義數量的中小框（h32-64、min4-8，~2000+ GT）兩邊**完全持平**。

**結論**：§6 的 +1.4/+2.4 真實、且**純 association**——同樣的檢測（recall 各 size
持平、DetA 平、Rcll 顯式還略低），顯式靠 T3→T1 staging 產出更跨幀一致的
box/score 讓 tracker 關聯更好；後處理只對兩邊等量補分，不改變相對結論。

## 7. Artifacts

**Checkpoints**（均在 `runs/`）：

| 路徑 | 內容 |
|---|---|
| `mamba_gt_v14replica_t3_t1/best.ckpt` | **T3→T1 production 候選（75.4）** |
| `mamba_gt_v14replica_t3/best.ckpt` | Phase A 中間產物（+ `best_flowgate0.ckpt` 探針副本） |
| `mamba_gt_v14replica_t3_{t1_,}s13/s14` | multi-seed 配對 |
| `mamba_gt_v14replica_t3t1_ssmft/` | 錯序疊加（MOTA 79.4 Pareto 點） |
| `mamba_gt_v14replica_ssmft_t3{,_t1}/` | 反向順序 |
| `mamba_soup_ssmft_t3t1_a{25,50,75}.ckpt` | 插值掃描 |
| `mamba_gt_v14replica_implicit_s{42,43,44}/` | **隱式 all-frames baseline，matched 3-seed（§6）** |
| `mamba_gt_v14replica_t3_t1_shared_s{43,44}/` | **顯式 T3→T1 shared-GT1 配對 43/44（§6）** |
| `mamba_gt_v14replica_t3t1_consist_cw{0,0p1,0p3,1p0}_t1/` | route-3 一致性 weight sweep（§5 表，NO-GO #37） |

**Scripts/Presets/代碼**：

- `scripts/train/temporal_yolo/run_v14replica_t3t1.sh`、`run_v14replica_t3t1_seed.sh`
- **§6 matched 配對**：`run_v14replica_implicit_{seed,sweep}.sh`（隱式 baseline，from shared GT1）、`run_v14replica_t3t1_shared_seed.sh`（顯式 shared-GT1 seed 版；**注意**舊 `_t3t1_seed.sh` 需 seed-specific GT1，配對請用 shared 版）、`scripts/tools/analyze_consistency_stats.py`（GT-only 密度統計）
- **§6.4 no-post robustness**：`scripts/eval/run_nopost_eval.sh`（6 ckpt × tracking `--no-interpolate-tracklets --no-relink-bridge-enabled` + detector-only size recall，附 summary）；recall json `report_data/recall_nopost_{impl,expl}_s{42,43,44}.json`；tracking `results/nopost_{impl,expl}_s{42,43,44}/`
- **§4.1 2×2 結構歸因**：`scripts/train/temporal_yolo/run_frozenT1_probe.sh`（凍結 T=1 續訓 discriminator）；`runs/mamba_gt_v14replica_frozenT1_30ep/`、`results/mamba_v14replica_frozenT1`（AssA 64.0 = C3 遺忘 −2.0 + C1 解凍 −1.2 分離）
- **§5 route-3**：`run_v14replica_consistency{,_sweep}.sh`、`train_mamba_gt.py::_temporal_consistency_loss` + `--consistency-weight`、`tests/unit/detection/test_mamba_consistency_loss.py`（5 tests）
- `scripts/tools/mamba_relink_features.py`（特徵 relink 探針）
- `configs/presets/mamba_eager_temporal_probe.yaml`（streaming temporal 探針 preset，已加入 preset 白名單）
- `src/saccade/perception/temporal_yolo/mamba_gated_detector.py`：builder temporal 三態聲明

**Eval 結果目錄**：`results/mamba_v14replica_t3t1*`、`results/mamba_v14replica_implicit_s{42,43,44}`、`results/mamba_v14replica_t3t1_shared_s{43,44}`、`results/mamba_v14replica_t3_T{1,3}eval`、`results/mamba_soup_a*`

**連動更新**：no-go registry #35、**#37（route-3 顯式一致性保護項）**；復刻協議
「Multi-seed 噪聲帶寬」「T3→T1 GT2 curriculum」「課程順序與增益保留」「成功因素
歸納」各節；`mamba-head-training.md` 總覽 callout；本文 §6 隱式 vs 顯式 matched
配對結案（翻盤「平手」）。
