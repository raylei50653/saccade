# Generalization 實驗 plan：train PersonPath22 / test MOT17

> 狀態：**規劃中，資料下載中**（2026-06-22）。
> 目的：把現行 in-sample headline（`frozen_v2`, IDF1 78.2）換成一個**leakage-free + 非過擬合**的
> 泛化數字。設計：**在大且多樣的 PersonPath22 上訓練，在從未見過的 MOT17 上測試**。
> 依據：[v14-R training protocol](../mamba-v14r-training-protocol.md)、[v14 replication](../mamba-v14-replication-protocol.md)、
> [T3→T1 curriculum](research/mamba-t3t1-curriculum-20260613.md)。

---

## 0. 為什麼是「train PersonPath22 / test MOT17」

確認事實：detection head + gated teacher（MOT17-finetuned YOLO backbone）+ feature cache **全吃 MOT17 7 條** →
78.2 是 training-set，**同時有過擬合（~5,300 樣本）與 leakage**。

此設計一次解掉兩者：
- **過擬合**：PersonPath22 大且多樣（138 train videos）→ head 學會泛化，不背特定幀。
- **leakage**：MOT17 完全不進訓練 → MOT17 數字是**合法 held-out，且直接可比文獻**。

它**取代**原本昂貴的 MOT17 LOO 重訓（LOO 降為 §9 optional 附錄）。

---

## 1. 設計

| 角色 | 資料 | 說明 |
|---|---|---|
| **訓練（全鏈）** | **PersonPath22 train**（138 videos） | teacher + cache + distill + GT1 + T3→T1，全部只吃 PersonPath22，**完全不碰 MOT17** |
| **主測試** | **MOT17 train / SDP 7 條**（dense GT，本機可 eval） | 從未訓練 → leakage-free 泛化數字，可比文獻與舊 78.2（誠實列「78.2 是 in-sample，此為 held-out」） |
| **補充測試** | **DanceTrack val**（25 videos，外觀均勻） | 零額外訓練，測 tracker 在 appearance-uniform 域的泛化 |

---

## 2. PersonPath22 recon 實證（2026-06-22）

| 項目 | 實測 |
|---|---|
| 規模 | 236 videos（**train 138 / test 98**），1920×1080, ~24–30fps，靜態相機為主、真實行人/監控（**域近 MOT17**） |
| **train 標註密度** | **稀疏：~21% 覆蓋、每 ~5 幀標一次**（`blob.frame_idx` gap 主要 5/4，4 支抽驗一致）。「4,736 annotated frames」即此 |
| test 標註 | dense（每幀，為 leaderboard eval 插值補密）；已附 MOT 格式 `person_path_22_data.zip`（`gt/gt.txt`+`seqinfo.ini`+`seqmaps`） |
| 原始標註 | `anno_visible.zip`（9.4MB，gluoncv JSON，**全 236 videos 都有**，train 138/138 覆蓋） |
| 容量/可用 | `videos.zip` **11.4 GB**（181 支，非幀）+ 標註幾十 MB。**videos.zip 僅 181/236**（55 支外部來源需另抓）→ **train 可直接用 107/138 支**（其餘 31 外部，不追） |
| 抽幀後 | **只抽 keyframes（~21%, ~25k train 幀）≈ ~6 GB**（見 §3 為何不用補密） |
| License | **CC-BY-NC-4.0（非商用）** → research 訓練/eval OK，**不可 redistribute / 商用 / bundle 進 release 倉** |

存放：`datasets/PersonPath22/{raw,annotation,mot_format}`（`datasets/` 已 gitignore）。

---

## 3. 為何稀疏標註不影響 T3→T1（不用內插）

**目前 T3→T1 機制（讀自 code）：**
- `--add-temporal` 在 Cross-Scan spatial 上注入 per-level **temporal MambaBlock（沿 T 軸的 SSM）**+ 可選 flow gate。
- Forward（`mamba_head.py:1726-1749`）：upsample 階段把 `(B*T,C,H,W)` reshape 成 `(B·H·W, T, C)`，
  **每個像素位置用 temporal SSM 殘差混 T 幀**，再 reshape 回去。
- **Phase A（T=3, clip-len 3, stride 6, 15ep）**：temporal ON，v8DetectionLoss **逐幀**（3 幀各對自己 GT），
  `--scan-stop-grad`、cache mode。**逼 spatial path 產出「時序混了仍可偵測」的一致特徵**。
- **Phase B（T=1, clip-len 1, 30ep）**：`T_frames>1` 為假 → temporal **bypass** → 純 spatial 再適應。
- **部署 whole-graph T=1 → temporal 全 bypass、零成本**；增益（+1.6~2.0 IDF1，幾乎全 AssA）活在 spatial 特徵。

→ T3→T1 本質是**一次性的時序一致性塑形（augmentation-like），不是 dense 監督**。clip 本來就 stride-6（幀隔 6）。

**所以 PersonPath22 sparse 的解法 = keyframe 當 temporal 軸，不內插：**
> 只抽**有標註的 keyframes（~5 幀間隔）**→ 重編號 1..N 連續 → clip-len 3 在 keyframe 序列上跑。
> 3 個連續 keyframe 間隔 ~5 真實幀 ≈ MOT17 stride-6 的 ~6 幀，**temporal 窗等價、塑形等價**。
> 每個抽出的幀都有**真 GT（零內插噪聲）**；Phase B 逐 keyframe 單幀訓練，稀疏完全 OK。
> （flow gate 可選；keyframe 間隔較大時建議 **關 flow gate**，靠 temporal SSM 塑形本身。）

---

## 4. Converter 規格：gluoncv JSON → MOT keyframe 格式

來源：`anno_visible/anno_visible_2022/<vid>.json`，`entities[]` 每筆：
`{ bb:[x,y,w,h], id, time(ms,忽略), confidence, labels:{...}, blob:{frame_idx} }`，`metadata:{number_of_frames,fps}`。

```
for each train video:
  1. 過濾 label（**用 anno_visible，不用 amodal**）：**保留 iff `labels` 含 'person'**。
     - 完全遮擋（零可見訊號）**本就不在 anno_visible**（實測 min area 24、zeroArea=0）→ User 決定「全沒訊號不傳」由「用 visible」天然滿足。
     - `severly_occluded_person`(618, 含 'person') = 部分遮擋仍有訊號 → **留**。
     - `reflection`(79) / `crowd`(1363) 不含 'person' → **自動丟**（reflection 非真人、crowd 是區域非個體）。
     - 一條 `'person' in labels` 同時做完三件事。(注意 PersonPath22 拼字為 `severly_occluded_person`。)
  2. keyframes = sorted(set(e.blob.frame_idx for 保留 e))；建 rank: frame_idx → 1..N。
  3. ffmpeg 只抽這些 frame_idx → img1/{rank:06d}.jpg（注意 ffmpeg n 從 0、frame_idx base 須對齊驗證）。
  4. gt.txt：每筆寫 `rank, id, x, y, w, h, 1, 1, 1`（id 連續化建議重映）。
  5. seqinfo.ini：seqLength=N、imWidth/Height、imExt=.jpg（frameRate 寫有效 keyframe rate）。
輸出佈局：datasets/PersonPath22/mot_train/<vid>/{img1,gt/gt.txt,seqinfo.ini}
```

dataloader 相容性已驗：`build_mot17_dataloader` 吃 `<root>/<split>/<seq>/{img1,gt/gt.txt,seqinfo.ini}`，
序列發現傳 **`detector=None`**（PersonPath22 非 `-SDP` 命名）或顯式 `--seqs`；每幀都有 GT，無空幀問題。

> test split 已有官方 dense MOT 格式（`person_path_22_data.zip`），converter 只需處理 **train** 138 條。

---

## 5. 全鏈訓練（只吃 PersonPath22，teacher 絕不碰 MOT17）

> 要讓 MOT17 測試乾淨，**最容易踩的坑**是 teacher/backbone：現行 gated teacher 是 MOT17-finetuned。
> 必須改。兩個選項：

**✅ 已選 Teacher = B（在 PersonPath22 上訓 gated teacher）**：域適應更佳；PP22-trained 仍**零 MOT17 leakage**（MOT17 測試乾淨）。
（A=base yolo26s 不 finetune 保留為 fallback。）

```bash
# Teacher B：mirror 部署 gated_det_v14replica recipe，改吃 PersonPath22（不 holdout，全 107 train）
.venv/bin/python scripts/train/temporal_yolo/train_gated_detector.py \
    --data-root datasets/PersonPath22 --yolo-weights models/yolo/yolo26s.pt \
    --run-dir runs/gated_det_pp22 --detector None \
    --epochs 30 --batch-size 4 --clip-len 2 --img-size 640 \
    --lr-gate 1e-3 --lr-yolo 1e-5 --gt-ratio 0.5 --seed 20260612 \
    --warmup-epochs 0 --save-every 1 --best-by train-loss
```
- 選 teacher epoch：在 PP22 held-out val 子集（或 DanceTrack）以 recall 選；部署 v14replica 因 e20 NaN 取 e12，PP22 較大可能更穩，依實況選。
- cache（fp16）從**選定的 PP22 teacher epoch** 建；gate-off（`gate_input=None`）→ cache = PP22-adapted backbone + detect head。
- ✅ 已驗整合細節：`train_gated_detector.py`（及 mamba trainers）的 `build_mot17_dataloader` 預設 `detector=SDP`、
  `resolve_training_sequences` 也按 `-SDP` 過濾 → PersonPath22 `uid_vid_*.mp4` 命名**不會被自動發現**。
  **解法：所有 PP22 訓練階段顯式傳 `--seqs <107 影片名逗號列>`**（dataset 在 seqs 非空時直接用、繞過 -SDP 過濾）。
  seqs 列存 `datasets/PersonPath22/train_seqs.txt` 供五階段重用。

**Cache 策略（建議）：建 cache，fp16。**
1. **建 cache（distill + GT2 用，GT1 維持 live）**：recipe 跑很多 epoch（distill 30 + T3 15 + T1 15），cache 把 teacher backbone forward 攤成一次、每 epoch 重用 → **整體快好幾倍**。GT1 依部署 recipe 維持 **live**（gate feedback @gt_ratio 0.5 需 live teacher）。
2. **fp16 cache**：107 影片 ~19k keyframes × ~3-6MB = fp32 ~57-114GB → **fp16 砍半（~30-57GB）**，且直接打 [[project_distill_cpu_h2d_bottleneck]]（cache H2D 是真瓶頸，commit `09eb174` 半精度傳輸 + GPU widen = 3.0× bit-exact）。建 cache + fp16 = 又快又不撞 H2D 牆。
3. **Leakage-critical / teacher 前置**：cache 烤進 teacher 的 P3/P4/P5 + cls/reg → **必須由不碰 MOT17 的 teacher 建**（base yolo26s 或 PP22-gated）。**cache 是 teacher-specific → teacher A/B 必須先定**（選 B 則先訓 PP22-gated teacher）才能建 cache。
4. **預處理一致**：PersonPath22 1920×1080 → **640 stretch-resize**（同 MOT17 部署口徑，preprocess none）；cache builder 寫 manifest（teacher hash / resize / gate=off / dtype）符合 v14-R §6.2 schema。

鏈（**精確對齊部署 lineage**，讀自部署 ckpt embedded args；全部 `--data-root datasets/PersonPath22 --detector None`，run-dir 加 `_pp22`）：

| 階段 | 命令要點（部署實值） |
|---|---|
| teacher | base `yolo26s.pt`（選 A）或 PP22-gated（選 B）= `gated_det_*_pp22/epoch_0012` |
| cache | `build_mamba_teacher_cache`（PersonPath22 keyframes，6 條→不，138 train videos） |
| **distill** | `train_mamba_head`：**cache mode**, `--clip-len 1`, `--lr 1e-3`, `--use-pixel-shuffle --use-cross-scan --d-state 16`, `--scan-stop-grad`, seed 20260612 |
| **GT1** | `train_mamba_gt`：**live teacher（不傳 --cache-dir）**, `--gt-ratio 0.5`, `--clip-len 4 --clip-stride 8`, `--lr 1e-4 --lr-gate 0`, `--scan-stop-grad`, 30ep, seed 20260612 |
| **T3 (Phase A)** | `train_mamba_gt`：`--add-temporal`, cache, `--clip-len 3 --clip-stride 6`, `--gt-ratio 0`, `--lr 1e-4 --lr-gate 0`, `--scan-stop-grad`, **15ep**, seed 42，warm from GT1 |
| **T1 (Phase B)** | `train_mamba_gt`：cache, `--clip-len 1 --clip-stride 2`, `--gt-ratio 0`, `--lr 1e-4 --lr-gate 0`, `--scan-stop-grad`, **15ep**, seed 42，warm from T3 = **最終部署候選** |

> 全程 **SSM 凍結**（`--scan-stop-grad`）、**GT 階段 `lr-gate 0`**、teacher 固定。架構 = PixelShuffle+CrossScan+temporal_mamba，
> d_model128/d_state16/num_blocks1，num_classes80。**注意 PersonPath22 keyframe 的有效 frameRate 已是 ~5fps，clip-stride 可考慮從 6 降（keyframe 間隔已 ~5 真實幀）——待 §3 機制等價性實測微調。**
> ⚠️ runner `run_v14replica_t3t1.sh`（15→30）**不是部署版**；部署 = T3 15ep → T1 15ep（以 ckpt 為準）。
**Blocker（v14-R §6.2）**：`train_mamba_head.py` 尚未接 provenance contract（seed/resume/selection/manifest）。
拿數字現在就能跑（label provenance-pending）；要蓋 strict-clean 章再補 contract。

候選選擇：在 **MOT17-02 或 DanceTrack val** 上用 `mamba_size_binned_recall` + `mot17.py` 選 epoch（不可用 train-loss）。

---

## 6. 測試與數字（2026-06-22 實測完成）

> ⚠️ **eval 必須用 `--preset mamba_pyt_backbone`（PyTorch backbone）**，不可用 `mamba_whole_graph`。
> 後者設 `fpn_backbone_engine: yolo26s_backbone_640_best.engine`，gate-free 部署下 backbone 完全來自
> 該 TRT engine（從 **MOT17 v14 teacher** 匯出 = 洩漏），`--mamba-teacher-ckpt` 被 **100% 繞過**。
> 見 memory `project_heldout_trt_backbone_leak`。

### 主數字（e30 PP22 teacher，全鏈 PP22-trained，PyTorch backbone）

| | in-sample（`frozen_v2`, MOT17-trained） | ~~洩漏版~~（MOT17 engine backbone + PP22 head） | **真 held-out**（PP22 backbone + PP22 head） |
|---|---:|---:|---:|
| MOT17 IDF1 | 78.2 | ~~64.2~~ | **50.2** |
| MOT17 HOTA / AssA | 70.2 / 69.7 | 52.8 / 52.2 | **42.7 / 46.6** |
| MOT17 DetA / Rcll | 70.9 / 81.0 | 53.8 / 69.5 | **39.7 / 53.5** |

**洩漏拆解（78.2 → 50.2，−28pp）**：換 head→PP22 = −14pp（head 洩漏 + 域偏移）；再換 backbone→PP22 = −14pp
（backbone 洩漏 + 域偏移）。head 與 backbone 各佔約一半洩漏。真實 leakage-free 泛化 = **IDF1 50.2**。

### 三個關鍵歸因實驗

1. **多訓 teacher（e5→e30，loss −19%）對 transfer 幾乎無效**：MOT17-02 乾淨 held-out 只 33.8→35.1（+1.3）。
   backbone 把 PP22 擬合得更好卻沒換來 MOT17 增益 → **不是 capacity/duration 問題**。多訓的真正價值是**揪出上面的
   TRT backbone 洩漏 bug**。

2. **GMC-only（剝掉 relink/OAO/stability/插值，只留 GMC）= IDF1 49.0**，僅比 full tracker 50.2 低 1.2，
   但 AssA −2.8 / IDs +217 / MOTA +1.9。→ **association 增強會 transfer**（在沒看過的域仍 +1.2 IDF1 / +2.8 AssA），
   不是 MOT17 過擬合；唯一在域外轉負的是**插值**（低 recall 時補洞引 FP，MOT17 高 recall 時淨正）。

3. **PP22 in-domain test（12 條 test split，從未訓練）反證「域差」假說**：

   | 測試域 | Rcll | DetA | IDF1 | HOTA |
   |---|---:|---:|---:|---:|
   | **PP22 in-domain** | **42.8%** | 32.3 | 45.1 | 37.8 |
   | MOT17 cross-domain | 53.5% | 39.7 | 50.2 | 42.7 |

   **模型在自己的域 recall 反而更低（42.8 < 53.5）**。若是域轉移問題應 in-domain 高、cross 低 → 恰相反。
   per-seq recall 23–69%（連稀疏場景都低），corr(密度, recall) = −0.45。

### 結論：瓶頸是**偵測器弱**，不是域 / tracker / 訓練長度

模型在哪個域都 recall-limited（23–69%）。根因 = 偵測訓練太弱：① sparse keyframe（~21%、~13k 樣本）；
② **lr-yolo 1e-5 太小，backbone 幾乎還是 base COCO yolo26s**（解釋 e5→e30 只 +1.3，且 loss 在 LR 被衰光時
仍在 −0.025/epoch 健康下降 = LR schedule 把學習掐死，非收斂）；③ mamba head 蒸餾自此弱 teacher。
**真正的槓桿在偵測訓練強度**（放大 lr-yolo + 拉長/晚衰 schedule、稠密採樣、augmentation、必要時換大 backbone），
不在混資料集 / tracker 調參 / 多訓 epoch。

> **更正（2026-06-22）**：曾列「converter 濾掉 crowd 標註」為根因 — **錯**。實查標籤共現，sitting/standing/
> **severly_occluded**/person_in_background 全部 co-label `person`，現行 filter `'person' in labels` 早就留著它們，
> 只丟掉純 `reflection`（樣本中 21 個，本就該丟）。crowd/遮擋/背景的人**沒被濾掉**。corr(密度,recall) −0.45 是
> 偵測器對密集小目標的真實能力上限，不是標註缺失。

### 復現

- pipeline：`scripts/train/temporal_yolo/run_pp22_heldout_e30.sh`（teacher e30 → cache → distill → GT1 → T3 → T1）
- eval（真 held-out）：`mot17.py --preset mamba_pyt_backbone --detector SDP --mamba-ckpt runs/mamba_gt_pp22_t3_t1_e30/best.ckpt --mamba-teacher-ckpt runs/gated_det_pp22/best.ckpt`
- PP22 in-domain：先 `personpath22_to_mot.py --split test --out-dir datasets/PersonPath22/mot_test`，再 eval `--data-root datasets/PersonPath22 --split mot_test --sequences <...>`

---

## 7. 回報紀律（誠實邊界）

- **預期比 78.2 低**，且這是**正確且想要的**：用虛高 in-sample 換**真實、可比、held-out** 的數字。
- **tracker 常數仍 MOT17-tuned**（`oao_tau`/bridge 門檻等在 MOT17 sweep）→ MOT17 測試時 detector 乾淨但
  tracker 超參有殘留 leakage（較輕，已揭露）。要全淨：在 PersonPath22 val 或 MOT17 held-out 子集 retune tracker 常數。
- **域差異**：PersonPath22 靜態相機為主；MOT17 含移動相機 → 測 MOT17 移動序列是公平的泛化壓力（GMC 在場）。
- **License**：PersonPath22 CC-BY-NC，eval/論文 OK，release 倉不可含。

---

## 8. 殘留風險

1. provenance contract（§5 blocker）— 數字可先跑，strict 章後補。
2. recipe 在 MOT17 調的（curriculum 長度/LR/stride）→ 搬 PersonPath22 可能要輕 retune。
3. label 過濾規則（crowd / person_in_background / occluded）對 recall 有影響 → 需小消融定規則。
4. keyframe frame_idx base（0/1）與 ffmpeg 抽幀對齊 → 實作時須驗第一幀對齊。
5. flow gate 在大間隔下的行為 → 建議關。

---

## 9. 開放決策與狀態

- **資料**：✅ `videos.zip`（11.4GB, 181 支）下載+驗證完成。**但 videos.zip 只含 181/236 支**（其餘 55 為外部來源，需按 external_dataset.md 各別抓）→ **可直接用 train 107/138 支**（仍 15× MOT17 多樣性，足夠，不追外部 31 支）。標註（全 236）/splits/test-MOT 包已下載。下一步=抽 107 train 影片的 keyframes + 寫 converter。
- **待 user 決策**：(1) teacher 選 A（base YOLO，先）或 B（PersonPath22-gated）；(2) 是否先補 provenance contract；
  (3) label 過濾規則；(4) 是否同時保留 DanceTrack 對照。
- **Optional 附錄（降級）**：MOT17 LOO 同域 leakage 隔離（teacher+cache+distill+GT 全鏈扣 02、eval 02）—
  控住域但有 7→6 shrinkage 混淆，僅在需要「同域記憶量測」時做。

---

## 10. 為何 MOT17 in-sample 快速收斂（= 過擬合指紋，非「好學」）

「快速收斂 + 高分（78.2）」和「泛化好」是**互斥**的——快收斂正是走了記憶捷徑的證據。三因疊加：
1. **train ≡ test**：head/teacher/cache 全在那 7 條評測序列上訓練，梯度直接 optimize 待評分的幀 → 收斂 = 記憶完成。
2. **7 個固定場景 + 連續幀高冗餘**：要擬合的 pattern 少、相鄰幀近重複 → 幾個 epoch 就飽和。對比 PP22 是
   107 場景 + keyframe 去重（隔 ~5 幀）→ 寬分佈、無免費記憶。**悖論**：PP22 樣本更多（~13k > ~5300）卻更難——
   快收斂的原因不是資料少，是**多樣性低 + train=test**。
3. **強 COCO teacher prior**：MOT17 正面直立行人 base yolo26s 本就偵測得不錯，fine-tune 只補小的場景特定 delta。
   frozen_yolo 量過此 delta：凍 backbone 73.4→55.8（−17.6 = 7 場景的場景特定記憶，不轉移）。

**佐證的非對稱**：PP22 teacher 訓到 e30 loss 仍在降（5.73，未飽和）；MOT17 teacher 早早 plateau/NaN（部署取 NaN-at-e20
的 e12）→ 7 場景幾 epoch 飽和 vs 107 場景持續在學。

## 11. 後續方向（2026-06-22 定）

**先在 PP22 訓練出「真的強」的偵測器，再開始做 held-out 比較。** 現行 50.2 是拿**弱模型**測的，對系統泛化能力不公平
（模型連 PP22 自己的 test 都 recall 42.8%）。比較要有意義，得先把偵測訓練做強。

**已實作（A+B，default off，bit-exact，2026-06-22；recipe = `run_pp22_teacher_strong.sh`，尚未跑）**：
- **A schedule/LR**：`--lr-yolo 1e-4`（從 1e-5）+ `--epochs 60` + 新增 `--min-lr-ratio 0.1`（cosine 不再衰到近 0，
  backbone 晚期繼續學）。直接打「LR schedule 把學習掐死」這個已證的瓶頸。
- **B 資料**：`--augment`（clip-consistent hflip + scale/translate jitter + brightness/contrast，一個 transform
  套整個 clip 保時序/ID 對齊，`_augment_clip` + 5 unit tests）+ `--balance-by-seq`（inverse-seq-freq 採樣，密集/長
  序列不再主導）。

**尚未做**：稠密採樣（keyframe 之外）、mosaic / copy-paste（對小+密最有效，但需設計 + 訓練驗證）、換 yolo26m
（`project_yolo26m_capacity`）、strategy E（切 PP22 val、按 recall 選 epoch，取代 train-loss）。

達到「PP22 in-domain recall 夠高」後，再跑 MOT17 held-out + PP22 held-out 雙向對照，才是公平的泛化量測。

---

## 一句話

**Train PersonPath22（keyframe 當 temporal 軸、零內插，治過擬合）→ test MOT17（沒看過、合法 held-out、可比文獻）。**
唯一真工程 = gluoncv→MOT keyframe converter + teacher 不碰 MOT17；T3→T1 是一次性塑形，sparse keyframe 直接餵。
**現況**：鏈跑通、leakage-free 數字 = 50.2，但瓶頸在偵測器弱 → 下一步是先把 PP22 偵測訓強再比較（§11）。
