# v14 Replication Protocol（V14-REPLICA-20260612）

> 目標：從原始 `yolo26s.pt` 完整復刻 legacy v14 的訓練鏈，驗證
> v14 recipe 是否可重現（若可，偶然發現轉正式方法；若不可，v14 視為
> lucky checkpoint）。
>
> 本協議**刻意保留歷史的 02 洩漏結構**（teacher/cache/distill 含全 7 序列），
> 因為復刻的對象就是含洩漏的歷史流程；不得當成 holdout 實驗報告。
> strict holdout 實驗見 [mamba-v14r-training-protocol.md](mamba-v14r-training-protocol.md)。

## 核心機制（frozen-SSM regime）

依 [frozen-SSM audit](../../../report_data/mamba_v14_frozen_ssm_audit.md)：
v14 全程訓練中 scan 無梯度，SSM 內部凍結在 init。復刻使用
`--scan-stop-grad`（distill、GT1、GT2 全程），以正確的 N=16 forward
重現歷史梯度拓撲。

唯一不復刻的歷史行為：05-27~05-31 期間 forward 的 N=1 + B/C 錯位讀取
（`A.shape[0]` bug）。該行為只影響凍結 mixer 的 forward dynamics 與
warm-start readout 學到的座標系，v14 最終段（05-31 後）已是 N=16 forward。

## 階段與命令

### Stage T — gated teacher（已完成 2026-06-12）

```bash
.venv/bin/python scripts/train/temporal_yolo/train_gated_detector.py \
    --data-root datasets/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --run-dir runs/gated_det_v14replica \
    --epochs 30 --batch-size 4 --clip-len 2 \
    --img-size 640 \
    --lr-gate 1e-3 --lr-yolo 1e-5 \
    --gt-ratio 0.5 --seed 20260612 \
    --warmup-epochs 0 \
    --save-every 1 --best-by train-loss \
    --protocol-revision V14-REPLICA-20260612
```

結果：epoch 20 batch 555 non-finite loss fail-fast 中止；epochs 1–19 已存。
**teacher 定為 `epoch_0012.ckpt`**（歷史 gated_det_v1 只訓到 epoch 12）。

復刻品質驗證：

| 指標 | 歷史 gated_det_v1 (e12) | replica e12 |
|---|---|---|
| learned-weight drift vs raw | 1.839% | 1.689% |
| BN running-stats drift | 17.17% | 16.98% |
| epoch-12 train loss | 4.660 | 4.714 |

### Stage 0 — feature cache（全 7 序列）

```bash
TEACHER_CKPT=runs/gated_det_v14replica/epoch_0012.ckpt \
SEQS="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP" \
scripts/train/temporal_yolo/build_mamba_teacher_cache.sh \
    runs/mamba_teacher_cache_v14replica
```

### Stage 1 — distill

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --teacher-ckpt runs/gated_det_v14replica/epoch_0012.ckpt \
    --cache-dir runs/mamba_teacher_cache_v14replica \
    --seqs MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP \
    --run-dir runs/mamba_distill_v14replica \
    --use-pixel-shuffle --use-cross-scan --d-state 16 \
    --scan-stop-grad \
    --epochs 30 --batch-size 8 --lr 1e-3 --seed 20260612
```

### Stage 2 — GT1（live teacher，gate feedback 有效）

歷史 `mamba_gt_pixelshuffle_crossscan` 的 args **沒有 cache_dir**：
teacher 每步 live forward，`gt_ratio=0.5` 的 gate feedback 真實作用。

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v14replica/epoch_0012.ckpt \
    --mamba-ckpt runs/mamba_distill_v14replica/best.ckpt \
    --run-dir runs/mamba_gt_v14replica_stage1 \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0.5 --seed 20260612 \
    --scan-stop-grad \
    --best-by train-loss --save-every 5
```

### Stage 3 — GT2（cache mode）= v14replica 最終

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v14replica/epoch_0012.ckpt \
    --mamba-ckpt runs/mamba_gt_v14replica_stage1/best.ckpt \
    --cache-dir runs/mamba_teacher_cache_v14replica \
    --run-dir runs/mamba_gt_v14replica_final \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 --seed 20260612 \
    --scan-stop-grad \
    --best-by train-loss --save-every 5
```

歷史 v14 在 cache mode 下 `gt_ratio=0.5` 本就無效（ungated cache），
現行腳本拒絕 cache + 非零 gt-ratio，故此處 `--gt-ratio 0` 等價。
歷史 epoch 31–60 的 resume 段 LR 實際只有 1e-6 量級（≈ no-op，
e30/e58 recall 相同），不復刻。

### Eval — 成敗判準

```bash
.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt runs/mamba_gt_v14replica_final/best.ckpt \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --output report_data/mamba_size_recall_v14replica.json

.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt runs/mamba_gt_v14replica_final/best.ckpt \
    --output results/mamba_v14replica_final
```

legacy v14 必須在**同 commit 同 config** 重跑對照，不得引用舊報告數字。
參考量級（MOT17-02 @0.25）：v14 all recall 0.940、4–8px 0.871；
tracking 全 7 序列 IDF1 72.4 / MOTA 77.0 量級。

判定：

- replica ≈ v14（IDF1 差 < 1pp、recall 結構一致）→ recipe 可重現，
  teacher prior + frozen-SSM + 多段 warm-start 轉正式方法；
  下一步單變因切 full-grad 對照，定罪/平反可訓練 SSM。
- replica 明顯低於 v14 → 殘餘未控因素（候選：N=1-era warm-start 座標系、
  scheduler/seed、05-19 失傳 artifacts），v14 降級為 lucky checkpoint。

## 結果（2026-06-12）

全鏈跑完（teacher e12 → cache → distill 30ep → GT1 30ep → GT2 30ep，
GT loss 3.221/2.548 vs 歷史 3.567/~2.87）。同 commit 同 config 對照：

| 指標 | legacy v14 | v14replica | strict v14-R（對照格） |
|---|---|---|---|
| MOT17-02 recall@0.25 all | 0.940 | **0.920** | 0.482 |
| MOT17-02 recall@0.25 4–8px | 0.871 | **0.826** | 0.093 |
| IDF1（全 7 序列） | 75.1 | **73.4** | — |
| MOTA | 77.7 | **77.3** | — |
| HOTA | 68.2 | **65.3** | — |

**判定：recipe 實質可重現。** teacher prior + frozen-SSM regime 兩因素
恢復了 strict 路線 46pp recall 崩壞的絕大部分；殘差 −1.7 IDF1 / −2pp
recall 歸於不可恢復因素（無歷史 seed、scheduler 差異、N=1-era warm-start
座標系刻意不復刻、歷史 distill/GT1 精確 epoch 數未知）。v14 不是 lucky
checkpoint；偶然發現（凍結 SSM）已轉為可控因素 `--scan-stop-grad`。

後續單變因對照（同 replica lineage 只切 full-grad）可直接定罪/平反
可訓練 SSM —— 唯一已有的 full-grad 資料點（strict run）與 teacher 混淆。

### 單變因延伸：SSM 解凍微調（2026-06-12）

`runs/mamba_gt_v14replica_ssmft_n16`：warm-start replica final，
`--no-scan-stop-grad`（SSM 內部首次有梯度），其餘同 GT2 再訓 30 epochs。

| | replica（frozen） | +SSM 解凍 ft |
|---|---|---|
| 02 recall@0.25 all / 4–8px | 0.920 / 0.826 | 0.920 / 0.825（e05 與 e29 均持平） |
| IDF1 / MOTA / HOTA | 73.4 / 77.3 / 65.3 | 73.3 / **78.1** / **66.1** |
| train loss | 2.548 | 2.149 |
| A_log max Δ | 0（凍結） | 0.010 |

**判讀**：從收斂點解凍 = 中性到微正（MOTA/HOTA +0.8）；train loss 改善
不轉化為 detector recall。full-grad 本身不是毒；strict run 的退化來源
收窄為 teacher prior 與/或「從非收斂點全程 full-grad 的收斂動態」。
待跑判別格：同 lineage 從 distill warm-start、GT 全程 full-grad。

### Multi-seed 噪聲帶寬（2026-06-13）

全 student 鏈（distill→GT1→GT2，teacher/cache 固定）reseed ×2
（`run_v14replica_seed.sh`，seeds 20260613/20260614）：

| Seed | IDF1 | MOTA | HOTA | AssA | IDs | FP | 02 recall@0.001 |
|---|---|---|---|---|---|---|---|
| 42（replica） | 73.4 | 77.3 | 65.3 | — | — | — | 0.958 |
| 20260613 | 73.2 | 76.2 | 65.3 | 62.8 | 578 | 4840 | 0.958 |
| 20260614 | 73.0 | 76.5 | 65.5 | 62.8 | 594 | 3928 | 0.959 |

**判讀**：IDF1 帶寬 [73.0, 73.4]，跨度 0.4pp。噪聲集中在 precision 側
（FP ±23%），recall 跨 seed 穩定。**v14 legacy 的 1.7pp 殘差是帶寬的
4 倍以上，非 seed noise**；歸因維持 teacher prior + 其他不可恢復因素。
配方判定為高度可重現。

### T3→T1 GT2 curriculum（2026-06-13）— 首次超越 legacy

假說（user 提出）：時序訓練增強空間一致性。GT2 改為兩段
（`run_v14replica_t3t1.sh`）：Phase A `--add-temporal` clip_len=3
stride=6 15ep（cache、stop-grad）；Phase B clip_len=1 stride=2 15ep
（temporal bypass，純空間再適應）。warm-start replica GT1，seed 42。

| | replica GT2 | **T3→T1** | legacy v14 |
|---|---|---|---|
| IDF1 | 73.4 | **75.4** | 75.1 |
| MOTA | 77.3 | **77.6** | 77.7 |
| HOTA | 65.3 | **67.7** | 68.2 |
| AssA | ~62.8 | **66.0** | — |
| IDs / FP | ~578 / ~3900 | **496 / 3272** | — |
| 02 recall@0.001 | 0.958 | 0.959 | — |

**Eval 語義（已驗證 + 代碼聲明）**：所有 t3t1 tracking eval 走
`mamba_whole_graph` 的單幀 forward，temporal blocks bypassed（effective
T=1）— 量到的是純空間塑形效果。注意 `mot17.py` 對 temporal checkpoint
預設 `temporal_T=3`：**非 whole-graph** eval 會啟用 streaming temporal
（sliding window，train/eval mismatch 風險），須傳 `--no-temporal`。
builder（`build_mamba_gated_detector`）現在對三種情況都印明確聲明
（streaming ACTIVE ⚠️ / whole-graph BYPASSED / T=0 BYPASSED）。

**判讀**：+2.0 IDF1 = 噪聲帶寬 5 倍，增益成立。增益幾乎全在
association（AssA +3.2、IDs −14%、FP −17%），recall 持平 —
符合「特徵跨幀一致性」機制：T=3 時 temporal blocks 跨幀混合迫使
spatial path 產出 temporal-consistent 特徵，T=1 推理時 temporal blocks
自動 bypass、一致性保留在 spatial path，部署零成本。

Per-sequence（vs legacy v14）：02 +4.7 / 13 +3.7 / 09 +2.4（crowd、
moving camera 受益最大）；10 −0.7、11 −1.8（低光、近飽和）微退。

**Multi-seed 配對驗證（2026-06-13 完成）**：每個 seed 從自己的 GT1
warm-start 跑 T3→T1，與自己的 plain GT2 配對（`run_v14replica_t3t1_seed.sh`）：

| Seed | plain GT2 | T3→T1 | 配對差 |
|---|---|---|---|
| 42 | 73.4 | 75.4 | +2.1 |
| 20260613 | 73.2 | 75.4 | +2.3 |
| 20260614 | 73.0 | 73.4 | +0.4 |

平均 +1.6。s14 的 +0.4 經 per-seq 歸因**不是塑形失敗**：困難序列
（02/05/10/13）三 seed 全部一致正向（02 +2.4~+3.3、05 +2.6~+3.4、
10 +1.2~+6.8、13 +2.2~+6.9）；s14 被 MOT17-11 稀釋 — 其 plain
baseline 在 11 抽到幸運高點（80.3 vs 其他 seed plain 74.4/75.2），
t3t1 拉回 75.0（與其他 seed t3t1 77.0/78.1 一致），計 −5.3；加上
最大權重序列 04 的 −0.8 噪聲。**判定：增益真實且結構穩定（困難
序列 12/12 正向），但 overall 幅度受單序列 baseline 運氣調制
（+0.4~+2.3）**。三 run train loss 幾乎相同（2.851/2.851/2.863），
再證 train loss 非部署品質 selector。

附帶 NO-GO（2026-06-13，registry #35）：T3→T1 特徵作 relink
embedding 離線探針 — hard pool AUC 0.438，與 plain replica 配對差
~0.001。**consistency ≠ discriminability**：AssA 增益全部經
box/score 穩定性（IoU 路徑）傳導，特徵空間不攜帶個體身分。
探針：`scripts/tools/mamba_relink_features.py`。

歷史脈絡：05-29~05-31 的 uncommitted T1/T3 dual-loss session
（`34cabc95`）可能是同機制雛形，但未進入 v14 lineage。

> 本日完整研究紀錄（含機制歸因、邊界實驗、插值掃描、artifacts 清單）：
> [mamba-t3t1-curriculum-20260613.md](research/mamba-t3t1-curriculum-20260613.md)

### 課程順序與增益保留（2026-06-13）

三個 follow-up 實驗劃清了 T3→T1 增益的邊界：

| | T3→T1 | T3T1→ssmft（錯序） | ssmft→T3T1(反向) | T=3 streaming eval |
|---|---|---|---|---|
| IDF1 | **75.4** | 73.8 | 74.3 | 69.8（PhaseA ckpt） |
| MOTA | 77.6 | **79.4** | 78.8 | 71.6 |
| DetA | 69.7 | **71.0** | 70.7 | 63.1 |
| AssA | **66.0** | 62.8 | 63.6 | 58.2 |

1. **SSM-ft 疊加拮抗**：T3→T1 之後解凍 SSM 再訓 30ep → DetA +1.3 /
   FN −1688 / MOTA 79.4（全系列最高）但 **AssA −3.2 退回 plain GT2
   水平** — v8 loss 無一致性保護項，後續全梯度訓練抹掉塑形。train
   loss 1.95 全場最低而 IDF1 反跌（loss 非 selector 第三證）。
2. **反向順序部分有效**：塑形放最後比錯序 +0.5 IDF1 / +0.8 AssA 且保
   住大部分檢測增益，但 AssA 只回 63.6 ≠ 66.0 —— **DetA↔AssA 沿 SSM
   自由度存在真實權衡軸**，排序無法消除。
3. **T=3 streaming 推理結案**：Phase A ckpt 以原生 T=3 評測（flow_gate
   歸零副本 + `mamba_eager_temporal_probe` preset）僅比 T=1 對照
   +0.7 IDF1，且 FP +23%/MOTA −0.4；checkpoint 本身 69.8 比 T3→T1
   product 低 5.6pp。**T=3 訓練的價值在塑形壓力，不在 temporal blocks
   的推理貢獻**；不投資 whole-graph temporal 配套。

設計規則（暫定）：**最後一段 curriculum 的目標決定特徵的最終構型；
想保留的塑形必須放在最後，或以顯式 loss 項保護。** IDF1/HOTA 北極星
下的 production 候選維持 T3→T1（`runs/mamba_gt_v14replica_t3_t1`）；
MOTA/recall 優先場景可選 T3T1→ssmft（79.4/71.0）。

## 成功因素歸納（為什麼這條訓練路徑有效）

基於復刻 + 歸因實驗的因果整理，按重要性排列：

1. **容量配比：凍結 backbone + 10M head**。~5,300 樣本下任何全網路
   fine-tune 都會過擬合；可訓練容量壓到 head，backbone 保留 COCO 先驗。
   旁證：teacher 只能訓 12 epochs（再訓 NaN）、learned drift 僅 1.84%。
2. **Frozen SSM = 隱式正則化器**（意外發現轉可控因素）。SSM 內部凍結
   在 init 等效隨機結構化 mixer（reservoir 式）；只學 gate/readout/heads
   比連 mixer 一起學更 sample-efficient。直接 full-grad 訓 N=16 退化；
   從收斂點解凍微調則中性偏正（上節 ssmft）。
3. **Teacher prior：溫和域適應監督**。拿掉 teacher prior → strict recall
   崩 46pp；replica 與 legacy 的 1.7pp 殘差主要在此。
4. **三段課程：稠密模仿→混合→純 GT**。distill 的 feature-level 稠密
   訊號是 5k 樣本下唯一能從零收斂的起步方式；GT1（gt_ratio 0.5）平滑
   過渡；GT2（gt_ratio 0）擺脫 teacher 的 systematic bias。
5. **T3→T1 時序塑形**（上節）。訓練時結構約束塑形特徵一致性，
   不增加部署容量。

反面清單（曾被誤認的成功要素）：「N=1→N=16 curriculum」= eval
artifact；「lucky checkpoint」= multi-seed 0.4pp 帶寬否證；「train loss
低 = 好」= replica loss 2.55 < v14 2.81 但 eval 較低；「02 洩漏給 v14
優勢」= per-seq 顯示 02 上兩者相等。

一句話：**在極小資料集上，把所有高容量元件凍成先驗（backbone、SSM、
teacher），只訓練低容量讀出層，用課程從稠密模仿漸進到稀疏 GT；
T3→T1 是同一哲學的延伸 —— 用訓練時的結構約束塑形特徵，而非增加
部署容量。**

## 與歷史的已知偏差

| 項目 | 歷史 | replica | 理由 |
|---|---|---|---|
| seed | 無 | 20260612 | 歷史不可恢復 |
| teacher LR schedule | 推定常數 1e-5（舊腳本無 scheduler） | cosine(30) warmup 0，取 e12 | 現行腳本固定行為 |
| distill/GT1 forward N | 1（且 B/C 錯位） | 16（正確） | bug 不值得復刻；v14 最終段本就 N=16 |
| scan 梯度 | 無（bug） | 無（--scan-stop-grad） | 忠實 |
| GT2 gt-ratio | 0.5（無效） | 0（顯式） | 等價 |
| clip-stride | 推定 8（舊預設） | 8 | 忠實 |
| epoch 31–60 resume 段 | LR≈1e-6 no-op | 不跑 | e30≈e58 |
| teacher 早期 run（05-19） | 已覆蓋失傳 | 不復刻 | 未進入 lineage（權重證據） |
