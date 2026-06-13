# Mamba Detection Head — 完整訓練流程

> Option F 的 `MambaDetectionHead` 從零到生產 checkpoint（`runs/mamba_gt_vgt_mamba_v14/best.ckpt`）的端到端訓練流程。
> 架構與 ablation 細節見 [option-f-mamba-head.md](option-f-mamba-head.md)；本文聚焦「**怎麼訓出來的**」。
>
> 新的 v14-R 實驗必須遵循
> [Mamba v14-R Training Protocol](mamba-v14r-training-protocol.md)。本文件中的
> 舊命令只保留歷史與架構背景。

> **Artifact audit（2026-06-12）**：現有 v14 `best.ckpt` 是 epoch 58，
> 但文件中的 loss `2.87` 對應 epoch 30。epoch 31-60 resume 時要求的
> `lr=3e-4` 未生效，實際 LR 只有 `1e-6` 至 `3e-6`；cache mode 也使
> `gt_ratio=0.5` gate feedback 無效。完整審計見
> [`report_data/mamba_v14_training_audit.md`](../../../report_data/mamba_v14_training_audit.md)。
>
> **Frozen-SSM audit（2026-06-12）**：v14 整條 lineage 訓練期間
> `_selective_scan` 走無 grad_fn 的 raw CUDA forward，`A_log/D/conv1d/
> x_proj/dt_proj` 從 distill 初始化起**從未被更新**（v14 與 parent 逐 bit
> 相同、`A_log == log(arange(1..16))`）。v14 實為「凍結結構化隨機 SSM
> mixer + 可學 gate/readout/heads」。「N=1→N=16 curriculum」是 eval
> artifact，非訓練機制。證據鏈見
> [`report_data/mamba_v14_frozen_ssm_audit.md`](../../../report_data/mamba_v14_frozen_ssm_audit.md)；
> 復刻流程見 [v14 Replication Protocol](mamba-v14-replication-protocol.md)
> （`--scan-stop-grad` 重現此 regime）。

## 0. 總覽

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  Teacher: GatedYOLODetector (runs/gated_det_v1/best.ckpt)│
                 │  = 整個 YOLO backbone/Detect head/gate 全程凍結           │
                 └───────────────┬─────────────────────────────────────────┘
                                 │ P3/P4/P5 FPN 特徵 (128/256/512 ch)
                                 ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Stage 0: 特徵快取 (precompute)                                          │
   │   train_mamba_head.py --precompute-dir  → cache_dir/SEQ/NNNNNN.pt      │
   │   每幀存 {p3,p4,p5} half；backbone 整段訓練期間不再跑                    │
   └───────────────┬──────────────────────────────────────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Stage 1: 蒸餾 (train_mamba_head.py)                                    │
   │   MSE loss：MambaHead 學 teacher Detect head 的 cls/reg 輸出           │
   │   產出 runs/mamba_distill*/best.ckpt                                   │
   └───────────────┬──────────────────────────────────────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Stage 2: GT 微調 (train_mamba_gt.py)  ← 生產 checkpoint 由此產出        │
   │   丟掉蒸餾 loss，改 MOT17 GT + Ultralytics v8DetectionLoss             │
   │   YOLO/Detect head/gate 權重與 BN 全凍結；只更新 Mamba student          │
   │   warm-start 自 Stage 1，gt-ratio 0.5                                  │
   └──────────────────────────────────────────────────────────────────────┘
```

兩階段的理由：蒸餾讓 head 快速得到「能用」的 cls/reg 表示（MSE 收斂穩）；GT 微調再用真值 + 追蹤指標導向的 v8 loss 把 MOTA/IDF1 推上去。

> **成功因素歸納與配方升級候選**：為什麼這條路徑在 5k 樣本上有效
> （容量配比、frozen-SSM 隱式正則化、teacher prior、三段課程），以及
> 2026-06-13 的 **T3→T1 GT2 curriculum**（IDF1 75.4，首次超越 legacy
> v14 的 75.1，增益全在 AssA）與 multi-seed 噪聲帶寬（0.4pp）——見
> [mamba-v14-replication-protocol.md](mamba-v14-replication-protocol.md)
> 的「成功因素歸納」與「T3→T1 GT2 curriculum」節。

## 1. Teacher（前置，非本流程訓練）

`runs/gated_det_v1/best.ckpt` = `GatedYOLODetector`，由
`train_gated_detector.py` 產出。進入任何 Mamba distill/GT 階段後，
teacher 是**純凍結依賴**：

- YOLO backbone、Detect head 與 spatial gate 的參數皆 `requires_grad=False`，
  不進 optimizer，也不接受 gradient update。
- 整個 teacher 維持 `eval()`；YOLO BatchNorm 的 running mean/variance 固定，
  不因 Mamba 訓練資料再次更新。
- GT 階段的 `v8DetectionLoss` 借用 teacher Detect head 的 stride、anchor/TAL
  設定，不代表 Detect head 參與訓練。
- cache 模式連 teacher forward 都不執行，只讀固定的 P3/P4/P5 與 cls/reg
  targets；每個 epoch 唯一更新的是 Mamba student。

這個「純凍結」描述限定於 **Mamba 訓練階段**。歷史
`gated_det_v1` 本身建立時的 checkpoint metadata 記錄 `lr_yolo=1e-5`，
因此它在自己的 gated-detector 訓練階段曾微調 YOLO；不能宣稱該 teacher
始終等同未修改的原始 `yolo26s.pt`。

### 1.1 Teacher 為何是核心變因

teacher 不只是提供一組固定標籤，而是定義 Mamba student 的整個學習座標系：

1. **輸入特徵分布**：Mamba 接收 teacher backbone 的 P3/P4/P5。backbone
   是否經 MOT17 domain adaptation，會改變小目標紋理、前景/背景分離與各尺度
   feature magnitude；後續 N=1/N=16 curriculum 都是在這個分布上學習。
2. **Stage-1 蒸餾目標**：teacher Detect head 的 cls/reg raw outputs 決定
   student 的初始分類信心、box regression 與尺度偏好。蒸餾不是只複製
   backbone feature，而是複製 teacher 的 detection prior。
3. **Stage-2 loss 語意**：GT fine-tune 雖使用真實標註，不使用 teacher
   pseudo-label，但 `v8DetectionLoss` 仍借用 teacher YOLO 的 class count、
   stride、anchor/TAL assigner 設定。這部分主要固定 loss geometry，不等同
   teacher Detect head 繼續教 student。
4. **可達收斂區域**：MOT17 資料量不足以讓 Mamba detection head 從任意
   initialization 穩定收斂。teacher distillation 與 N=1 GT stabilization
   共同把 student 放進較好的 basin，再切換到 N=16 微調。

歷史 `gated_det_v1` 的實際 teacher prior 很強：

- 使用全部七個 MOT17-SDP sequence，YOLO 以 `lr_yolo=1e-5` 訓練到 epoch 12。
- 相對原始 `yolo26s.pt`：learned weights 偏移 **1.84%** relative L2，
  BN running stats 偏移 **15–17%**（早先報告的 7.18% 把兩者混在一起，被
  BN 統計量灌水）。teacher prior 的主要成分是 BN 的 MOT17 重校準 +
  1.8% 權重微調。
- 權重軌跡（e1=0.48% 平滑升至 e12=1.84%）與 epoch_0001 optimizer step
  count（665 ≈ 單 epoch）證實現存 artifact 為單段、從原始 yolo26s.pt
  起訓；更早（05-19/20）的 gated run 已被覆蓋，未進入 lineage。
- gate alpha 很小：P3 `0.00230`、P4 `0.01182`、P5 `0.00231`。
- `trt_feat_cache_v2` 以 `gate_input=None` 建立，因此 cache 是 **ungated**
  P3/P4/P5；歷史 Mamba 流程實際繼承的是 MOT17-finetuned YOLO backbone 與
  Detect head，而不是 gate feedback。

因此必須區分三種 teacher：

| teacher | YOLO 狀態 | 對 Mamba 的實際意義 |
|---|---|---|
| 原始 YOLO | `yolo26s.pt`，未做 MOT17 adaptation | 通用特徵與原始 Detect prior |
| 歷史 `gated_det_v1` | YOLO 曾用全部 MOT17 微調 | MOT17 domain-adapted feature/Detect prior；含 02 洩漏 |
| strict frozen teacher | 排除 02，但 YOLO 權重與 BN 固定 | 若 cache ungated，幾乎等同原始 YOLO；只訓 gate 對 cache/distill 幫助有限 |

這帶來一個必要的實驗判斷：若規範要求「原始 YOLO 全程純凍結」，就不能期待
只重訓 gate 重現 `gated_det_v1` 的提升，因為 gate 不會進入 ungated cache。
要同時保持 strict holdout 並重建歷史 teacher prior，必須在六個 training
sequences 上進行受控的 YOLO domain adaptation，或新增可訓練 adapter/Detect
head；兩者都應視為獨立 teacher ablation，不能與純 frozen-YOLO 結果混稱。

~~建議將 teacher 與 curriculum 分開做 2×2 對照~~（**2026-06-12 修正**：
frozen-SSM audit 證明「N=1 GT stabilization」不是訓練機制——scan 參數從未
被訓練，N=1 只是 `A.shape[0]` bug 的 forward/eval artifact。curriculum 軸
改為 **gradient regime**）：

| Teacher prior | Gradient regime | 狀態 |
|---|---|---|
| 原始 frozen YOLO | scan full-grad（現行） | ✅ 已跑 = strict v14-R（02 recall 0.48，最弱格） |
| 原始 frozen YOLO | scan stop-grad（v14 regime） | 待跑 |
| domain-adapted teacher | scan full-grad | 待跑 |
| domain-adapted teacher | scan stop-grad | ★ v14 最忠實 replica |

★ 格的全洩漏復刻版（全 7 序列，對照歷史數字）正在執行：見
[v14 Replication Protocol](mamba-v14-replication-protocol.md)。strict
holdout 版須改用六序列 domain-adapted teacher。若只跑純 frozen-YOLO
路徑，結論應限定為「原始 YOLO prior 下的 v14-R」，不能直接否定歷史 v14。

## 2. Stage 0 — 特徵快取

backbone 對每幀的輸出在整個訓練期間不變，所以先一次算好存檔，訓練時只讀 cache、跳過 backbone forward（這是「cache mode」）。

```bash
scripts/train/temporal_yolo/build_mamba_teacher_cache.sh \
    runs/mamba_teacher_cache_v14r_holdout02
```

- 用 **eager PyTorch backbone**（非 TRT engine）在 `--img-size` 跑，逐幀存
  `P3/P4/P5` 與 frozen Detect head 的 `cls/reg` targets (half)。
- 預設只建立 strict-clean 的六個 training sequences，不包含
  `MOT17-02-SDP`。
- `manifest.json` 記錄 schema、YOLO/teacher SHA-256、sequence list、
  frame counts、resize、dtype 與完整狀態；來源不一致或 cache 未完成時拒絕訓練。
- `--precompute-dir` 模式 cache 完即 `return`，不續訓。
- Stage 1 使用 `--cache-dir` 時不再執行 YOLO backbone、Detect head 或其
  BatchNorm；每個 epoch 只執行 Mamba forward/backward。舊版只有
  `P3/P4/P5` 的 cache 會被拒絕，必須重新 precompute。
- 容量：加入 teacher cls/reg targets 後，640 約 5.3 MB/幀；MOT17 SDP
  7 seq = 5316 幀，約 28 GB。低於 40 GB VRAM 時預設載入 CPU RAM。
- 訓練時 `train_mamba_gt.py` 預設把整個 cache 預載進 VRAM（夠的話）或 CPU RAM（移除 per-step disk 讀）。`--no-preload-cache` 改逐步從 disk 串流（RAM 不足時用）。

> ⚠️ train(eager backbone) vs eval(TRT engine) 的特徵有 FP16 級別差異，但實測可互通（v14 train eager-640 → eval TRT-640 正常）。

## 3. Stage 1 — 蒸餾（`train_mamba_head.py`）

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "" \
    --cache-dir runs/mamba_teacher_cache_v14r_holdout02 \
    --seqs MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP \
    --use-pixel-shuffle --use-cross-scan \
    --epochs 20 --batch-size 8 --lr 1e-3
```

- Loss = MSE：MambaHead 的 cls/reg 對齊 teacher Detect head 的輸出。
- teacher cls/reg 已包含在 cache；訓練迴圈不重算 YOLO head。
- 架構旗標在此決定：`--use-cross-scan`、`--use-pixel-shuffle`、`--d-state`、`--spatial-reduction` 等。
- 產出 `runs/mamba_distill*/best.ckpt`，作為 Stage 2 的 warm-start。

## 4. Stage 2 — GT 微調（`train_mamba_gt.py`）★ 生產 checkpoint

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --mamba-ckpt runs/mamba_gt_pixelshuffle_crossscan/best.ckpt \
    --cache-dir runs/trt_feat_cache_v2 \
    --run-dir runs/mamba_gt_vgt_mamba_v14 \
    --img-size 640 --epochs 30 --batch-size 4 --lr 1e-4 \
    --gt-ratio 0.5
```

上面依 `runs/mamba_gt_vgt_mamba_v14/epoch_0001.ckpt` 的 metadata 還原第一段
訓練，不應直接當成可重現 recipe。現有 checkpoint 實際分成 30 epoch
`lr=1e-4`，再從自身 checkpoint resume 到 epoch 60；第二段雖指定
`lr=3e-4`，但錯誤保留舊 optimizer state，實際 LR 只有約 `1e-6` 至
`3e-6`。此外使用 `--cache-dir` 時 gate feedback 不會作用。

- **監督 = 真實 MOT17 GT**（`_make_yolo_batch`），**不是 teacher pseudo-label**。Loss = Ultralytics `v8DetectionLoss`（DFL + CIoU + BCE），其 anchor/TAL assigner 取自 teacher.yolo_model（隨 feature map 尺寸自動生 anchor）。
- **gate-feedback**：`_build_gate_inputs` 以機率 `gt-ratio` 把前一幀 GT 框渲染成高斯熱圖（`TrackerGateInput`）餵進 spatial gate，模擬追蹤閉環。
- 架構參數從 `--mamba-ckpt` 的 `mamba_args` 繼承（`use_cross_scan` 等）。
  一般 Stage 2 載入 Stage 1 distill 權重；歷史 v14 則載入已完成 GT 微調的
  `mamba_gt_pixelshuffle_crossscan`，形成額外一段 GT warm-start。
- **YOLO 純凍結不變量**：teacher 權重不更新、BN 不更新；optimizer 只包含
  Mamba student。`--freeze-temporal`、`--freeze-spatial` 只控制 Mamba
  內部哪些模組更新，不會解凍 YOLO。

### 4.1 Controlled v14-R training

`runs/mamba_distill_cs_n16/best.ckpt` 是可用的 Stage-1 起點，架構為
PixelShuffle + Cross-Scan + N=16。v14-R 不從任何舊 Mamba GT checkpoint
warm-start。

這是 **controlled re-finetune**，不是 strict clean holdout：

- `gated_det_v1` 曾用全部 MOT17-SDP 訓練；
- `trt_feat_cache_v2` 由該 teacher 對全部序列產生；
- `mamba_distill_cs_n16` 也用全部 cache 蒸餾。

因此 GT 階段排除 `MOT17-02-SDP` 可以控制 checkpoint selection，但 02 已透過
teacher/distillation lineage 被模型看過。strict clean 實驗必須從
`gated_det` 開始排除 holdout，再重建 cache 與 distill checkpoint。

第一輪在 GT re-finetune 階段排除 `MOT17-02-SDP`，作為小目標導向的
model-selection sequence：

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --mamba-ckpt runs/mamba_distill_cs_n16/best.ckpt \
    --cache-dir runs/trt_feat_cache_v2 \
    --run-dir runs/mamba_gt_v14r_holdout02 \
    --holdout-seqs MOT17-02-SDP \
    --img-size 640 --clip-len 4 --clip-stride 4 \
    --epochs 30 --batch-size 4 --accum-steps 1 \
    --lr 1e-4 --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 --seed 20260612 \
    --best-by none --save-every 5
```

這個設定的語意：

- `clip-stride=4` 覆蓋 MOT17 的 5308/5316 frames（99.85%），不是舊流程固定
  stride 8 的約 50%。
- cache 是 ungated FPN feature，因此必須 `gt-ratio=0`；腳本現在會拒絕
  cache + nonzero gt-ratio。
- `best-by none` 不產生誤導性的 training-loss `best.ckpt`。候選為
  `epoch_0005.ckpt`、`epoch_0010.ckpt` ... `epoch_0030.ckpt`。
- `seed` 會套用 Python、PyTorch、CUDA 與 DataLoader shuffle。

每個候選先跑 held-out detector recall：

```bash
.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt runs/mamba_gt_v14r_holdout02/epoch_0020.ckpt \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --save-frame-records \
    --output report_data/mamba_size_recall_v14r_holdout02_e20.json
```

再對 detector recall 較好的候選跑完整 held-out tracking：

```bash
.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_whole_graph --detector SDP \
    --sequences MOT17-02-SDP \
    --mamba-ckpt runs/mamba_gt_v14r_holdout02/epoch_0020.ckpt \
    --output results/mamba_gt_v14r_holdout02_e20
```

以 selection sequence 的 4-8/8-16 px recall、HOTA、IDF1、FP/FN 選定
epoch，不以 training loss 選定。選完 epoch 數後，用同一 seed、LR、stride
在七個序列上從相同 Stage-1 checkpoint 重訓固定 epoch 數，產生 final
controlled v14-R。

strict clean v14-R 的 dependency chain：

1. `train_gated_detector.py` 排除 selection sequence。
2. 用該 gated checkpoint 重建只含 training sequences 的 feature cache。
3. 從該 cache 重訓 Cross-Scan + PixelShuffle + N=16 distill checkpoint。
4. 用相同 split/seed 進行 Mamba GT fine-tune。
5. selection sequence 僅用於 detector recall/HOTA/IDF1 選模。

### 4.2 Resume semantics

- `--resume` 現在是 exact resume，必須同時存在 optimizer 與 scheduler state；
  它會恢復 checkpoint 內的實際 LR。
- 改 LR、延長舊 checkpoint 或載入沒有 scheduler state 的歷史 checkpoint，
  必須加 `--resume-reset-optimizer`，明確建立新的 optimizer 與單一
  warmup+cosine schedule。
- checkpoint 現在記錄 `epoch_loss`、`epoch_lr`、scheduler state、holdout
  sequences 與 selection status。

## 5. 版本譜系（純空間、temporal 分支與 v14）

「**vgt**」是 run 命名標籤（指 GT-ft 的 gate-feedback「虛擬追蹤訊號」），
**不是程式旗標**。v14 不是 temporal v2 的直接後繼；兩者來自不同實驗分支。

### 5.1 Checkpoint 直接譜系

以下關係來自各 checkpoint 的 `args.teacher_ckpt`、`args.mamba_ckpt` 與
`mamba_args`，需區分「teacher 來源」和「student 權重 warm-start」：

```text
gated_det_v1
  ├─ teacher ─> mamba_distill_vgt_mamba_v2
  │                └─ warm-start ─> mamba_gt_vgt_mamba_v2
  │
  └─ teacher ─> mamba_distill_pixelshuffle_crossscan
                   └─ warm-start ─> mamba_gt_pixelshuffle_crossscan
                                        └─ warm-start ─> mamba_gt_vgt_mamba_v14
```

依 artifact 時間與 checkpoint metadata，實際研究順序為：

1. **2026-05-27：純空間基線**
   `mamba_distill_pixelshuffle_crossscan` →
   `mamba_gt_pixelshuffle_crossscan`。架構已是 PixelShuffle + Cross-Scan，
   沒有 temporal blocks，實際是 **N=1** 模型。checkpoint 雖記錄
   `d_state=16`，但當時 CUDA kernel 誤用 `A.shape[0]`，推論只執行一個
   state。GT best checkpoint 完成於 21:29。
2. **2026-05-29：temporal 分支**
   從另一個 distill run 建立 `mamba_distill_vgt_mamba_v2` →
   `mamba_gt_vgt_mamba_v2`。這條分支加入 temporal SSM，並產生意外強的
   eval-time FP suppression，但不是 v14 的父權重。
3. **2026-05-31：v14 回到純空間分支**
   修正 CUDA selective-scan 將 `A.shape[0]=1` 誤當成 `dstate` 的問題，
   讓原本 configured `d_state=16` 真正以**有效 N=16** 執行，再從
   `mamba_gt_pixelshuffle_crossscan/best.ckpt` GT warm-start 重訓純空間
   Cross-Scan 模型。

因此使用者記憶中的「先有純空間，T 是後來加入；純空間意外很穩，最後 v14
又補了一項修正」與 artifacts 一致。v14 補的是 **CUDA selective-scan 的
SSM state dimension N=16 修正**，不是新增 temporal 模組。

> **Frozen-SSM 修正（2026-06-12）**：上述「N=16 修正」只作用於
> forward/推論；該年代的 scan 輸出無 grad_fn，**整條 v14 lineage 的
> SSM 內部參數（A_log/D/conv1d/x_proj/dt_proj）從未被訓練**（v14 與
> parent 逐 bit 相同、A_log 等於確定性初始值）。「修正後不重訓即 72.2」
> 的原因是 scan 內部本來就是 N=16-shaped 的未訓練 init。v2/v14 各 GT
> 階段實際更新的只有 in_proj（z 半邊）、out_proj、input_proj/downsample/
> upsample 與 cls/reg heads。§5.2 的「temporal in_proj/out_proj 有變、
> A_log/conv 不變」異常與此梯度拓撲一致。完整證據見
> [`report_data/mamba_v14_frozen_ssm_audit.md`](../../../report_data/mamba_v14_frozen_ssm_audit.md)。

| checkpoint | 直接權重起點 | teacher | 架構 |
|---|---|---|---|
| `mamba_distill_vgt_mamba_v2` | 隨機初始化 student | `gated_det_v1` | temporal SSM，無 Cross-Scan/PixelShuffle |
| `mamba_gt_vgt_mamba_v2` | `mamba_distill_vgt_mamba_v2` | `gated_det_v1` | temporal SSM |
| `mamba_distill_pixelshuffle_crossscan` | 隨機初始化 student | `gated_det_v1` | Cross-Scan + PixelShuffle + N=1，純空間 |
| `mamba_gt_pixelshuffle_crossscan` | `mamba_distill_pixelshuffle_crossscan` | `gated_det_v1` | Cross-Scan + PixelShuffle + N=1，純空間 |
| `mamba_gt_vgt_mamba_v14` epoch 1–30 | `mamba_gt_pixelshuffle_crossscan` | `gated_det_v1` | Cross-Scan + PixelShuffle + N=16，純空間 |
| `mamba_gt_vgt_mamba_v14` epoch 31–60 | 自身 checkpoint resume | `gated_det_v1` | 同上 |

因此「v14 是從 v2 發展而來」只適用於研究時間線中 v2 提供的 temporal
對照經驗；**v14 並未直接載入 `mamba_gt_vgt_mamba_v2` 或
`mamba_distill_vgt_mamba_v2` 的權重**。v14 的直接父 checkpoint 是
`mamba_gt_pixelshuffle_crossscan/best.ckpt`。

目前 workspace 保有 `mamba_gt_pixelshuffle_crossscan/best.ckpt`，但其 metadata
所指的 `mamba_distill_pixelshuffle_crossscan/best.ckpt` 已不在 workspace。
GT checkpoint 的 `mamba_args` 證實其父 distill 架構是純空間
Cross-Scan + PixelShuffle；現有 artifacts 足以驗證架構與 v14 的直接
warm-start，但不足以從該 distillation 權重原樣重播整條歷史鏈。

### 5.2 `mamba_gt_vgt_mamba_v2` 歷史訓練紀錄

#### 已驗證事實

- 訓練時間：distill 約為 2026-05-29 13:38–14:39，GT fine-tune 約為
  14:42–14:56。
- 當時 HEAD 為 `9b5e9d94`（`feat: P1/P2/P3 Mamba head optimizations +
  VGT Flow-Gated temporal Mamba`）。
- 2026-05-29 12:34 的 unreachable WIP stash 已從 Git object database 找回；
  它只修改 evaluator/metrics，未修改 Mamba head 或兩個訓練腳本。
- checkpoint 未保存 Git SHA，因此仍無法證明啟動訓練後工作樹沒有被修改。

shell history 中的實際啟動命令如下。未明寫的參數使用當時腳本預設值：

```bash
# Stage 1: teacher distillation
uv run scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --use-temporal-mamba \
    --cache-dir runs/trt_feat_cache_v2 \
    --run-dir runs/mamba_distill_vgt_mamba_v2 \
    --epochs 30 --batch-size 8

# 中途曾以 latest.ckpt resume
uv run scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --use-temporal-mamba \
    --cache-dir runs/trt_feat_cache_v2 \
    --run-dir runs/mamba_distill_vgt_mamba_v2 \
    --epochs 30 --batch-size 8 \
    --resume runs/mamba_distill_vgt_mamba_v2/latest.ckpt

# Stage 2: MOT17 GT fine-tune
uv run scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --mamba-ckpt runs/mamba_distill_vgt_mamba_v2/best.ckpt \
    --run-dir runs/mamba_gt_vgt_mamba_v2 \
    --epochs 30 --batch-size 4 --clip-len 4
```

checkpoint metadata 補足：

| 階段 | 架構/資料 | 主要參數 | 結果 |
|---|---|---|---|
| Distill | temporal SSM；無 Cross-Scan/PixelShuffle；全部 MOT17 | `T=3`, `lr=1e-3`, cache | epoch 30，loss `1.22769` |
| GT fine-tune | 從 distill best warm-start；全部 MOT17 | `T=4`, `lr=1e-4`, warmup 5, `gt_ratio=0.5`, 無 cache | epoch 30，loss `4.75512` |

#### 已確認的訓練異常

1. Distill 的 temporal 分支確實以 `T=3` 執行，但當時用
   `torch.cat([t0_batch, t1_batch, t2_batch])` 組成 time-major tensor，
   head 卻以 `reshape(B, T, ...)` 解讀為 batch-major。當 `B>1` 時，每條
   temporal sequence 會混入不同 batch sample，並非正確的單一 clip。
2. GT 腳本逐幀呼叫 `mamba(feats)`，沒有傳 `T`，因此提交版本中
   `T_frames=1`，完整 temporal SSM 與 flow gate 應被 bypass；GT loss
   主要優化 spatial path 與 detection heads。
3. checkpoint tensor 比對顯示 GT fine-tune 後 flow-gate 參數完全不變，
   temporal block 的 `A_log`、`D`、conv/SSM projection 也不變；但每層
   temporal `in_proj`、`out_proj` 有變化，且 optimizer 保存了這六個 tensor
   的 state。這與 `9b5e9d94` 的逐幀 bypass 程式不完全一致。

#### 目前結論與復現要求

可確認 v2 是「錯序 temporal distillation + 逐幀 GT spatial fine-tune」產生的
checkpoint，eval-time temporal SSM 卻意外帶來強 FP suppression。尚不能確認
六個 temporal projection tensor 為何在 GT 階段更新；可能是訓練期間存在未保存
的工作樹版本，不能把現有 HEAD 直接宣稱為 bit-exact recipe。

> **已解（2026-06-12）**：Claude session `34cabc95`（標題
> `temporal-alternating-t1-t3-training`，05-29→05-31）證實當時 GT 訓練
> 跑的是未 commit 的「每 batch T=3+T=1 雙 loss」工作樹（log 格式
> `loss=… T3=… T1=…`），temporal blocks 確實執行；配合 no-grad scan
> 的梯度拓撲（只有 in_proj/out_proj 可更新），完整解釋此異常。該實驗
> 於 05-31 02:58 全部回滾，v14 建立於回滾後、為純空間模型，未受影響。
> 詳見 [`report_data/mamba_v14_frozen_ssm_audit.md`](../../../report_data/mamba_v14_frozen_ssm_audit.md)。

正式完整復現前，先在 `9b5e9d94` 的獨立 worktree 執行：

1. 單 batch gradient probe，記錄 spatial、temporal、flow-gate 的 gradient。
2. 一個 epoch GT 訓練，對比歷史 `epoch_0001.ckpt` 的 loss 與逐 tensor 變化。
3. 只有行為可對齊後，才投入完整 30 epoch distill + 30 epoch GT。

### 5.3 架構演進

| ckpt | 架構 | IDF1 | 結論 |
|---|---|---|---|
| `mamba_gt_vgt_mamba_v2` (vgt_v2) | temporal SSM（無 cross-scan）| 69.4% | 舊代，**重度依賴 temporal**（拔掉只剩 49.4%）|
| Cross-Scan N=1 | 四向空間掃描（純空間）| 71.6% | 空間掃描取代 temporal 的作用 |
| **`mamba_gt_vgt_mamba_v14`** | **Cross-Scan + PixelShuffle + N=16（純空間）** | **72.4% / MOTA 77.3%** | ★ **生產版** |
| v15 (+SSM temporal) | v14 + 時序 | 58.6% | NO-GO（−13.8pp）|
| v17 (+TemporalAttention) | v14 + 時序 | 69.2% | NO-GO（−3.2pp）|

**關鍵轉折**：Cross-Scan 純空間模型早於 temporal v2 存在。v2 證明
eval-time temporal SSM 可以強力壓 FP，但訓練語意不穩定；既有純空間模型在
修正 CUDA `N=1 → N=16` 後已達 72.2% IDF1，不重訓就接近最佳。再從該純空間
GT checkpoint 重訓得到 v14（72.4%），且不需 temporal buffer。後續 v15-v17
重新加入時序皆退步，**時序方向結案**。

## 6. 評估與部署

```bash
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP
```

- preset `configs/presets/mamba_whole_graph.yaml`：`mamba_ckpt: v14`、`fpn_backbone_engine: yolo26s_backbone_640_best.engine`、whole-graph CUDA graph、ReID off。
- **eval 用 TRT backbone engine**（非 eager）；engine 由 `scripts/model/export_yolo_backbone_ckpt.py` 從 yolo26s.pt export。
- selective scan 在 eval/inference 走 C++ CUDA op（`saccade::selective_scan_fwd`），訓練走可微分 JIT（見下）。

## 7. 高解析度重訓（1024/1280）

v14 是 640 訓練；直接拿去跑高解析會壞或退步（precision 崩）。要用高解析**必須在該解析度重訓 head**。

### 7.1 幾何約束：input 必須 ÷128
head `spatial_reduction=4`（downsample conv k4s4 + PixelShuffle(4)），每個 FPN 層邊長須 ÷4 → input/32 須 ÷4 → **input 須 ÷128**。
- 640 ✓(÷128=5)、1024 ✓(8)、1280 ✓(10)、**960 ✗**(7.5) → 960 在 head `torch.cat` 直接 crash（30 vs 28）。

### 7.2 重訓流程
1. **Cache**：`train_mamba_head.py --precompute-dir runs/trt_feat_cache_1024 --img-size 1024`（eager backbone，~39 GB）。
2. **Train**：`train_mamba_gt.py --cache-dir runs/trt_feat_cache_1024 --img-size 1024 --mamba-ckpt <v14> --run-dir <out>`（warm-start v14；head conv/mamba 權重與解析度無關，可轉移）。
3. **Eval**：`mot17.py --tiling native_1024 --fpn-backbone-engine yolo26s_backbone_1024_best.engine`。

### 7.3 已知地雷（本流程踩過並修復）
- **selective_scan 無 backward**：C++ `mamba_scan.cu` 只有 forward kernel；CUDA op 沒註冊 autograd → 訓練 backward 直接報錯。修法：`_selective_scan` 在 `grad_needed` 時走可微分 JIT scan，inference 仍用 CUDA op（`mamba_head.py`，eval 零影響）。
- **JIT scan 慢**：純 Python 序列迴圈，1024 序列長 → ~75 min/epoch。對策：`--freeze-spatial` 只訓 cls/reg head → frozen scan 走快的 CUDA forward → **~90 s/epoch（~50×）**；多數高解析退化是 confidence 校準問題，head-only recalibration 即可探信號。
- **座標雙重縮放**：whole-graph detector 內部已用 `set_whole_graph_img_dims` 的 sx/sy 把框映射回原圖座標；eval 的 `detect_single_patch_960` 不可再除一次（會 IDF1 崩到 1%）。正解：whole-graph 分支直接餵原圖、回傳即原圖座標。
- **anchor FEAT_SHAPES 寫死**：`mamba_gated_detector.py` 的 whole-graph anchor 原寫死 80/40/20（640），須改依 `img_size` 推算。
- **tiling wiring**：`mot17.py` img-size map、`config/detection.py` --tiling choices、`evaluator.py` detect_fn dispatch 都要加 native_1024/1280。

## 關鍵檔案

| 檔案 | 角色 |
|---|---|
| `scripts/train/temporal_yolo/train_mamba_head.py` | Stage 0 cache + Stage 1 蒸餾 |
| `scripts/train/temporal_yolo/train_mamba_gt.py` | Stage 2 GT 微調（生產 ckpt）|
| `src/saccade/perception/temporal_yolo/mamba_head.py` | `MambaDetectionHead`、`_selective_scan`（JIT/CUDA 雙路）|
| `src/saccade/perception/temporal_yolo/mamba_gated_detector.py` | 推論封裝、whole-graph、anchor |
| `src/tracking/mamba_scan.cu` | C++ selective scan **forward only**（無 backward）|
| `scripts/model/export_yolo_backbone_ckpt.py` | export TRT backbone engine |
| `configs/presets/mamba_whole_graph.yaml` | 生產 eval preset |
