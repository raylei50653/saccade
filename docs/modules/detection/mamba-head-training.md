# Mamba Detection Head — 完整訓練流程

> Option F 的 `MambaDetectionHead` 從零到生產 checkpoint（`runs/mamba_gt_vgt_mamba_v14/best.ckpt`）的端到端訓練流程。
> 架構與 ablation 細節見 [option-f-mamba-head.md](option-f-mamba-head.md)；本文聚焦「**怎麼訓出來的**」。

## 0. 總覽

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  Teacher: GatedYOLODetector (runs/gated_det_v1/best.ckpt)│
                 │  = 凍結 yolo26s backbone (layers 0-22) + Gated Detect head│
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
   │   backbone/gate 凍結；gate-feedback = 前一幀 GT 渲染高斯熱圖           │
   │   warm-start 自 Stage 1，gt-ratio 0.5                                  │
   └──────────────────────────────────────────────────────────────────────┘
```

兩階段的理由：蒸餾讓 head 快速得到「能用」的 cls/reg 表示（MSE 收斂穩）；GT 微調再用真值 + 追蹤指標導向的 v8 loss 把 MOTA/IDF1 推上去。

## 1. Teacher（前置，非本流程訓練）

`runs/gated_det_v1/best.ckpt` = `GatedYOLODetector`：凍結的 yolo26s backbone + Gated Detect head（`train_gated_detector.py` 產出）。訓練 Mamba head 時 teacher **全程凍結**，只當 (a) 特徵來源 (b) v8DetectionLoss 的 anchor/TAL assigner 來源 (c) spatial gate。

## 2. Stage 0 — 特徵快取

backbone 對每幀的輸出在整個訓練期間不變，所以先一次算好存檔，訓練時只讀 cache、跳過 backbone forward（這是「cache mode」）。

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --img-size 640 \
    --precompute-dir runs/trt_feat_cache_v2
```

- 用 **eager PyTorch backbone**（非 TRT engine）在 `--img-size` 跑，逐幀存 `cache_dir/SEQ/NNNNNN.pt = {p3,p4,p5}` (half)。
- `--precompute-dir` 模式 cache 完即 `return`，不續訓。
- 容量：640 約 2.9 MB/幀、1024 約 7.3 MB/幀；MOT17 SDP 7 seq = 5316 幀 → 640 ~15 GB、1024 ~39 GB。
- 訓練時 `train_mamba_gt.py` 預設把整個 cache 預載進 VRAM（夠的話）或 CPU RAM（移除 per-step disk 讀）。`--no-preload-cache` 改逐步從 disk 串流（RAM 不足時用）。

> ⚠️ train(eager backbone) vs eval(TRT engine) 的特徵有 FP16 級別差異，但實測可互通（v14 train eager-640 → eval TRT-640 正常）。

## 3. Stage 1 — 蒸餾（`train_mamba_head.py`）

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --cache-dir runs/trt_feat_cache_v2 \
    --use-pixel-shuffle --use-cross-scan \
    --epochs 20 --batch-size 8 --lr 1e-3
```

- Loss = MSE：MambaHead 的 cls/reg 對齊 teacher Detect head 的輸出。
- 架構旗標在此決定：`--use-cross-scan`、`--use-pixel-shuffle`、`--d-state`、`--spatial-reduction` 等。
- 產出 `runs/mamba_distill*/best.ckpt`，作為 Stage 2 的 warm-start。

## 4. Stage 2 — GT 微調（`train_mamba_gt.py`）★ 生產 checkpoint

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --mamba-ckpt runs/mamba_distill_vgt_mamba_v2/best.ckpt \
    --cache-dir runs/trt_feat_cache_v2 \
    --run-dir runs/mamba_gt_vgt_mamba_v14 \
    --img-size 640 --epochs 60 --batch-size 4 --lr 3e-4 \
    --gt-ratio 0.5 --use-pixel-shuffle
```

- **監督 = 真實 MOT17 GT**（`_make_yolo_batch`），**不是 teacher pseudo-label**。Loss = Ultralytics `v8DetectionLoss`（DFL + CIoU + BCE），其 anchor/TAL assigner 取自 teacher.yolo_model（隨 feature map 尺寸自動生 anchor）。
- **gate-feedback**：`_build_gate_inputs` 以機率 `gt-ratio` 把前一幀 GT 框渲染成高斯熱圖（`TrackerGateInput`）餵進 spatial gate，模擬追蹤閉環。
- 架構參數從 `--mamba-ckpt` 的 `mamba_args` 繼承（`use_cross_scan` 等），warm-start 載入 Stage 1 權重。
- 凍結選項：`--freeze-teacher`(預設)、`--freeze-temporal`、`--freeze-spatial`（只訓 cls/reg head）。

## 5. 版本譜系（vgt_v2 → v14）

「**vgt**」是 run 命名標籤（指 GT-ft 的 gate-feedback「虛擬追蹤訊號」），**不是程式旗標**。整個 v2→v17 系列都是上面同一條 distill→GT-ft pipeline，差別在**架構**：

| ckpt | 架構 | IDF1 | 結論 |
|---|---|---|---|
| `mamba_gt_vgt_mamba_v2` (vgt_v2) | temporal SSM（無 cross-scan）| 69.4% | 舊代，**重度依賴 temporal**（拔掉只剩 49.4%）|
| Cross-Scan N=1 | 四向空間掃描（純空間）| 71.6% | 空間掃描取代 temporal 的作用 |
| **`mamba_gt_vgt_mamba_v14`** | **Cross-Scan + PixelShuffle + N=16（純空間）** | **72.4% / MOTA 77.3%** | ★ **生產版** |
| v15 (+SSM temporal) | v14 + 時序 | 58.6% | NO-GO（−13.8pp）|
| v17 (+TemporalAttention) | v14 + 時序 | 69.2% | NO-GO（−3.2pp）|

**關鍵轉折**：v2 靠 eval-time sliding-window temporal SSM 壓 FP；改用 cross-scan（四向空間掃描）後純空間就追平且不需時序 buffer → N=16 SSM 修正重訓 = v14。後續所有時序擴充（v15-v17）皆退步，**時序方向結案**。

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
