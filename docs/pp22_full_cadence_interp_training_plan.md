# PP22 全-cadence interp 訓練 — 計畫 / Handoff

> 開新對話做。這份文檔自足:讀完即可接手。
> 分支:`feat/pp22-keyframe-aware-eval-and-training`(基於 `996386d4`)

## 0. 目標 / 假設

在 **PersonPath22(只用 PP22,不碰 MOT17 GT)** 上,用「**全 cadence 訓練 + GT 線性插值 + loss 只在關鍵幀**」重訓 mamba head curriculum,測試「修正訓練 cadence 是否提升 recall」。
**MOT17 只作 post-hoc transfer 診斷,不進訓練**(GT 不進 loss)。⚠️ 用 MOT17 選 checkpoint 本質仍 touching target domain,故**不算 zero-shot**:zero-shot PP22→MOT17 與「為部署做 target-domain calibration/選模」必須分開報告,別混為一談。

### 為什麼
PP22 標註是 ~5fps 關鍵幀。現有訓練資料把關鍵幀重編號 1..N → 訓練時「相鄰幀」其實隔 ~5 真實幀 → temporal SSM 看到大跳躍(非真實運動)。修法 = 跑全 cadence(每幀),讓 SSM 看真實逐幀運動;但中間幀的 GT 是線性插值(近似)→ **loss 只算在真關鍵幀**,中間幀只餵 temporal context。

### ⚠️ 重要 caveat(務必記得)
1. **Deploy 是 T=1 non-temporal**(`--no-temporal`,temporal blocks 被 bypass)→ interp 主要是清理 T3 stage 的 cadence、間接 shape backbone/head 特徵;對「單幀 deploy recall」的直接收益**未證實、可能有限**。這是個要實測的假設,不是已知會贏。
2. **已診斷的 recall 主槓桿其實是 score 校準,不是 cadence**:detector raw recall 93%@0.001 → 76%@0.25 → tracking ~38%,人都偵測到了、被門檻層層篩掉。interp 是在測 cadence 這條**次要**假設。便宜的對照組 = 直接降 `new_track_thresh`/`conf` 門檻下掃(見 §6)。
3. **中間幀特徵必須是「真 teacher 算的」**,不能也線性插值 —— 若插值特徵,SSM 看到的是線性運動 = 它本來就能從關鍵幀推出來的 → 全 cadence 白做。

---

## 1. 目前狀態

### 已 commit(`996386d4`)
- **`--score-on-gt-frames`**(eval):跑全 cadence、只在 gt 關鍵幀算分。`metrics.run_motmetrics_evaluation(score_on_gt_frames=)` 把預測檔過濾到 gt 幀寫 `<output>/_gt_frames/`,motmetrics + TrackEval/HOTA 都指向過濾後。MOT17 dense GT 上 bit-exact no-op。
- **訓練 interp + keyframe-loss**:`dataset.py` `interpolate_gt` + `_interpolate_gt_for_frame`(per-ID 相鄰關鍵幀線性內插、不外推、births/deaths 排除)+ 每幀 `is_keyframe`;`train_mamba_gt.py` `--interpolate-gt` 把主 all-frames loss 與 t1 loss **按 sample 切到關鍵幀子集**(丟 target≠mask,必須切 preds/feats/gt)。全關鍵幀時 bit-exact。`tests/unit/detection/test_gt_interpolation.py`(5 tests)。

### 未 commit(工作區,需保留)
- `scripts/train/temporal_yolo/train_mamba_gt.py`:加了 **`--no-preload-images`**(live 模式 per-step decode,免 RAM OOM)+ **`--num-workers`**(並行 decode)。loader build 已改成 `live_mode/preload/need_images` 邏輯。
- `scripts/eval/select_ckpt_by_recall.py`:加了 **`--extra-eval-args`** passthrough(PP22 須傳 `--no-temporal --no-compile` + `--preset mamba_pyt_backbone`)。**注意:此工具用的是 tracking eval(含後處理),不該拿來守退化 —— 改用 §5 detector-only。**

### 資料已備妥
- **全 cadence 訓練資料已抽好**:`datasets/PersonPath22_full/train/`,**107 seqs / 78,608 frames / 32G**。結構 = img1 全幀(原始編號)+ gt 只在關鍵幀(真實 index 1,6,11…)+ seqinfo。
- 產生指令(已跑完,留存供重現):
  ```
  .venv/bin/python scripts/train/temporal_yolo/personpath22_to_mot.py \
    --videos-zip datasets/PersonPath22/raw/videos.zip \
    --anno-dir datasets/PersonPath22/annotation/anno_visible/anno_visible_2022 \
    --splits datasets/PersonPath22/annotation/splits.json --split train \
    --out-dir datasets/PersonPath22_full/train --keep-all-frames
  ```
- 既有 keyframe 資料 `datasets/PersonPath22/train/` 保留不動。

---

## 2. 硬限制(實測)

| 資源 | 值 |
|---|---|
| 全 cadence 總幀數 | 78,608 |
| RAM | **54G 總**(~46G avail) |
| Disk free | 171G |
| Cache per-frame(實測) | 4.28MB full / **2.87MB p3p4p5-only** fp16(cls/reg GT stage 不用) |

→ **全 cadence cache p3p4p5 ≈ 226G > disk** ❌(fp8 ~113G 可塞但需改 precompute + 精度風險)
→ **RAM preload 全幀 ≈ 96G > RAM** ❌ → 必須 per-step decode
→ **結論:不 cache,live teacher + per-step decode**。但 1080p CPU decode 重 → **要 GPU 硬解(§3,待建)**。

---

## 3. GPU-decode 訓練 pipeline ✅(已建 + smoke 通過,2026-07-01)

**狀態**:已實作並驗證。`dataset.py` `return_jpeg_bytes`(worker 回傳 raw JPEG **bytes**,非 tensor)+ `gpu_decode_clip_batch`(batched nvJPEG → `(B,T,3,S,S)` float[0,255]);`train_mamba_gt.py` `--gpu-decode`。
- **坑(已修)**:worker 若回傳 torch byte-tensor,經 DataLoader queue 走 fd-based shared-memory,大量小 tensor → `os.dup Bad file descriptor` 崩。**改回傳 python `bytes`**(正常 pickle,無 fd 共享),decode 時 `torch.frombuffer(bytearray(b))` 包成 uint8。
- **smoke(50 batch,B4 T4,GT1 recipe)**:GPU-decode ACTIVE + interp ACTIVE + loss 有限(avg~3.0),**VRAM 1.5GB(無 OOM)**。**吞吐:首 batch ~26s(nvJPEG/worker 暖機)但 batch 2+ ~0.2s/batch** → ~8min/epoch,GT1 30ep ≈ 4hr,整鏈 ~overnight。ETA 顯示被首 batch 暖機灌大,實際不慢。
- **測試**:`tests/unit/detection/test_gpu_decode_clip.py`(shape/dtype/range + batch-major ordering,CUDA-gated);live 驗過 bytes-mode vs decode-mode GT/is_keyframe 一致、pixel diff ~2/255(bilinear stretch vs antialias)。
- **stage 交接 ckpt**:`--best-by none` 不寫 best.ckpt,每 epoch 寫 `epoch_XXXX.ckpt`+`latest.ckpt`。GT1→T3→T1 交接用各 stage `latest.ckpt`(cosine LR 已 anneal 到底=常規 warm-start),最終選模仍靠 §5 detector-only 在 T1 各 epoch 上做。

### 3a. Profiling(cuda-synced step 拆解,2026-07-01)— 結論:流程已到底,不用再優化
穩態拆解(b60→b80 delta,GT1 recipe B4 T4):**teacher backbone forward 0.080s/batch(~44%)** > backward 0.036(20%)> decode 0.026(14%)> loss 0.020(11%)> mamba 0.018(10%);**data(loader wait)穩態 ≈0.003s(~0%)**。
- **data I/O 完全 overlap**(GPU-decode + bytes + 8 workers + persistent_workers 成功):`data=24.5s` 是一次性 worker-spawn/prefetch fill(persistent 只付一次,非每 epoch),穩態 loader 不等 → **不是 I/O-bound**。
- **compute-bound 在 frozen teacher backbone(44%)**。**`--compile` NO-GO**:teacher 反而變慢(gate_input None-vs-list 兩路徑 + box 數變動 → dynamic shapes 狂 recompile;58s 前置吃不回,穩態僅 0.080→0.065)。
- **唯一剩的槓桿 = 把 teacher forward 從 T-serial 批成 B*T-一次**(省 launch),但 gate_input 是 per-t 建、gt_ratio 的 random() gating 是每 t 對整個 B 共用一次 → 批起來會改 gate-dropout 的 RNG stream = **擾動被訓模型**(這是要拿去評估的 run,不可擾動)→ 不做。
- **結論**:~0.18s/batch = ~8min/epoch,GT1 30ep ≈ 4hr = teacher-forward 地板。流程已優化到位,直接開 §4。

### 3b. Bug fix(no-keyframe batch backward crash,commit 5ab711cc)
首跑 GT1 在 epoch1 batch330 崩:`element 0 does not require grad`。根因=**interp + clip_len=4 < PP22 關鍵幀間隔~5** → 某些 clip 4 幀全 interpolated(無關鍵幀),當整個 batch 的 B×T 位置全非關鍵幀時 keyframe-mask 把每個 t 都跳過 → `batch_loss` 停在 grad-less 常數 → backward 崩(機率~(1/5)^4≈0.16%,幾百 batch 撞一次)。**修法**=backward 前 `if not batch_loss.requires_grad: continue`(無監督訊號就跳)。已驗證同 seed 過 batch330。**driver = `run_pp22_full_cadence_chain.sh`**(commit ad9bac07,GT1→T3→T1 latest.ckpt 交接,背景 `nohup`,log `runs/pp22_full_cadence_chain.log`)。

<details><summary>原設計筆記(存查)</summary>

## (原)3. 待建:GPU-decode 訓練 pipeline(主任務)

**動機**:PP22 是 1920×1080,CPU JPEG decode 重;`--no-preload-images` 的 CPU per-step decode 會變瓶頸。用 **nvJPEG(torchvision GPU decode)**。

**已驗證**:`torchvision.io.decode_jpeg(bytes, device='cuda')` → uint8 CUDA tensor ✅。DALI pipeline 是給**影片**的(`src/saccade/media/dali_pipeline.py` 用 nvv4l2decoder),JPEG 幀用 torchvision nvJPEG 即可,不需 DALI。

### 設計(workers 讀 bytes / 主迴圈 GPU 解)
- **DataLoader workers**:只 `torchvision.io.read_file(path)` 回傳**原始 JPEG bytes**(不 decode、不碰 CUDA → num_workers>0 safe)。
- **主訓練迴圈(已有 CUDA)**:把一個 batch 的 B×T bytes 一次 `decode_jpeg(list, device='cuda')`(batched nvJPEG)→ `F.interpolate` resize 到 640 → reshape `(B,T,3,640,640)` float[0,255]。

### 實作點
1. `dataset.py`:加 `return_jpeg_bytes: bool`。`__getitem__` 在此模式下用 `read_file` 回傳 `jpeg_bytes: list[Tensor]`,不 decode、frames 空。**GT 仍在 dataset 用 seqinfo 原始尺寸 scale 到 640(與 decode 位置無關,不變)**。`collate_fn` 帶上 `jpeg_bytes`。
2. `dataset.py` 加 helper:
   ```python
   def gpu_decode_clip_batch(jpeg_bytes_batch, img_size, device) -> Tensor:
       # list[B] of list[T] of uint8 1D -> (B,T,3,img,img) float [0,255] on cuda
       from torchvision.io import decode_jpeg
       import torch.nn.functional as F
       flat = [bt for clip in jpeg_bytes_batch for bt in clip]
       dec = decode_jpeg(flat, device=device)        # batched nvJPEG; fallback: loop
       out = [F.interpolate(d.float().unsqueeze(0), (img_size,img_size),
                            mode="bilinear", align_corners=False).squeeze(0) for d in dec]
       B, T = len(jpeg_bytes_batch), len(jpeg_bytes_batch[0])
       return torch.stack(out).view(B, T, 3, img_size, img_size)
   ```
3. `train_mamba_gt.py`:加 `--gpu-decode`。set 時 loader `return_jpeg_bytes=True` + `num_workers`;迴圈把 `frames = batch["frames"].to(device,float32)/255` 換成 `frames = gpu_decode_clip_batch(batch["jpeg_bytes"], img_size, device)/255`。

### 風險 / 注意
- `decode_jpeg` 接 list 的 batched API 若該版本不支援 → fallback 逐張 decode(仍 GPU)。
- resize 用 bilinear stretch,與 dataset 的 `TF.resize([640,640], antialias=True)` 有微小 antialias 差 → 訓練無妨。
- GT scale = 640/orig(seqinfo),與 GPU stretch-resize 640×640 一致。
- gt-injection(`gate_input`)用 prev-frame GT boxes,不受 decode 影響。
- 寫個 test:bytes 模式 vs 既有 decode 模式,GT/is_keyframe 一致 + GPU decode 出的 frame shape/range 正確。

</details>

---

## 4. 全訓練流程(全 cadence + interp + live + gpu-decode)

**Distill stage(clip_len 1)不受 cadence 影響 → 直接重用** `runs/mamba_distill_pp22_augment_e30/best.ckpt`。只重跑 GT1 → T3 → T1。

共同 flags:
```
--data-root datasets/PersonPath22_full \
--yolo-weights models/yolo/yolo26s.pt \
--teacher-ckpt runs/gated_det_pp22_augment/best.ckpt \
--img-size 640 --seqs "$(paste -sd, datasets/PersonPath22/train_seqs.txt)" \
--interpolate-gt --gpu-decode --num-workers 12 \
--best-by none --save-every 1            # 存每個 epoch 供 §5 選模
# 注意:NO --cache-dir(live teacher)
```

| Stage | from | 關鍵 flags(recipe 來自既有 ckpt args) |
|---|---|---|
| **GT1** | `mamba_distill_pp22_augment_e30/best.ckpt` | `--clip-len 4 --clip-stride 8 --lr 1e-4 --lr-gate 0 --gt-ratio 0.5 --scan-stop-grad --d-state 16 --d-model 128 --epochs 30 --warmup-epochs 5 --seed 20260612 --run-dir runs/mamba_gt_pp22_aug_full_stage1` |
| **T3** | GT1 best | `--add-temporal --clip-len 3 --clip-stride 6 --lr 1e-4 --lr-gate 0 --gt-ratio 0 --scan-stop-grad --d-state 16 --d-model 128 --epochs 15 --warmup-epochs 3 --seed 42 --run-dir runs/mamba_gt_pp22_aug_full_t3` |
| **T1** | T3 best | `--clip-len 1 --clip-stride 2 --lr 1e-4 --lr-gate 0 --gt-ratio 0 --scan-stop-grad --d-state 16 --d-model 128 --epochs 15 --warmup-epochs 3 --seed 42 --run-dir runs/mamba_gt_pp22_aug_full_t3_t1` |

**ETA**:live teacher × 全 cadence,估 GT1 數小時、整鏈 ~overnight(視 GPU-decode 後瓶頸落在 teacher forward)。先跑 §3 的 smoke(3 batch)確認不 OOM、interp ACTIVE、loss 有限,再開全跑(背景)。

---

## 5. Checkpoint 選擇 — ✅ 已跑,結論 **NO-GO**(2026-07-01)

跑完全鏈(GT1 30ep→T3 15ep→T1 15ep,B=16,~6hr)+ detector-only sweep(MOT17-02+13-SDP,@0.001/@0.25,15 個 T1 epoch vs baseline):

| ckpt | @0.001 | @0.25 | Δ@0.25 |
|---|---|---|---|
| baseline (keyframe t3t1 best) | 92.75 | **75.76** | — |
| full t1_ep01 | 93.26 | 78.19 | +2.43 |
| full t1_ep02 | 92.75 | 76.64 | +0.88 |
| full t1_ep03 | 92.83 | 73.36 | −2.40 |
| full t1_ep04–07 | ~92.5 | 74.6–76.6 | ±1 |
| full t1_ep08–15 | 92.2–92.5 | 74.2–75.7 | −0.1 … −0.95 |

**PP22 自身域(held-out `mot_test_kf` 8 seqs)複驗 — NO-GO 更硬**:baseline @0.001=45.43/@0.25=34.92;full-cadence ep01 45.02/35.34(+0.42)、ep04 45.06/35.10、ep15 44.16/33.96(−0.96)。**@0.001 全 ≤ baseline**(沒改善偵測、還略降)、**@0.25 在 baseline ±1 噪聲**、ep1 微峰隨訓練跌 —— 與 MOT17 同型且更弱。**在目標域本身也打不過 baseline → 假設「cadence 對自身域有意義」被證偽,NO-GO 決定性。**

**判讀 NO-GO(MOT17)**:① **@0.001(真檢測 recall)全平 92.2–93.3、≈baseline** → full-cadence 沒改變 detector 實際找人能力。② @0.25 是**校準噪聲**:epoch 間 ±2.4 亂跳,峰值在 **ep1(最沒訓的 ckpt)**、隨訓練**往下掉**(ep15 −0.95),與「正確 cadence 越訓越好」相反。③ ep1 +2.43 落在 epoch-to-epoch 噪聲帶內 = calibration 運氣,非 cadence 能力增益。**符合早標 caveat**:deploy T=1 non-temporal,cadence 只塑形 T3 而 T3 不轉移到單幀 deploy;recall 真槓桿是 calibration(§2/§6)非 cadence。→ **interp 對 T=1 deploy recall 無利,結案**。ckpt 留 `runs/mamba_gt_pp22_aug_full_*`。

<details><summary>(原)5. Checkpoint 選擇方法論</summary>

## 5. Checkpoint 選擇(只用 MOT17,且必須 detector-only)

**方法論(已結案)**:`select_ckpt_by_recall.py` 包的是**完整 tracking pipeline(含 interpolation/relink/private-continuation 後處理)** → 後處理會補回退化的 detector、**遮住退化**,不該拿來守門。
**正解 = detector-only recall**(`scripts/eval/detector/mamba_size_binned_recall.py`:detect→NMS→match IoU0.5,無 tracker)。

掃 T1 各 epoch、aggregate `all` recall:
```
# 每個 epoch_*.ckpt:
.venv/bin/python scripts/eval/detector/mamba_size_binned_recall.py \
  --mamba-ckpt <epoch.ckpt> --teacher-ckpt runs/gated_det_pp22_augment/best.ckpt \
  --data-root datasets/MOT17 --split train \
  --sequences MOT17-02-SDP,MOT17-13-SDP --score-thresholds 0.001,0.25 \
  --output <json>
# 解析 sequences[*].thresholds["0.25"].bins["all"] 的 matched/gt → aggregate recall
```
(driver 範本在舊 scratchpad `det_select.py`,可重寫。)選 **@0.25 不退化**的 epoch(目的=避免退化,非搶分)。

**基線參考(現有 train-loss best.ckpt = augment T1 ep15)**:detector-only MOT17(02+13)@0.001 ~92.8% / @0.25 ~75.8%,14 epoch 幾乎平(無退化)。新 full-cadence 訓練後拿同一把尺比,看有沒有贏。

</details>

---

## 6. 對照組(✅ 已跑,2026-07-01)— 結論:interp 對 MOT17 transfer 先擱置

門檻下掃(現有 `mamba_gt_pp22_augment_t3_t1_e30/best.ckpt`,`--preset mamba_pyt_backbone --no-temporal --no-compile`,MOT17 全 7-SDP,降 `--new-track-thresh` + `--confirm-score-thresh`):

| nt/cs | Rcll | Prcn | IDF1 | MOTA | IDs | FP |
|---|---|---|---|---|---|---|
| **0.28/0.50** baseline | 55.4 | 81.1 | **53.6** | **42.2** | 1215 | 43k |
| 0.20/0.35 | 61.0 | 73.9 | 51.9 | 38.8 | 1980 | 73k |
| 0.12/0.25 | 63.0 | 69.7 | 50.5 | 34.8 | 2691 | 92k |
| 0.06/0.12 | 64.2 | 65.4 | 48.7 | 29.3 | 3117 | 114k |
| 0.02/0.02 | 64.2 | 65.1 | 48.5 | 28.9 | 3201 | 116k |

**判讀**:recall 門檻-gated、可回血 +8.8pp(55→64)但 **~64% 飽和**(0.06→0.02 不再漲、只灌 FP);天花板 64% 而非 detector 的 76%(那 12pp 被 tracker confirm/association 吃掉)。**回血完全不划算**:每降一格 IDF1↓、MOTA 42→29(近腰斬)、precision 81→65、FP 43k→116k(2.7×)。**baseline 0.28/0.50 就是 IDF1/MOTA 最優。** 低分帶 = 真檢測+大量 junk 混雜(domain-shift + 背景人 over-detection)。

→ **MOT17 transfer 的 recall 缺口是 calibration/domain-shift 綁的,與 interp 修的 cadence 正交。interp 全跑(§4)對 MOT17 transfer 八成無效,先擱置。**

**未決(下次開跑前先定)**:這條線真正目標是 **MOT17 transfer** 還是 **PP22 自身域**?
- transfer → 槓桿是 score 重校 / domain adaptation(混 MOT17 進訓練)+ label-policy FP,**不是 interp**。
- 自身域 → 背景人在 PP22 不算 FP,門檻/recall 帳不同,**interp 仍可能有意義** → 才值得跑 §4。

---

## 7. 快速參考

- 分支:`feat/pp22-keyframe-aware-eval-and-training` @ `996386d4`(+ 未 commit 的 train_mamba_gt / select_ckpt_by_recall 改動)
- PP22 teacher(backbone):`runs/gated_det_pp22_augment/best.ckpt`
- Distill 起點:`runs/mamba_distill_pp22_augment_e30/best.ckpt`
- 全 cadence 資料:`datasets/PersonPath22_full/train/`(107 seq / 78,608 frame)
- 既有 keyframe 模型(基線):`runs/mamba_gt_pp22_augment_t3_t1_e30/best.ckpt`(ep15,train-loss 選)+ `best_recall.ckpt`(ep12,MOT17 tracking-recall 選,**已知此選法被後處理污染**)
- Eval PP22 自身域:`--preset mamba_pyt_backbone --mamba-teacher-ckpt <pp22 teacher> --no-temporal --data-root datasets/PersonPath22 --split mot_test_kf --score-on-gt-frames`
- env 坑:`.venv/bin/python`(別用 bare python);cublas LD_LIBRARY_PATH;PP22 eval 一律 `mot_test_kf` 全 cadence + `--score-on-gt-frames`,**別用 mot_test_full 評 tracking(cadence 崩)**
- 相關記憶:`project_pp22_s_pipeline_eval`、`project_pp22_conversion_label_policy`、`project_distill_cpu_h2d_bottleneck`、`project_recall_lever_separability`

## 8. 建議順序(新對話)
0. **先定目標:MOT17 transfer 還是 PP22 自身域?**(§6 未決)。決定 interp 還跑不跑。
   - 若 **transfer**:interp 擱置,改做 score 重校 / 混 MOT17 / label-policy FP(§6 結論)。
   - 若 **自身域**:往下 ↓
1. ~~§6 門檻下掃~~ ✅ 已跑(2026-07-01,結論見 §6:transfer 無效)。
2. ~~建 §3 GPU-decode pipeline + smoke~~ ✅ 已建+驗(2026-07-01,見 §3:0.2s/batch、VRAM 1.5GB、interp ACTIVE)。**目標已定=PP22 自身域**。
3. **§4 全鏈訓練(背景,overnight)← 下一步**。GT1→T3→T1,各 stage `latest.ckpt` 交接。
4. §5 detector-only MOT17 選模 → 跟基線比。
5. 若贏:部署 + 更新記憶;若平/輸:記 NO-GO(interp 對 T=1 deploy 無利),回 score 校準主線。

> **本輪狀態(2026-07-01 收)**:eval flag + interp/keyframe-loss 已 commit;train_mamba_gt(`--no-preload-images`/`--num-workers`)+ select_ckpt(`--extra-eval-args`)+ 本文檔 已 commit;全 cadence 資料已抽(`PersonPath22_full`);§6 門掃證 interp 對 transfer 無效。**GPU-decode pipeline(§3)+ §4 全跑 = 下次,且僅在目標=自身域時才做。**
