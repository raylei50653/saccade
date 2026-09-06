# Production frame path：代碼閱讀地圖

> 閱讀日期：2026-09-06。固定 source commit：`806c52cf8ced0836c80606559f7c38a5fcc546a3`。
> 範圍：`mot17.py --preset mamba_whole_graph --detector SDP --double-buffer`，s backbone、native extension 可用、無額外實驗環境變數。
> 本文是指定提交的閱讀快照；持續維護的資料流入口仍是 [DATAFLOW.md](../DATAFLOW.md)。沒有執行 inference、benchmark 或 profiler；複雜度與尺寸由代碼推導，不能當作 latency 或瓶頸量測。

閱讀期間工作區的 `pipeline.py`、`pool.py`、`mamba_gated_detector.py` 出現其他已暫存修改，因此後續用上述提交的獨立 source snapshot 核對。以下行號均對應該提交；本機未提交修改不在本報告範圍。可用 `git show 806c52cf:src/tracking/tracker_gpu.cu` 取得同版來源。

## 1. 先確認正在分析哪一條路徑

本 repo 的 production baseline 指 MOT17 evaluation 主路徑。`--detector SDP` 選取 MOT17 sequence variant；指定 Mamba checkpoint 後實際仍執行影像 detector，不能解讀成直接使用 SDP detection 檔。RTSP／NVDEC、workbench、C++ concurrent runner、tiled detector、ReID ablation 是不同入口。

設定先經 CLI 載入 base／preset，再由 `configure_runtime_env()` 決定 decode 與排程。不能只看 Python 類別預設或 YAML 某一列。

| 決策 | 此次命令的路徑 | 代碼證據 |
|:--|:--|:--|
| Decode | torchvision `decode_jpeg(device="cuda")`；CPU 執行檔案讀取，背景 thread 預取 | [mot17_args.py](../../scripts/eval/mot17_args.py) `configure_runtime_env():166`；[streaming.py](../../src/saccade/perception/eval/streaming.py) `TorchvisionGpuStreamer:116` |
| 替代 decode | `--no-gpu-decode` → `DALIStreamerStream` 的 CPU JPEG decoder → `.gpu()` | 同上；`DALIStreamerStream:72`。另一個 `JpgPipe(device="mixed")` 不是這個 wrapper |
| Detector | YOLO26s TensorRT backbone + PyTorch Mamba detection head；whole-detect CUDA graph | [mot17.py](../../scripts/eval/mot17.py) Mamba 建構分支；[preset](../../configs/presets/mamba_whole_graph.yaml) |
| Tracker | native GPUByteTracker + `GraphedTrackerUpdate` | [pipeline.py](../../src/saccade/perception/eval/pipeline.py) `1472`；[tracker_gpu.py](../../src/saccade/perception/tracking/tracker_gpu.py) `1740` |
| ReID | `off`；沒有 production ROI embedding extraction、bank sync 或 appearance association | preset；[config.py](../../src/saccade/perception/eval/config.py) `reid_work_enabled:1675` |
| Camera motion | C++ cuFFT phase correlation，downscale=4，獨立 GMC graph | [pipeline.py](../../src/saccade/perception/eval/pipeline.py) `_build_gmc_estimator:1778` |
| Output | GPU 幾何 bridge → compact → deferred pinned D2H → CPU MOT emit；sequence 結束後補插值 | [evaluator.py](../../src/saccade/perception/eval/evaluator.py) `1688`、`3157` |

`--double-buffer` 會設置 `SACCADE_DETECT_BARRIER=event`。不帶它的標準 CLI 會設成 `full`。`--profile-stages` 使 double-buffer 不符合 eligibility，因此不能將兩種模式的時間直接當成相同 pipeline 的量測。

## 2. 實際 call graph

```text
scripts/eval/mot17.py
  ├─ load config / preset → configure_runtime_env
  ├─ build_mamba_gated_detector
  └─ evaluator.run_eval
       ├─ EvalPipeline：per-sequence tracker / GMC / pools / output buffers
       └─ frame scheduler
            ├─ next(stream_iter)                       [CPU file read + GPU JPEG decode]
            ├─ _launch_double_buffer_detect            [side CUDA stream]
            │    └─ _run_detect
            │         ├─ RGB HWC u8 → CHW f32 / 255 → pool
            │         └─ detect_native_640 → detect_single_patch_640
            │              └─ detector.detect_raw → forward → _forward_whole_graph
            │                   ├─ stretch resize 640×640
            │                   ├─ TRTYoloBackbone.infer_graph → P3/P4/P5
            │                   ├─ MambaDetectionHead._forward_eager (T=1)
            │                   └─ _postprocess_mamba_fixed → decode + top-k
            │    └─ clone detection outputs → ready_event
            └─ _run_frame                              [main CUDA stream + CPU orchestration]
                 ├─ wait_event(ready_event)
                 ├─ _run_native_tensor_prep
                 │    └─ build_track_priors_gpu        [previous tracker state]
                 ├─ _run_nms
                 │    ├─ main NMS graph (nocopyback)
                 │    └─ wider NMS + private append + count D2H/sync
                 ├─ _run_post_nms_finalize → _run_detection_filters
                 ├─ _run_reid_and_gmc                  [ReID skipped; GMC runs]
                 ├─ _run_track → graph replay → C++ update_into → run_update_device
                 │    ├─ quality kernel (disabled params)
                 │    ├─ fused GMC control + Kalman predict + S inverse
                 │    ├─ inter-track occlusion
                 │    ├─ gate + cost + sparse candidate append
                 │    ├─ five-stage top-3 construction
                 │    ├─ S0 → S1 → S1b → S1c → S2 bid/commit
                 │    ├─ state update + Kalman correction
                 │    ├─ free slots → births → init covariance
                 │    ├─ history update → bridge propose → bridge commit
                 │    └─ compact_results
                 └─ pinned output copy → event → next-iteration CPU emit
       └─ sequence tail: interpolate_tracklets → write MOT results / metrics
```

核心入口：[evaluator.py](../../src/saccade/perception/eval/evaluator.py) `_run_frame:261`、`run_eval:1965`；[stages.py](../../src/saccade/perception/eval/stages.py) `_run_detect:1809`、`_run_track:517`；[tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) `run_update_device:3466`。

**順序上的重點：** feature extraction 是 detector 內的 backbone／Mamba 計算；沒有 detector 後額外 ReID stage。Private continuation 使用前一輪 tracker state，發生在本幀 GMC／motion prediction 之前。幾何 bridge 在 birth／Kalman update 之後、GPU output compact 之前。

## 3. 每階段的演算法、尺寸、複雜度、位置

記號：`P=H×W`；`A=80²+40²+20²=8400` detector locations；`R=300` raw top-k（未 override）；`M` 為 postprocess 列數，包含 score 被設為 -1 的列；`N` 為 active tracks，包含 lost；`Tcap=2048` track slots；`Dcap=1024` association／NMS 容量；`C=16` sparse candidates；每個 pass `K=3`；`Q=⌊H/4⌋⌊W/4⌋`。

表中 big-O 是工作量／儲存量的結構推導，GPU 平行化不代表這些操作消失；不是 wall-time 模型。

| 階段 | 演算法 | 輸入 → 輸出尺寸／layout | 計算複雜度與資料位置 |
|:--|:--|:--|:--|
| Fetch／decode | JPEG read + nvJPEG，背景 queue | compressed bytes → CUDA RGB `uint8[H,W,3]` | decode 隨 pixels／compressed stream 而變；約 `O(P)` 影像工作量。CPU I/O → decoder → GPU |
| Ingest | `permute` + `torch.div(...,out=pool)` | HWC u8 → contiguous CHW `float32[3,H,W]`，值域 `[0,1]` | `O(P)`，GPU；preprocess=none 不做 gamma／contrast |
| Detector resize | bilinear stretch | `[1,3,H,W]` → `[1,3,640,640]` | `O(640²)` 輸出採樣，GPU；不是 letterbox |
| Backbone | YOLO26s layers 0–22，TRT enqueue | P3 `[1,128,80,80]`、P4 `[1,256,40,40]`、P5 `[1,512,20,20]` | convolution 約 `Σ H_l W_l k_l² C_in C_out`；GPU。Binding buffer 為 f32；engine 內部精度未在本次反序列化查驗 |
| Mamba head | projection → reduction → 四方向 spatial scan → PixelShuffle → cls/reg conv | coarse grids `20²,10²,5²`；cls `[1,80,h_l,w_l]`，reg `[1,4,h_l,w_l]` | scan 約 `O(4 ΣL_l d_inner d_state)`，另加 projection／conv／upsample 成本；GPU |
| Box decode／top-k | distance-to-box，sigmoid/class max，top-k | 8400 locations → `float32[1,300,6]`，`xyxy,score,class` | decode/class max `O(A×80)`；top-k 成本依 backend，排序上界可用 `O(A log A)`；GPU |
| Native prep／priors | dtype cast、contiguous，GPU track prior compaction | boxes `[R,4]` f32；scores `[R]` f32；classes `[R]` i32；prior capacity `[Tcap,4]` | `O(R+Tcap)`；GPU |
| Main NMS | filter、stable score sort、IoU suppression／bitmask select | fixed padded inputs `[Dcap,...]` → kept detections + device count | pairwise 上界 `O(Dcap²)`、bitmask 約 `O(Dcap²/64)` words；GPU，實際有效 count 會跳過 padding |
| Private continuation | NMS=0.70 取額外 survivors，prior gate，再 score clamp | main survivors + 最多 50 額外框 | 第二份 NMS；append 最壞 `O(R×Tcap)`，**單一 GPU thread** 掃 candidate／prior；最後 count D2H |
| FP pruning | elementwise score／area reject，score mask | `[M]` → 同 shape，reject score=-1 | `O(M)` GPU；保留索引對齊，不 compact |
| GMC | gray/downsample/Hann → FFT cross power → inverse FFT／peak → translation | CHW 原圖 → gray `[H/4,W/4]` → f32 warp `[6]` | `O(Q log Q)` FFT + `O(Q)`；GPU，另有原圖 `O(P)` D2D staging |
| Motion prediction | 8D CV Kalman + GMC control，4×4 innovation inverse | state `[Tcap,8]`；cov `[Tcap,8,8]`；S inverse `[Tcap,4,4]` | fixed-dimensional `O(Tcap)`，GPU |
| Occlusion | 每 active track 掃所有 slots，max IoU + duration／front latch | states → coeff／TTL 各 `[Tcap]` | slot scan `O(N×Tcap)`，有效 pairs `O(N²)`；GPU |
| Gate／cost | IoU OR Mahalanobis；multiplicative penalties | dense f32 cost `[Tcap,Dcap]` + sparse candidate buffers | **graph 路徑 `O(Tcap×Dcap)`**，GPU；fused no-ReID cost kernel 本身沒有 active-track early exit |
| Stage top-k | `exp(-λcost)×aspect_factor`，五組 top-3 | 每 track 最多 16 個候選 → `[5,Tcap,3]` indices/probs | 固定 stage/thread 常數下 `O(Tcap×C)`，GPU；candidate stride=32 |
| Assignment | 各 pass shared/global atomicMax bid + separate commit | top-3 → trk→det／det→trk | 每 pass `O(Tcap×K + ceil(Tcap/32)×Dcap)`，包含每 block price 初始化；GPU |
| State／KF update／birth | match update、confirmation、free-slot prefix/scan、spawn | matches → persistent track slots | baseline `O(Tcap+Dcap)`；可選 proximity birth gate 另增 pair scan；GPU |
| Geometric bridge | history regression、雙向距離、margin、claim/commit | young candidates × live lost slots → output ID adoption | `O(B×Tcap)` slot scan，`B`=本幀達觸發條件的 candidates；GPU |
| Compact／materialize | confirmed output compaction、fixed buffer copy | boxes `[2048,4]`；4 組 scalar arrays `[2048]`；count scalar | `O(Tcap)` GPU → pinned CPU；不是每個 frame 僅傳有效 N 列 |
| Emit／sequence tail | MOT format／ID mapping；gap linear interpolation | CPU track rows → MOT text；sequence rows `U` | emit `O(Nout)`；tail 約 `O(U log U+新增列數)`，CPU |

容量證據：[tracker_gpu.py](../../src/saccade/perception/tracking/tracker_gpu.py) `GPUByteTracker:527`、`GraphedTrackerUpdate:1740`；[pipeline.py](../../src/saccade/perception/eval/pipeline.py) NMS buffers `1430`；[tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) allocation `3100`、candidate stride `3155`。

## 4. Detector 與 tensor 的具體內容

### 4.1 原圖與 layout

`H,W` 來自 sequence `seqinfo.ini`，不是全 repo 固定 1080p。torchvision decoder 先回傳 CHW u8；streamer 做 `permute(1,2,0)` 得 HWC **view，未保證 contiguous**。Ingest 以 `torch.div(frame_gpu.permute(2,0,1),255,out=pool.frame_buffer)` 寫入 contiguous f32 CHW。見 [streaming.py](../../src/saccade/perception/eval/streaming.py) `116`、[stages.py](../../src/saccade/perception/eval/stages.py) `1843`、[pool.py](../../src/saccade/perception/eval/pool.py) `63`。

`native_640 + preprocess:none` 將長寬各自伸縮至 640，再分別乘回 `W/640`、`H/640`，輸出原圖 pixel 座標。沒有 letterbox padding。NV12 是可選分支，不是本命令必要中介格式。

### 4.2 本機 checkpoint 已讀取的 metadata

檔案：`runs/mamba_gt_v14replica_t3_t1/best.ckpt`；`epoch=15`；SHA-256：

```text
c161c88e50b894d8b51cc614c46c3700370373decf05a15825bdf00ccf0e0876
```

使用 `torch.load(map_location="cpu", weights_only=True)` 讀 metadata／weight shapes；這是本機 artifact 證據，並非 Git 自動綁定的 checkpoint identity。

| 欄位 | 讀取結果 |
|:--|:--|
| `d_model / d_state / num_blocks` | `128 / 16 / 1`；MambaBlock expand=2 → d_inner=256 |
| `spatial_reduction` | 4 → P3/P4/P5 coarse lengths=400/100/25 |
| Spatial routing | `use_cross_scan=true`，原方向、雙軸 flip、水平 flip、垂直 flip 合批運算，還原後平均 |
| Upsample／hybrid | `use_pixel_shuffle=true`；`use_hybrid_head=false` |
| Temporal／detail | checkpoint 的 `use_temporal_mamba=true`，但 whole graph `_forward_eager` 未傳 T，故 T=1，不進 temporal branch；detail fusion=false |
| Input projections | weight shapes `[128,128,1,1]`、`[128,256,1,1]`、`[128,512,1,1]` |
| Head outputs | cls 最後 conv `[80,128,1,1]`；reg `[4,128,1,1]`；reg_max 預設=1，這個 artifact 不需要多-bin DFL softmax |

各尺度由 `x_proj` 與 upsampled Mamba context concat 成 256 channels 後送 cls／reg head。FPN 是 detection features，不能當作已抽出的 ReID embeddings。

證據：[mamba_gated_detector.py](../../src/saccade/perception/temporal_yolo/mamba_gated_detector.py) `TRTYoloBackbone:325`、head init `701`、`_whole_graph_fn:1134`；[mamba_head.py](../../src/saccade/perception/temporal_yolo/mamba_head.py) `MambaBlock:451`、`_cross_scan_mamba:586`、`_forward_eager:1883`、head output `2050`。

### 4.3 Whole graph 的邊界

此處 whole graph 包含 resize、TRT backbone、Mamba head、box decode／top-k／座標還原。`_postprocess_mamba_fixed_eager:242` 沒有呼叫 NMS，也沒有以 `conf_thr` 做可變長 compaction；它回傳固定 top-k。真正 threshold/filter/NMS 是後面的 native postprocess。

GPU graph 以 shape／原圖尺寸／input slot 等建 key。此提交的 pool 可 lease detector static input surface（`pool.bind_detect_input_surface`），讓 ingest 直接寫該 surface，避免額外原圖複製；無 lease 時 callable 仍可能有 input staging copy。Double-buffer 的兩個 input pools 不共用 writer。

## 5. Filtering／NMS／低分 detection

基準 NMS IoU threshold=0.50（`eval/config.py:1710,1798`），`track_person_only=false`，native main NMS 依 class 抑制，不跨 class 直接合併。前置 filter 的比較是嚴格 `score>0.05`；本 preset 不做 person class 限制、不做 tiled center-in-frame gate，person geometry prior、quality scaling 也關閉。Predicate → exclusive scan → stable scatter 保留原始順序，再排序送 NMS。見 [box_ops.hpp](../../include/tracking/box_ops.hpp) `detection_keep:193`、[pipeline.cpp](../../src/tracking/pipeline.cpp) `process_detections_main_nms_graph_nocopyback:1171`。

Private continuation 在同一批 filter 後 raw boxes 上再做 IoU=0.70 NMS，排除 main NMS 已保留者，檢查 prior IoU≥0.30；min score=0.10，最多加 50 個。GPU priors 來自上幀 active tracker states（預設 age≤2）、固定容量、空槽為零框。這些框的 score 上限是 `new_track_thresh - epsilon = 0.28 - 0.0001`，因此可延續 track、不能直接 birth。基準 `private_low_stage_only=false`，不能把這些框全部叫成 S2 low-confidence：它們通常屬於 S1b／S1c 的中分集合。

Private append 是 `blockIdx==0 && threadIdx==0` 的串行 kernel，逐 candidate 掃 prior；它在 GPU 上也可能呈現低 SM utilization，不能因此推論應搬到 CPU。見 [tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) `append_private_continuation_kernel:6265`；[pipeline.cpp](../../src/tracking/pipeline.cpp) private ceiling `742`；[stages.py](../../src/saccade/perception/eval/stages.py) `_run_nms:589`。

NMS 後 FP hard filter 預設仍 ON：`score<0.10`，或 `area>40000` 且 `score<0.40`，設 score=-1，保留 shape。**因此 baseline 雖有 `[0.05,0.10)` S2 pass，一般 detector 低分框會在此前被清掉；不能僅因 S2 kernel 有執行就宣稱有有效 low-score 候選。** 見 [detection_filters.py](../../src/saccade/perception/eval/detection_filters.py) `828`、[stages.py](../../src/saccade/perception/eval/stages.py) `2592`。

Tiling merge、extra birth gates、external FP model、ONMS prior immunity 等有獨立 opt-in，不應從 repo 中存在其函式推論 production 有執行。

## 6. Motion、cost、assignment

### 6.1 GMC 與 Kalman

GMC 使用 downscaled Hann-windowed grayscale，前後幀 R2C FFT → normalized conjugate cross-power → C2R inverse FFT → peak／3×3 subpixel refinement。輸出 `[1,0,tx,0,1,ty]`，是平移估計，並非完整 camera pose。PCR threshold 預設 5，低信心會縮小 translation；超過 downscaled 圖幅 25% 的位移拒絕。見 [gmc.cpp](../../src/tracking/gmc.cpp) `estimate_into_direct:488`、[gmc_kernel.cu](../../src/tracking/gmc_kernel.cu) warp construction `240`。

Kalman state：`[cx,cy,a=w/h,h,vx,vy,va,vh]`；8×8 covariance；measurement `[cx,cy,a,h]`。每 frame constant-velocity predict，噪聲依高度縮放；4×4 innovation inverse 供 gating 與 correction。基準 R scale=2.8、Kalman adaptation mode=0。

真正呼叫的是 fused kernel：**先將 GMC translation 加入位置，再做 CV predict**，使速度學習 residual object motion。它也處理 GMC 線性部分對速度／局部 covariance block 的變換。不能用旁邊未被此路徑呼叫的 `gmc_kernel` 代替解釋。見 [tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) `predict_gmc_sinv_fused_kernel:109`；[kalman_gpu.cuh](../../include/tracking/kalman_gpu.cuh)。

### 6.2 Gate 與 cost

先通過 `IoU>0.30 OR Mahalanobis²<9.4877`；不通過寫 cost=1。此處 Euclidean distance 不是額外加權 cost 項。Baseline 沒有 appearance cost，velocity-direction weight=0、fuse-score weight=0。

基準可寫成：

```text
penalty = 0.50 × duration_ramped_max_overlap
        + 0.50 × max(0, (predicted_foot_y - detection_bottom_y) / predicted_h)
                   [only when front TTL > 0]
        - (0.20 / 10) / (1 + abs(predicted_h - detection_h) / max(detection_h,1e-3))

cost = clamp(1 - IoU × exp(-penalty), 0, 1)
```

OAO overlap duration ramp 25 frames；front latch 條件為 max overlap≥0.45、腳點差在 0.15 reference-height 內且本 track 在前方，TTL=4。其他 additive／velocity／energy 形式存在，但不在此基準設定。Cost 尾端仍有 clamp，不是完全 unclamped energy。

可被任何 stage 選取的 cost 才 append 到 sparse list，最多讀前 16 個、儲存 stride=32；這是有容量上限的 candidate list，不是全部 gated pairs 的無損稀疏表示。Candidate count 超過 16 時不能假設仍完整。

證據：[tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) `compute_track_occlusion_kernel:319`、`stage1_cost_fused_kernel:483`、`run_update_device:3525`、IoU／Mahalanobis defaults `4942`。

### 6.3 五個 pass：集合與順序

每個 pass 僅允許仍未分配的 track／detection；後一 pass 不能搶走前一 pass 的 assignment。Top-3 在開始時一次算好，各 pass auction 再檢查 `trk_to_det`／`det_to_trk`，不因前一輪取走候選而重新做 top-k。

| 順序 | Native 名稱 | Track 集合 | Detection score 集合 | Cost 上限 |
|:--|:--|:--|:--|:--|
| 0 | S0 / Unambiguous / DDA | confirmed，包含仍 active 的 lost | `[0.45,1.1)` | 0.12；`SACCADE_ENABLE_DDA` 預設 ON |
| 1 | S1 / HiConf | 剩餘 confirmed | `[0.45,1.1)` | 0.50 |
| 2 | S1b / MidConf | 剩餘 confirmed | `[0.10,0.45)` | 0.50 |
| 3 | S1c / Tentative | tentative | `[0.10,1.1)` 正常 score 範圍 | 0.50 |
| 4 | S2 / LoConf | 剩餘 confirmed | `[0.05,0.10)` | 0.50；前述 FP filter 通常使此集合為空 |

S0 雖名為 Unambiguous，選取條件是 state／score／cost，並沒有「candidate count 必須等於 1」判斷。

`fused_sinkhorn_multistage_kernel` 實際以 `p=exp(-10×cost)×aspect_factor` 排名、取 top-3。Aspect factor 對 `w/h>0.8` 或 `<0.15` 的框降權，下限 0.5。**沒有 Sinkhorn row/column normalization 迭代，沒有 CPU Hungarian。**

每 pass 呼叫一次 `parallel_auction_shmem_kernel`，以 best-minus-second+epsilon 形成 bid，另有預設 0.1 的 height-stability bid bias（與 cost 的 0.20 是不同項）。先 shared atomicMax，再 global atomicMax，最後獨立 commit kernel 解決跨 block 結果。沒有拍賣直到收斂的迴圈；輸掉者可等下一 pass，不保證該 pass 找到全域最優 assignment。Association bid 的 tie-breaker=`n_trk-t`，同 bid 時較小 slot t 優先。

證據：[tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) `fused_sinkhorn_multistage_kernel:882`、`parallel_auction_shmem_kernel:1002`、`commit_auction_results_kernel:1089`、`run_stage:3611`。

## 7. Track lifecycle 與 occlusion recovery

### 7.1 狀態機

Native state 只有 EMPTY=0、TENTATIVE=1、CONFIRMED=2；lost 是 `active + confirmed + age>0` 的組合，不是另一個 state enum。

```text
EMPTY
  └─ unmatched detection score >= 0.28 + free slot → TENTATIVE (streak=1, velocity=0)
TENTATIVE
  ├─ consecutive match, streak>=3 且平均 score>=0.50 → CONFIRMED
  ├─ match 但 confirmation 證據不足 → TENTATIVE
  └─ 一次 unmatched → inactive / EMPTY
CONFIRMED
  ├─ match → age=0，Kalman correction → output
  └─ unmatched → 保持 active/confirmed，streak=0，age 隨 predict 增加
       ├─ 後續 association match → age=0 → output
       ├─ bridge commit 採用其 output ID → 舊 lost slot inactive
       └─ predict 時 age>=track_buffer (預設30) → inactive → slot 可再使用
```

到期判斷發生在 association 前。`bridge_ttl=120` 不會讓已經被 track_buffer=30 移除的 slot 復活；實際 bridge 候選必須仍 active。基準不啟用 predict-through-occlusion output（coast_max_age=0），所以 lost 保留於記憶體不代表本幀會輸出預測框。默认輸出匹配後 Kalman box，`SACCADE_OUTPUT_MEASUREMENT` 才切成 detection measurement。

證據：[tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) `track_state_update_post_kernel:1121`、`spawn_new_tracks_kernel:1506`、`compact_results_kernel:2730`；[lifecycle config](../../scripts/eval/config/lifecycle.py) `track_buffer:110`。

### 7.2 幾何 bridge：不是 ReID cache lookup

每個 active、age=0 track 維護 8 筆 `(cx,cy,h)` 歷史；滿時 shift-left 再 append，EMA height=`0.95×old+0.05×current`。名稱雖為 foot ring，實際 anchor 由 center／foot／adaptive 模式選取；此命令預設 adaptive。Candidate 第一次 matched streak 達 `bridge_at=4` 且至少四個歷史點時觸發一次。

Candidate 掃描仍 active、confirmed、當幀 unmatched、age 在 `[2,120]` 的 lost slots。用 lost 尾端與 candidate 起始四點回歸速度，計算雙向完整外推殘差及空間距離，除以 reference height；lost 越慢，越偏向空間距離。**代碼的外推乘數是 lost `age`，並非另行定義的缺失 frame 數 `age-bridge_at+1`。**

高度 EMA ratio=`lost/candidate` 必須落在閉區間 `[0.75,1.33]`。Direction bonus=0.8 在兩端速度方向一致且速度可信時，將距離向 cross-track residual 混合。最終距離≤0.25，best／second distance margin≥0.05 才 propose；只有一個有效候選時 second 是 sentinel。`bridge_spatial_gate=0` 關閉額外 spatial hard gate；physical max-speed gate、gap-occupancy gate 與 appearance veto 在本基準也關閉。

若多個 candidate claim 同一 lost，claim winner 由量化 detection score、再由 candidate slot 決定；**不是按最小 bridge distance 做全域 matching**。Commit 將 candidate 的 output track ID 改成 lost ID，停用舊 lost slot；它沒有將整份舊 Kalman state／covariance 複製進 candidate。

證據：[tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) `update_foot_history_kernel:1856`、`relink_bidir_propose_kernel:2063`、claim key `2594`、`relink_bidir_commit_kernel:2642`。

### 7.3 ReID／cache 與最終輸出

本 preset：embedding extraction／ROI crop／cache lookup／EMA appearance aggregation 都不執行；tracker graph 傳 embeddings pointer=0。不能因 CUDA 類別仍配置 feature buffer、或 evaluator 仍有 `reid_bank_sync` stage 名字，就推論有每幀 ReID 計算。保留的 persistent data 是 Kalman state、lifecycle counters、geometry history、GMC previous image 與各種 graph/buffer。

若未來研究 ReID，要另外從 [stages.py](../../src/saccade/perception/eval/stages.py) `_run_reid_and_gmc:2871`、[feature_extractor.py](../../src/saccade/perception/feature_extractor.py)、[tracker_gpu.py](../../src/saccade/perception/tracking/tracker_gpu.py) appearance bank 實作追讀；不同 model／budget／bank 模式不能混為此 production path 的單一 embedding shape。

CPU MOT emit 之後，sequence tail 會對長度至少 5 的 track、缺失≤35 frames 的 gap 做線性插值（min_h=0）。它使用 gap 後端觀測，因此是 sequence-level 後處理，不屬於 causal per-frame tracker；最終 MOT 檔不等於逐 frame 的原始 compact output。見 [post_merge.py](../../src/saccade/perception/eval/post_merge.py) `interpolate_tracklets:359`。

## 8. CPU↔GPU 同步與跨幀 overlap

| 邊界 | 實際動作 | 對排程的含義 |
|:--|:--|:--|
| File read／decode | CPU read bytes；decode_jpeg 到 CUDA；prefetch queue | CPU I/O、decode thread 與 compute 可排程並行；實際 hardware engine／重疊率本次未量測 |
| Decode producer → detect stream | `input_ready.record(main)`、side `wait_event`、tensor `record_stream` | GPU dependency；不是 host 等整張卡 |
| Detect → postprocess | output clones **完成 enqueue 後** record ready；main wait_event | 防止下一 replay 覆寫本幀 detection；clone 本身是 D2D traffic |
| Main NMS → private append | main NMS graph nocopyback，private append 在 graph 外 | main graph 並非完整 postprocess graph |
| Postprocess → Python count | `process_detections_split_pipeline_graphed` count D2H + `cudaStreamSynchronize` | 每 frame 真正 host-blocking 邊界，見 `pipeline.cpp:1243` |
| GMC input | copy 原圖 CHW 到固定 GMC buffer，再 replay | 是 `12HW` bytes payload 的 D2D staging；不是只傳六個 warp floats |
| GMC → tracker | device warp `[6]` copy 到 tracker graph input | 同 stream ordering，不必為讀 warp 轉 NumPy／CPU |
| Association passes | 同一 CUDA stream 上順序 bid/commit kernels | 存在 device dependencies，但沒有每 pass 的 CPU result readback |
| Tracker → output | fixed-capacity GPU arrays nonblocking copy 到 parity pinned buffers，record event | D2H 可與下一幀 GPU work 重疊 |
| CPU emit | `_flush_db_tracker_out` event.synchronize 後讀 count／arrays | 若 D2H 尚未完成，此處 host wait；不是整個 pipeline 零同步 |
| Warmup／capture | detector、NMS、GMC、tracker 各自 warmup/capture，含 synchronize | 首幀／換 shape 成本與 steady state 分開 |
| 非 double-buffer／profiling | CLI full barriers；profiling 另加多處 synchronize | 會改變 pipeline overlap，不適合直接推回正常吞吐 |

依據：[stages.py](../../src/saccade/perception/eval/stages.py) `_launch_double_buffer_detect:1699`、`_run_gmc_estimate:196`、`_flush_db_tracker_out:419`；[evaluator.py](../../src/saccade/perception/eval/evaluator.py) pinned copies `1693`、double-buffer loop `2609`。

```text
CPU scheduler:  fetch N → enqueue detect N+1 → consume N → ... → flush emit N
side stream:             detect N+1 [resize/backbone/head/top-k/clone]
main stream:             post N → GMC N → tracker N → output D2H N
                                      ↓ sequential tracker state dependency
                         post N+1 → GMC N+1 → tracker N+1
```

這是可排程的 concurrency，不是已量測的完全 overlap。Postprocess count sync、GPU 資源競爭、decoder queue 等都可能縮小重疊窗口。

## 9. 給後續 profiling 的可驗證假設

以下僅是從代碼得到的問題清單，尚無實測瓶頸排序。

| 觀察對象 | 演算法／資料流原因 | 需要再量測的內容 |
|:--|:--|:--|
| 固定 tracker cost grid | `[2048,1024]` f32 cost=8 MiB；graph 以 max_assoc 當 n_det，並非只處理活躍 `N×M` | cost kernel duration、padding 比例、L2／DRAM traffic、active count sweep |
| Occlusion／bridge | active tracks 或 candidates 掃固定 track slots | duration 對 active／lost／bridge-trigger 數的曲線 |
| Sparse candidate cap | 每 track 16、每 pass top-3 | candidate overflow、pass 後候選耗盡；效能與 assignment 行為一起確認 |
| Private append | 第二個 NMS + single-thread prior scan | main NMS／private NMS／append 各自時間；有效 prior 和 capacity 的差距 |
| 多個小 graph／kernel | detector、main NMS、GMC、tracker 是不同 graph，private tail 另跑 | CUDA API launch timeline、graph gaps、host count wait |
| 原圖 memory traffic | 1080p CHW f32 payload=24,883,200 bytes≈23.73 MiB；GMC staging 每幀搬一次 | 是否接近 memory bandwidth、是否有額外 graph staging copy；D2D 讀寫流量至少為 payload 的兩倍 |
| Output D2H | `2048×(4+1+1+1+1)×4+4=65,540` bytes／frame | memcpy overlap、flush wait、CPU formatting |
| 跨幀 throughput | detect N+1 可與 tracker N overlap，state update 仍有因果順序 | 同時記錄 end-to-end latency 與 throughput，不能用 `1000/FPS` 取代單幀 latency |
| ReID／Hungarian | 此路徑沒有這些計算 | 若 profiler 真出現，先核對 preset／入口／extension／環境，而非立即優化它 |

## 10. 驗證範圍與舊文件差異

已直接讀取 Python orchestration、preset/config、CUDA kernels、C++ facade，以及本機 checkpoint metadata／weight shapes。尺寸表中的 runtime tensor shape 是這些契約的推導；沒有執行模型 hook，也沒有量測 CUDA engine 內部 precision、GPU hardware decoder 選擇或記憶體 peak。300 FPS 級吞吐在本次未重新驗證。

閱讀既有文件時，以下敘述需以此提交代碼校正：

- [DATAFLOW.md](../DATAFLOW.md) §3.3 將 whole graph 輸入寫 HWC u8 並包含 NMS；實際 graph 輸入已是 CHW f32，fixed postprocess 僅 top-k，NMS 在後續階段。
- 該文件 §8.4 的 clone/event 文字順序與代碼不同；代碼是先 clone、再 record ready event。
- [早期演算法參考](../../report_data/pipeline_algorithms_reference.md) 有 960 letterbox、完整 Sinkhorn 迭代、ReID、較早 IoU gate 與 score formula，不宜當本 preset 的執行規格。

這份快照可供下一階段逐段對照 profiler trace；變更 preset、source commit、model artifact、native build 或排程 flags 後，應先重新核對對應路徑。
