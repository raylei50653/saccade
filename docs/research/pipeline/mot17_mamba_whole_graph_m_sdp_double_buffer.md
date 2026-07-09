# MOT17 eval path map: `mamba_whole_graph_m` + SDP + double-buffer

## Scope

本文件只描述：

```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer
```

不代表所有 mamba / whole_graph / MOT17 eval 路徑。其他 preset、`--cpp-threads`、`--workbench`、非 SDP detector、CPU decode、或未開 double-buffer 的組合，可能走完全不同的 detector / barrier / graph 分支。

---

## Known caveats

- `--profile-stages` 會讓 double-buffer eligibility 失效，回到 serial。
- `cpp_threads > 0` 會切到 C++ eval path，且 whole_graph 會被關掉。
- GPU decode 與 CPU decode 可能造成 detector 分佈差異；此 runbook 預設 GPU decode。

其他常見 silently-different 分支：

- `--workbench`：double-buffer 不合格；且 `private_continuation` 與此 hot-path 不相容。
- 手動設 `SACCADE_DETECT_BARRIER=full`（或非 `event`）且仍開 `--double-buffer`：CLI 會拒絕；若只改 env 而不經 `configure_runtime_env`，`_double_buffer_eligible` 會回 `False`。
- shell 殘留的 `SACCADE_STREAM_MODE` 會被 entrypoint **清掉**，避免實驗 stream mode 污染正式 eval。
- argparse 預設 `--engine models/yolo/yolo26m_960_batch1.engine` 在此指令下**不進 detect 熱路徑**（有 `mamba_ckpt` 時使用預建 `MambaGatedDetector`）。

---

## Validation status

- 本文件描述的是已解析出的實際 runtime path（以同一 CLI 跑 `_load_config_defaults` + `build_parser` + `configure_runtime_env` 核對）。
- 若 preset YAML、engine path、env 設定或 evaluator 分支有變動，需要重新核對。
- 文件撰寫時本機對應模型 / SDP 資料 / `build/*.so` 皆存在；**不**保證後續 checkout 或 slim release 仍完整。

核對方式（不跑完整 eval）：

```bash
uv run python -c "
import sys, os
from pathlib import Path
sys.path[:0] = ['.', 'src', 'scripts/eval']
sys.argv = ['mot17.py', '--preset', 'mamba_whole_graph_m', '--detector', 'SDP', '--double-buffer']
from mot17 import _load_config_defaults
from mot17_args import build_parser, configure_runtime_env
root = Path('.').resolve()
parser = build_parser()
parser.add_argument('--no-compile', action='store_true')
parser.add_argument('--mamba-trt', action='store_true', default=None)
parser.add_argument('--no-mamba-trt', action='store_false', dest='mamba_trt')
parser.add_argument('--mamba-head-engine', default=None)
parser.set_defaults(**_load_config_defaults(root))
args = parser.parse_args()
configure_runtime_env(args)
print('preset', args.preset, 'tiling', args.tiling)
print('mamba_ckpt', args.mamba_ckpt)
print('fpn_backbone_engine', args.fpn_backbone_engine)
print('mamba_head_engine', args.mamba_head_engine)
print('env', {k: os.environ.get(k) for k in
  ['SACCADE_DOUBLE_BUFFER','SACCADE_DETECT_BARRIER','SACCADE_GPU_DECODE','SACCADE_MAIN_NMS_GRAPHED']})
"
```

---

## 1. 指令實際解析結果

| 項目 | 解析值 |
|------|--------|
| preset | `mamba_whole_graph_m` |
| detector filter | `SDP` → 自動 7 條 sequence |
| data_root / split | `datasets/MOT17` / `train` |
| output | `results/MOT17_eval` |
| detector | **MambaGatedDetector**（`engine=mamba`） |
| tiling | `native_640` |
| preprocess | `none` |
| reid | `off` |
| graphs | `use_whole_graph` + `use_cuda_graph` + `use_tracker_graph` + `main_nms_graphed` |
| double-buffer | 開 → env 強制 `event` barrier |

**自動 env（`configure_runtime_env`）：**

| Env | 值 | 意義 |
|-----|-----|------|
| `SACCADE_DOUBLE_BUFFER` | `1` | 允許 detect(N+1) ‖ track(N) |
| `SACCADE_DETECT_BARRIER` | `event` | double-buffer 必要條件 |
| `SACCADE_GPU_DECODE` | `1` | torchvision/nvJPEG GPU decode（預設） |
| `SACCADE_MAIN_NMS_GRAPHED` | `1` | preset 的 `main_nms_graphed: true` |
| `SACCADE_STREAM_MODE` | 清掉 | 避免 shell 殘留實驗 stream mode |

---

## 2. 設定載入鏈（優先序）

```text
argparse defaults
  < --config YAML（此指令沒有）
  < --module-* YAML（此指令沒有）
  < configs/presets/mamba_whole_graph_m.yaml   ← 主力
  < CLI flags（--detector SDP, --double-buffer）
```

| 角色 | 檔案 |
|------|------|
| Entry | `scripts/eval/mot17.py` |
| CLI / env | `scripts/eval/mot17_args.py` |
| 參數群組 | `scripts/eval/config/{core,detection,geometry,motion,reid,semantic,lifecycle,pipeline}.py` |
| Preset | `configs/presets/mamba_whole_graph_m.yaml` |
| Eval 本體 | `src/saccade/perception/eval/runner.py` → `evaluator.py` |
| Stage helpers | `src/saccade/perception/eval/{pipeline,stages,detection,streaming}.py` |
| Mamba detector | `src/saccade/perception/temporal_yolo/mamba_gated_detector.py` |
| Metrics | `src/saccade/perception/eval/metrics.py` + `third_party/TrackEval` |
| Native ext | `build/saccade_tracking_ext*.so` 等（`SACCADE_BUILD_PATH` 或 `build/`） |

---

## 3. 路徑總表

### 3.1 必用模型 / 權重

| 用途 | 路徑 | 誰讀 |
|------|------|------|
| Mamba head ckpt | `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt` | `build_mamba_gated_detector` |
| Teacher（gated）ckpt | `runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt` | teacher backbone 結構 / 權重 |
| YOLO26m base | `models/yolo/yolo26m.pt` | YOLO 結構 + sha 校驗 |
| FPN backbone TRT | `models/yolo/yolo26m_backbone_640_best.engine` | `TRTYoloBackbone`（whole-graph L1） |
| Mamba head TRT | `models/yolo/mamba_head_26m.engine` | `TRTMambaHead`（preset 有設 → 走 TRT head，不必再加 `--mamba-trt`） |

### 3.2 預設有、但此路徑不進熱路徑

| 路徑 | 說明 |
|------|------|
| `models/yolo/yolo26m_960_batch1.engine` | argparse 預設 `--engine`；有 `mamba_ckpt` 時 detector 已預建，不會用這支做 detect |

### 3.3 資料集（SDP 7 seq，共 5316 frames）

```text
datasets/MOT17/train/
  MOT17-{02,04,05,09,10,11,13}-SDP/
    seqinfo.ini          # 長度、解析度、fps
    img1/*.jpg           # 輸入影格
    gt/gt.txt            # metrics GT
```

| Sequence | Frames | 解析度 |
|----------|--------|--------|
| MOT17-02-SDP | 600 | 1920×1080 |
| MOT17-04-SDP | 1050 | 1920×1080 |
| MOT17-05-SDP | 837 | 640×480 |
| MOT17-09-SDP | 525 | 1920×1080 |
| MOT17-10-SDP | 654 | 1920×1080 |
| MOT17-11-SDP | 900 | 1920×1080 |
| MOT17-13-SDP | 750 | 1920×1080 |

選序邏輯：`--detector SDP` 且沒給 `--sequences` → 掃 `data_root/split/*-SDP`。

### 3.4 輸出（預設 `results/MOT17_eval/`）

| 檔案 | 意義 |
|------|------|
| `MOT17-*-SDP.txt` | MOT 格式軌跡結果 |
| `_fps_summary.txt` / `_latency_profile*.json` | 吞吐與延遲 |
| `_global_id_map.txt` | global ID map |
| （可選）`_frame_ledger_*.csv` | 需 `--profile-frame-csv` |
| （可選）`renders/*_visualized.mp4` | 需 `--visualize` |
| MLflow | `http://localhost:5000` experiment `mot17`（失敗只印 warning） |

### 3.5 執行時 native / Python 依賴

| 路徑 | 用途 |
|------|------|
| `build/`（或 `SACCADE_BUILD_PATH`） | `saccade_tracking_ext`、perception/eval so |
| `src/` | Python package root |
| `third_party/TrackEval/` | HOTA 等 |
| GPU decode | `torchvision.io.decode_jpeg(device="cuda")`（`streaming.py`） |

---

## 4. 此指令的可用 Pipeline

### 4.1 高層控制流

```text
mot17.py
  ├─ load preset YAML
  ├─ configure_runtime_env  → DOUBLE_BUFFER=1, BARRIER=event, GPU_DECODE=1, MAIN_NMS_GRAPHED=1
  ├─ auto sequences = *-SDP
  ├─ build_mamba_gated_detector(...)   # engine="mamba"
  │     TRT backbone + TRT head + whole_graph + cuda_graph
  └─ run_eval(**kwargs)                # cpp_threads=0 → Python path
        per seq:
          decode → detect → post/NMS → (ReID OFF) → GMC → track → emit
        post-seq:
          interpolate_tracklets → write *.txt
        overall:
          motmetrics + TrackEval HOTA
```

### 4.2 每幀 stage（此 preset 的 ON/OFF）

| Stage | 狀態 | 對應檔案 / 條件 |
|-------|------|-----------------|
| fetch / decode | **ON** GPU JPEG | `streaming.py`；`SACCADE_GPU_DECODE=1` |
| ingest / preprocess | **ON**，`preprocess=none` | stretch/resize 進 640；無 gamma/contrast |
| detect | **ON** whole-graph L1 | `mamba_gated_detector.py` + `detect_native_640` |
| postprocess / NMS | **ON**；main NMS 可 graphed | `stages.py` / `pipeline.py`；`SACCADE_MAIN_NMS_GRAPHED=1`；有 ONMS priors 時 fallback eager |
| private continuation | **ON** | preset；擴 NMS 候選可續軌、不可 birth |
| ReID (bank/crop/extract) | **OFF** | `reid_mode=off` |
| GMC | **ON** GPU，`downscale=4` | tracker/GMC path |
| track | **ON** tracker CUDA graph L3 | `use_tracker_graph=true` |
| relink bridge | **ON**（m 放寬 gate） | `relink_bridge_*`：px=0.4, h∈[0.6,1.7] |
| interpolate | **ON** post-seq | max_gap=35, min_len=5 |
| double-buffer overlap | **ON**（eligible） | 見 §4.4 |

### 4.3 CUDA Graph 三層（m / whole_graph）

| Layer | 內容 | 開關 |
|-------|------|------|
| L1 Whole-Detect | resize → TRT backbone → Mamba head → decode | `use_whole_graph` + `fpn_backbone_engine` |
| L2 NMS graph | main NMS nocopyback | `main_nms_graphed` / env |
| L3 Tracker graph | Kalman + association | `use_tracker_graph` |

### 4.4 Double-buffer eligibility

`src/saccade/perception/eval/pipeline.py::_double_buffer_eligible`：

1. `SACCADE_DOUBLE_BUFFER=1`
2. CUDA available
3. 非 `profile_stages`
4. 非 `workbench`
5. detect 可 frame-independent：`use_whole_graph=True` **或** `_temporal_T==0`
6. `SACCADE_DETECT_BARRIER == event`

此指令全部滿足 → detect(N+1) 與 tracker(N) 重疊。文件標稱為 bit-exact vs 串行、主要提吞吐。

---

## 5. 模組 ↔ 檔案 ↔ 此指令 knobs

| 模組 | 主要 knobs（preset） | 關鍵檔案 |
|------|----------------------|----------|
| Detection | tiling=native_640, preprocess=none, private_continuation*, max_det=300 | `scripts/eval/config/detection.py`, `mamba_gated_detector.py`, `eval/detection.py` |
| Geometry / Assoc | match_thresh=0.50, multiplicative_cost, oao_*, sinkhorn_lambda=10, stability_cost_w=0.20 | `scripts/eval/config/geometry.py`, tracker CUDA |
| Motion / GMC | gmc=true, gmc_downscale=4, kalman_r_scale=**3.5**（m 特調） | GMC + Kalman |
| ReID | **off** | 不載 ReID engine / cropper |
| Lifecycle | confirm_streak=3, confirm_score_thresh=0.50, interpolate_*, relink_bridge_* | lifecycle + post_merge |
| Scheduling | double-buffer + event barrier + GPU decode | `mot17_args.py`, `pipeline.py` |

**m 相對 s（`mamba_whole_graph`）的關鍵差異：**

- 權重 / engine 全走 **yolo26m**（P3/P4/P5 = 256/512/512）
- `kalman_r_scale: 3.5`（s 是 2.8）
- relink bridge 較鬆：`h_lo/h_hi = 0.6/1.7`, `px=0.4`
- backbone 指向 m 的 TRT：`yolo26m_backbone_640_best.engine`

---

## 6. 決策樹（此指令會選的分支）

```text
mamba_ckpt 有值？
  YES → build MambaGatedDetector
        ├─ fpn_backbone_engine 有 → TRTYoloBackbone (TRT)
        ├─ mamba_head_engine 有 → TRTMambaHead (TRT)
        ├─ use_whole_graph → L1 whole detect graph
        └─ tiling native_640 → detect_native_640
  NO  → 才會用 --engine 的 YOLO TRT

cpp_threads > 0？
  NO  → run_eval (Python)          # 本指令
  YES → run_eval_cpp（且 whole_graph 會被關掉）

double_buffer eligible？
  YES → side-stream detect 預取
  NO  → 每幀 serial

reid_mode == off？
  YES → 跳過 bank/crop/extract/async_reid
```

---

## 7. 最小可跑依賴檢查清單

```text
configs/presets/mamba_whole_graph_m.yaml
runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt
runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt
models/yolo/yolo26m.pt
models/yolo/yolo26m_backbone_640_best.engine
models/yolo/mamba_head_26m.engine
datasets/MOT17/train/MOT17-*-SDP/{img1,gt/gt.txt,seqinfo.ini}
build/saccade_tracking_ext*.so   # 與其他 perception/eval so
CUDA + 可用 GPU
```

---

## 8. 和「不用的」路徑（避免混淆）

| 容易誤以為會用 | 實際 |
|----------------|------|
| `--engine yolo26m_960_batch1.engine` | Mamba 路徑忽略 |
| `configs/mot17_baseline.yaml` | 有 preset 時不載 fallback baseline |
| `configs/modules/*.yaml` | 沒給 `--module-*` 不載 |
| C++ `EvaluatorPool` / workbench | `cpp_threads=0`、無 `--workbench` |
| SigLIP / FPN-ReID engine | `reid_mode=off` |
| DALI CPU decode | 預設 GPU decode；只有 `--no-gpu-decode` 才切 |
| multi-process | 無 `--processes` |

---

## Related docs

- [docs/PIPELINE.md](../../PIPELINE.md) — 產品主路徑概覽
- [report_data/mamba_whole_graph_pipeline_flow.md](../../../report_data/mamba_whole_graph_pipeline_flow.md) — whole_graph stage 流程
- [perf_attribution_whole_graph_m.md](perf_attribution_whole_graph_m.md) — m preset 每幀開銷歸因
- [sync_audit_20260706.md](sync_audit_20260706.md) — stream / barrier 決定論審計
- [configs/presets/mamba_whole_graph_m.yaml](../../../configs/presets/mamba_whole_graph_m.yaml) — 本文件對應的 preset 來源
