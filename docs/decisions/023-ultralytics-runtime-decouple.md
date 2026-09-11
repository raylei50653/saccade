<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-11 -->
<!-- doc-module: detection -->

# ADR 023: 評估 detector 路徑與 Ultralytics 的解耦邊界

## Status

**Proposed** (2026-09-11)

這是 [issue #391](https://github.com/raylei50653/saccade/issues/391) 的可行性與邊界分析。PR #390（`10ba516e`）把專案自身程式碼改為 Apache-2.0，並刻意留下 `ultralytics` 這條 runtime 相依。本文回答「能不能解耦、解耦到哪、代價是什麼」，**不授權**換 detector、改權重、改 tracker policy、改 benchmark claim。

本文**不做** AGPL 衍生作品邊界的法律判斷。Upstream 授權立場只當作工程後果的輸入；實際散佈／SaaS 路徑請自行評估或徵詢法律意見。Upstream 口徑見 [Ultralytics licensing](https://www.ultralytics.com/license)，已在 [README.md](../../README.md) 與 [NOTICE](../../NOTICE) 陳述。

對照時點 2026-09-11，`main` = `10ba516e`，鎖定套件 `ultralytics==8.4.37`（`uv.lock`）。

---

## 1. Context

Headline 評測路徑是 `mamba_whole_graph`：YOLO26s **backbone**（TRT engine）+ 專案自己的 **Mamba detection head** + 專案自己的 tracker。Python 套件 `ultralytics` 仍是 `pyproject.toml` 的預設 runtime 依賴，而且 `MambaGatedDetector` 建構時一定 `from ultralytics import YOLO` 去載 `.pt`，推論 decode 仍 `from ultralytics.utils.tal import dist2bbox`。

這留下兩件不同的事，不應混成一件：

1. **Python 套件耦合**：預設 install 與 production 建構路徑 import AGPL 套件。這是 #391 能用工程手段收斂的。
2. **YOLO26 權重／架構血統**：upstream 的 FAQ 把 YOLO 程式、架構、訓練管線、以及用該管線產出的模型都放在 AGPL-3.0 預設下。把 `.pt` 編成 TensorRT engine **不自動**等於這條血統消失。這不是「改 import」能單獨結案的。

本文只把 (1) 收成可執行的解耦邊界，並標出 (2) 仍須商業授權或法律意見的殘餘。

---

## 2. Import inventory

分類規則：`runtime` = production / eval 推論會走到；`training` = 訓練 loss／assigner；`eval-baseline` = 外部對照、不是 Saccade tracker；`export/tooling` = 導出 engine 或診斷。`transitive` = 自己不 import，但建構 `MambaGatedDetector` / `GatedYOLODetector` 時一定會載入。

### 2.1 `src/saccade/perception/temporal_yolo/`（直接 import）

| 檔案 | 符號 | 角色 | 熱路徑？ |
|---|---|---|---|
| `yolo_gated_detector.py` | `from ultralytics import YOLO` | **runtime** 載入 `yolo26s.pt` → `nn.Module`，再把 gate 打進 layer 16/19/22 | 是。`build_mamba_gated_detector` **即使**已給 TRT backbone 也會建構 teacher |
| `mamba_gated_detector.py` | `ultralytics.utils.tal.make_anchors`, `dist2bbox` | **runtime** DFL/LTRB → xyxy decode | 是。`_postprocess_mamba` 與 `_postprocess_mamba_fixed_eager`（whole-graph 走後者） |
| `yolo_joint.py` | `from ultralytics import YOLO` | **runtime** Option C FPN wrapper | 否。Option C 已被 Option F 取代 |
| `ngla_assigner.py` | `TaskAlignedAssigner` 子類 | **training** | 否 |
| `train_config.py` | `v8DetectionLoss` | **training** | 否 |

同目錄其餘模組（`mamba_head.py`、`yolo_conditioned.py`、`teacher_head_detector.py`、dataset/loss 等）**沒有**直接 import。`teacher_head_detector.py` 是 matched-baseline 控制組，經 `build_gated_yolo_detector` **間接**載入 YOLO。

### 2.2 `scripts/eval/`（直接 import）

| 檔案 | 符號 | 角色 |
|---|---|---|
| `baselines/ultralytics_official_mot17.py` | `from ultralytics import YOLO`；`model.track(...)` | **eval-baseline**。跑 Ultralytics 官方 BoTSORT，不是 Saccade tracker |
| `detection_map.py` | `YOLO` fallback（`--backend ultralytics` / `.pt`） | **eval** 偵測 mAP 工具；TRT 路徑不需要 |
| `appearance/export_external_fp_rows.py` | 同上 | **eval** 外觀列導出；TRT 路徑不需要 |

不直接 import 套件、但名字含 ultralytics：

- `scripts/eval/ultralytics_official_mot17.py` — 相容 wrapper，轉呼叫 baseline
- `scripts/eval/compare_framework_ultralytics.py` — 只比兩個結果目錄，**零**套件 import

`scripts/eval/` 裡大量 `build_mamba_gated_detector(...)` 腳本屬 **transitive runtime**（建構時仍走 §2.1 的 YOLO loader）。

### 2.3 同 repo 其餘直接 import（超出 issue 點名範圍，但分類完整）

| 檔案 | 角色 |
|---|---|
| `scripts/train/temporal_yolo/train_mamba_gt.py` | **training** `v8DetectionLoss` / `BboxLoss` |
| `scripts/train/temporal_yolo/train_gated_detector.py` | **training** `v8DetectionLoss` |
| `scripts/train/temporal_yolo/train_gated_tp.py` | **training** `make_anchors` + `v8DetectionLoss` |
| `scripts/model/export_yolo_{pose,backbone,person,backbone_ckpt}.py`、`export_yoloe_embedding.py` | **export** 把 `.pt` 編成 TRT / 抽 embedding |
| `scripts/tools/mamba_assigner_diagnostics.py` | **tooling** assigner 診斷 |
| `scripts/benchmarks/mamba_train_prof.py`、`debug_postprocess.py` | **tooling** |
| `scratch/verify_whole_detect_graph.py` | **tooling**（非發行路徑） |
| `pyproject.toml` `dependencies` | **runtime packaging**：預設 `uv sync` 必裝 |

### 2.4 已經不依賴 Python `ultralytics` 的路徑

- `TRTYoloDetector` / `ConcurrentTRTDetector`（`detector_trt.py`）：native TensorRT，`main.py` 與部分 bench 走這條。Engine 仍由 `scripts/model/export_yolo_*.py` 從 Ultralytics `.pt` 導出。
- C++ `saccade::MambaGatedDetector`（`include/perception/mamba_gated_detector.hpp`、`src/perception/mamba_gated_detector.cpp`）：TRT backbone + TorchScript head + **自己寫的** LTRB decode，零 ultralytics 符號。
- `BaseDetector`（`include/perception/base_detector.hpp`）已經是 C++ 側的窄介面。

### 2.5 洩漏到 detector 層以外的假設

這些不是 import，但換實作時必須對齊，否則 tracker／ReID／eval 會 silently drift：

| 假設 | 落點 | 說明 |
|---|---|---|
| YOLO26s sequential layer index `p3=16, p4=19, p5=22` | `yolo_gated_detector._GATE_LAYER_IDX`；`fpn_reid.py`、多個 `scripts/eval/appearance/*` 直接 import | FPN 抽層寫死 Ultralytics 模組序 |
| FPN 通道 `(128, 256, 512)` | `mamba_head.DEFAULT_MAMBA_IN_CHANNELS`；C++ `get_fpn_dim()` | yolo26s 專用；yolo26m/l 走 Python whole-graph |
| stride `{8, 16, 32}` | Python / C++ decode | P3/P4/P5 |
| DFL / LTRB 通道排列（4 sides × `reg_max` bins） | `_dfl_decode` 註解寫明對齊 `v8DetectionLoss.bbox_decode` | `reg_max==1` 時是直接回歸（YOLO26 預設） |
| `detect_raw` → `[B, max_det, 6]` padded xyxy/conf/cls | eval `detect_fn`、tracker 前處理 | 分數=0 的 pad 列是契約，不是「無偵測就回空 tensor」 |
| `yolo.model.save` 與 `m.f` skip 連線 | `_forward_pytorch_backbone` | 重放 Ultralytics sequential graph |
| `.pt` pickle 類名 | `YOLO(path)` | `torch.load` 需要 `ultralytics.nn.tasks.DetectionModel` 才能 unpickle；不是純 state_dict |

---

## 3. Detector ↔ tracker 契約（交替實作必須對準）

Eval 真正消費的不是 `ultralytics.Results`，而是下面這條鏈。`TeacherHeadDetector` 的模組 docstring 已寫過窄介面；此處把它提升成 production 契約。

### 3.1 輸入

| 項目 | Headline（`mamba_whole_graph`） | 契約 |
|---|---|---|
| 像素 | `pool.frame_buffer` CHW `float32`，範圍 `[0, 1]`，原圖解析度 | 不是 Ultralytics letterbox 114-pad（headline `preprocess: none`） |
| 幾何 | `native_640`：whole-graph 在 detector 內 `interpolate` 到 `img_size`，再用 `set_whole_graph_img_dims` 把 box 乘回原圖 | 非 letterbox。letterbox 是 opt-in，TTA 才強制 |
| Device / stream | 與 `torch.cuda.current_stream()` 同一條；ingest→detect 有顯式 barrier（NVJPEG/DALI 與 TRT enqueue 不在 default stream） | 交替實作不得自己 `synchronize()` 打亂 double-buffer |
| Batch | production 單流 batch-1；靜態 engine 不可吃 tile batch 時 `detect_raw` 逐片 clone | `is_dynamic` / `input_shape` 是 eval 探測的屬性 |
| Gate | headline 部署 gate-free（`gate_input=None`，gt-ratio-0 lineage） | 有 TRT 時 gate 在 PyTorch 做 `feat * (1 + alpha * heatmap)`；headline 等價 identity |

### 3.2 輸出

Eval `detect_fn`（`detect_native_640`）回傳，座標已是**原圖**：

```text
boxes:  Tensor[N, 4] float32 xyxy   # 原圖像素
scores: Tensor[N]    float32
classes: Tensor[N]   (float in raw[:, 5], 隨後當 class id)
is_tiled: bool
keypoints: Optional[Tensor]         # headline 為 None
```

`detect_raw` 的 raw 契約（`TeacherHeadDetector` / `MambaGatedDetector` / `TRTYoloDetector` 共用）：

```text
Tensor[B, max_det, 6]  # x1, y1, x2, y2, conf, cls
```

- **pre-NMS**。NMS、private continuation、quality/birth gate 是 eval postprocess / tracker 的事。
- 不足 `max_det` 的列 **zero-pad**（conf=0），不是變長。
- Headline `conf_thr=0.001`（detector）與 `new_track_thresh=0.28`（tracker）是兩層門檻，換實作不得把 detector 門檻偷偷抬到 tracker 門檻。
- Optional：`extract_fpn_embeddings(boxes_xyxy) -> [N, fpn_dim]` L2-normalized。Headline ReID off，**不是**關鍵路徑，但 C++ `BaseDetector` 仍暴露它。

Tracker 消費（`GPUByteTracker.update`）：

```text
boxes[N,4] float32, scores[N] float32, classes[N] int32,
embeddings[N, D]? , gmc[2,3]?
```

換 detector **只要**守住 `detect_fn` 的 `(boxes, scores, classes)` 原圖契約，tracker 語意不變。不得把 Ultralytics `Results`、end2end NMS、或 80-class 內部 top-k 格式漏過這層。

### 3.3 Lifecycle

| 階段 | 現況 | 交替實作義務 |
|---|---|---|
| Load | `build_mamba_gated_detector(yolo_pt, teacher_ckpt, mamba_ckpt, trt_backbone_engine, ...)` | 血統 SHA256 fail-closed（`base_yolo_sha256`、`teacher_checkpoint_sha256`）。TRT 解耦後 `.pt` 仍可當 blob 做 checksum，不必 unpickle |
| Warmup | `_whole_graph_warmup` 後才 capture | 形狀／`img_size`／NMS pad 是 graph key |
| Capture | `use_whole_graph`：TRT backbone + Mamba head + postprocess 一張 CUDA graph | decode 必須 graph-safe（已有 `_precompute_anchor_grid`；`torch.arange` 不可進 replay） |
| Replay | `graphed_callables` 以 shape key 重放；`stages.py` 在跨幀前 **clone** static buffer | 不得回傳會被下一 replay 覆寫的 view 給 tracker |
| Reset | `reset_tracker()` + `StreamState.reset()`；pipeline 擁有真正的 tracker | detector 內的 `GPUByteTracker` 是歷史殘留，eval 不靠它 |
| Fallback | `SACCADE_TRT_BACKEND=auto\|cpp\|python`；無 engine 時走 PyTorch backbone | 解耦後「無 TRT → 載入 ultralytics」只能活在 extra，不能當預設 runtime 後門 |

### 3.4 錯誤行為

- 血統 SHA 不符 → `ValueError`（fail-closed）。
- `detect_raw_preprocessed` 無 whole-graph TRT → `RuntimeError`。
- C++ detector：不支援 detail fusion；非 `(128,256,512)` 通道拒絕。
- 空偵測：仍回 `[B, max_det, 6]` 全零，不是 exception。

---

## 4. 選項比較

三條技術上可信的路。都不在本 issue 實作。

### Option A — 維持現狀 + Ultralytics 商業授權

- **做什麼**：Python 依賴與 YOLO26 血統都不動；散佈／SaaS 走 [Enterprise License](https://www.ultralytics.com/license)。
- **碰到的程式面**：零。
- **效能 / capture**：零。
- **重訓 / export**：零。
- **評測**：零。
- **packaging**：維持 `ultralytics` 在預設 `dependencies`。
- **代價**：授權成本；工程耦合不變；`uv sync` 的預設環境仍是 AGPL 套件。
- **適合**：只想解散佈問題、不想動 production 建構路徑。

### Option B — 只解耦 Python runtime（建議）

生產熱路徑其實已經是 TRT backbone + 自有 Mamba head。剩下的 Python 套件耦合是建構期 loader 與約十行 decode。

具體步驟（後續 issue，不是本 ADR 的授權）：

1. **獨立實作** LTRB→xyxy（C++ 路徑已有一份：`src/perception/mamba_gated_detector.cpp` 的 `c_xy = anchor + (rb-lt)/2`）。**禁止**把 `ultralytics.utils.tal.dist2bbox` vendoring 進 Apache-2.0 tree（那會把 AGPL 片段寫進發行物，比現在的 import 更糟）。
2. TRT backbone 存在時，**不要**呼叫 `YOLO(yolo_pt_path)`。`TrackSpatialGate` 已是專案程式；從 `GatedYOLODetector` 拆出來即可。`.pt` 只做 SHA256 blob check。
3. `ultralytics` 從預設 `dependencies` 改到 optional extra（`ultralytics` / `train`）。訓練、export、official MOT baseline、teacher-head 控制組、無 TRT 的 PyTorch backbone 走 extra。
4. 加 fail-closed 測試：headline `build_mamba_gated_detector(..., trt_backbone_engine=...)` 過程中不得 `import ultralytics`。

- **碰到的程式面**：`yolo_gated_detector.py` 建構、`mamba_gated_detector.py` decode、`pyproject.toml`、一組 import 契約測試。不碰 tracker、不碰 ckpt、不碰 preset 數字。
- **效能 / capture**：decode 必須與現有 `_postprocess_mamba_fixed` bit-comparable，否則 whole-graph 與 MOTA 都會動。C++ 已證明這段算術不必靠套件。
- **重訓**：不需要。訓練繼續用 extra 裡的 `v8DetectionLoss` / TAL。
- **export**：仍用 extra 裡的 `scripts/model/export_yolo_*.py` 產 engine。Engine 檔本身不 import Python 套件。
- **評測閘**：MOT17-04-SDP smoke；golden `detect_raw` bit-compare（同一幀、同一 engine）。過了才考慮 7-seq。不得在本路徑改 claim。
- **runtime-identity / packaging 閘**：
  - decode Python bytes 若變，`implementation` 軸會動，須依 ADR 022 重出版 runtime identity（不得把舊 probe 當 equivalence）。
  - `pyproject.toml` 從預設依賴拿掉 `ultralytics` 會動 environment recipe，同樣要重出版。
  - math-model attestation 只在 source anchor 被改到時才欠；純 packaging 通常不欠。
- **明確不解決**：YOLO26 權重／架構血統（§1 第 2 點）。TRT engine 仍從 `yolo26s.pt` 編出。若問題是「整條血統」而不是「pip 套件」，Option A 仍在。

### Option C — 換 detector（YOLOX / RT-DETR / 其他寬鬆授權 runtime）

- **做什麼**：換掉 YOLO26s backbone（以及可能的 head），改用寬鬆授權的偵測器。
- **碰到的程式面**：訓練、ckpt lineage、FPN 通道、layer index、DFL 佈局、TRT engine、Mamba head 輸入、`fpn_reid`、全部 frozen 指標。
- **效能 / capture**：whole-graph 重做；延遲與 box 分佈都會變，tracker 門檻（`new_track_thresh=0.28` 等）不再有依據。
- **重訓**：必要。Mamba head 是對 YOLO26s FPN 蒸餾／GT-ft 出來的。
- **評測**：完整 7-seq + 凍結指標重釘。這就是換 detector。
- **packaging**：可去掉 `ultralytics`，但違反本 issue 的 non-goal。
- **不採**：成本與語意風險都遠大於 #391 要問的問題。若將來有人要放棄 YOLO26 血統，另開 issue，並從訓練協議重跑。

「把 YOLO26 backbone 用純 PyTorch 重寫、對進 state_dict」是 Option B 的放大版，不是 Option C。它只在「必須支援無 TRT 的 PyTorch backbone、又不准裝 ultralytics」時才需要；headline 不需要。估時是數週加上靜默精度漂移風險，列為 B 的非目標。

---

## 5. 建議

**選 Option B（只解耦 Python runtime）。不要在本線換 detector。Option A 保留給「YOLO26 血統／散佈」這條法律問題，與 B 並行、不互斥。**

理由：

1. Headline 熱路徑已經不跑 Ultralytics `predict` / `track`。套件留在 runtime 是因為建構期 `YOLO()` 與 `dist2bbox` 兩處懶惰 import。
2. C++ detector 已經示範 decode 可以完全自寫，且 eval 契約是 `detect_raw([B,N,6])`，不是 `ultralytics.Results`。
3. 換 detector（C）會改 tracker 語意與凍結指標，直接違反 #391 non-goals。
4. 商業授權（A）解散佈，不解工程耦合；B 解耦合，不解血統。兩者回答的問題不同。

### 5.1 本 ADR 授權什麼 / 不授權什麼

| 授權 | 不授權 |
|---|---|
| 寫下契約與盤點 | 改 `src/` 推論行為 |
| 開後續 issue / PR 做 B | 改權重、preset、門檻、claim |
| 把 `ultralytics` 標成「訓練／export extra」的計畫 | 把 AGPL 原始碼 vendoring 進 tree |
| | 宣稱 TRT engine 在法律上不再是 YOLO26 衍生 |

---

## 6. Follow-up issues / PRs（僅在採 B 時）

建議拆成獨立 issue，不要塞回 #391：

1. **獨立 decode**：用自有 LTRB→xyxy 取代 `ultralytics.utils.tal.dist2bbox` / `make_anchors`。對齊 C++ 實作與現有 `_precompute_anchor_grid`。Golden `detect_raw` bit-compare + MOT17-04-SDP smoke。欠 runtime-identity republication（implementation 軸）。
2. **TRT 建構跳過 `YOLO()`**：`trt_backbone_engine` 非空時不載入 Ultralytics `nn.Module`；gate 模組獨立建構；`.pt` 只做 SHA256。Import 契約測試：該路徑 `sys.modules` 不得出現 `ultralytics`。
3. **packaging**：`ultralytics` 移出預設 `dependencies`，改 `optional-dependencies.ultralytics`（訓練 / export / official baseline / teacher-head / PyTorch backbone）。更新 README／NOTICE 的「預設 install 仍含 AGPL」句。欠 environment-recipe republication。
4. **訓練仍耦合（park）**：`v8DetectionLoss` + `TaskAlignedAssigner` 子類留在 extra。若要 AGPL-free **訓練**，另開 issue；那是新的 loss/assigner 工程，不是 runtime 解耦。
5. **不要開**「換 YOLOX/RT-DETR」issue，除非明確放棄 YOLO26s + Mamba head 血統並接受全量重訓與 7-seq 重釘。

---

## 7. Consequences

- **Positive**：#391 的五條 acceptance 可以靠本文件勾完；後續 B 的 PR 有可測試的契約，而不會滑成換模型。
- **Negative**：B 落地前預設環境仍裝 AGPL 套件。B 落地後訓練／export 需要顯式 extra。YOLO26 血統問題仍在。
- **Neutral**：ADR 004 / 006（YOLO26、native TRT）仍成立；本 ADR 只收斂「Python 套件要不要出現在 production import 圖」這件事。
