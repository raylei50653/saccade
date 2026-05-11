# scripts/eval

評估、調參、延遲分析的腳本目錄。

---

## 入口點

| 腳本 | 用途 |
|---|---|
| `mot17.py` | MOT17 主評估入口 |
| `dancetrack.py` | DanceTrack 跨資料集評估 |
| `sportsmot.py` | SportsMOT 跨資料集評估 |
| `ablation_mot17.py` | 分組參數 ablation runner |
| `pipeline_contribution.py` | 模組累積貢獻分析 |
| `latency_report.py` | 延遲 profile 事後分析與比較 |
| `module_benchmark.sh` | 完整實驗流程封裝 |

---

## 工作流

### 1. 標準評估

```bash
uv run python scripts/eval/mot17.py \
  --engine models/yolo/yolo26s_960_batch1.engine \
  --detector SDP
```

載入設定的優先順序（低 → 高）：
`configs/mot17_baseline.yaml` → `--module-<name>` YAML → `--preset` → CLI flags

### 2. 延遲分析

**快速 profile**（~3s，跳過 MOTMetrics）：

```bash
uv run python scripts/eval/mot17.py \
  --profile-stages --latency-only \
  --sequences MOT17-04-SDP \
  --max-frames 150 --warmup-frames 50 \
  --output runs/my_profile
```

輸出：console ASCII waterfall + `runs/my_profile/_stage_profile.json`

**事後分析**（不重跑）：

```bash
python scripts/eval/latency_report.py runs/my_profile/
python scripts/eval/latency_report.py runs/before/ --compare runs/after/
python scripts/eval/latency_report.py runs/my_profile/ --seq MOT17-04-SDP
```

**Stage 解讀**：

- `detect` — TRT 推論，通常最大瓶頸（~40-50%）
- `postprocess` — NMS + filter，第二大（~20-25%）
- `relink_write` — 含 background async 寫入，std 偏高屬正常
- `[unaccounted]` — frame_total 扣掉所有 stage 的殘差，< 1ms 正常
- `p95 >> mean` — 偶發長尾，值得調查

### 3. Ablation / 參數調整

```bash
# 分組 ablation
uv run python scripts/eval/ablation_mot17.py --category detection,geometry

# Bayesian 最佳化
uv run python scripts/eval/bayesian_optimizer.py

# 彙整 ablation 結果
uv run python scripts/eval/summarize_ablation_mot17.py results/ablation/
```

### 4. 模組貢獻分析

```bash
uv run python scripts/eval/pipeline_contribution.py --detector SDP
```

執行 `tracker_core → +gmc → +semantic → +bank → full_default` 累積切割，
每欄 Δprev 為單一模組的裸增益。輸出 `contribution_report.md/csv`。

### 5. 跨資料集驗證

MOT17 調參完成後，用作泛化 gate：

```bash
uv run python scripts/eval/sportsmot.py
uv run python scripts/eval/dancetrack.py
```

從 `configs/mot17_baseline.yaml` 出發，只在泛化確認後才引入資料集專用參數。

### 6. 完整實驗流程（module_benchmark.sh）

```bash
# 完整跑（profile → ablation → validate → contribution）
scripts/eval/module_benchmark.sh

# 單步
scripts/eval/module_benchmark.sh --mode profile --sequences MOT17-09-SDP --max-frames 80
scripts/eval/module_benchmark.sh --mode ablation --ablation-categories detection,geometry
scripts/eval/module_benchmark.sh --mode validate --engine models/yolo/yolo26s_960_batch1.engine
scripts/eval/module_benchmark.sh --mode contribution -- --match-thresh 0.78
```

Outputs: `results/module_benchmark/<timestamp>/`（含 `summary.txt`、`commands.txt`、`notes.md`）

---

## 輔助工具

| 腳本 | 用途 |
|---|---|
| `calculate_mota.py` | 對已有 .txt 結果重新計算追蹤指標 |
| `convert_mot17.py` | MOT17 格式轉換 |
| `analyze_fn.py` | FN 根因診斷（逐框追蹤） |
| `bench_yolo_batch.py` | detector engine batch 吞吐量 benchmark |
| `ablation_experiments.py` | ablation_mot17.py 使用的實驗定義集合 |
| `mot17_args.py` | 內部：mot17.py 的 argparse 委派 |

---

## 封存 / 實驗性

以下腳本為一次性實驗或外部 baseline 比較，不屬於常規工作流：

| 腳本 | 說明 |
|---|---|
| `mot17_public.py` | 用 MOT17 公開 detections 評估（det/det.txt） |
| `ultralytics_official_mot17.py` | Ultralytics 官方追蹤作為外部 baseline |
| `compare_framework_ultralytics.py` | 比較 Saccade vs Ultralytics 結果目錄 |
| `validate_last_vit_phase0.py` | LaSt-ViT Phase 0 原型驗證（已結案，No-Go） |
| `sweep_a7_quality.py` | A7 quality gate 參數掃描（已結案） |
| `coordinate_optimizer.py` | 座標最佳化實驗 |
