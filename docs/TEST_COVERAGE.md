# 測試覆蓋率報告

> 最後更新：2026-05-15（v10）  
> 執行指令：`uv run pytest --cov=saccade --cov-report=term-missing`

## 總覽

| 指標 | 數值 |
|------|------|
| **總覆蓋率** | **66%** |
| 總語句數 | 7,751 |
| 已覆蓋 | 5,109 |
| 未覆蓋 | 2,642 |
| 通過測試 | 647 |
| 跳過測試 | 2 |
| 執行時間 | ~18s |

## 覆蓋率分佈

### ✅ 高覆蓋率模組（≥ 80%）

| 模組 | 覆蓋率 | 未覆蓋行 |
|------|--------|----------|
| `perception/eval/config.py` | **99%** | 3 |
| `storage/redis_cache.py` | **99%** | 1 |
| `perception/eval/multi_birth.py` | **98%** | 3 |
| `perception/eval/quality.py` | **100%** | 0 |
| `perception/calibrator.py` | **96%** | 2 |
| `perception/biometric.py` | **96%** | 3 |
| `perception/eval/external_fp_rows.py` | **95%** | 6 |
| `perception/dispatcher.py` | **94%** | 7 |
| `perception/eval/lifecycle.py` | **92%** | 14 |
| `perception/eval/metrics.py` | **92%** | 14 |
| `perception/eval/gmc.py` | **90%** | 5 |
| `perception/eval/helpers.py` | **91%** | 20 |
| `perception/eval/external_fp_model.py` | **85%** | 88 |
| `perception/entropy.py` | **84%** | 13 |
| `perception/eval/post_merge.py` | **82%** | 39 |
| `perception/eval/reporting.py` | **93%** | 18 |
| `perception/eval/relink.py` | **88%** | 55 |
| `media/dali_pipeline.py` | **75%** | 20 |
| `perception/tracking/dynamic_reid.py` | **72%** | 72 |
| `perception/eval/streaming.py` | **71%** | 17 |
| `storage/chroma_store.py` | **64%** | 20 |
| `perception/tracking/tracker_gpu.py` | **62%** | 129 |
| `perception/eval/output_bank.py` | **60%** | 20 |
| `cognition/orchestrator.py` | **52%** | 61 |
| `perception/drift_handler.py` | **100%** | 0 |

### ⚠️ 中等覆蓋率模組（30% ~ 80%）

| 模組 | 覆蓋率 | 未覆蓋行 |
|------|--------|----------|
| `perception/cropper.py` | **77%** | 24 |
| `perception/eval/detection.py` | **49%** | 286 |
| `perception/eval/evaluator.py` | **40%** | 734 |
| `resource/resource_manager.py` | **42%** | 57 |
| `perception/zero_copy.py` | **43%** | 55 |
| `perception/eval/preprocess.py` | **46%** | 30 |
| `perception/feature_extractor.py` | **48%** | 84 |
| `perception/eval/runner.py` | **50%** | 6 |
| `perception/eval/scene_adapt.py` | **32%** | 41 |
| `perception/detector_trt.py` | **37%** | 122 |

### ❌ 低覆蓋率模組（< 30%）

| 模組 | 覆蓋率 | 未覆蓋行 |
|------|--------|----------|
| `perception/eval/quality.py` | **11%** | 34 |
| `media/mediamtx_client.py` | **26%** | 151 |

### 🔴 零覆蓋模組（0%）

| 模組 | 語句數 |
|------|--------|
| `api/server.py` | 49 |
| `media/ffmpeg_utils.py` | 38 |
| `media/rtsp_dali_pipeline.py` | 38 |
| `perception/embedding_dispatcher.py` | 92 |
| `perception/roi_selector.py` | 29 |
| `perception/text_encoder.py` | 20 |
| `storage/ablation_store.py` | 69 |

> 註：部分 0% 模組為 CUDA/native wrapper、需 GPU 硬體或特定環境的模組。

## 模組群組覆蓋率

| 模組群組 | 覆蓋率 | 備註 |
|----------|--------|------|
| `perception/eval/` | **64%** | 評估工具為測試重點，多數核心函數已覆蓋 |
| `perception/tracking/` | **62%** | ReID / reorder 覆蓋佳，tracker_gpu 部分未覆蓋 |
| `perception/`（不含 eval） | **52%** | biometric、calibrator、cropper、dispatcher、drift_handler、entropy 覆蓋佳 |
| `perception/eval/` | **67%** | quality 100%、reporting 93%、multi_birth 98%、config 99%；evaluator/detection 待補 |
| `media/` | **44%** | dali_pipeline 覆蓋好；ffmpeg/mediamtx 未覆蓋 |
| `storage/` | **71%** | chroma_store + redis_cache 覆蓋佳 |
| `cognition/` | **52%** | orchestrator 部分覆蓋 |
| `resource/` | **42%** | resource_manager 部分覆蓋 |
| `pipeline/` | **40%** | health 部分覆蓋 |
| `api/` | **0%** | 尚未有測試 |

## 測試檔案清單

| 測試檔案 | 測試數 | 類別 |
|----------|--------|------|
| `test_core.py` | 17 | 核心功能 |
| `test_biometric.py` | 12 | 生物辨識 |
| `test_dali_pipeline.py` | 12 | DALI 管线 |
| `test_detection.py` | 28 | 偵測（2 skip） |
| `test_e2e.py` | 3 | 端到端 |
| `test_eval_utils.py` | 30 | 評估工具 |
| `test_external_fp_model.py` | 14 | 外部 FP 模型 |
| `test_external_fp_rows.py` | 4 | 外部 FP 行 |
| `test_feature_bank.py` | 14 | 特徵庫 |
| `test_gmc.py` | 17 | GMC |
| `test_identity_resolver_parity.py` | 5 | 身份解析 Parity |
| `test_label_boosted_birth_rows.py` | 3 | 標籤增強出生行 |
| `test_last_vit_cpp_vs_python.py` | 18 | C++/Python 比對 |
| `test_metrics_map.py` | 6 | 指標映射 |
| `test_multi_birth.py` | 25 | 多重生成 |
| `test_phase_b_parity.py` | 8 | Phase B Parity |
| `test_pipeline.py` | 3 | 管線 |
| `test_post_merge.py` | 36 | 後合併 |
| `test_relink_motion_candidates.py` | 4 | Relink 運動候選 |
| `test_reorder.py` | 7 | 重排序 |
| `test_reporting.py` | 10 | 報告 |
| `test_runner_batch_helpers.py` | 7 | Runner 批次輔助 |
| `test_runner_budgeting.py` | 5 | Runner 預算 |
| `test_runner_materialization.py` | 3 | Runner 實值化 |
| `test_analyze_near_miss_*` | 20 | 近失分析 |
| `test_dispatcher.py` | 22 | 感知分派器（P2-2） |
| `test_helpers.py` | 11 | 評估工具函數（P2-2） |
| `test_detection_extra.py` | 39 | 偵測工具函數（P2-3） |
| `test_relink_extra.py` | 73 | 重連線器測試（P2-4） |
| `test_drift_handler.py` | 43 | 漂移處理器測試（P2-5） |
| `test_redis_cache.py` | 53 | Redis 快取測試（P2-6） |
| `test_calibrator.py` | 17 | INT8 校準器測試（P2-7） |
| `test_cropper.py` | 32 | 零拷貝裁切器測試（P2-8） |
| `test_quality.py` | 22 | 檢測品質評分測試（P2-9） |
| `test_reporting_extended.py` | 30 | 報告模組擴充測試（P2-10） |

## 覆蓋率成長曲線

| 日期 | 覆蓋率 | 備註 |
|------|--------|------|
| 2026-05-15（首次） | 9% | 僅 `perception/eval/` 部分模組有測試 |
| 2026-05-15（v1） | **56%** | 大量測試覆蓋核心評估模組 |
| 2026-05-15（v2） | **58%** | 新增 `test_dispatcher.py` (22 tests) + `test_helpers.py` (11 tests)，覆蓋率提升 2pp |
| 2026-05-15（v3） | **58%** | 新增 `test_detection_extra.py` (39 tests)，`detection.py` 40% → 49% |
| 2026-05-15（v4） | **59%** | 新增 `test_relink_extra.py` (73 tests)，`relink.py` 51% → 88% |
| 2026-05-15（v5） | **60%** | 新增 `test_drift_handler.py` (43 tests)，`drift_handler.py` 0% → 100% |
| 2026-05-15（v6） | **63%** | 修復 `test_redis_cache.py` 10+ NameError + 重複 decorator；新增 53 tests，`redis_cache.py` 27% → 99% |
| 2026-05-15（v7） | **64%** | 新增 `test_calibrator.py` (17 tests)，`calibrator.py` 0% → 96% |
| 2026-05-15（v8） | **64%** | 新增 `test_cropper.py` (32 tests)，`cropper.py` 23% → 77% |
| 2026-05-15（v9） | **64%** | 新增 `test_quality.py` (22 tests)，`quality.py` 11% → 100% |
| 2026-05-15（v10） | **66%** | 新增 `test_reporting_extended.py` (30 tests)，`reporting.py` 49% → 93%；`_print_stage_waterfall` 首次測試 |

## 後續優先事項

1. **評估模組補強**：
   - `perception/eval/evaluator.py`（40%，734 行未覆蓋）— P2-1 待實作
   - `perception/eval/detection.py`（49%，286 行未覆蓋）

2. **其他中等覆蓋模組**：
   - `perception/feature_extractor.py`（48%，84 行未覆蓋）
   - `perception/zero_copy.py`（43%，55 行未覆蓋）
   - `perception/eval/preprocess.py`（46%，30 行未覆蓋）
   - `perception/eval/scene_adapt.py`（32%，41 行未覆蓋）
   - `perception/eval/streaming.py`（71%，17 行未覆蓋）

3. **API 與串流模組**：
   - `api/server.py`（0%，49 行）
   - `media/mediamtx_client.py`（26%，151 行）

4. **目標**：70%+ 總覆蓋率
   - 距離目標：66% → 70% 需要再覆蓋 ~310 行
