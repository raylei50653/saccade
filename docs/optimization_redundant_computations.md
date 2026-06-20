# 重複計算分析與修復記錄

分析日期: 2026-06-20

---

## 高影響

### #1 偵測品質指標被重複計算 3-5 次/幀 ✅ 已修復
- **檔案:** `quality.py:5-44` → `evaluator.py:2845, 3428, 3543, 4668, 4880`
- **問題:** `_compute_detection_quality_batch` 在同一幀內被呼叫多次
- **修復:** `_run_birth_config` 接受可選的 `fused_quality_factors` 參數；stage2 quality gate 計算後的品質直接傳遞給 birth config（當 stage2 未移除 boxes 時）

### #2 GPU tracker IoU/Mahalanobis 在 count_stage1 和 cost kernel 之間重算 ⚠️ 暫緩
- **檔案:** `tracker_gpu.cu:274/278` → `tracker_gpu.cu:480/484, 616/620`
- **問題:** `count_stage1_candidates_kernel` 計算 IoU/Mahalanobis 做 gate check，結果丟棄後在 cost kernel 重算
- **備註:** 需較大架構重構 (sinkhorn fusion 或 compact candidate 傳遞)，暫緩

### #3 `compute_conditional_cost_kernel` cosine similarity 算了兩次 ✅ 已修復
- **檔案:** `tracker_gpu.cu:646-652` → `tracker_gpu.cu:673-679 (multiplicative) / :721-724 (additive)`
- **問題:** 512 維 embedding dot product 計算兩次
- **修復:** 將 dot product 提前計算到 `cos_sim_pre` / `norm_sq_pre`，multiplicative 和 additive 分支直接複用

### #4 `relink_births_kernel` embedding dot product 算了兩次 ❌ 已復原
- **檔案:** `tracker_gpu.cu:1336-1346` → `tracker_gpu.cu:1365-1369`
- **問題:** Pass 1 計算 dot/norm 建立統計分佈後丟棄，Pass 2 重複一樣計算
- **復原原因:** 本想用 stack array `pass1_dot[256]` 快取，但 256×2 floats = 2048 bytes/thread 會造成 local memory spill 比重新計算還慢

---

## 中影響

### #5 mahal_sq_det 內的 det_cx/det_cy 在 vel_dir penalty 區塊被重算 ⏭️ 跳過
- **檔案:** `tracker_gpu.cu:213-214` → `tracker_gpu.cu:521-522, 558-559, 690-691, 746-747`
- **備註:** 每次僅 ~4 flops per candidate，效能影響極小，不進行修復

### #6 `merge_cross_tile_duplicates` 迴圈內每次都重算全部候選的 center + wh ✅ 已修復
- **檔案:** `detection.py:789-794`
- **修復:** while 迴圈前預先計算 `all_centers` 和 `all_wh`，迴圈內透過索引取值而非重算

### #7 `_tile_seam_mask_for_boxes` 與 `tile_seam_mask` 是兩個完全重複的函數 ✅ 已修復
- **檔案:** `detection.py:319-356` 與 `utils.py:163-202`
- **修復:** `_tile_seam_mask_for_boxes` 簡化為 wrapper，內部呼叫 `tile_seam_mask`；`tile_seam_mask` 補上遺漏的 `mamba_global_2x2` tiling
- **行為變更註記:** 補上 `mamba_global_2x2` 後，**直接**呼叫 `tile_seam_mask` 的 caller（`count_tile_seam_boxes` / tile diagnostics）對 `mamba_global_2x2` 由回傳全 0 改為回傳真實 seam mask。`merge_cross_tile_duplicates` 不受影響（它走 `_tile_seam_mask_for_boxes`→`_is_tiled_tiling`，本來就含 `mamba_global_2x2`），故僅影響診斷計數器

---

## 低影響

### #8 `fused_scores.clone()` 在 `_run_birth_config` 可合併 ✅ 已修復
- **檔案:** `evaluator.py:3515, 3560, 3590`
- **修復:** 使用 `scores_cloned` flag，第一個觸發的 gate 做 clone，後續 gate 直接在已 clone 的 tensor 上操作

### #9 `_detect_tiles` 在 static batch-1 engine 時浪費一次 batch detect ✅ 已修復
- **檔案:** `detection.py:1291-1301`
- **修復:** 先用 `_get_detector_static_batch_size` 檢查，若 batch size < n_tiles 直接走 sequential 路徑
- **dynamic 引擎修正:** `_get_detector_static_batch_size` 對 dynamic 引擎回傳 1（保守值，line 149 需要此語意），但 dynamic 引擎可一次處理整個 tile batch。故 short-circuit 加上 `not is_dynamic` 條件，避免 dynamic 引擎被誤逼成逐 tile sequential（正確但喪失 batching）

### #10 Birth quality gate / stage2 quality gate 對已過濾 boxes 重算品質 ✅ 已修復
- **檔案:** `evaluator.py:3543, 4880`
- **修復:** 隨 #1 修復，stage2 quality gate 計算的 `_s2_quality` 在 boxes 不變時直接傳遞給 `_run_birth_config`

### #11 S_inv 在 kalman update mode 2 時重複計算 ❌ 已復原
- **檔案:** `tracker_gpu.cu:159` → `tracker_gpu.cu:1172-1174`
- **復原原因:** 4x4 矩陣逆 (~50 flops) 在暫存器內完成，從 `d_s_inv_` 全域記憶體讀取延遲反而更大。且 `light_factor` 參數不一致有正確性問題
