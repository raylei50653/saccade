# GPU Pipeline M4-a / M4-b 實驗紀錄

Date: 2026-04-29

## 目標

在 `M3.5` 完成後，主瓶頸已不再是 detector / tracker 的大塊 D2H，而是 runner 內剩餘的 host materialization 與 Python-side lifecycle / output control flow。這一輪的目標是：

1. 縮減 semantic relink 所需的 motion host materialization。
2. 把 output assembly / post-lifecycle side effects 從 frame loop 內的大段 Python 控制流收斂成更窄的 batch helper。
3. 不改變追蹤語義與實驗指標，只重構資料流與邊界。

---

## 問題定義

`M3.5` 之後，runner 的 hot path 仍有兩個明顯 CPU 面瓶頸：

- **M4-a 問題**：`relinker.update_motion_snapshots(detector.tracker.get_state_snapshots())`
  - 每幀把 tracker active state/covariance 整批 materialize 回 host。
  - 即使 relinker 只會查少量 candidate IDs，也會拉整批 snapshot。

- **M4-b 問題**：tracker result 雖然已是窄 D2H，但 runner 仍在 Python loop 內做：
  - per-track dataclass / scalar unpack
  - lifecycle merge / global-id mapping / output line 拼接
  - dynamic reid observation dict 構建
  - id stability / appearance bank side effects

這使得 CPU 雖然不再承接整幀張量，但仍承接太多逐筆流程決策。

---

## 技術方向

### M4-a：把 motion snapshot 從全量 push 改成 candidate pull

原本路徑：

`GPUByteTracker active states -> full host snapshots -> relinker.motion map`

改成：

`relinker.motion_candidate_ids(frame_id) -> tracker.get_motion_snapshots_for_track_ids(ids) -> relinker.update_motion_snapshots(...)`

設計原則：

- relinker 先決定這一幀真的需要哪些 canonical IDs 的 motion state。
- tracker 只為這些 IDs 做窄查詢，不再全量 D2H 所有 active tracks。
- 若 `semantic_mahalanobis_threshold <= 0`，整條 motion path 直接跳過。

### M4-b：把 runner Python control flow 轉成 staged batch pipeline

原本是單一大 loop 混合：

- geometry suspect gate
- id stability gate
- appearance bank update
- relink resolve
- lifecycle resolve
- global id mapping
- MOT line assembly
- output appearance side effects
- dynamic reid observations

改成分階段資料流：

1. `HostTrackResultView`
   tracker compact result 的 CPU materialization。
2. `HostTrackBatch`
   把 host-side boxes / scores / ids / det_idx / dynamic observations 固定成單一 shape。
3. `PreparedTrackCandidate`
   通過 pre-resolution gating、並完成 embedding attachment / consistency gate 的候選。
4. `RelinkedTrackCandidate`
   完成 semantic relink、但尚未做 lifecycle merge 的候選。
5. `ResolvedTrack`
   完成 relink + lifecycle merge 的結果。
6. output / side effects
   global id mapping、MOT line、output appearance bank。

---

## 已落地實作

### M4-a：motion 窄查詢

修改檔案：

- `include/tracking/tracker_gpu.hpp`
- `src/tracking/tracker_gpu.cu`
- `src/tracking/tracker_gpu_python.cpp`
- `perception/tracking/tracker_gpu.py`
- `perception/eval/relink.py`
- `perception/eval/runner.py`

具體變更：

- `GPUByteTracker` 新增 `get_motion_snapshots_for_track_ids(track_ids, stream)`。
- `get_state_snapshots()` 不再是空實作，補回正確 full snapshot path。
- `SemanticRelinker` / `PythonSemanticRelinker` 新增 `motion_candidate_ids(frame_id)`。
- runner 改成：
  - 先向 relinker 問 candidate ids。
  - 再向 tracker 拉窄版 motion snapshots。
- `mahalanobis_threshold <= 0` 時，`motion_candidate_ids()` 直接回空，整條 motion snapshot D2H 關閉。

效果：

- 從「每幀全量 active snapshot push」收斂成「candidate-id pull」。
- 在未開 Mahalanobis gate 的配置下，完全不付這條 motion 查詢成本。

### M4-b：output / lifecycle path 分段化

修改檔案：

- `perception/eval/runner.py`
- `perception/tracking/tracker_gpu.py`

具體變更：

- tracker result 不再 fan-out 成 per-track Python dataclass；改成 bulk host tensors / lists。
- MOT output line assembly 抽成 `_mot_result_line(...)`。
- dynamic reid observation 構建抽成 `_build_dynamic_reid_observations(...)`。
- 新增資料形狀：
  - `HostTrackBatch`
  - `PreparedTrackCandidate`
  - `CandidateAppearanceUpdate`
  - `RelinkedTrackCandidate`
  - `ResolvedTrack`
- 新增流程 helper：
  - `_prepare_host_track_batch(...)`
  - `_collect_stability_candidates(...)`
  - `_build_prepared_candidates(...)`
  - `_apply_consistency_gate(...)`
  - `_prepare_track_candidates(...)`
  - `_resolve_frame_tracks(...)`
  - `_emit_resolved_tracks(...)`
- `IdStabilityFilter` 新增 `accept_many(...)`。
- `TrackAppearanceBank` 新增 `update_many(...)`。
- `OutputAppearanceBank` 新增 `update_many(...)`。
- frame-end side effects 收斂成 `_finalize_frame_side_effects(...)`。

額外微優化：

- geometry suspect / bank match IoU 改成直接重用 tracker result GPU box slice，不再每 track 建新的 GPU 小 tensor。

### M4-b：identity path packed 化

在 staged pipeline 成形後，又把 relink / lifecycle 的 batch 邊界再收窄一層：

- `PythonSemanticRelinker` 新增：
  - `inject_references_many(...)`
  - `resolve_many(...)`
  - `resolve_many_packed(...)`
- C++ / pybind `SemanticRelinker` 對應新增：
  - `inject_references_many(...)`
  - `resolve_many(...)`
  - `resolve_many_packed(...)`
- `TrackletLifecycleMerger` 新增：
  - `resolve_many(...)`
  - `resolve_many_packed(...)`

runner 端對應變更：

- output appearance side-effects 改走 `OutputAppearanceBank.update_many(...)`
- semantic bank inject on death 改走 `_inject_lost_track_references(...)`
- frame-end side-effects 改集中成 `_finalize_frame_side_effects(...)`
- `_relink_prepared_candidates(...)` 優先走：
  - `resolve_many_packed(raw_ids, embeddings, boxes, scores, ...)`
  - 再退回 `resolve_many(...)`
  - 最後才退回逐筆 `resolve(...)`
- `_resolve_frame_tracks(...)` 的 lifecycle 階段也優先走：
  - `resolve_many_packed(local_ids, boxes, scores, embeddings, ...)`

---

## 最終資料流（本輪結束）

frame loop 內目前已經收斂成：

1. tracker `update_into(...)`
2. `_materialize_gpu_track_results(...)`
3. `_prepare_host_track_batch(...)`
4. `_prepare_track_candidates(...)`
   - `_collect_stability_candidates(...)`
   - `IdStabilityFilter.accept_many(...)`
   - `_build_prepared_candidates(...)`
   - `TrackAppearanceBank.update_many(...)`
   - `_apply_consistency_gate(...)`
5. `_resolve_frame_tracks(...)`
   - `_relink_prepared_candidates(...)`
   - `SemanticRelinker.resolve_many_packed(...)`
   - `TrackletLifecycleMerger.resolve_many_packed(...)`
6. `_emit_resolved_tracks(...)`
7. 後處理：
   - lifecycle prune
   - `_finalize_frame_side_effects(...)`
     - semantic bank inject on death
     - dynamic reid observe
     - primary appearance bank prune

這表示 frame loop 裡的主要 Python 熱路徑，已從「大段混合流程」轉成「可替換的 staged pipeline」。

---

## 驗證

本輪每個階段都以相同組合驗證：

- `uv run pytest tests/test_runner_materialization.py -q`
- `uv run pytest tests/test_relink_motion_candidates.py -q`
- `uv run pytest tests/test_runner_batch_helpers.py -q`
- `uv run pytest tests/test_e2e.py -q`

結果：

- unit tests 持續通過
- `tests/test_e2e.py` 3/3 通過
- 包含 `semantic` 路徑的 integration test 未出現 import/runtime regression
- 本輪結束時，batch/helper/relink 測試已擴充到 `16 passed`

---

## 結論

`M4-a / M4-b` 這一輪沒有把 semantic relink 或 lifecycle/output 真正移到 GPU，但已完成更重要的前置工作：

- CPU 資料流已從「逐筆散落邏輯」整理成明確 batch boundary。
- motion sync 已從 full snapshot push 收斂為 on-demand candidate pull。
- runner 第 2 段已可被視為獨立的 staged pipeline，而不是不可拆的 Python loop。
- identity path 已有可直接 native 化的 packed 介面草稿：
  - relinker：`resolve_many_packed(raw_ids, embeddings, boxes, scores, ...)`
  - lifecycle：`resolve_many_packed(local_ids, boxes, scores, embeddings, ...)`

這讓後續優化有清楚方向：

1. 先替換單一 stage 的內部實作，而不是再碰整段 frame loop。
2. 如果要 native 化，最合理的切點已收斂到 identity path：
   - `PreparedTrackCandidate` → `RelinkedTrackCandidate`
   - `RelinkedTrackCandidate` → `ResolvedTrack`
3. Python 端 batch / side-effect boundary 已大致收乾淨，下一步應該是單一 native identity resolve pass。

---

## 下一步

- **M4-b 狀態**：Python-side runner / identity / side-effects batching 已大致完成。
- **下一步主軸**：設計單一 native identity resolve pass，直接消費 packed candidates，輸出 resolved IDs。
- **M4-c 候選**：再評估是否還值得消除 `post_count` / result-count 之類單點回讀。
- **長線方向**：若 native identity pass 穩定，再決定是否把 `PreparedTrackCandidate` 生成再往前推，或維持目前「host prepare + native resolve」的分層。
