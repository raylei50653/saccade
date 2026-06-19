# Track Lifecycle — 模組 TODO

> 全局進度矩陣與 Baseline 見 [docs/TODO.md](../../TODO.md)。跨模組測試覆蓋率總任務見主 TODO 的「跨模組待辦」節。

## 待辦

- [ ] **提升測試覆蓋率（lifecycle 切片）**：`saccade/perception/eval/evaluator.py`（目前 40% → 目標 70%+）中，專注於 lifecycle 狀態轉移（Tentative / Confirmed / Lost）與超時驅逐邏輯的測試。隸屬主 TODO 跨模組測試覆蓋率任務。
  - 2026-06-19：已補 `tests/unit/eval/test_runner_batch_helpers.py` lifecycle helper slice：Tentative reset、score EMA confirmation、Lost track TTL reuse / prune、candidate preparation 與 emit path。剩餘：`evaluator.py` monolith 的 `run_eval` branch coverage 與端到端 lifecycle 分支覆蓋。
