# Saccade Documentation

這個目錄的文件已整理成明確分工。若你是新進開發者，**不要從這裡隨便挑一份開始看**，先依下面入口順序進入。

---

## First Read

1. [DEVELOPMENT.md](/DEVELOPMENT.md:1)
   - 開發主入口
   - source-of-truth 順序
   - 當前主路徑
   - 目前優先問題
   - 文件更新規則

2. [architecture.md](/docs/architecture.md:1)
   - 穩定架構形狀
   - 模組責任邊界
   - 系統合約

3. [pipeline_flow.md](/docs/pipeline_flow.md:1)
   - 目前實作主路徑的資料流
   - 對應 `runner.py` 與 MOT17 evaluation path

4. [api_spec.md](/docs/api_spec.md:1)
   - Redis event / stream contract
   - Chroma metadata contract
   - FastAPI request / response shape

5. [TODO.md](/docs/TODO.md:1)
   - 當前待辦
   - 近期 ablation 結論
   - 下一輪 backlog

---

## What Each File Owns

### Stable Documents

- [architecture.md](/docs/architecture.md:1)
  - 目前穩定的系統形狀與責任邊界
- [pipeline_flow.md](/docs/pipeline_flow.md:1)
  - 目前實作主路徑的資料流
- [api_spec.md](/docs/api_spec.md:1)
  - API / event / storage 合約

### Planning / Direction

- [TODO.md](/docs/TODO.md:1)
  - 當前待辦與近期結論
- [TODO_history.md](/docs/TODO_history.md:1)
  - 已完成項、已放棄方向、歷史路線圖

### Decisions

- [decisions/README.md](/docs/decisions/README.md)
  - ADR 索引

重點 ADR：

- [ADR 013: GPUByteTracker + Saccade Heartbeat](/docs/decisions/013-gpubytetracker-saccade-heartbeat.md)
- [ADR 014: Agentic RAG with LlamaIndex](/docs/decisions/014-agentic-rag-llama-index.md)
- [ADR 015: Sinkhorn-Auction Hybrid GPU Association](/docs/decisions/015-sinkhorn-auction-hybrid-association.md)
- [ADR 016: Rerank Phase 3 - Reference Quality and False-Accept Filtering](/docs/decisions/016-rerank-phase3-reference-quality.md)

### Evaluation / Experiments

- [experiments/README.md](/docs/experiments/README.md)
- [benchmarks/README.md](/docs/benchmarks/README.md)
- [../scripts/eval/README.md](/scripts/eval/README.md:1)

主 evaluation 入口：

- `scripts/eval/mot17.py`
- `scripts/eval/ablation_mot17.py`

### Deep Dives / Legacy Context

- [layers/README.md](/docs/layers/README.md)
- [layers/gpubytetracker_deep_dive.md](/docs/layers/gpubytetracker_deep_dive.md)
- [progress/README.md](/docs/progress/README.md)

說明：

- `layers/` 適合看子系統深度背景
- `progress/` 是狀態快照，不是最高權威文件

### Operations

- [runbooks/README.md](/docs/runbooks/README.md)
- [runbooks/hot_swap_model.md](/docs/runbooks/hot_swap_model.md)
- [runbooks/stream_recovery.md](/docs/runbooks/stream_recovery.md)
- [runbooks/vram_oom.md](/docs/runbooks/vram_oom.md)

### Maintenance

- [DOC_MAINTENANCE.md](/docs/DOC_MAINTENANCE.md)
- [TESTING.md](/docs/TESTING.md:1)

---

## Reading Paths

### 如果你要開發 tracking / relink / MOT

先看：

1. [DEVELOPMENT.md](/DEVELOPMENT.md:1)
2. [architecture.md](/docs/architecture.md:1)
3. [pipeline_flow.md](/docs/pipeline_flow.md:1)
4. [TODO.md](/docs/TODO.md:1)
5. `src/saccade/perception/eval/runner.py`

### 如果你要改事件 / API / storage schema

先看：

1. [DEVELOPMENT.md](/DEVELOPMENT.md:1)
2. [api_spec.md](/docs/api_spec.md:1)
3. [architecture.md](/docs/architecture.md:1)

### 如果你要理解舊決策或歷史脈絡

先看：

1. [TODO_history.md](/docs/TODO_history.md:1)
2. [decisions/README.md](/docs/decisions/README.md)
3. [progress/README.md](/docs/progress/README.md)

---

## Notes

- 若文件與 `src/saccade/perception/`, `src/tracking/`, `scripts/eval/`, `tests/` 衝突，以目前主路徑程式碼為準。
- `TODO.md` 是當前方向，不是穩定架構規格。
- `TODO_history.md` 是歷史，不是現在待辦。
- `progress/` 與部分 `experiments/` 可能保留舊階段敘事，閱讀時請先確認是否仍對應目前主路徑。
