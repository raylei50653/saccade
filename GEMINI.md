# GEMINI.md - Saccade Project Instructions

Welcome to the **Saccade** project. This file serves as the foundational mandate for all AI-assisted development and maintenance.

## Project Overview
**Saccade** is a dual-path video perception system designed for high-efficiency edge AI. It mimics the human visual system's "saccades" by splitting processing into two distinct loops:
- **Fast Path (Perception):** Continuous, low-latency object detection and tracking using YOLO11 (TensorRT prioritized).
- **Vector Path (Memory):** On-demand semantic feature extraction using Jina-CLIP-v2 (TensorRT) and vector storage in ChromaDB.
- **Orchestrator:** A centralized pipeline that triggers semantic indexing based on Information Entropy and Semantic Drift.

### Core Technologies
- **Language:** Python 3.12+ (Strict Type Hinting)
- **Environment:** Docker (CUDA, TensorRT, GStreamer) & `uv` for Python dependencies.
- **Inference:** `ultralytics` (YOLO26), `SigLIP2` (Vector Embedding).
- **Infrastructure:** MediaMTX (RTSP/WebRTC), Redis (Events/Real-time Tracks), ChromaDB (Vector Memory).

## Building and Running
The project uses Docker to ensure environmental reproducibility.

### Environment Setup
```bash
# Build and enter the development container
docker-compose up -d --build
docker-compose exec saccade zsh

# Install dependencies inside container (if not already synced)
uv sync
```

### Key Commands
- **Install Dependencies:** `uv sync`
- **Run Perception:** `python main.py --mode perception`
- **Run Orchestrator:** `python main.py --mode orchestrator`
- **Run Tests:** `pytest`
- **Type Checking:** `mypy .`
- **VRAM Monitor:** `./scripts/vram_monitor.sh`

## Development Conventions

### 1. Documentation-First (文檔先行)
**強制性流程**: 在進行任何代碼變更、功能開發或架構重構前，必須先在 `docs/` 目錄下提交相應的**實作規劃或決策紀錄 (ADR)**。
- 若為架構決策，請更新 `docs/decisions/`。
- 若為模組功能開發，請更新 `docs/progress/`。
- 規劃中必須明確說明：**目標**、**技術路徑**、以及如何符合 **Zero-Copy** 原則與 **L1-L5 分層架構**。

### 2. Architecture Decision Records (ADR)
All significant technical choices must be documented in `docs/decisions/` using the Background/Decision/Trade-offs format. Refer to these before proposing architectural changes.

### 2. Type Safety
- **Strict Mode:** `mypy` is configured in `strict` mode in `pyproject.toml`. All new code MUST have complete type hints.
- **No Casts:** Avoid `typing.cast` or `Any` unless absolutely necessary and documented.

### 3. Model Management
- **Large Binaries:** Weights (`.pt`, `.engine`, `.bin`) are NEVER committed to Git.
- **Documentation:** Always update `models/README.md` when adding new models or changing versions.
- **Gitignore:** Ensure `models/**/*` patterns cover any new binary formats.

### 4. Zero-Copy First
- All image processing (Decoding -> Detection -> Cropping -> Embedding) must stay in GPU VRAM. Use `torch.Tensor` and CUDA Streams to prevent CPU/GPU sync bottlenecks.

### 5. Maintenance
- **Systemd:** Service templates in `infra/systemd/` support hot-swapping and watchdog integration.
- **Progress Tracking:** Update `docs/progress/<module>.md` after completing tasks to maintain a clear roadmap.

### 6. Git Commit Conventions
- **Conventional Commits:** AI agents MUST use the Conventional Commits format (`type(scope): subject`) for all Git commits.
- **Atomic Commits:** Combine related changes but separate unrelated tasks into distinct commits.
- **Descriptive Bodies:** Always include a commit body (using bullet points) detailing the "why" and "what" for any refactoring, feature additions, or bug fixes.
- **Pre-commit Validation:** Before staging or committing ANY Python code changes, you MUST run type checks (`uv run mypy .` or `docker-compose exec saccade uv run mypy .`) and ensure they pass. Do not commit code with unresolved typing errors unless explicitly instructed.

## Directory Map
- `perception/`: YOLO11 inference, Jina-CLIP embedding, Semantic Drift filtering.
- `cognition/`: Resource management and frame selection logic.
- `media/`: MediaMTX client, FFmpeg/GStreamer Zero-Copy utilities.
- `storage/`: Vector DB (Chroma) and Redis real-time track cache.
- `pipeline/`: High-level orchestration and health monitoring.
- `configs/`: Hardware-specific profiles and model thresholds.
- `scripts/`: Devops, TensorRT building, and export tools.
