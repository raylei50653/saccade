# GEMINI.md - Saccade Project Instructions

Welcome to the **Saccade** project. This file serves as the foundational mandate for all AI-assisted development and maintenance.

## Project Overview
**Saccade** is a dual-path video perception system designed for high-efficiency edge AI. It mimics the human visual system's "saccades" by splitting processing into two distinct loops:
- **Fast Path (Perception):** Continuous, low-latency object detection and tracking using YOLO26 (TensorRT prioritized).
- **Slow Path (Cognition):** On-demand, deep visual reasoning using LLMs (llama-cpp/GGUF) and VLMs (Qwen/SigLIP).
- **Orchestrator:** A centralized pipeline that triggers the slow path based on information entropy and VRAM availability.

### Core Technologies
- **Language:** Python 3.12+ (Strict Type Hinting)
- **Environment:** Nix Flakes (CUDA, TensorRT, GStreamer) & `uv` for Python dependencies.
- **Inference:** `ultralytics` (YOLO), `llama-cpp-python` (LLM/GGUF).
- **Infrastructure:** MediaMTX (RTSP/WebRTC), Redis (Events/State), ChromaDB (Vector Memory).

## Building and Running
The project uses Nix to ensure environmental reproducibility.

### Environment Setup
```bash
# Enter the development shell
nix develop

# Or use the convenience script (automatically runs uv sync)
./scripts/nix_enter.sh
```

### Key Commands
- **Install Dependencies:** `uv sync`
- **Run Perception:** `python main.py --mode perception`
- **Run Tests:** `pytest`
- **Type Checking:** `mypy .`
- **VRAM Monitor:** `./scripts/vram_monitor.sh`

## Development Conventions

### 1. Architecture Decision Records (ADR)
All significant technical choices must be documented in `docs/decisions/` using the Background/Decision/Trade-offs format. Refer to these before proposing architectural changes.

### 2. Type Safety
- **Strict Mode:** `mypy` is configured in `strict` mode in `pyproject.toml`. All new code MUST have complete type hints.
- **No Casts:** Avoid `typing.cast` or `Any` unless absolutely necessary and documented.

### 3. Model Management
- **Large Binaries:** Weights (`.pt`, `.gguf`, `.engine`, `.bin`) are NEVER committed to Git.
- **Documentation:** Always update `models/README.md` when adding new models or changing versions.
- **Gitignore:** Ensure `models/**/*` patterns cover any new binary formats.

### 4. Async First
- The `pipeline/orchestrator.py` uses `asyncio`. Ensure any blocking I/O (like model loading or heavy inference) is handled in appropriate executors to prevent stalling the fast path.

### 5. Maintenance
- **Systemd:** Service templates in `infra/systemd/` support hot-swapping and watchdog integration.
- **Progress Tracking:** Update `docs/progress/<module>.md` after completing tasks to maintain a clear roadmap.

## Directory Map
- `perception/`: YOLO inference, CV tracking, entropy calculation.
- `cognition/`: LLM/VLM engines, VRAM resource management.
- `media/`: MediaMTX client, FFmpeg/NVENC utilities.
- `storage/`: Vector DB (Chroma) and Redis cache.
- `pipeline/`: High-level orchestration and health monitoring.
- `configs/`: Hardware-specific profiles and model thresholds.
- `scripts/`: Devops and stress-testing tools.
