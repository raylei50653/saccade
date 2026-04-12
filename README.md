# Saccade

Continuous visual perception with vector DB memory — every detection tagged, timestamped, and queryable.

---

## Overview

Saccade is a dual-track visual reasoning system designed to run on constrained GPU hardware (12GB VRAM). It combines continuous real-time perception with event-driven cognitive analysis, storing every detection as a vector-indexed, time-tagged memory queryable by semantic description.

Like the saccadic motion of the eye — fast scanning, then focused understanding — Saccade separates perception from cognition into two asynchronous tracks that run in parallel without blocking each other.

## Architecture

![架構圖](./assets/images/architecture.svg)

**Fast track** runs continuously, evaluating every frame for information entropy. Only high-value events trigger the slow track — keeping the LLM off the hot path entirely.

**Slow track** pulls keyframes from MediaMTX on demand, runs deep visual-language analysis, and writes structured detections to ChromaDB with semantic tags and timestamps.

## Key Design Decisions

**Bifurcated pipeline** — perception and cognition run as independent services. A Redis queue is the only coupling point between them. Either service can restart without interrupting the video stream.

**Dynamic compute provisioning** — llama.cpp `-c` (context) and `-ngl` (GPU offload layers) are tuned at runtime based on available VRAM, with smooth fallback to 64GB system RAM when needed.

**Zero-copy GPU path** — video frames travel `NVDEC → NVMM → CUDA Tensor` without touching CPU memory, minimising PCIe bandwidth usage.

**Stateless inference** — inference units hold no persistent state. All memory lives in ChromaDB (vector store) and Redis (event queue), so services are safe to hot-swap via Systemd without data loss.

**Vector-indexed memory** — every detection is embedded, tagged, and stored with a Unix timestamp. Query examples: *"show all people detected near the entrance after 18:00"*, *"find frames where a vehicle was stationary for more than 30 seconds"*.

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Detection | YOLO26, CV tracking |
| Cognition | Qwen-3.5 (GGUF), CLIP, SigLIP, llama.cpp |
| Media | MediaMTX, FFmpeg (NVDEC/NVENC), GStreamer |
| Memory | ChromaDB (vector), Redis (cache/queue) |
| Compute | TensorRT, PagedAttention, NVML |
| Environment | Nix Flakes, uv |

## Getting Started

**Requirements:** NVIDIA GPU (12GB+ VRAM), Nix with flakes enabled, CUDA 12.x

```bash
# Enter the development environment (pins CUDA, GStreamer, system deps)
nix develop

# Install Python dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env

# Start the pipeline
python main.py
```

**Stream pressure test:**
```bash
ffmpeg -re -i source.mp4 -c copy -f rtsp rtsp://localhost:8554/stream
```

## Project Structure

Detailed functional mapping of each directory can be found in [**docs/architecture.md**](docs/architecture.md).

```bash
saccade/
├── perception/    # Fast track: YOLO detection & Entropy evaluation
├── cognition/     # Slow track: VLM analysis & VRAM resource management
├── pipeline/      # Orchestrator: Event routing & System health
├── media/         # Streaming: MediaMTX client & FFmpeg utilities
├── storage/       # Memory: ChromaDB (Vector) & Redis (State)
├── infra/         # DevOps: Systemd units & MediaMTX config
├── configs/       # Settings: LLM profiles & model thresholds
├── scripts/       # CLI Tools: Service management & VRAM monitor
└── tests/         # Quality: Unit tests & Performance benchmarks
```

## Development Conventions

- **Performance first** — all new Python operators should prefer a CUDA/TensorRT implementation where one exists.
- **Stateless inference** — inference units must be restartable at any time; state belongs in ChromaDB or Redis.
- **Type safety** — strict type hinting enforced via mypy in CI.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system design and ADRs
- [`docs/progress/`](docs/progress/) — per-module development status
- [`docs/runbooks/`](docs/runbooks/) — operational procedures (hot swap, stream recovery, OOM handling)
- [`docs/benchmarks/`](docs/benchmarks/) — latency, VRAM, and throughput measurements
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — full development guide