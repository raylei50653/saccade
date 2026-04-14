# Saccade

Continuous visual perception with vector DB memory — every detection tagged, timestamped, and queryable.

---

## Overview

Saccade is a dual-track visual reasoning system designed to run on constrained GPU hardware (12GB VRAM). It combines continuous real-time perception with event-driven semantic extraction, storing every detection and feature vector as a time-tagged memory queryable by semantic description.

Like the saccadic motion of the eye — fast scanning, then focused understanding — Saccade separates bounding box perception from deep semantic feature extraction into two asynchronous tracks that run in parallel without blocking each other.

## Architecture

**Fast track (Perception)** runs continuously at 140+ FPS, evaluating every frame using YOLO26 and TensorRT. It handles Zero-Copy hardware decoding (NVDEC) to keep the pipeline entirely on the GPU via a C++ fast path.

**Slow track (Semantic Extraction)** operates purely on GPU via an asynchronous CUDA stream. It uses `torchvision.ops.roi_align` for microsecond-level cropping and a TensorRT-optimized **SigLIP 2** model to extract high-dimensional semantic features. These are filtered for semantic drift and written to ChromaDB alongside structured metadata.

## Key Design Decisions

**Pure Vision-Vector Pipeline** — We transitioned to a pure YOLO26 + SigLIP 2 (TensorRT) architecture, reducing VRAM usage to ~1.5GB while scaling throughput massively.

**Semantic Drift Handling** — To prevent database bloat, extracted features are compared against a GPU-based hot cache using Cosine Similarity. Only features indicating a significant semantic shift are written to the vector database.

**Zero-copy GPU path** — video frames travel `NVDEC → NVMM → CUDA Tensor → TensorRT` without touching CPU memory, minimising PCIe bandwidth usage and CPU load.

**Vector-indexed memory** — every novel detection is embedded using SigLIP 2, tagged, and stored with a Unix timestamp in ChromaDB. Supported by Hybrid Search (Semantic + Metadata + Temporal filtering).

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Detection | YOLO26 (TensorRT Engine, C++ Fast Path), GPU tracking |
| Extraction | SigLIP2 (TensorRT Engine) |
| Media | MediaMTX, GStreamer (nvh264dec), FFmpeg |
| Memory | ChromaDB (vector), Redis (cache/queue) |
| Compute | TensorRT, CUDA Streams, NVDEC |
| Environment | Docker, uv |

## Getting Started

**Requirements:** NVIDIA GPU (12GB+ VRAM), Docker + NVIDIA Container Toolkit, CUDA 12.x

```bash
# Start and enter the development environment
docker-compose up -d --build
docker-compose exec perception bash

# Install Python dependencies (inside container)
uv sync

# Start the pipeline
./scripts/saccade up
```

## Project Structure

Detailed functional mapping of each directory can be found in [**docs/architecture.md**](docs/architecture.md).

```bash
saccade/
├── perception/    # Perception: YOLO TRT, Zero-Copy Cropping, SigLIP TRT, Semantic Drift
├── pipeline/      # Orchestrator: Event routing, metadata indexing & System health
├── media/         # Streaming: MediaMTX client (GStreamer Zero-Copy) & FFmpeg utilities
├── storage/       # Memory: ChromaDB (Vector + Metadata) & Redis (State)
├── infra/         # DevOps: Systemd units & MediaMTX config
├── scripts/       # CLI Tools: Service management, TRT compilation, VRAM monitor
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
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — full development guideelopment guide