# Saccade Documentation

This directory contains architecture notes, ADRs, evaluation references,
experiments, runbooks, and maintenance guidance for the current Saccade codebase.

The repo has recently been simplified around a MOT17-centered evaluation
workflow, so this index now prioritizes the documents that still map cleanly to
the active code.

## Architecture

- [System Architecture](architecture.md)
- [Pipeline Flow](pipeline_flow.md)
- [API Specification](api_spec.md)

## Evaluation and Experiments

- [Experiments Index](experiments/README.md)
- [Benchmarks Index](benchmarks/README.md)
- [Eval Script Guide](../scripts/eval/README.md)

Use these together with:

- `scripts/eval/mot17.py`
- `scripts/eval/ablation_mot17.py`

## Layers and Subsystems

- [Layers Index](layers/README.md)
- [L1 Perception](layers/L1_perception.md)
- [L2 Vector Path](layers/L2_vector_path.md)
- [L3 / L4 Storage](layers/L3_L4_storage.md)
- [L5 / L6 Cognition](layers/L5_L6_cognition.md)
- [GPUByteTracker Deep Dive](layers/gpubytetracker_deep_dive.md)

## Architecture Decisions

- [ADR Index](decisions/README.md)
- [ADR 002: MediaMTX Gateway](decisions/002-mediamtx-gateway.md)
- [ADR 003: Zero-Copy Pipeline](decisions/003-zero-copy-pipeline.md)
- [ADR 004: YOLO26 Perception Upgrade](decisions/004-yolo26-perception.md)
- [ADR 005: YOLO26 + SigLIP 2 Upgrade](decisions/005-yolo26-siglip2-upgrade.md)
- [ADR 006: Native TensorRT YOLO](decisions/006-native-trt-yolo.md)
- [ADR 007: C++ Core Migration](decisions/007-cpp-migration-spec.md)
- [ADR 010: NVIDIA DALI GPU Preprocessing](decisions/010-dali-gpu-preprocessing.md)
- [ADR 013: GPUByteTracker + Saccade Heartbeat](decisions/013-gpubytetracker-saccade-heartbeat.md)
- [ADR 014: Agentic RAG with LlamaIndex](decisions/014-agentic-rag-llama-index.md)
- [ADR 015: Sinkhorn-Auction Hybrid GPU Association](decisions/015-sinkhorn-auction-hybrid-association.md)

## Progress Tracking

- [Progress Index](progress/README.md)
- [Perception Progress](progress/perception.md)
- [Storage Progress](progress/storage.md)
- [Cognition Progress](progress/cognition.md)
- [Media Progress](progress/media.md)
- [Infrastructure Progress](progress/infra.md)

These files are status snapshots, not the best source for stable user-facing
entry points. Prefer the architecture, eval, and ADR documents when you need to
understand the current intended system shape.

## Operations

- [Runbooks Index](runbooks/README.md)
- [Hot Swap Model](runbooks/hot_swap_model.md)
- [Stream Recovery](runbooks/stream_recovery.md)
- [VRAM OOM Mitigation](runbooks/vram_oom.md)

## Maintenance

- [Documentation Maintenance Rules](DOC_MAINTENANCE.md)
- [TODO](TODO.md)
- [TODO History](TODO_history.md)

## Notes

- Some documents still capture historical milestones or earlier design phases.
- If a document conflicts with current code under `src/saccade/perception/`, `scripts/eval/`,
  `src/`, or `tests/`, treat the code as the source of truth.
- The most actively maintained evaluation path is now:
  - `scripts/eval/mot17.py`
  - `scripts/eval/ablation_mot17.py`
