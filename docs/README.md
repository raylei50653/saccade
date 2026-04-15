# Saccade Documentation

Welcome to the Saccade project documentation. Saccade is a dual-path video perception system designed for high-efficiency edge AI.

## 🏗 核心架構 (Core Architecture)
- [System Architecture Overview](architecture.md)
- [Pipeline Flow](pipeline_flow.md)
- [API Specification](api_spec.md)

## 🛠 實作進度 (Progress)
- [Perception (L1-L2)](progress/perception.md) - YOLO26 & SigLIP 2
- [Storage (L3-L4)](progress/storage.md) - Redis & ChromaDB
- [Cognition (L5)](progress/cognition.md) - Agentic RAG
- [Media](progress/media.md) - GStreamer & Zero-Copy
- [Infrastructure](progress/infra.md) - Docker & Systemd
- [**Optimization Breakthrough**](progress/optimization.md) - 3000 FPS Achievement

## 📜 架構決策紀錄 (ADR)
- [ADR 002: MediaMTX Gateway](decisions/002-mediamtx-gateway.md)
- [ADR 003: Zero-Copy Pipeline](decisions/003-zero-copy-pipeline.md)
- [ADR 005: YOLO26 Perception](decisions/005-yolo26-perception.md)
- [ADR 006: Native TensorRT YOLO](decisions/006-native-trt-yolo.md)
- [ADR 007: C++ Core Migration](decisions/007-cpp-migration-spec.md)
- [ADR 008: Cognition Layer](decisions/008-cognition-layer-definition.md)
- [ADR 009: Industrial Zero-Copy V2](decisions/009-industrial-zero-copy-v2.md)
- [ADR 010: NVIDIA DALI GPU Preprocessing](decisions/010-dali-gpu-preprocessing.md)

## 📊 效能基準 (Benchmarks)
- [Latency Log](benchmarks/latency_log.md)
- [Throughput](benchmarks/throughput.md)
- [VRAM Usage](benchmarks/vram_usage.md)

## 📚 操作指南 (Runbooks)
- [Hot Swap Model](runbooks/hot_swap_model.md)
- [Stream Recovery](runbooks/stream_recovery.md)
- [VRAM OOM Mitigation](runbooks/vram_oom.md)
