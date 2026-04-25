# ADR Index

本目錄保存架構決策紀錄 (Architecture Decision Records)。每份 ADR 都應說明背景、決策內容與影響。

## 索引

- [ADR 002: MediaMTX Gateway](002-mediamtx-gateway.md)
- [ADR 003: Zero-Copy Pipeline](003-zero-copy-pipeline.md)
- [ADR 004: YOLO26 Perception (L1) Upgrade](004-yolo26-perception.md)
- [ADR 005: YOLO26 + SigLIP 2 整合升級](005-yolo26-siglip2-upgrade.md)
- [ADR 006: Native TensorRT YOLO](006-native-trt-yolo.md)
- [ADR 007: C++ Migration Spec](007-cpp-migration-spec.md)
- [ADR 008: Cognition Layer Definition](008-cognition-layer-definition.md)
- [ADR 009: Industrial Zero-Copy V2](009-industrial-zero-copy-v2.md)
- [ADR 011: FastTracker Reference Adaptation](011-fasttracker-reference-adaptation.md)
- [ADR 012: FastTracker Selective Adaptation](012-fasttracker-selective-adaptation.md)
- [ADR 013: GPUByteTracker 與 Saccade Heartbeat](013-gpubytetracker-saccade-heartbeat.md)
- [ADR 014: Agentic RAG — LlamaIndex 整合](014-agentic-rag-llama-index.md)

> ADR 001、010 未建立（對應決策已直接合併至後續 ADR 中）。

## 使用方式

- 技術選型變更、核心模組重構、效能架構調整，應新增或更新 ADR。
- 若舊決策已被新決策取代，應在文件內標註 `Superseded by ADR XXX`。
- ADR 一旦 `Accepted` 並落地，不應回頭修改決策內容，只可新增 ADR 說明演進。
