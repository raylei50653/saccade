---
type: "query"
date: "2026-06-19T10:33:13.146631+00:00"
question: "How does SemanticRelinkerCpp depend on TRTEngine or detection outputs for kalman gate computation?"
contributor: "graphify"
source_nodes: ["SemanticRelinkerCpp", "TRTEngine", "FeatureExtractor", "MambaGatedDetector", "EvalPipeline", "FrameCtx"]
---

# Q: How does SemanticRelinkerCpp depend on TRTEngine or detection outputs for kalman gate computation?

## Answer

SemanticRelinkerCpp has no direct EXTRACTED edge to TRTEngine - only one INFERRED conceptually_related_to edge. The real dependency is indirect: TRTEngine.infer() → FeatureExtractor → MambaGatedDetector → detection boxes → SequenceRunner/EvalPipeline → FrameCtx → SemanticRelinkerCpp. Core methods like kalman_gate_dist_2d, evaluate_candidate_gates, build_gate_table all consume detection outputs that originate from TRTEngine. This hidden dependency chain means any change to TRTEngine output format (confidence, embedding dims) silently affects kalman gate correctness. The recent r_scale/calman gate fix sits exactly at the end of this chain.

## Source Nodes

- SemanticRelinkerCpp
- TRTEngine
- FeatureExtractor
- MambaGatedDetector
- EvalPipeline
- FrameCtx