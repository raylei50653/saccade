---
type: "query"
date: "2026-06-19T10:35:05.784800+00:00"
question: "What is the detection-to-tracking data contract through FrameCtx and EvalPipeline?"
contributor: "graphify"
source_nodes: ["EvalPipeline", "FrameCtx", "SemanticRelinkerCpp", "GPUByteTracker", "PythonSemanticRelinker"]
---

# Q: What is the detection-to-tracking data contract through FrameCtx and EvalPipeline?

## Answer

EvalPipeline (line 1212, evaluator.py) is a per-sequence state bucket with ~30 Any-typed fields including detector, cropper, extractor, perception_pipeline, _fpn_reid_dim. FrameCtx (line 2209) carries only 7 fields: raw_boxes/scores/classes, post_boxes/scores/classes, geometry_suspect_mask, num_priors. The FPN embeddings used by kalman gate/relink are NOT in FrameCtx - they flow through a separate implicit path. Three God Nodes (SemanticRelinkerCpp, GPUByteTracker, PythonSemanticRelinker) all depend on detection outputs through chains of INFERRED edges, with FrameCtx and EvalPipeline as the only bridge nodes. No typed contract exists between detection output schema and tracking input expectations.

## Source Nodes

- EvalPipeline
- FrameCtx
- SemanticRelinkerCpp
- GPUByteTracker
- PythonSemanticRelinker