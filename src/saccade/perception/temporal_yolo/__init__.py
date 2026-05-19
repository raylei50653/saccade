"""
temporal_yolo package — YOLO + Transformer Track Queries 混合追蹤架構。

Modules:
  model     — TemporalYOLOHybrid 模型結構（Backbone + Cross-Attention Decoder + Lifecycle）
  evaluator — MOTREvaluator，驅動逐幀推論與 Track Queries 跨幀回饋迴圈

Quick start:
    from saccade.perception.temporal_yolo import build_temporal_yolo_model, MOTREvaluator
"""

from .model import (
    TemporalYOLOConfig,
    TemporalYOLOHybrid,
    YOLOFeatureExtractor,
    TrackQueryDecoder,
    TrackQueryLifecycle,
    build_temporal_yolo_model,
    load_temporal_yolo_model,
)
from .evaluator import MOTREvaluator
from .yolo_joint import (
    JointConfig,
    YOLOFeaturePyramid,
    FPNSequenceProjection,
    TemporalYOLOJoint,
    build_temporal_yolo_joint,
)
from .yolo_conditioned import (
    TrackerGateInput,
    GaussianHeatmapRenderer,
    TrackSpatialGate,
    ConditionedConfig,
    TemporalYOLOConditioned,
    build_temporal_yolo_conditioned,
)

__all__ = [
    "TemporalYOLOConfig",
    "TemporalYOLOHybrid",
    "YOLOFeatureExtractor",
    "TrackQueryDecoder",
    "TrackQueryLifecycle",
    "build_temporal_yolo_model",
    "load_temporal_yolo_model",
    "MOTREvaluator",
    "JointConfig",
    "YOLOFeaturePyramid",
    "FPNSequenceProjection",
    "TemporalYOLOJoint",
    "build_temporal_yolo_joint",
    "TrackerGateInput",
    "GaussianHeatmapRenderer",
    "TrackSpatialGate",
    "ConditionedConfig",
    "TemporalYOLOConditioned",
    "build_temporal_yolo_conditioned",
]
