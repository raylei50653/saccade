from .core import CoreConfig, add_core_args
from .detection import DetectionConfig, add_detection_args
from .geometry import GeometryConfig, add_geometry_args
from .motion import MotionConfig, add_motion_args
from .reid import ReIDConfig, add_reid_args
from .semantic import SemanticConfig, add_semantic_args
from .trigger import TriggerConfig, add_trigger_args
from .lifecycle import LifecycleConfig, add_lifecycle_args
from .pipeline import PipelineConfig

__all__ = [
    "CoreConfig",
    "DetectionConfig",
    "GeometryConfig",
    "MotionConfig",
    "ReIDConfig",
    "SemanticConfig",
    "TriggerConfig",
    "LifecycleConfig",
    "PipelineConfig",
    "add_core_args",
    "add_detection_args",
    "add_geometry_args",
    "add_motion_args",
    "add_reid_args",
    "add_semantic_args",
    "add_trigger_args",
    "add_lifecycle_args",
]
