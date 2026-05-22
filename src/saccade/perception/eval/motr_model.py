# This file is kept for backward compatibility.
# The Temporal YOLO Hybrid architecture has moved to:
#   saccade.perception.temporal_yolo
#
# Please update your imports to:
#   from saccade.perception.temporal_yolo import build_temporal_yolo_model
from saccade.perception.temporal_yolo.model import (  # noqa: F401
    TemporalYOLOConfig,
    TemporalYOLOHybrid,
    build_temporal_yolo_model,
    load_temporal_yolo_model,
)
