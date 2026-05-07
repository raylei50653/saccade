# mypy: ignore-errors
import configparser
import csv
import math
import time
from collections import OrderedDict
import dataclasses
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from typing import Any, TypedDict

from .types import (
    IdStabilityState,
    TrackletLifecycleState,
    MotRecord,
    HostTrackResultView,
    HostTrackBatch,
    PreparedTrackCandidate,
    CandidateAppearanceUpdate,
    ResolvedTrack,
    OutputTracklet,
)
from .lifecycle import (
    IdStabilityFilter,
    TrackletLifecycleMerger,
    UnionFind,
)
from .quality import (
    compute_detection_quality_batch as _compute_detection_quality_batch,
    compute_bank_quality_score as _compute_bank_quality_score,
)
from .utils import (
    parse_debug_frame_ranges as _parse_debug_frame_ranges,
    debug_frame_selected as _debug_frame_selected,
    append_stage_dump_rows as _append_stage_dump_rows,
    mot_box as _mot_box,
    box_iou_tuple as _box_iou_tuple,
    box_center as _box_center,
    shift_box as _shift_box,
    mot_result_line as _mot_result_line,
    safe_cpp_ptr as _safe_cpp_ptr,
    apply_narrow_person_score_bonus as _apply_narrow_person_score_bonus,
    tile_seam_mask as _tile_seam_mask,
    count_tile_seam_boxes as _count_tile_seam_boxes,
)
from .output_bank import OutputAppearanceBank
from .post_merge import (
    post_merge_output_tracklets,
    filter_low_quality_tracklets,
)
from .helpers import (
    materialize_gpu_track_results as _materialize_gpu_track_results,
    build_dynamic_reid_observations as _build_dynamic_reid_observations,
    prepare_host_track_batch as _prepare_host_track_batch,
    resolve_frame_tracks as _resolve_frame_tracks,
    prepare_track_candidates as _prepare_track_candidates,
    collect_stability_candidates as _collect_stability_candidates,
    build_prepared_candidates as _build_prepared_candidates,
    apply_consistency_gate as _apply_consistency_gate,
    emit_resolved_tracks as _emit_resolved_tracks,
    inject_lost_track_references as _inject_lost_track_references,
    finalize_frame_side_effects as _finalize_frame_side_effects,
    budget_reid_candidates as _budget_reid_candidates,
)

# Perception/eval modules load local extensions before any torchvision fallback.
from saccade.perception.cropper import ZeroCopyCropper
from saccade.perception.detector_trt import TRTYoloDetector
from saccade.perception.feature_extractor import TRTFeatureExtractor

from saccade.perception.eval.detection import (
    _box_iou_single,
    _box_iou_matrix,
    _box_iou_pairwise_diag,
    detect_adaptive_960_tiled,
    detect_960p_3x2_tiled,
    detect_native_960,
    filter_detections_fast,
    merge_cross_tile_duplicates_fast,
    nms_fast,
)
from saccade.perception.eval.gmc import SparseOpticalFlowGMC
from saccade.perception.eval.pool import AdaptiveFramePool
from saccade.perception.eval.preprocess import (
    GeometryScaleState,
    apply_frame_preprocess,
    geometry_mid_thresh_scale,
    parse_preprocess,
)
from saccade.perception.eval.relink import (
    IdentityResolver,
    PythonSemanticRelinker,
    SemanticRelinker,
)

try:
    from saccade_tracking_ext import (
        TrackletLifecycleMerger as _CppTrackletLifecycleMerger,
        IdentityResolver as _CppIdentityResolver,
    )

    _LIFECYCLE_CLS: type | None = _CppTrackletLifecycleMerger
except ImportError:
    _CppIdentityResolver = None
    _LIFECYCLE_CLS = None
from saccade.perception.eval.streaming import DALIStreamerStream
from saccade.perception.eval.tracking import GlobalTrackIdMapper
from saccade.perception.tracking.tracker_gpu import (
    DynamicReIDController,
    ReIDTrackObservation,
    TrackAppearanceBank,
    need_reid_frame,
)

try:
    from saccade_tracking_ext import PerceptionPipeline, PerceptionPipelineConfig
except ImportError:
    PerceptionPipeline = None
    PerceptionPipelineConfig = None


# IdStabilityState removed


# IdStabilityFilter removed


# Utility functions removed


# _apply_narrow_person_score_bonus removed


# TrackletLifecycleMerger moved to lifecycle.py

# Orphaned methods removed


# Dataclasses moved to types.py


# Functions moved to output_bank.py and helpers.py


# Frame tracking helpers moved to helpers.py and utils.py


# Internal helpers moved to helpers.py, quality.py, and utils.py


# Post-merge functions moved to post_merge.py


from .evaluator import run_eval
