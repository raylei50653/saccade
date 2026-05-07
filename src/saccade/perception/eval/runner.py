# mypy: ignore-errors

from saccade.perception.eval.evaluator import run_eval as run_eval  # noqa: F401

# Perception/eval modules load local extensions before any torchvision fallback.


try:
    from saccade_tracking_ext import (
        TrackletLifecycleMerger as _CppTrackletLifecycleMerger,
        IdentityResolver as _CppIdentityResolver,
    )

    _LIFECYCLE_CLS: type | None = _CppTrackletLifecycleMerger
except ImportError:
    _CppIdentityResolver = None
    _LIFECYCLE_CLS = None

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
