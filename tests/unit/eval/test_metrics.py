"""Tests for TrackEval metrics discovery (perception.eval.metrics)."""

# scope: eval
# function: behavior
# lifecycle: active

import importlib
import sys

from saccade.perception.eval.metrics import _find_trackeval_root


def test_vendored_trackeval_mot_dataset_is_importable():
    trackeval_root = _find_trackeval_root()
    assert trackeval_root is not None

    root_str = str(trackeval_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    trackeval = importlib.import_module("trackeval")

    assert hasattr(trackeval, "datasets")
    assert hasattr(trackeval.datasets, "MotChallenge2DBox")
    cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    assert "GT_FOLDER" in cfg
    assert "TRACKERS_FOLDER" in cfg
