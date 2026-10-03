"""GPUByteTracker/GMC/pipeline parameters through the legacy pybind front-end (#465 PR-4b).

The native objects no longer read ``SACCADE_*``; the binding resolves the legacy
hatches and hands them over as explicit parameters. These tests check that the
eval harness path still lands every value in the one parameter state the kernels
read (``snapshot()``), and that the first CUDA graph capture of the tracker
update freezes it. The shipping builders are covered natively
(``tests/native/test_shipping_native_build.cpp``).
"""

# scope: tracking
# function: contract
# lifecycle: active

from __future__ import annotations

import pytest
import torch

pytest.importorskip("saccade_tracking_ext", exc_type=ImportError)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import saccade_tracking_ext as ext  # noqa: E402

from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    GPUByteTracker,
    GraphedTrackerUpdate,
)

_HATCHES = {
    "SACCADE_ENABLE_DDA": ("0", False, True),
    "SACCADE_DDA_MAX_COST": ("0.5", 0.5, 0.12),
    "SACCADE_STABILITY_W": ("0", 0.0, 0.1),
    "SACCADE_FRESHNESS_W": ("0.25", 0.25, 0.0),
    "SACCADE_COAST_MAX_AGE": ("3.9", 3, 0),
}


def _f32(x: float) -> float:
    return float(torch.tensor(x, dtype=torch.float32).item())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        *_HATCHES,
        "SACCADE_KALMAN_ADAPT_MODE",
        "SACCADE_GMC_PCR_THRESH",
        "SACCADE_DETERMINISTIC_FILTER_COMPACTION",
        "SACCADE_ATOMIC_FILTER_BASELINE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_snapshot_is_one_flat_view() -> None:
    snap = ext.GPUByteTracker(256, 16, 128).snapshot()
    assert snap["constructor.max_objects"] == 256
    assert snap["constructor.max_assoc"] == 128
    assert snap["set_homography.h"] is None
    assert snap["config_frozen"] is False
    assert snap["embeddings_forbidden"] is False  # only the shipping builder sets it
    assert not any(
        v for k, v in snap.items() if k.startswith(("research.", "diagnostic."))
    )
    assert len(snap) == 102


@pytest.mark.parametrize("name", sorted(_HATCHES))
def test_constructor_resolves_legacy_hatches(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    raw, set_value, unset_value = _HATCHES[name]
    key = f"native_env.{name}"
    assert ext.GPUByteTracker(64, 8, 64).snapshot()[key] == pytest.approx(unset_value)
    monkeypatch.setenv(name, raw)
    tracker = ext.GPUByteTracker(64, 8, 64)
    assert tracker.snapshot()[key] == pytest.approx(set_value)
    # Read at construction only: a later change does not reach this tracker.
    monkeypatch.delenv(name)
    assert tracker.snapshot()[key] == pytest.approx(set_value)


def test_kalman_override_is_read_on_every_set_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = GPUByteTracker(max_objects=64, embedding_dim=8)
    tracker.set_params(0.1, 0.5, 0.8, 30, kalman_adapt_mode=0)
    assert tracker.tracker.snapshot()["set_params.kalman_adapt_mode"] == 0
    monkeypatch.setenv("SACCADE_KALMAN_ADAPT_MODE", "2")
    tracker.set_params(0.1, 0.5, 0.8, 30, kalman_adapt_mode=0)
    assert tracker.tracker.snapshot()["set_params.kalman_adapt_mode"] == 2


def test_setters_write_the_snapshot_and_keep_negative_score_w() -> None:
    tracker = ext.GPUByteTracker(64, 8, 64)
    tracker.set_oao_params(0.5, -1.0, -1.0, 0, 0.0, 0.0, 0.0, 25.0)
    tracker.set_params(0.05, 0.45, 0.5, 30, 0.1, 0, 0.5, False, 0.28)
    tracker.set_frame_size(1920, 1080)
    snap = tracker.snapshot()
    assert snap["set_oao_params.score_w"] == -1.0  # was canonicalized to 0 before PR-4b
    assert snap["set_oao_params.tau"] == 0.5
    assert snap["set_oao_params.ramp_frames"] == 25.0
    assert snap["set_params.confirm_streak"] == 1  # legacy canonicalization kept
    assert snap["set_params.new_track_thresh"] == _f32(0.28)
    assert (snap["set_frame_size.w"], snap["set_frame_size.h"]) == (1920, 1080)
    tracker.set_oao_params(0.5, -1.0, 1.5)
    assert tracker.snapshot()["set_oao_params.score_w"] == 1.0


def test_graph_capture_freezes_the_configuration() -> None:
    tracker = GPUByteTracker(max_objects=256, embedding_dim=16)
    tracker.set_frame_size(1920, 1080)
    graphed = GraphedTrackerUpdate(tracker)
    before = tracker.tracker.snapshot()
    assert before["config_frozen"] is False

    empty = torch.zeros(0, 4, device="cuda")
    graphed.copy_inputs(
        empty,
        torch.zeros(0, device="cuda"),
        torch.zeros(0, dtype=torch.int32, device="cuda"),
    )
    graphed.replay()
    torch.cuda.synchronize()

    after = tracker.tracker.snapshot()
    assert after["config_frozen"] is True
    for call in (
        lambda: tracker.set_frame_size(640, 480),
        lambda: tracker.set_params(0.1, 0.5, 0.8, 30),
        lambda: tracker.set_oao_params(0.2),
        lambda: tracker.tracker.set_sinkhorn_lambda(20.0),
        lambda: tracker.tracker.set_research_bridge_shadow(True),
    ):
        with pytest.raises(
            RuntimeError, match="frozen after the first CUDA graph capture"
        ):
            call()
    # Nothing reached the parameters the captured graph replays.
    del after["config_frozen"], before["config_frozen"]
    assert after == before


def test_gmc_and_pipeline_resolve_their_hatches_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ext.GMC(4).snapshot()["native_env.SACCADE_GMC_PCR_THRESH"] == 5.0
    monkeypatch.setenv("SACCADE_GMC_PCR_THRESH", "3.5")
    assert ext.GMC(4).snapshot()["native_env.SACCADE_GMC_PCR_THRESH"] == 3.5

    cfg = ext.PerceptionPipelineConfig()
    assert (
        ext.PerceptionPipeline(0, 0, cfg).snapshot()["filter_compaction"]
        == "stable_scan"
    )
    monkeypatch.setenv("SACCADE_ATOMIC_FILTER_BASELINE", "1")
    assert (
        ext.PerceptionPipeline(0, 0, cfg).snapshot()["filter_compaction"]
        == "atomic_baseline"
    )
    monkeypatch.setenv("SACCADE_DETERMINISTIC_FILTER_COMPACTION", "1")
    snap = ext.PerceptionPipeline(0, 0, cfg).snapshot()
    assert snap["filter_compaction"] == "serial_stable"  # deterministic wins
    assert snap["constructor.config.max_detections"] == cfg.max_detections


# ── former process-wide statics ───────────────────────────────────────────
#
# Before PR-4b, SACCADE_FRESHNESS_W / SACCADE_STABILITY_W (tracker auction bid)
# and SACCADE_GMC_PCR_THRESH (GMC phase correlation) were function-local
# ``static const`` values: read from env once per process, at the first
# update/estimate, then used by every instance. They were const after that
# first read, so instances never communicated through them; the only way the
# old "shared" semantics could be observed is env changing between two
# constructions inside one process. No harness does that (the oracle sets env
# in configure_runtime_env before any tracker exists; ablations pass env to
# subprocesses). PR-4b makes the value per instance, read at construction.


def _scene(frame: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.tensor([100.0 + 6 * frame, 420.0 + 2 * frame, 760.0 - 5 * frame])
    boxes = torch.stack(
        [xs, torch.full_like(xs, 200.0), xs + 60.0, torch.full_like(xs, 380.0)], 1
    )
    scores = torch.tensor([0.92, 0.81, 0.66])
    return boxes.cuda(), scores.cuda(), torch.zeros(3, dtype=torch.int32).cuda()


def _step(tracker: object, frame: int, gmc: torch.Tensor) -> list[tuple]:
    boxes, scores, classes = _scene(frame)
    out = tracker.update(  # type: ignore[attr-defined]
        boxes.data_ptr(),
        scores.data_ptr(),
        classes.data_ptr(),
        int(boxes.shape[0]),
        torch.cuda.current_stream().cuda_stream,
        0,
        gmc.data_ptr(),
    )
    return [(r.obj_id, r.x1, r.y1, r.x2, r.y2, r.score) for r in out]


def _identity_gmc() -> torch.Tensor:
    return torch.eye(2, 3, dtype=torch.float32, device="cuda").flatten().contiguous()


def test_former_statics_are_per_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    gmc = _identity_gmc()
    first = ext.GPUByteTracker(64, 8, 64)
    for frame in range(3):  # the old statics were fixed by the first update
        _step(first, frame, gmc)
    gmc_first = ext.GMC(4)

    monkeypatch.setenv("SACCADE_STABILITY_W", "0")
    monkeypatch.setenv("SACCADE_FRESHNESS_W", "0.25")
    monkeypatch.setenv("SACCADE_GMC_PCR_THRESH", "3.5")
    second = ext.GPUByteTracker(64, 8, 64)
    gmc_second = ext.GMC(4)

    a, b = first.snapshot(), second.snapshot()
    assert a["native_env.SACCADE_STABILITY_W"] == pytest.approx(0.1)
    assert a["native_env.SACCADE_FRESHNESS_W"] == 0.0
    assert b["native_env.SACCADE_STABILITY_W"] == 0.0
    assert b["native_env.SACCADE_FRESHNESS_W"] == 0.25
    assert gmc_first.snapshot()["native_env.SACCADE_GMC_PCR_THRESH"] == 5.0
    assert gmc_second.snapshot()["native_env.SACCADE_GMC_PCR_THRESH"] == 3.5
    # Constructing the second instance did not reach the first one.
    _step(first, 3, gmc)
    assert first.snapshot() == a


def test_instances_under_one_environment_do_not_interfere() -> None:
    """The harness condition (env fixed for the process): per-instance values
    equal the old process-wide ones, and interleaving two trackers gives each
    the output it has alone."""
    gmc = _identity_gmc()
    solo = ext.GPUByteTracker(64, 8, 64)
    expected = [_step(solo, frame, gmc) for frame in range(20)]

    left = ext.GPUByteTracker(64, 8, 64)
    right = ext.GPUByteTracker(64, 8, 64)
    assert (
        left.snapshot() == right.snapshot() == ext.GPUByteTracker(64, 8, 64).snapshot()
    )
    for frame in range(20):
        assert _step(left, frame, gmc) == expected[frame]
        assert _step(right, frame, gmc) == expected[frame]
    assert any(expected), "the synthetic scene must produce tracks"
