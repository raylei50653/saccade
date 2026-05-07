import torch
from dataclasses import dataclass
from typing import TypedDict, Any


@dataclass
class IdStabilityState:
    frame_id: int
    box: tuple[float, float, float, float]
    stable_hits: int
    total_hits: int
    score_ema: float


@dataclass
class TrackletLifecycleState:
    output_id: int
    frame_id: int
    box: tuple[float, float, float, float]
    score: float
    embedding: torch.Tensor | None


@dataclass
class MotRecord:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float
    tail: list[str]


class HostTrackResultView(TypedDict):
    count: int
    boxes: torch.Tensor
    scores: torch.Tensor
    ids: torch.Tensor
    classes: torch.Tensor | None
    det_idx: torch.Tensor | None


@dataclass(frozen=True)
class HostTrackBatch:
    boxes_gpu: torch.Tensor
    boxes: list[tuple[float, float, float, float]]
    scores: list[float]
    ids: list[int]
    classes: list[int] | None
    det_idx: list[int] | None
    person_observations: dict[int, Any] | None


@dataclass(frozen=True)
class PreparedTrackCandidate:
    local_track_id: int
    box: tuple[float, float, float, float]
    score: float
    embedding: torch.Tensor | None


@dataclass(frozen=True)
class CandidateAppearanceUpdate:
    track_id: int
    embedding: torch.Tensor
    det_score: float
    iou: float
    frame_id: int
    geometry_clean: bool
    suspect_box: bool
    aspect_ratio: float = 0.0
    bank_quality_score: float = 0.0


@dataclass(frozen=True)
class ResolvedTrack:
    local_track_id: int
    resolved_track_id: int
    box: tuple[float, float, float, float]
    score: float
    embedding: torch.Tensor | None


@dataclass
class OutputTracklet:
    track_id: int
    records: list[MotRecord]
    start: int
    end: int
    start_box: tuple[float, float, float, float]
    end_box: tuple[float, float, float, float]
    start_velocity: tuple[float, float]
    end_velocity: tuple[float, float]
    mean_score: float
