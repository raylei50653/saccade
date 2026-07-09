from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from ._helpers import _help, _tier


@dataclasses.dataclass
class ReIDConfig:
    # Mode + model
    reid_mode: str = "off"
    reid_model: str = "siglip2"
    reid_engine_path: str = ""
    fpn_reid_ckpt: str = ""
    # Budget + scheduling
    reid_budget: float = 0.2
    reid_interval: int = 20
    # Matching gates
    reid_cos_threshold: float = 0.90
    reid_iou_low: float = 0.30
    reid_iou_high: float = 0.60
    reid_weight: float = 0.80
    # Appearance cost blend (affects Hungarian matcher cost matrix)
    reid_cost_cos_w: float = 0.55
    reid_cost_iou_w: float = 0.30
    reid_cost_score_w: float = 0.15
    # Cropping
    reid_crop_mode: str = "tight"
    reid_crop_padding: float = 0.0
    reid_crop_layout: str = "full"
    # Lazy ReID
    lazy_reid_min_hit_streak: int = 2
    lazy_reid_self_threshold: float = 0.85
    # Async pipeline.
    # NOTE: defaults mirror the live argparse defaults (store_true => False).
    # The runtime reads the argparse namespace, not these fields; keep in sync
    # (tests/unit/test_config_consistency.py enforces it). Whether these *should*
    # default on (perf-only, see project_e2e_latency_opt) is a separate decision.
    async_reid: bool = False
    pipeline_relink: bool = False
    # Advanced GMC (appearance-aware controls)
    gmc_fg_mask: bool = False
    gmc_pcr_uncertain_thresh: float = 8.0
    homography_root: str = ""
    # Profiling
    profile_lazy_reid_candidates: bool = False
    profile_lazy_reid_embeddings: bool = False
    # Experimental
    exp_velocity_aligned_bank: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReIDConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_flat_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_reid_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group(
        _tier("ReID backbone and embedding policy", "Tier 1/2")
    )
    grp.description = (
        "Appearance extraction and ReID scheduling. "
        "Load with: --module-reid configs/modules/reid.yaml"
    )
    grp.add_argument(
        "--reid-mode",
        choices=("off", "tracker", "semantic", "hybrid", "extract"),
        default="off",
        help=_help(
            "ReID stack to enable. NOTE: semantic/hybrid modes are NO-GO when GMC is ON "
            "(GMC eliminates the main relink use case). extract = run the appearance "
            "extractor for downstream consumers (online handover / crop ring) WITHOUT "
            "feeding embeddings into association — the tracker output is identical to "
            "off, so the handover sees the same track-fragmentation base as the offline "
            "reid-off run. Pair with a reid config that disables appearance_bank + "
            "relink_enabled (e.g. reid_mnv4_extract.yaml).",
            range_hint="off/tracker/semantic/hybrid/extract",
            edge="off = baseline (no appearance); extract = extract-only, tracker "
            "unaffected; semantic = add semantic relink",
        ),
    )
    grp.add_argument(
        "--reid-model",
        choices=(
            "siglip2",
            "siglip2_reid",
            "dinov2",
            "transreid",
            "osnet",
            "fastreid",
            "mobilenetv4_reid",
            "fpn_raw",
            "fpn_trained",
        ),
        default="siglip2",
        help="Embedding model family.",
    )
    grp.add_argument(
        "--fpn-reid-ckpt",
        default="",
        help="Trained DimReduceHead checkpoint for fpn_trained mode.",
    )
    grp.add_argument(
        "--fpn-backbone-engine",
        default="",
        help="TRT backbone engine path for FPN feature extraction (used when detector has no teacher backbone).",
    )
    grp.add_argument(
        "--reid-engine-path",
        default="",
        help="Optional TensorRT/engine path for the selected ReID backend.",
    )
    grp.add_argument(
        "--reid-budget",
        type=float,
        default=0.2,
        help=_help(
            "Max detections to ReID per frame; <1 = ratio, >=1 = fixed count, 0 = unlimited.",
            range_hint=">=0",
            edge="lower saves FPS but risks missing identity signals",
        ),
    )
    grp.add_argument(
        "--reid-interval",
        type=int,
        default=20,
        help=_help(
            "Fixed heartbeat interval for appearance refresh when dynamic triggering is off.",
            range_hint=">=1",
            edge="smaller is more responsive but costs more latency",
        ),
    )
    grp.add_argument(
        "--reid-cos-threshold",
        type=float,
        default=0.90,
        help=_help(
            "Cosine similarity gate for embedding matches.",
            range_hint="0-1",
            edge="higher is purer but breaks under pose/domain shift",
        ),
    )
    grp.add_argument(
        "--reid-iou-low",
        type=float,
        default=0.30,
        help=_help(
            "Low IoU bound where appearance can compensate for weak geometry.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--reid-iou-high",
        type=float,
        default=0.60,
        help=_help(
            "High IoU bound where geometry alone is usually sufficient.",
            range_hint="0-1",
            edge="keep above reid-iou-low",
        ),
    )
    grp.add_argument(
        "--reid-weight",
        type=float,
        default=0.80,
        help=_help(
            "Blend weight for appearance in combined matching cost.",
            range_hint="0-1",
            edge="near 1 lets embeddings dominate geometry",
        ),
    )
    grp.add_argument(
        "--reid-cost-cos-w",
        type=float,
        default=0.55,
        help=_help(
            "Appearance cost weight (cosine similarity term) in the Hungarian matcher.",
            range_hint="0-1",
            edge="lower = rely more on IoU, higher = trust embeddings more",
        ),
    )
    grp.add_argument(
        "--reid-cost-iou-w",
        type=float,
        default=0.30,
        help=_help(
            "IoU cost weight in the appearance-aware Hungarian matcher.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--reid-cost-score-w",
        type=float,
        default=0.15,
        help=_help(
            "Detection score cost weight in the appearance-aware Hungarian matcher.",
            range_hint="0-1",
        ),
    )
    grp.add_argument(
        "--reid-crop-mode",
        choices=("tight", "square", "square_mean"),
        default="tight",
        help="Crop geometry for embedding extraction.",
    )
    grp.add_argument(
        "--reid-crop-padding",
        type=float,
        default=0.0,
        help=_help(
            "Extra crop padding ratio for embeddings.", range_hint=">=0, usually 0-0.3"
        ),
    )
    grp.add_argument(
        "--reid-crop-layout",
        choices=("full", "parts"),
        default="full",
        help="Embedding crop layout.",
    )
    grp.add_argument(
        "--lazy-reid-min-hit-streak",
        type=int,
        default=2,
        help=_help(
            "Hits before a track is eligible for lazy self-ReID checks.",
            range_hint=">=1",
        ),
    )
    grp.add_argument(
        "--lazy-reid-self-threshold",
        type=float,
        default=0.85,
        help=_help("Self-similarity threshold for lazy ReID reuse.", range_hint="0-1"),
    )
    grp.add_argument(
        "--async-reid",
        action="store_true",
        help="Pipeline ReID extract on a side CUDA stream (~1.5ms/reid-frame gain).",
    )
    grp.add_argument(
        "--pipeline-relink",
        action="store_true",
        help="Inter-frame pipelining: run relink_write for frame N while detecting frame N+1.",
    )
    grp.add_argument(
        "--gmc-fg-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_help(
            "Zero out previous-frame track boxes in GMC downscaled image to reduce "
            "foreground bias. Does not fix PCR-dominated background; headline keeps "
            "false (registry #20).",
            policy="NO-GO",
        ),
    )
    grp.add_argument(
        "--gmc-pcr-uncertain-thresh",
        type=float,
        default=8.0,
        help=_help(
            "PCR ratio below which GMC shift is flagged as uncertain, triggering ReID boost.",
            range_hint=">5.0 (PCR threshold is 5.0)",
        ),
    )
    grp.add_argument(
        "--homography-root",
        default="",
        help="Optional directory containing sequence-specific .txt homography matrices (3x3).",
    )
    grp.add_argument(
        "--profile-lazy-reid-candidates",
        action="store_true",
        help="Profile candidate generation for lazy ReID triggering.",
    )
    grp.add_argument(
        "--profile-lazy-reid-embeddings",
        action="store_true",
        help="Profile embedding extraction for lazy ReID triggering.",
    )
    grp.add_argument(
        "--exp-velocity-aligned-bank",
        action="store_true",
        help="[Experimental] Only update Appearance Bank when the detection's instantaneous motion direction is aligned with its history.",
    )
