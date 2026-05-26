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
    # Budget + scheduling
    reid_budget: float = 0.2
    reid_interval: int = 20
    # Matching gates
    reid_cos_threshold: float = 0.90
    reid_iou_low: float = 0.30
    reid_iou_high: float = 0.60
    reid_weight: float = 0.80
    # Cropping
    reid_crop_mode: str = "tight"
    reid_crop_padding: float = 0.0
    reid_crop_layout: str = "full"
    # Lazy ReID
    lazy_reid_min_hit_streak: int = 2
    lazy_reid_self_threshold: float = 0.85
    # Async pipeline
    async_reid: bool = True
    pipeline_relink: bool = True
    # Advanced GMC (appearance-aware controls)
    gmc_fg_mask: bool = False
    gmc_pcr_uncertain_thresh: float = 8.0
    homography_root: str = ""
    # Profiling
    profile_lazy_reid_candidates: bool = False
    profile_lazy_reid_embeddings: bool = False
    # LaSt-ViT (experimental within reid)
    last_vit_embed: bool = False  # NO-GO (2026-05-02): SigLIP2 not trained with LaSt-ViT objective; IDF1 gain +0.09pp < +1.0pp threshold
    last_vit_gate: float = 0.0
    last_vit_sigma_embed: float = 0.015
    last_vit_sigma_gate: float = 0.040
    last_vit_top_k: float = 0.5
    # Pose-guided box expansion
    pose_box_expand: bool = False  # NO-GO (2026-05-10): FPS -60%, box expansion引发ID switches; root cause in detector training data
    pose_expand_ankle_conf: float = 0.30
    pose_expand_margin: float = 0.05
    pose_expand_flat_aspect: float = 0.0

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
        choices=("off", "tracker", "semantic", "hybrid"),
        default="off",
        help=_help(
            "ReID stack to enable. NOTE: semantic/hybrid modes are NO-GO when GMC is ON "
            "(GMC eliminates the main relink use case).",
            range_hint="off/tracker/semantic/hybrid",
            edge="off = baseline (no appearance); semantic = add semantic relink",
        ),
    )
    grp.add_argument(
        "--reid-model",
        choices=(
            "siglip2",
            "dinov2",
            "transreid",
            "osnet",
            "fastreid",
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
        help="Zero out previous-frame track boxes in GMC downscaled image to reduce foreground bias.",
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
        "--last-vit-embed",
        action="store_true",
        help="Replace image_embeds with LaSt-ViT V1 frequency-filtered embedding.",
    )
    grp.add_argument(
        "--last-vit-gate",
        type=float,
        default=0.0,
        help="Stability gate threshold [0,1] for LaSt-ViT embeddings. 0 disables.",
    )
    grp.add_argument(
        "--last-vit-sigma-embed",
        type=float,
        default=0.015,
        help="Gaussian sigma for Top-K patch selection (LaSt-ViT).",
    )
    grp.add_argument(
        "--last-vit-sigma-gate",
        type=float,
        default=0.040,
        help="Gaussian sigma for stability gating (LaSt-ViT).",
    )
    grp.add_argument(
        "--last-vit-top-k",
        type=float,
        default=0.5,
        help="Fraction of N patches to pool in LaSt-ViT (top_k_ratio).",
    )
    grp.add_argument(
        "--pose-box-expand",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="NO-GO (2026-05-10): Extend detection box bottom to ankle keypoints. FPS -60%,引发ID switches.",
    )
    grp.add_argument(
        "--pose-expand-ankle-conf",
        type=float,
        default=0.30,
        help=_help(
            "Minimum ankle keypoint confidence for box expansion.",
            range_hint="0-1",
            edge="too low expands on noisy ankle predictions",
        ),
    )
    grp.add_argument(
        "--pose-expand-margin",
        type=float,
        default=0.05,
        help=_help(
            "Extra margin below ankle as fraction of box height.", range_hint="0-0.2"
        ),
    )
    grp.add_argument(
        "--pose-expand-flat-aspect",
        type=float,
        default=0.0,
        help=_help(
            "Only expand boxes with h/w < this value (flat/under-segmented boxes). "
            "0 = expand all boxes with visible ankles. Typical upright person h/w ≈ 2-3; "
            "set ~1.5 to restrict to clearly flat boxes.",
            range_hint="0 (off) or 1.0-2.0",
        ),
    )
