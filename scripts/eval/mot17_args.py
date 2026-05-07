import argparse


def _help(text: str, *, range_hint: str | None = None, edge: str | None = None) -> str:
    parts = [text]
    if range_hint:
        parts.append(f"Range: {range_hint}.")
    if edge:
        parts.append(f"Boundary: {edge}.")
    return " ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MOT17 tracking runs. Parameters are grouped by capability area so "
            "the --help output shows which stage each knob affects and the practical edge "
            "where aggressive tuning usually starts to trade recall for precision, or vice versa."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("I/O and dataset scope")
    io_group.description = (
        "Controls which engine, split, and sequence slice to evaluate. These define job scope, "
        "not tracker behavior."
    )
    io_group.add_argument(
        "--engine",
        default="models/yolo/yolo26s_960_batch1.engine",
        help="TensorRT detector engine path.",
    )
    io_group.add_argument(
        "--output",
        default="results/MOT17_eval",
        help="Directory for outputs, metrics, and dumps.",
    )
    io_group.add_argument(
        "--data-root", default="datasets/MOT17", help="MOT17 dataset root."
    )
    io_group.add_argument(
        "--split",
        default="train",
        help=_help(
            "Dataset split name.",
            range_hint="train/val/test or repo-defined split names",
        ),
    )
    io_group.add_argument(
        "--sequences",
        default="",
        help="Comma-separated sequence names. Empty means all sequences in the split.",
    )
    io_group.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=_help(
            "Optional per-sequence frame cap for smoke tests.",
            range_hint=">=1 or unset",
            edge="small caps under-sample relink and long-gap behavior",
        ),
    )
    io_group.add_argument(
        "--debug-dump-seq",
        default="",
        help="Optional single sequence name whose detection stages should be dumped.",
    )
    io_group.add_argument(
        "--debug-dump-frames",
        default="",
        help="Optional frame list/ranges for stage dump, e.g. 172-235,300,301.",
    )
    io_group.add_argument(
        "--debug-dump-csv",
        default="",
        help="Optional CSV path for raw/post_filter/post_nms/post_merge box dumps.",
    )
    io_group.add_argument(
        "--detector",
        choices=["SDP", "DPM", "FRCNN"],
        default=None,
        help=_help(
            "Filter sequences by detector suffix. Overrides --sequences only when --sequences is empty.",
            range_hint="SDP/DPM/FRCNN",
        ),
    )

    detect_group = parser.add_argument_group("Detection and preprocessing")
    detect_group.description = (
        "Front-end controls for detector filtering and image preparation. These mostly set the "
        "recall/precision envelope before tracking starts."
    )
    detect_group.add_argument(
        "--conf-threshold",
        type=float,
        default=0.05,
        help=_help(
            "Detector confidence floor.",
            range_hint="0-1, usually 0.01-0.3",
            edge="lower increases recall/noise; higher suppresses weak people",
        ),
    )
    detect_group.add_argument(
        "--person-class",
        type=int,
        default=0,
        help="Person class id in detector output.",
    )
    detect_group.add_argument(
        "--track-person-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only person-class detections for tracking.",
    )
    detect_group.add_argument(
        "--preprocess",
        default="letterbox,gamma,contrast",
        help="Comma-separated preprocessing pipeline.",
    )
    detect_group.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help=_help(
            "Gamma correction factor.",
            range_hint="0.6-1.4",
            edge="too low brightens shadows but amplifies sensor noise",
        ),
    )
    detect_group.add_argument(
        "--gamma-luma-threshold",
        type=float,
        default=0.35,
        help=_help("Only apply gamma below this luma threshold.", range_hint="0-1"),
    )
    detect_group.add_argument(
        "--contrast",
        type=float,
        default=1.2,
        help=_help(
            "Linear contrast multiplier.",
            range_hint="0.8-1.6",
            edge="too high clips highlights and destabilizes embeddings",
        ),
    )
    detect_group.add_argument(
        "--tiling",
        choices=["960p_2x2", "960p_3x2", "native_960"],
        default="native_960",
        help="Inference tiling preset.",
    )
    detect_group.add_argument(
        "--cross-tile-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge duplicate detections across tile boundaries.",
    )
    detect_group.add_argument(
        "--cross-tile-score-penalty",
        type=float,
        default=1.0,
        help=_help(
            "MOT17-b: multiply the score of cross-tile-merged boxes by this factor "
            "(1.0 = no change, <1 makes merged boxes less aggressive in association).",
            range_hint="0-1",
        ),
    )
    detect_group.add_argument(
        "--tile-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit per-sequence diagnostics for tiled detection: seam-near boxes, merge compression, and merged clusters.",
    )
    detect_group.add_argument(
        "--tile-seam-score-penalty",
        type=float,
        default=1.0,
        help=_help(
            "Multiply scores of seam-near tiled detections by this factor.",
            range_hint="0-1",
            edge="lower suppresses tile-boundary artifacts but can also hide true partial people",
        ),
    )
    detect_group.add_argument(
        "--tile-seam-margin-canvas-px",
        type=float,
        default=24.0,
        help=_help(
            "Half-width of the seam-near band on the 960 canvas used for seam-aware diagnostics and score penalty.",
            range_hint=">=0",
        ),
    )
    detect_group.add_argument(
        "--cross-tile-seam-center-scale",
        type=float,
        default=1.8,
        help=_help(
            "Scale factor that widens center-distance gating for seam-near duplicate merge.",
            range_hint=">=1",
        ),
    )
    detect_group.add_argument(
        "--cross-tile-seam-area-ratio-threshold",
        type=float,
        default=0.30,
        help=_help(
            "Minimum area-ratio for seam-near duplicate merge candidates.",
            range_hint="0-1",
        ),
    )
    detect_group.add_argument(
        "--cross-tile-seam-min-overlap-ratio",
        type=float,
        default=0.45,
        help=_help(
            "Minimum overlap ratio on both axes for seam-near duplicate merge.",
            range_hint="0-1",
        ),
    )
    detect_group.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=None,
        help=_help(
            "Override detector NMS IoU.",
            range_hint="0-1 or unset",
            edge="lower removes duplicates earlier; higher preserves crowded detections",
        ),
    )
    detect_group.add_argument(
        "--crowd-low-score-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable lower detector/tracker score floors only on crowded frames.",
    )
    detect_group.add_argument(
        "--crowd-low-score-trigger",
        type=int,
        default=25,
        help=_help(
            "Activate crowd low-score mode when post-merge detections reach this count.",
            range_hint=">=1",
        ),
    )
    detect_group.add_argument(
        "--crowd-conf-threshold",
        type=float,
        default=0.02,
        help=_help(
            "Detector confidence floor used during crowd low-score mode.",
            range_hint="0-1",
        ),
    )
    detect_group.add_argument(
        "--narrow-person-score-bonus",
        type=float,
        default=0.0,
        help=_help(
            "Additive score bonus for narrow person-like raw detections before post-filter.",
            range_hint=">=0",
            edge="too high will promote background slivers into tracks",
        ),
    )
    detect_group.add_argument(
        "--narrow-person-max-width-ratio",
        type=float,
        default=0.018,
        help=_help(
            "Maximum box width / frame width for narrow-person score bonus.",
            range_hint=">0",
        ),
    )
    detect_group.add_argument(
        "--narrow-person-min-height-ratio",
        type=float,
        default=0.045,
        help=_help(
            "Minimum box height / frame height for narrow-person score bonus.",
            range_hint=">0",
        ),
    )
    detect_group.add_argument(
        "--narrow-person-min-aspect",
        type=float,
        default=2.0,
        help=_help(
            "Minimum h/w aspect ratio for narrow-person score bonus.",
            range_hint=">0",
        ),
    )
    detect_group.add_argument(
        "--narrow-person-max-aspect",
        type=float,
        default=4.8,
        help=_help(
            "Maximum h/w aspect ratio for narrow-person score bonus.",
            range_hint=">0",
        ),
    )
    detect_group.add_argument(
        "--warmup-frames",
        type=int,
        default=50,
        help=_help(
            "Frames to ignore for warmup/profiling effects.",
            range_hint=">=0",
            edge="too short includes engine cold-start noise",
        ),
    )

    assoc_group = parser.add_argument_group("Core tracking and association")
    assoc_group.description = (
        "Primary thresholds that decide whether detections become tracks, match existing tracks, "
        "or get promoted into stable identities."
    )
    assoc_group.add_argument(
        "--track-thresh",
        type=float,
        default=0.05,
        help=_help(
            "Low-confidence association floor.",
            range_hint="0-1, usually <= high-thresh",
        ),
    )
    assoc_group.add_argument(
        "--high-thresh",
        type=float,
        default=0.45,
        help=_help(
            "High-confidence matching threshold.",
            range_hint="0-1",
            edge="higher reduces false links but raises fragmentation",
        ),
    )
    assoc_group.add_argument(
        "--mid-thresh",
        type=float,
        default=0.10,
        help=_help(
            "Mid-tier confidence bucket between track-thresh and high-thresh.",
            range_hint="0-1",
            edge="keep between track-thresh and high-thresh for predictable behavior",
        ),
    )
    assoc_group.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.35,
        help=_help(
            "Minimum score for starting a new track.",
            range_hint="0-1",
            edge="higher suppresses weak true positives and tends to increase FN",
        ),
    )
    assoc_group.add_argument(
        "--crowd-track-thresh",
        type=float,
        default=0.02,
        help=_help(
            "Low-confidence association floor used during crowd low-score mode.",
            range_hint="0-1",
        ),
    )
    assoc_group.add_argument(
        "--crowd-mid-thresh",
        type=float,
        default=0.05,
        help=_help(
            "Mid-tier confidence bucket used during crowd low-score mode.",
            range_hint="0-1",
        ),
    )
    assoc_group.add_argument(
        "--crowd-new-track-thresh",
        type=float,
        default=0.25,
        help=_help(
            "New-track score threshold used during crowd low-score mode.",
            range_hint="0-1",
        ),
    )
    assoc_group.add_argument(
        "--match-thresh",
        type=float,
        default=0.78,
        help=_help(
            "Association similarity gate.",
            range_hint="0-1",
            edge="near 1 is strict and favors purity over continuity",
        ),
    )
    assoc_group.add_argument(
        "--confirm-streak",
        type=int,
        default=1,
        help=_help(
            "Hits needed before confirming a tentative track.",
            range_hint=">=1",
            edge="higher removes flicker but delays births",
        ),
    )
    assoc_group.add_argument(
        "--confirm-score-thresh",
        type=float,
        default=0.0,
        help=_help("Minimum score for counting toward confirmation.", range_hint="0-1"),
    )
    assoc_group.add_argument(
        "--adaptive-confirmation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable score/geometry-aware confirmation logic.",
    )
    assoc_group.add_argument(
        "--gmc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable global motion compensation.",
    )
    assoc_group.add_argument(
        "--gmc-mode",
        choices=["cpu", "gpu"],
        default="gpu",
        help=_help(
            "GMC algorithm mode. 'cpu' uses OpenCV LK; 'gpu' uses cuFFT phase correlation.",
            range_hint="cpu/gpu",
        ),
    )
    assoc_group.add_argument(
        "--gmc-downscale",
        type=int,
        default=8,
        help=_help(
            "Downscale factor for GMC estimation.",
            range_hint=">=1",
            edge="too large is faster but loses fine camera motion",
        ),
    )
    assoc_group.add_argument(
        "--gmc-fg-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(A4) Zero out previous-frame track boxes in the downscaled gray image before "
        "phase correlation, reducing foreground bias in crowded scenes.",
    )
    assoc_group.add_argument(
        "--gmc-pcr-uncertain-thresh",
        type=float,
        default=8.0,
        help=_help(
            "(A4) PCR ratio below which a valid GMC shift is flagged as 'uncertain', "
            "triggering a 1.5× ReID priority boost for all active tracks.",
            range_hint=">5.0 (PCR threshold is 5.0)",
        ),
    )
    assoc_group.add_argument(
        "--homography-root",
        default="",
        help="Optional directory containing sequence-specific .txt homography matrices (3x3 row-major).",
    )
    assoc_group.add_argument(
        "--nsa-kalman",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Noise Scale Adaptive Kalman: scale R by (1-score)^2 per detection.",
    )
    assoc_group.add_argument(
        "--per-seq-adapt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-scale time-like tracker parameters by sequence frameRate.",
    )

    geom_group = parser.add_argument_group("Geometry priors and ID stability")
    geom_group.description = (
        "Shape, scale, and short-term motion guards that reject implausible person boxes or "
        "unstable ID transitions before appearance logic runs."
    )
    geom_group.add_argument(
        "--geometry-mid-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adaptive middle-scale geometry gating.",
    )
    geom_group.add_argument(
        "--geometry-ref-height-ratio",
        type=float,
        default=0.12,
        help=_help(
            "Reference person height ratio for geometry normalization.",
            range_hint="0-1",
        ),
    )
    geom_group.add_argument(
        "--geometry-min-scale",
        type=float,
        default=0.875,
        help=_help(
            "Allowed lower scale multiplier around geometry reference.",
            range_hint=">0, usually 0.7-1.0",
        ),
    )
    geom_group.add_argument(
        "--geometry-max-scale",
        type=float,
        default=1.20,
        help=_help(
            "Allowed upper scale multiplier around geometry reference.",
            range_hint=">1, usually 1.0-1.5",
        ),
    )
    geom_group.add_argument(
        "--geometry-ema-beta",
        type=float,
        default=0.80,
        help=_help(
            "EMA smoothing for geometry reference updates.",
            range_hint="0-1",
            edge="near 1 is stable but slow to adapt",
        ),
    )
    geom_group.add_argument(
        "--geometry-loosen-step",
        type=float,
        default=0.08,
        help=_help("Per-update rate for relaxing geometry gates.", range_hint="0-1"),
    )
    geom_group.add_argument(
        "--geometry-tighten-step",
        type=float,
        default=0.03,
        help=_help("Per-update rate for tightening geometry gates.", range_hint="0-1"),
    )
    geom_group.add_argument(
        "--geometry-min-samples",
        type=int,
        default=5,
        help=_help(
            "Samples needed before geometry adaptation is trusted.", range_hint=">=1"
        ),
    )
    geom_group.add_argument(
        "--id-stability-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter unstable identity handoffs with short-term spatial consistency checks.",
    )
    geom_group.add_argument(
        "--id-stability-min-hits",
        type=int,
        default=2,
        help=_help(
            "Minimum hits before stability checks are considered reliable.",
            range_hint=">=1",
        ),
    )
    geom_group.add_argument(
        "--id-stability-min-iou",
        type=float,
        default=0.05,
        help=_help("Minimum IoU to treat a continuation as stable.", range_hint="0-1"),
    )
    geom_group.add_argument(
        "--id-stability-max-center-shift",
        type=float,
        default=2.0,
        help=_help(
            "Maximum normalized center shift for a stable continuation.",
            range_hint=">0",
            edge="too low breaks fast movers, too high lets jumps through",
        ),
    )
    geom_group.add_argument(
        "--id-stability-max-gap",
        type=int,
        default=1,
        help=_help(
            "Maximum gap in frames still considered short-term stable.",
            range_hint=">=0",
        ),
    )
    geom_group.add_argument(
        "--id-stability-score-ema",
        type=float,
        default=0.70,
        help=_help("EMA decay for stability confidence.", range_hint="0-1"),
    )
    geom_group.add_argument(
        "--id-stability-min-score-ema",
        type=float,
        default=0.15,
        help=_help(
            "Minimum EMA score to keep a stable ID hypothesis.", range_hint="0-1"
        ),
    )
    geom_group.add_argument(
        "--person-geometry-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hard geometric filtering for person boxes.",
    )
    geom_group.add_argument(
        "--person-min-height-ratio",
        type=float,
        default=0.018,
        help=_help(
            "Minimum bbox height as image-height ratio.",
            range_hint="0-1",
            edge="higher drops far-away pedestrians",
        ),
    )
    geom_group.add_argument(
        "--person-min-aspect",
        type=float,
        default=1.0,
        help=_help("Minimum h/w aspect ratio for person boxes.", range_hint=">0"),
    )
    geom_group.add_argument(
        "--person-max-aspect",
        type=float,
        default=5.5,
        help=_help(
            "Maximum h/w aspect ratio for person boxes.",
            range_hint="> person-min-aspect",
        ),
    )
    geom_group.add_argument(
        "--person-min-area-ratio",
        type=float,
        default=0.00006,
        help=_help("Minimum bbox area as image-area ratio.", range_hint="0-1"),
    )
    geom_group.add_argument(
        "--person-max-area-ratio",
        type=float,
        default=0.0,
        help=_help(
            "Maximum bbox area as image-area ratio; 0 disables the cap.",
            range_hint="0-1",
            edge="use only when very large false positives dominate",
        ),
    )
    geom_group.add_argument(
        "--detection-quality-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(A6) Scale detection scores by a continuous quality factor (aspect+center+area) "
        "instead of binary suspect capping.",
    )
    geom_group.add_argument(
        "--detection-quality-w-aspect",
        type=float,
        default=0.50,
        help="(A6) Weight for aspect ratio in detection quality factor.",
    )
    geom_group.add_argument(
        "--detection-quality-w-center",
        type=float,
        default=0.30,
        help="(A6) Weight for center bias (truncation) in detection quality factor.",
    )
    geom_group.add_argument(
        "--detection-quality-w-area",
        type=float,
        default=0.20,
        help="(A6) Weight for area ratio in detection quality factor.",
    )
    geom_group.add_argument(
        "--geometry-suspect-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow suspicious geometry only with extra tracker support.",
    )
    geom_group.add_argument(
        "--geometry-suspect-score",
        type=float,
        default=None,
        help=_help(
            "Override score gate for suspect geometry cases.", range_hint="0-1 or unset"
        ),
    )

    reid_group = parser.add_argument_group("ReID backbone and embedding policy")
    reid_group.description = (
        "Appearance extraction and coarse ReID scheduling. These settings define when embeddings "
        "exist at all and how strict identity similarity should be."
    )
    reid_group.add_argument(
        "--reid-budget",
        type=float,
        default=0.2,
        help=_help(
            "Maximum detections to ReID per frame; <1.0 means ratio of detections, >=1 means fixed count, 0 means unlimited.",
            range_hint=">=0",
            edge="lower saves FPS but risks missing identity signals",
        ),
    )
    reid_group.add_argument(
        "--async-reid",
        action="store_true",
        help=(
            "Pipeline ReID extract on a side CUDA stream, overlapping with "
            "tracker.update_into on the main stream. Bank/relink still receive "
            "fresh embeddings; tracker cost matrix loses detection-side appearance "
            "for the pipelined frames. Expected throughput gain: ~1.5ms/reid-frame."
        ),
    )
    reid_group.add_argument(
        "--pipeline-relink",
        action="store_true",
        help=(
            "Inter-frame pipelining: run relink_write for frame N in a background "
            "thread while the main thread runs detect+postprocess for frame N+1. "
            "All GPU tensors pre-materialized to CPU before submit to avoid stream "
            "conflicts. Incompatible with --profile-stages (auto-disabled). "
            "Expected gain: ~5ms/frame (~62 FPS from ~51 FPS)."
        ),
    )
    reid_group.add_argument(
        "--profile-lazy-reid-candidates",
        action="store_true",
        help="Profile candidate generation for lazy ReID triggering.",
    )
    reid_group.add_argument(
        "--profile-lazy-reid-embeddings",
        action="store_true",
        help="Profile embedding extraction for lazy ReID triggering.",
    )
    reid_group.add_argument(
        "--lazy-reid-min-hit-streak",
        type=int,
        default=2,
        help=_help(
            "Hits before a track is eligible for lazy self-ReID checks.",
            range_hint=">=1",
        ),
    )
    reid_group.add_argument(
        "--lazy-reid-self-threshold",
        type=float,
        default=0.85,
        help=_help("Self-similarity threshold for lazy ReID reuse.", range_hint="0-1"),
    )
    reid_group.add_argument(
        "--reid-mode",
        choices=("off", "tracker", "semantic", "hybrid"),
        default="semantic",
        help=_help(
            "ReID stack to enable.",
            range_hint="off/tracker/semantic/hybrid",
            edge="off removes appearance recovery entirely",
        ),
    )
    reid_group.add_argument(
        "--reid-model",
        choices=("siglip2", "dinov2", "transreid", "osnet", "fastreid"),
        default="siglip2",
        help="Embedding model family.",
    )
    reid_group.add_argument(
        "--reid-engine-path",
        default="",
        help="Optional TensorRT/engine path for the selected ReID backend.",
    )
    reid_group.add_argument(
        "--last-vit-embed",
        action="store_true",
        help="Phase 2C: replace image_embeds with LaSt-ViT V1 frequency-filtered embedding.",
    )
    reid_group.add_argument(
        "--last-vit-gate",
        type=float,
        default=0.0,
        help="Phase 2C: stability gate threshold [0,1]. Embeddings below this are zeroed "
        "out before bank/tracker injection. 0 disables.",
    )
    reid_group.add_argument(
        "--last-vit-sigma-embed",
        type=float,
        default=0.015,
        help="Gaussian sigma for Top-K patch selection (LaSt-ViT).",
    )
    reid_group.add_argument(
        "--last-vit-sigma-gate",
        type=float,
        default=0.040,
        help="Gaussian sigma for stability gating (LaSt-ViT).",
    )
    reid_group.add_argument(
        "--last-vit-top-k",
        type=float,
        default=0.5,
        help="Fraction of N patches to pool in LaSt-ViT (top_k_ratio).",
    )
    reid_group.add_argument(
        "--reid-interval",
        type=int,
        default=10,
        help=_help(
            "Fixed heartbeat interval for appearance refresh when dynamic triggering is off.",
            range_hint=">=1",
            edge="smaller is more responsive but costs more latency",
        ),
    )
    reid_group.add_argument(
        "--reid-cos-threshold",
        type=float,
        default=0.90,
        help=_help(
            "Cosine similarity gate for embedding matches.",
            range_hint="0-1",
            edge="higher is purer but breaks under pose/domain shift",
        ),
    )
    reid_group.add_argument(
        "--reid-iou-low",
        type=float,
        default=0.30,
        help=_help(
            "Low IoU bound where appearance can compensate for weak geometry.",
            range_hint="0-1",
        ),
    )
    reid_group.add_argument(
        "--reid-iou-high",
        type=float,
        default=0.60,
        help=_help(
            "High IoU bound where geometry alone is usually sufficient.",
            range_hint="0-1",
            edge="keep above reid-iou-low",
        ),
    )
    reid_group.add_argument(
        "--reid-weight",
        type=float,
        default=0.80,
        help=_help(
            "Blend weight for appearance in combined matching cost.",
            range_hint="0-1",
            edge="near 1 lets embeddings dominate geometry",
        ),
    )
    reid_group.add_argument(
        "--reid-crop-mode",
        choices=("tight", "square", "square_mean"),
        default="tight",
        help="Crop geometry for embedding extraction.",
    )
    reid_group.add_argument(
        "--reid-crop-padding",
        type=float,
        default=0.0,
        help=_help(
            "Extra crop padding ratio for embeddings.", range_hint=">=0, usually 0-0.3"
        ),
    )
    reid_group.add_argument(
        "--reid-crop-layout",
        choices=("full", "parts"),
        default="full",
        help="Embedding crop layout.",
    )

    semantic_group = parser.add_argument_group("Semantic relink and appearance memory")
    semantic_group.description = (
        "Higher-level relinking and appearance-bank logic for recovering IDs after gaps, death/birth "
        "events, and ambiguous crowded matches."
    )
    semantic_group.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.91,
        help=_help("Similarity gate for semantic relinking.", range_hint="0-1"),
    )
    semantic_group.add_argument(
        "--semantic-ttl",
        type=int,
        default=45,
        help=_help(
            "Frames to keep lost tracks in semantic memory.",
            range_hint=">=1",
            edge="larger values help long occlusion but raise stale-ID risk",
        ),
    )
    semantic_group.add_argument(
        "--semantic-ema",
        type=float,
        default=0.83,
        help=_help("EMA decay for semantic appearance updates.", range_hint="0-1"),
    )
    semantic_group.add_argument(
        "--semantic-spatial-gate",
        type=float,
        default=0.20,
        help=_help(
            "Spatial gate for semantic relink candidates.",
            range_hint=">=0",
            edge="too small blocks cross-lane recovery",
        ),
    )
    semantic_group.add_argument(
        "--semantic-min-lost-frames",
        type=int,
        default=2,
        help=_help(
            "Track must be lost this many frames before semantic relink is considered.",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-min-iou",
        type=float,
        default=0.20,
        help=_help("Minimum IoU for semantic relink acceptance.", range_hint="0-1"),
    )
    semantic_group.add_argument(
        "--semantic-mahalanobis-threshold",
        type=float,
        default=0.0,
        help=_help(
            "Optional motion gate for semantic relink; 0 disables.", range_hint=">=0"
        ),
    )
    semantic_group.add_argument(
        "--semantic-debug",
        action="store_true",
        help="Print extra semantic relink diagnostics.",
    )
    semantic_group.add_argument(
        "--semantic-buffer-size",
        type=int,
        default=10,
        help=_help("Appearance vectors kept for semantic reference.", range_hint=">=1"),
    )
    semantic_group.add_argument(
        "--semantic-min-consistency",
        type=float,
        default=0.0,
        help=_help(
            "Minimum internal consistency for semantic references.", range_hint="0-1"
        ),
    )
    semantic_group.add_argument(
        "--semantic-rerank-mode",
        choices=("mean", "max", "top2_mean", "weighted"),
        default="mean",
        help="How to aggregate multiple reference embeddings.",
    )
    semantic_group.add_argument(
        "--semantic-reciprocal-margin",
        type=float,
        default=0.0,
        help=_help(
            "Reject relink if top-1 and top-2 similarities are too close; 0 disables.",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-iou-weight",
        type=float,
        default=0.0,
        help=_help(
            "MOT17-a: blend IoU into joint ranking score (0 = pure cosine).",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-mahalanobis-weight",
        type=float,
        default=0.0,
        help=_help(
            "MOT17-a: blend normalised motion evidence into joint ranking score (0 = off).",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-dynamic-margin-crowd",
        type=float,
        default=0.0,
        help=_help(
            "MOT17-a: add this × crowd_factor to reciprocal margin (crowd_factor = min(1, (n_competitors-1)/8)).",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-dynamic-margin-age",
        type=float,
        default=0.0,
        help=_help(
            "MOT17-a: add this × (lost_frames/ttl) to reciprocal margin; penalises older lost tracks.",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-w-sim-base",
        type=float,
        default=0.80,
        help=_help(
            "A1 Unified Score: Base weight for appearance similarity.", range_hint="0-1"
        ),
    )
    semantic_group.add_argument(
        "--semantic-w-iou-base",
        type=float,
        default=0.34,
        help=_help("A1 Unified Score: Base weight for spatial IoU.", range_hint="0-1"),
    )
    semantic_group.add_argument(
        "--semantic-w-maha-base",
        type=float,
        default=0.31,
        help=_help(
            "A1 Unified Score: Base weight for Mahalanobis score.", range_hint="0-1"
        ),
    )
    semantic_group.add_argument(
        "--semantic-shift-ambiguity",
        type=float,
        default=0.34,
        help=_help(
            "A1 Unified Score: Shift weight from IoU to Sim under crowd ambiguity.",
            range_hint="0-1",
        ),
    )
    semantic_group.add_argument(
        "--semantic-shift-lost-age",
        type=float,
        default=0.18,
        help=_help(
            "A1 Unified Score: Shift weight from IoU to Sim as lost age increases.",
            range_hint="0-1",
        ),
    )
    semantic_group.add_argument(
        "--force-python-relinker",
        action="store_true",
        default=False,
        help="Force Python relinker path (confound control: same algorithm as C++, different impl).",
    )
    semantic_group.add_argument(
        "--semantic-bank-inject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject TrackAppearanceBank representative when a track dies.",
    )
    semantic_group.add_argument(
        "--appearance-bank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Top-K appearance bank for primary association (Phase 1).",
    )
    semantic_group.add_argument(
        "--appearance-bank-size",
        type=int,
        default=5,
        help=_help(
            "Top-K samples kept per track in the appearance bank.", range_hint=">=1"
        ),
    )
    semantic_group.add_argument(
        "--appearance-bank-min-score",
        type=float,
        default=0.45,
        help=_help("Minimum detection score to admit a bank sample.", range_hint="0-1"),
    )
    semantic_group.add_argument(
        "--appearance-bank-min-iou",
        type=float,
        default=0.35,
        help=_help(
            "Minimum track/detection IoU to admit a bank sample.", range_hint="0-1"
        ),
    )
    semantic_group.add_argument(
        "--appearance-bank-consistency-threshold",
        type=float,
        default=0.75,
        help=_help(
            "Minimum pairwise cosine mean to trust the bank for relink.",
            range_hint="0-1",
        ),
    )
    semantic_group.add_argument(
        "--appearance-bank-high-quality-min-score",
        type=float,
        default=0.75,
        help=_help(
            "Minimum detection score for a bank sample to qualify as high-quality "
            "(used for reference injection gate, Phase 3).",
            range_hint="0-1",
        ),
    )
    semantic_group.add_argument(
        "--appearance-bank-min-aspect",
        type=float,
        default=1.2,
        help=_help(
            "Minimum h/w aspect ratio for a high-quality bank sample (person tracking).",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--appearance-bank-max-aspect",
        type=float,
        default=4.5,
        help=_help(
            "Maximum h/w aspect ratio for a high-quality bank sample (person tracking).",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--bank-quality-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(A6) Use composite quality score (det+iou+aspect+center+area) for bank sample "
        "ranking instead of the legacy 0.5*det+0.3*iou formula.",
    )
    semantic_group.add_argument(
        "--bank-quality-w-det",
        type=float,
        default=0.45,
        help="(A6) Weight for detection score in composite quality score.",
    )
    semantic_group.add_argument(
        "--bank-quality-w-iou",
        type=float,
        default=0.20,
        help="(A6) Weight for IoU in composite quality score.",
    )
    semantic_group.add_argument(
        "--bank-quality-w-aspect",
        type=float,
        default=0.15,
        help="(A6) Weight for aspect ratio in composite quality score.",
    )
    semantic_group.add_argument(
        "--bank-quality-w-center",
        type=float,
        default=0.10,
        help="(A6) Weight for center bias (truncation) in composite quality score.",
    )
    semantic_group.add_argument(
        "--bank-quality-w-area",
        type=float,
        default=0.10,
        help="(A6) Weight for area ratio in composite quality score.",
    )
    semantic_group.add_argument(
        "--semantic-clean-score-threshold",
        type=float,
        default=0.65,
        help=_help(
            "Detection score below which current observation is treated as low-quality "
            "(triggers strict_sim_threshold gate, Phase 3).",
            range_hint="0-1",
        ),
    )
    semantic_group.add_argument(
        "--semantic-clean-min-aspect",
        type=float,
        default=1.2,
        help=_help(
            "Minimum h/w aspect ratio for current observation to be treated as clean.",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-clean-max-aspect",
        type=float,
        default=4.5,
        help=_help(
            "Maximum h/w aspect ratio for current observation to be treated as clean.",
            range_hint=">=0",
        ),
    )
    semantic_group.add_argument(
        "--semantic-clean-margin-ratio",
        type=float,
        default=0.0,
        help=_help(
            "Frame-edge margin: observations whose bounding box touches within this "
            "fraction of frame width/height from the edge are treated as low-quality "
            "(triggers strict_sim_threshold gate). 0=disabled.",
            range_hint="0-0.15",
        ),
    )
    semantic_group.add_argument(
        "--semantic-strict-sim-threshold",
        type=float,
        default=0.0,
        help=_help(
            "Similarity threshold for low-quality observations (0=use sim_threshold). "
            "Set higher than semantic-threshold to enforce strict false-accept filter.",
            range_hint="0-1",
        ),
    )

    trigger_group = parser.add_argument_group("Dynamic ReID trigger policy")
    trigger_group.description = (
        "Controls when appearance extraction fires. These are throughput-vs-recovery knobs: "
        "more frequent triggers recover IDs better but cost more compute."
    )
    trigger_group.add_argument(
        "--need-reid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dynamic need_reid_frame() triggering instead of a fixed heartbeat.",
    )
    trigger_group.add_argument(
        "--reid-history-size",
        type=int,
        default=5,
        help=_help("BBox history length for trigger state.", range_hint=">=1"),
    )
    trigger_group.add_argument(
        "--reid-trigger-mode",
        choices=(
            "count_jump",
            "event_any",
            "event_persist",
            "event_strict",
            "event_memory",
            "score_ema",
            "score_ema_geom",
            "score_ema_conf",
        ),
        default="score_ema",
        help="Dynamic ReID trigger logic.",
    )
    trigger_group.add_argument(
        "--reid-long-memory-decay",
        type=float,
        default=0.80,
        help=_help("EMA decay for long-memory trigger state.", range_hint="0-1"),
    )
    trigger_group.add_argument(
        "--reid-long-memory-trigger",
        type=float,
        default=1.25,
        help=_help("Trigger threshold for long-memory mode.", range_hint=">0"),
    )
    trigger_group.add_argument(
        "--reid-score-decay",
        type=float,
        default=0.80,
        help=_help("EMA decay for weighted trigger signals.", range_hint="0-1"),
    )
    trigger_group.add_argument(
        "--reid-score-threshold",
        type=float,
        default=2.0,
        help=_help(
            "Fire trigger when weighted score crosses this threshold.", range_hint=">0"
        ),
    )
    trigger_group.add_argument(
        "--reid-score-threshold-low",
        type=float,
        default=2.0,
        help=_help(
            "Lower hysteresis threshold for staying active; unset reuses high threshold.",
            range_hint=">=0 or unset",
        ),
    )
    trigger_group.add_argument(
        "--reid-weight-new",
        type=float,
        default=1.0,
        help=_help("Weight for new-track trigger signal.", range_hint=">=0"),
    )
    trigger_group.add_argument(
        "--reid-weight-lost",
        type=float,
        default=1.4,
        help=_help("Weight for lost-track trigger signal.", range_hint=">=0"),
    )
    trigger_group.add_argument(
        "--reid-weight-geom",
        type=float,
        default=0.5,
        help=_help("Weight for geometry-instability trigger signal.", range_hint=">=0"),
    )
    trigger_group.add_argument(
        "--reid-weight-conf",
        type=float,
        default=0.5,
        help=_help("Weight for confidence-jitter trigger signal.", range_hint=">=0"),
    )
    trigger_group.add_argument(
        "--reid-birth-death-boost",
        type=float,
        default=1.0,
        help=_help(
            "Extra score when birth and death happen together.", range_hint=">=0"
        ),
    )
    trigger_group.add_argument(
        "--reid-lost-age-cap",
        type=int,
        default=30,
        help=_help("Saturating age cap for lost-track weighting.", range_hint=">=1"),
    )
    trigger_group.add_argument(
        "--reid-unstable-shift-weight",
        type=float,
        default=1.0,
        help=_help("Shift contribution weight in instability score.", range_hint=">=0"),
    )
    trigger_group.add_argument(
        "--reid-unstable-iou-weight",
        type=float,
        default=1.0,
        help=_help(
            "IoU-drop contribution weight in instability score.", range_hint=">=0"
        ),
    )
    trigger_group.add_argument(
        "--reid-conf-jitter-gate",
        type=float,
        default=0.10,
        help=_help("Deadzone for per-track confidence jitter.", range_hint=">=0"),
    )
    trigger_group.add_argument(
        "--reid-trigger-persist-frames",
        type=int,
        default=2,
        help=_help(
            "Frames score must stay high before trigger fires.", range_hint=">=1"
        ),
    )
    trigger_group.add_argument(
        "--reid-cooldown-frames",
        type=int,
        default=4,
        help=_help("Frames to suppress ReID after each trigger.", range_hint=">=0"),
    )
    trigger_group.add_argument(
        "--reid-birth-death-lost-min",
        type=float,
        default=0.1,
        help=_help(
            "Minimum lost-signal before birth/death boost applies.", range_hint=">=0"
        ),
    )
    trigger_group.add_argument(
        "--reid-geom-smooth-window",
        type=int,
        default=1,
        help=_help(
            "Moving-average window before geometry-instability scoring.",
            range_hint=">=1",
        ),
    )

    merge_group = parser.add_argument_group("Lifecycle merge and tracklet cleanup")
    merge_group.description = (
        "Late-stage identity stitching and output pruning. These knobs act after core association, "
        "so they trade IDF1 cleanup against accidental over-merge."
    )
    merge_group.add_argument(
        "--lifecycle-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable pre-output lifecycle merge.",
    )
    merge_group.add_argument(
        "--lifecycle-ttl",
        type=int,
        default=45,
        help=_help("Frames to keep dead tracks mergeable.", range_hint=">=1"),
    )
    merge_group.add_argument(
        "--lifecycle-min-gap",
        type=int,
        default=2,
        help=_help(
            "Minimum dead/alive gap before lifecycle merge is considered.",
            range_hint=">=0",
        ),
    )
    merge_group.add_argument(
        "--lifecycle-spatial-gate",
        type=float,
        default=0.08,
        help=_help("Spatial gate for lifecycle merge candidates.", range_hint=">=0"),
    )
    merge_group.add_argument(
        "--lifecycle-min-iou",
        type=float,
        default=0.0,
        help=_help("Minimum IoU for lifecycle merge.", range_hint="0-1"),
    )
    merge_group.add_argument(
        "--lifecycle-sim-threshold",
        type=float,
        default=0.90,
        help=_help(
            "Appearance similarity threshold for lifecycle merge.", range_hint="0-1"
        ),
    )
    merge_group.add_argument(
        "--lifecycle-require-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require appearance embedding to allow lifecycle merge.",
    )
    merge_group.add_argument(
        "--lifecycle-ema",
        type=float,
        default=0.83,
        help=_help("EMA decay for lifecycle merge appearance state.", range_hint="0-1"),
    )
    merge_group.add_argument(
        "--post-lifecycle-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable post-output lifecycle merge.",
    )
    merge_group.add_argument(
        "--post-lifecycle-ttl",
        type=int,
        default=60,
        help=_help("Frames to keep tracklets for post merge.", range_hint=">=1"),
    )
    merge_group.add_argument(
        "--post-lifecycle-min-gap",
        type=int,
        default=1,
        help=_help("Minimum gap before post merge.", range_hint=">=0"),
    )
    merge_group.add_argument(
        "--post-lifecycle-velocity-samples",
        type=int,
        default=5,
        help=_help("Samples used to estimate post-merge motion.", range_hint=">=1"),
    )
    merge_group.add_argument(
        "--post-lifecycle-spatial-weight",
        type=float,
        default=0.35,
        help=_help("Spatial term weight in post-merge cost.", range_hint=">=0"),
    )
    merge_group.add_argument(
        "--post-lifecycle-motion-weight",
        type=float,
        default=0.45,
        help=_help("Motion term weight in post-merge cost.", range_hint=">=0"),
    )
    merge_group.add_argument(
        "--post-lifecycle-time-weight",
        type=float,
        default=0.10,
        help=_help("Time-gap term weight in post-merge cost.", range_hint=">=0"),
    )
    merge_group.add_argument(
        "--post-lifecycle-direction-weight",
        type=float,
        default=0.25,
        help=_help(
            "Direction-consistency weight in post-merge cost.", range_hint=">=0"
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-max-cost",
        type=float,
        default=1.25,
        help=_help(
            "Maximum total post-merge cost to accept a stitch.",
            range_hint=">0",
            edge="lower is safer but leaves fragments",
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-appearance-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require appearance gate during post merge.",
    )
    merge_group.add_argument(
        "--post-lifecycle-appearance-threshold",
        type=float,
        default=0.90,
        help=_help("Appearance threshold for post merge.", range_hint="0-1"),
    )
    merge_group.add_argument(
        "--post-lifecycle-appearance-min-samples",
        type=int,
        default=1,
        help=_help(
            "Minimum appearance samples before post-merge gating activates.",
            range_hint=">=1",
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-appearance-max-samples",
        type=int,
        default=5,
        help=_help(
            "Maximum appearance samples used in post-merge gating.", range_hint=">=1"
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-appearance-min-score",
        type=float,
        default=0.0,
        help=_help(
            "Minimum sample score for post-merge appearance gating.", range_hint="0-1"
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-appearance-min-consistency",
        type=float,
        default=0.0,
        help=_help(
            "Minimum consistency for post-merge appearance gating.", range_hint="0-1"
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-appearance-weight",
        type=float,
        default=0.0,
        help=_help(
            "(A5) Add appearance cost (1-sim) as a soft term in post-merge cost. "
            "0 disables. Combined with gate for full quality control.",
            range_hint="0-2",
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-gap-uncertainty-weight",
        type=float,
        default=0.0,
        help=_help(
            "(A5) Scale appearance weight by (1 + k * gap/ttl). "
            "Longer gaps require stronger appearance match.",
            range_hint="0-2",
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-consistency-weight",
        type=float,
        default=0.0,
        help=_help(
            "(A5) Penalise low-consistency tracklets in post-merge cost.",
            range_hint="0-1",
        ),
    )
    merge_group.add_argument(
        "--post-lifecycle-missing-appearance-cost",
        type=float,
        default=0.5,
        help=_help(
            "(A5) Appearance cost assigned when embedding is unavailable.",
            range_hint="0-1",
        ),
    )
    merge_group.add_argument(
        "--min-tracklet-len",
        type=int,
        default=1,
        help=_help("Drop tracklets shorter than this length.", range_hint=">=1"),
    )
    merge_group.add_argument(
        "--min-tracklet-score",
        type=float,
        default=0.0,
        help=_help("Drop tracklets below this mean detection score.", range_hint="0-1"),
    )

    debug_group = parser.add_argument_group("Profiling and diagnostics")
    debug_group.description = "Instrumentation flags. These should not change tracker output semantics unless the underlying profiled path is conditional."
    debug_group.add_argument(
        "--profile-stages", action="store_true", help="Profile stage-level runtime."
    )

    return parser
