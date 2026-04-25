# ADR 012: FastTracker Selective Adaptation (Phase 2 Execution)

## Status
Accepted

## Context
Current MOT17 validation in this repo shows:
- ReID disabled (`--no-reid`) causes severe identity fragmentation (`IDs` explosion).
- SigLIP preprocessing/normalization fixes recover large portions of `MOTA/IDF1`.
- Additional spatial-fused SigLIP embedding gives no clear gain under current thresholds.

This indicates the next bottleneck is tracker policy (association lifecycle and gating), not full backbone replacement.

## Decision
Adopt FastTracker ideas in a selective, implementation-first sequence, without replacing the existing C++/CUDA pipeline:

1. Add `init_iou_suppress` on new track initialization.
2. Strengthen occlusion-aware lifecycle (occluded-specific TTL and reactivation rules).
3. Refine class-aware motion/gating table by class profile and motion bounds.
4. Keep ROI/direction constraints optional and deploy-scene specific (not default MOT17 path).

We explicitly do **not** replace SigLIP or rewrite the tracker architecture at this stage.

## Scope & Implementation Order

### Step A (highest priority): New-track duplicate suppression
- Add IoU suppression before creating newborn tracks.
- If unmatched detection overlaps an active track above threshold, do not initialize a new ID.
- Target file: `src/tracking/tracker_gpu.cu`.

### Step B: Occlusion-state policy hardening
- Split removal policy by state (`tracked/lost/occluded`) with separate frame budgets.
- Add conservative recovery ordering and stricter confidence for occluded reactivation.
- Keep current staged association, tune thresholds with MOT17 sweep.

### Step C: Class-aware motion/gating retune
- Keep per-class KF noise profiles.
- Add per-class velocity/size-change gate to reject implausible associations.

### Step D (optional): Scene constraints
- ROI/lane/direction constraints only for fixed-camera traffic deployments.
- Not enabled in generic MOT17 benchmark pipeline.

## Validation Plan (must-pass)
- Dataset: MOT17 (same detector engine and same command template).
- Compare to latest C++ baseline with SigLIP preprocessing fix.
- Primary KPIs:
  - `IDs` must decrease.
  - `IDF1` must increase or stay stable with meaningful `IDs` reduction.
  - `MOTA` must not regress more than an agreed tolerance.
- Secondary checks:
  - No major FPS regression in eval runtime.
  - No class-id continuity regressions in output logs.

## Rollback Criteria
- Revert step if any change causes:
  - clear `MOTA` drop without compensating `IDF1/IDs` improvement, or
  - substantial runtime regression.

## Non-goals
- No end-to-end migration to external FastTracker codebase.
- No detector/backbone swap in this ADR.
- No cross-camera/global ReID work in this phase.
