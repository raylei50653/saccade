# ADR 011: FastTracker Reference Adaptation Plan

## Status
Accepted (Phase 1-2 implemented)

## Context
Saccade already runs a hybrid production pipeline (`Python orchestration + C++/CUDA core`) and has a BoT-SORT/ByteTrack-style tracker foundation.
We want to reference FastTracker ideas without replacing the whole stack.

FastTracker methods we intend to borrow:
- Occlusion-aware identity recovery.
- Class-aware motion modeling.
- Scene-constraint-based trajectory refinement (optional, domain-specific).

## Decision
Adopt a phased, low-risk adaptation strategy:

1. Phase 1 (now): class-aware motion + class-consistent association in C++ tracker core.
2. Phase 2: occlusion-state handling and explicit reactivation policies.
3. Phase 3: optional road-structure constraints (only for traffic deployments).

We explicitly avoid full architecture replacement.

## Phase 1 Scope (Implemented)
- Add class-aware motion profiles in `src/tracking/tracker_gpu.cu`.
- Enforce class-consistent matching in association to reduce cross-class ID switches.
- Preserve tracked class id in output (`TrackResult.class_id`) instead of fixed placeholder value.
- Replace fixed lost timeout with per-track TTL (derived from class profile).

## Phase 2 Scope (Implemented)
- Add explicit `occluded` state in tracker lifecycle (tracked/lost/occluded).
- Add staged reactivation order:
  - tracked pool first,
  - then lost pool (stricter threshold),
  - then occluded pool (strictest threshold).
- Add overlap-based occlusion candidate marking for unmatched tracked targets.
- Add occlusion grace window before final removal.

## Why this path
- Keeps current production architecture stable.
- Targets known MOT failure modes (ID switches under multi-class scenes).
- Easy to benchmark with current MOT17/real-stream scripts.

## Validation Plan
- Run MOT17 evaluation before/after:
  - `MOTA`, `IDF1`, `HOTA`, `ID Sw.`
- Run one real-stream regression:
  - verify no FPS regression beyond tolerance.
  - verify class-id continuity in logs.

## Non-goals
- No scene-semantic lane/crosswalk module in this phase.
- No rewrite of orchestrator/media ingestion path.
