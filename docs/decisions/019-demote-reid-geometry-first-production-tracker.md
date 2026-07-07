# ADR 019: Demote ReID and Adopt Geometry-First Production Tracker

## Status

Accepted (2026-07-07)

## Context

The tracker was originally designed with ReID as the main identity recovery
direction. Many parts of the handover, dead-track, requery, crop, and candidate
recovery infrastructure were built around the assumption that ReID would provide
the strongest signal for occlusion recovery and ID switch reduction.

However, subsequent geometry-only work showed that most of the practical ReID
gains could be absorbed by explicit temporal, geometric, GMC-compensated, and
lifecycle policies.

The latest geometry-only result is:

```text
Overall throughput: 351.83 FPS
Mean latency: 5.67 ms

IDF1: 80.3%
MOTA: 81.9%
HOTA: 74.3%
DetA: 76.1%
AssA: 72.8%
IDs: 359
FP: 2067
FN: 17895
Recall: 84.1%
Precision: 97.9%
```

This matches the best ReID-assisted IDF1 while outperforming the ReID path in
throughput, latency, HOTA, AssA, MOTA, maintainability, and interpretability.

The main observation is that ReID started as a high-capacity identity signal,
but after the geometry path matured, its relative value degraded into a single
auxiliary signal with no standout production advantage. In complex crowded
scenes, ReID is also vulnerable to crop contamination, occluder contamination,
nearby-identity contamination, box jitter, and embedding-bank pollution.

In contrast, the geometry route is complex but explicit. Its failures can
usually be traced to lifecycle policy, motion prediction, GMC, candidate gating,
birth/confirm/death policy, or matching thresholds.

## Decision

The production tracker will adopt a geometry-first, ReID-free architecture.

The production path must not depend on:

```text
ReID embeddings
MNv4 inference
crop ring
crop stash
embedding bank
appearance score
k-reciprocal ReID score
requery extraction
borderline ReID requery
```

The production path may keep and further develop:

```text
lost-track state
short-occlusion recovery
handover candidates
geometry candidate scoring
GMC-compensated motion
candidate gating
birth / confirm / death policy
snapshot-score-apply architecture
determinism / shadow compare / profiling tools
```

ReID-related infrastructure is demoted to an experimental or archived extension.
It should remain available for reference and future research, but it must not be
initialized, imported, or executed by the production preset.

The production tracker is geometry-first and ReID-free. ReID is retained only as
an experimental/archive extension.

The architecture transition is:

```text
from:
  ReID-centered tracker with geometry support

to:
  Geometry-first tracker with archived ReID extension
```

## Rationale

ReID was not removed merely because of latency. It was demoted because it is now
Pareto-dominated by the geometry-only path in the current deployment regime.

The geometry-only path provides:

```text
same IDF1
better MOTA
better HOTA
better AssA
much higher FPS
lower latency
clearer failure modes
lower maintenance cost
no crop/embedding contamination risk
```

ReID remains a valid research direction for regimes such as long occlusion, low
FPS, cross-camera tracking, severe camera cuts, or weak detector/geometry
settings. However, those are not the current production target.

For this project, the production value of a signal is not determined only by raw
signal capacity. It must also be controllable, interpretable, stable, and
maintainable. Under that criterion, geometry-first wins.

## Consequences

### Positive

- Production path becomes simpler and faster.
- No ReID engine loading or crop-ring allocation.
- No embedding-bank maintenance.
- Fewer hidden failure modes from polluted appearance features.
- Better debugability and interpretability.
- Short-occlusion recovery remains explicit and geometry-driven.
- Mainline architecture aligns with the best current benchmark result.

### Negative

- The tracker gives up potential ReID benefits in long-occlusion or low-FPS
  regimes.
- Some existing code and naming are still ReID-oriented and need cleanup.
- ReID experiments will require explicit opt-in paths.
- Historical code may be confusing until the semantic split is complete.

### Neutral / Follow-up

- Some infrastructure originally created for ReID should not be deleted because
  it has become generic geometry recovery infrastructure.
- The cleanup must separate generic recovery logic from true appearance-specific
  logic.

## Alternatives Considered

### Keep ReID in production behind a flag

Rejected.

Even if disabled by default, ReID-related code in the production path increases
configuration complexity, import risk, maintenance cost, and conceptual
ambiguity.

### Delete all ReID-related infrastructure immediately

Rejected.

Some systems originally created for ReID are now valuable generic geometry
recovery infrastructure. Blind deletion risks removing the mechanisms that made
the geometry-only path strong.

### Keep ReID as a fallback for hard cases

Rejected for current production.

The hard cases where ReID is expected to help are also where crop contamination
and embedding pollution are most likely. A fallback path would reintroduce
latency, threshold tuning, and opaque error modes without proven aggregate gain.

## Migration Notes

The implementation should proceed as a cleanup and split, not as a blind
deletion. The main risk is accidentally removing generic recovery mechanisms
that were originally named or packaged as ReID infrastructure.

### Phase 0: Freeze Evidence

Create a document recording:

```text
best ReID-assisted metrics
best geometry-only metrics
reason for ReID demotion
known ReID failure modes
production rule: no appearance dependency
```

Suggested file:

```text
docs/archive/reid_demoted.md
```

### Phase 1: Inventory

Audit all symbols and files related to:

```text
reid
requery
crop
embedding
bank
mnv4
appearance
cheb
handover
lost track
dead track
recovery
```

Classify each item as:

```text
KEEP_GENERIC
ARCHIVE_REID
DELETE_OR_DEPRECATE
```

### Phase 2: Semantic Rename

Rename ReID-oriented generic concepts into geometry-first concepts.

Examples:

```text
requery_candidate   -> recovery_candidate
requery_band        -> handover_uncertainty_band / recovery_margin_band
dead_reid_entry     -> lost_track_entry
reid_context        -> recovery_context
score_handover_geom -> score_handover_geometry
requery_rescore     -> appearance_rescore
```

### Phase 3: Config Split

Production config should only expose geometry/recovery knobs.

ReID knobs should move to an experimental namespace and must not affect
production presets.

### Phase 4: Code Path Split

Keep generic recovery in the mainline:

```text
tracking/recovery/
tracking/handover/
tracking/lifecycle/
```

Move appearance-specific code to experimental/archive:

```text
tracking/experimental/reid/
```

### Phase 5: Evaluator and Binding Cleanup

Production evaluator must not automatically import or initialize ReID hooks.

ReID-specific hooks such as crop store, ring params, embedding bank, and
appearance rescoring should be isolated behind explicit experimental mode.

### Phase 6: Regression Guard

After the split, production preset must preserve the geometry-only result:

```text
IDF1: 80.3%
MOTA: 81.9%
HOTA: 74.3%
AssA: 72.8%
FPS: ~351.83
Latency: ~5.67 ms
```

Production grep guard should ensure no hot-path dependency on:

```text
mnv4
crop_store
embedding_bank
appearance_score
requery_extract
```

Allowed terms:

```text
handover
recovery
lost_track
geometry
lifecycle
```
