# Tracker Decision Research

This directory documents the **tracker decision-layer semantics** of the production path: geometric, motion, lifecycle, and association signals that drive matching, continuation, rejection, relink, and ID preservation.

It is **cross-module** by design — not a mirror of `scripts/eval/config/geometry.py` or `docs/modules/geometry/`. Knobs may live under geometry, lifecycle, detection (private pool), or core config; what matters is that they change association / identity decisions.

This is **not** the place for full runtime pipeline maps, detector engine paths, CUDA graph scheduling, or benchmark runbooks. Those belong under `docs/research/pipeline/`.

---

## Scope

`docs/research/tracker-decision/` is responsible for documenting:

* Geometry and motion signals used by the tracker
* Association scoring semantics
* Gate / score / weight / normalization separation
* Kalman and GMC assumptions that affect matching
* Relink and handover rules
* Occlusion recovery behavior
* Geometry-related lifecycle policy
* Native bridge behavior for geometry config fields
* Failure modes and audit notes for geometry-driven tracking

In short:

> Pipeline docs explain **which path runs**.
> Tracker-decision docs explain **why the tracker makes a decision on that path**.

---

## Non-scope

This directory should not become a dumping ground for general benchmark or runtime documentation.

Do **not** put the following here unless they are directly needed to explain geometry decision semantics:

* Full CLI runbooks
* Detector weight / engine path inventories
* CUDA graph layer descriptions
* Double-buffer scheduling details
* MLflow / output directory documentation
* Full MOT17 benchmark summaries
* Training logs
* General pipeline execution maps
* ReID implementation notes, unless they are used to compare against geometry-only relink behavior

Those should live in more appropriate locations such as:

```text
docs/research/pipeline/
docs/research/training/
docs/research/reid/
docs/archive/
```

---

## Recommended Files

A minimal useful structure is:

```text
docs/research/tracker-decision/
├── README.md
├── assoc_knobs.md
├── scoring_semantics.md
├── relink_bridge.md
├── kalman_gmc_motion.md
├── failure_modes.md
└── audit/
    ├── config_surface.md
    ├── callpoints.md
    └── native_bridge.md
```

---

## File Responsibilities

### `assoc_knobs.md`

Inventory of geometry / association / lifecycle knobs that affect tracker decisions.

Each knob should document:

* Config field name
* Source file
* Preset value, if relevant
* Runtime callpoint
* Whether it acts as a gate, score term, normalization, policy weight, or lifecycle rule
* Expected effect when increased or decreased
* Known risks
* Whether it reaches Python-only path, native path, or both

Example categories:

```text
match_thresh
multiplicative_cost
sinkhorn_lambda
stability_cost_w
oao_*
relink_bridge_*
kalman_r_scale
gmc_downscale
confirm_score_thresh
private_continuation_*
```

---

### `scoring_semantics.md`

Explains what the association score means.

This file should separate heterogeneous concepts that may currently be mixed inside one score matrix:

```text
measurement signal
normalization / calibration
policy weight
decision threshold
lifecycle rule
```

The goal is to make clear which parts are actual geometric evidence and which parts are policy choices.

This file should answer:

* What is being measured?
* What is being normalized?
* What is being weighted?
* What is only a gate?
* What can change identity decisions?
* Which terms are redundant or overlapping?
* Which terms should eventually be refactored apart?

---

### `relink_bridge.md`

Documents geometry-only relink and handover behavior.

This file should explain:

* What a relink bridge is
* When a dead track may reconnect
* Which gates are relaxed during relink
* Which gates remain strict
* How height ratio, center distance, velocity, and recent history affect relink
* How geometry relink differs from ReID-based relink
* Known false-relink and missed-relink cases

This is the right place for notes about fields such as:

```text
relink_bridge_px
relink_bridge_h_lo
relink_bridge_h_hi
relink_mode
occlusion / lost-track recovery policy
```

---

### `kalman_gmc_motion.md`

Documents the motion model assumptions.

This file should cover:

* Kalman state meaning
* Measurement noise assumptions
* `kalman_r_scale` effects
* GMC role and limitations
* GMC downscale tradeoffs
* How motion prediction interacts with geometry matching
* Failure cases caused by camera motion, jitter, or bad compensation

This file should avoid becoming a generic Kalman tutorial. It should stay tied to the tracker’s actual decision path.

---

### `failure_modes.md`

Tracks known geometry-side failure modes.

Recommended format:

```text
## Failure mode name

### Symptom

### Likely cause

### Related knobs

### Probe / evidence

### Current status

### Possible fix
```

Useful examples:

```text
near-person crossing
long occlusion recovery
box jitter amplification
private continuation false keepalive
late birth from low-score true positive
camera motion under-compensation
wrong relink after disappearance
height-ratio gate too strict / too loose
```

---

## Audit Subdirectory

### `audit/config_surface.md`

Documents the decision-layer config surface (cross-module).

This file should group fields by subdomain rather than by file location.

Example groups:

```text
kalman + motion
geometry scale
association scoring
id stability
crowd geometry
occlusion
OAO
Sinkhorn / multi-assignment
person geometry
native bridge fields
```

The purpose is to answer:

> How large is the geometry parameter surface, and which parts are active, inactive, redundant, or risky?

---

### `audit/callpoints.md`

Maps config fields to actual usage sites.

Recommended format:

```text
knob → config schema → preset override → Python callpoint → native bridge → runtime effect
```

This helps catch:

* Parameters that exist but are not used
* Parameters that only affect Python path
* Parameters that silently affect native path
* Parameters with ambiguous names
* Parameters whose effect differs from their documentation

---

### `audit/native_bridge.md`

Documents Python → C++ / CUDA config bridging for decision-layer fields.

This file should include:

* Which geometry fields cross into native code
* How they are packed or translated
* Whether names are exact or substring-matched
* Known collision risks
* Whether Python and native paths are behaviorally equivalent
* Required tests before changing bridge fields

---

## Relationship to Pipeline Docs

A pipeline runbook may mention geometry knobs, but it should not define their semantics.

For example, a pipeline document may say:

```text
preset mamba_whole_graph_m sets relink_bridge_px=0.4 and h ratio gate to [0.6, 1.7]
```

But the meaning of those fields belongs here:

```text
docs/research/tracker-decision/relink_bridge.md
```

Similarly, a pipeline document may say:

```text
the run uses tracker CUDA graph
```

But the meaning of the tracker’s association policy belongs here:

```text
docs/research/tracker-decision/scoring_semantics.md
```

---

## Writing Rules

Tracker-decision documents should prefer decision semantics over implementation trivia.

Each document should make clear:

```text
What decision is being made?
What signal supports the decision?
What policy modifies the signal?
What knob controls the behavior?
Where is the knob used?
What failure mode does it address?
What failure mode can it introduce?
```

Avoid only writing:

```text
parameter X = value Y
```

Prefer:

```text
parameter X controls gate Y at decision point Z.
Increasing it makes association more permissive, which can recover more occlusions but may increase false relinks.
```

---

## Document Index (status)

| File | Status | Role |
|:--|:--|:--|
| [audit/config_surface.md](audit/config_surface.md) | **done** | Cross-module decision surface + ACTIVE/LATENT/NO-GO |
| [audit/callpoints.md](audit/callpoints.md) | **done** | schema → preset → inject → native → effect |
| [audit/native_bridge.md](audit/native_bridge.md) | **done** | Python↔CUDA setters, remaps, packing risks |
| [audit/math_model_drift_2026-07-09.md](audit/math_model_drift_2026-07-09.md) | **done** | P3 static drift check vs math_model.md |
| [audit/dual_stability_cleanup.md](audit/dual_stability_cleanup.md) | **done (RFC)** | P4-1 dual stability architecture A/B/C |
| [audit/dual_stability_ablation_protocol.md](audit/dual_stability_ablation_protocol.md) | **done (P7 protocol)** | 4-way matrix A–D; smoke → 04 → 7-seq; no default flip |
| [audit/no_go_guardrails.md](audit/no_go_guardrails.md) | **done (RFC)** | P4-2 NO-GO/LATENT promotion + future validator rules |
| [audit/active_contract_healthcheck.md](audit/active_contract_healthcheck.md) | **done** | C1–C8 checklist + P5 script link |
| [scripts/tools/check_headline_decision_contract.py](../../../scripts/tools/check_headline_decision_contract.py) | **done (P5+P5.1)** | YAML contract + inject-map C8 (CI) |
| [assoc_knobs.md](assoc_knobs.md) | **done** | Knob cards (gate / score / weight / lifecycle) |
| [scoring_semantics.md](scoring_semantics.md) | **done** | What association cost means on headline path |
| [relink_bridge.md](relink_bridge.md) | **done** | Geometry-only bridge reconnect semantics |
| [kalman_gmc_motion.md](kalman_gmc_motion.md) | **done** | Motion model assumptions for matching |
| [failure_modes.md](failure_modes.md) | **done** | Geometry-side failure catalog |

Baseline for ACTIVE labels: `mamba_whole_graph` / `mamba_whole_graph_m`, `reid_mode=off`.

Canonical math companion (not replaced by this tree): [`docs/reference/math_model.md`](../../reference/math_model.md)
（source audit **2026-07-09**，已對齊本 tree 的 active contract）。

---

## Current Active Contract

Locked facts from the static decision audit. **Do not “rediscover” these ad hoc** — update this section when presets or inject paths change.

```text
1. ACTIVE 決策面很小
   Match/birth/confirm + multiplicative cost + OAO duration-ramp + occ_state
   + bridge relink + private continuation + kalman_r_scale/GMC.
   Schema still carries a large LATENT/NO-GO surface; only the ACTIVE subset
   defines production identity policy.

2. s/m association 主閾值相同
   match_thresh, new_track_thresh, confirm_*, oao_*, multiplicative_cost,
   sinkhorn_lambda, stability_cost_w, private_continuation_* are shared.

3. m 的決策差異主要在 motion + bridge gates
   kalman_r_scale: s=2.8 / m=3.5
   relink_bridge_px + h_lo/h_hi: m looser (small-box recovery)
   relink_bridge_dir_bonus: s=0.8 / m=0.0 (explicit in presets)

4. private continuation 改的是 association 輸入 det set
   Not a GPUByteTracker setter. Score-clamped below new_track_thresh so
   candidates may CONTINUE tracks but must not BIRTH ghosts.

5. occ_state_* is production-ACTIVE
   Schema defaults (enabled + iou/foot/ttl/cost_weight) are part of the
   headline path; presets now write them explicitly so silence ≠ accidental off.

6. Pipeline vs decision split
   docs/research/pipeline/*  → which path runs
   docs/research/tracker-decision/* → why match/birth/keep/relink
```

Cross-module path (not geometry-only):

```text
detector input set → association cost → lifecycle (birth/confirm/lost)
  → relink bridge → native kernels
```

---

## Current Priority

Guardrails (P0–P6) closed. Next is **P7 dual-stability ablation** (measure first):

```text
audit/dual_stability_ablation_protocol.md  # P7 4-way matrix A–D; no default flip
audit/dual_stability_cleanup.md            # architecture options after results
audit/no_go_guardrails.md
audit/active_contract_healthcheck.md       # C1–C9
scripts/tools/check_headline_decision_contract.py
.github/pull_request_template.md
```

Still keep fresh when presets or cost kernels change:

```text
audit/config_surface.md
scoring_semantics.md
assoc_knobs.md
```

### Hygiene (done / do next)

| Item | Status |
|:--|:--|
| Explicit `occ_state_*` in headline presets | **done** (behavior-preserving) |
| Explicit `relink_bridge_dir_bonus` on m (`0.0`) | **done** (behavior-preserving; documents s≠m) |
| NO-GO / LATENT prohibition table | [audit/config_surface.md](audit/config_surface.md) |
| Dual stability architecture debt | **P7 protocol** — [audit/dual_stability_ablation_protocol.md](audit/dual_stability_ablation_protocol.md); cleanup options [dual_stability_cleanup.md](audit/dual_stability_cleanup.md) |
| NO-GO guardrail process | **RFC** — [audit/no_go_guardrails.md](audit/no_go_guardrails.md) |
| Active-path healthcheck | **done** — [audit/active_contract_healthcheck.md](audit/active_contract_healthcheck.md) |
| `math_model.md` drift check + align | **done** (PR #62 audit, PR #63 align) |
| P5: preset validator / CI healthcheck script | **done** — YAML C1–C7 |
| P5.1: inject-map C8 in same checker | **done** — `pipeline.py` setters + private det-set |
| P6: surface hardening | **done** — C9 allowlist/forbid, help `[NO-GO]`/`[LATENT]`, PR template |
| P7: dual stability 4-way ablation | **protocol** — [audit/dual_stability_ablation_protocol.md](audit/dual_stability_ablation_protocol.md); results PR later |

Open maintainability questions:

1. Large LATENT/NO-GO surface — **guarded** (C9 + help + promotion bar); schema delete deferred.
2. Dual height-stability — **P7 measure** (matrix A–D) before any default flip.

The main open question is not just whether the tracker is accurate, but whether the decision layer is understandable, auditable, and maintainable.
