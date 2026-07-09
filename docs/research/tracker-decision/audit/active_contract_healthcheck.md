# Active Decision Contract — Healthcheck

**Status:** manual checklist + **P5 static script** (YAML only, no GPU)  
**Date:** 2026-07-09  
**Contract owner:** [../README.md](../README.md) § Current Active Contract  
**Sources of truth:**  
headline presets → schema defaults → `pipeline.py` inject → native  
(see [callpoints.md](callpoints.md), [native_bridge.md](native_bridge.md)).

**Automated checker (C1–C5, C7, C9 hard; C6 NOTE; C8 inject-map hard):**

```bash
uv run python scripts/tools/check_headline_decision_contract.py
# YAML only (skip C8):
uv run python scripts/tools/check_headline_decision_contract.py --skip-inject
```

CI: `contracts` job runs this script. Unit tests:
`tests/unit/test_headline_decision_contract.py`.

**Goal:** After any change to preset / schema / inject / decision docs, confirm
the **production decision contract** has not drifted — without running 7-seq
unless behavior actually changes.

---

## When to run

| Trigger | Mode |
|:--|:--|
| Edit `mamba_whole_graph*.yaml` | **required** before merge |
| Edit geometry / lifecycle / detection decision schema defaults | **required** |
| Edit `pipeline.py` tracker inject (`set_*`) | **required** |
| Edit cost / auction / occ kernels with policy comments | recommended |
| Docs-only under `tracker-decision/` | recommended (spot-check) |
| Behavior PR | this checklist **plus** smoke → MOT17-04-SDP → 7-seq |

**Docs-only / guardrail PRs:** static pass is enough.  
**Behavior PRs:** do not substitute this checklist for eval.

---

## How to run

```bash
# Automated (P5 + P5.1) — preferred first step on preset / inject PRs
uv run python scripts/tools/check_headline_decision_contract.py

# Manual supplements
diff -u configs/presets/mamba_whole_graph.yaml configs/presets/mamba_whole_graph_m.yaml

uv run python scripts/tools/check_doc_stale_paths.py
uv run python scripts/tools/check_doc_freshness.py   # warn-only
```

---

## Check matrix

Legend: **PASS** must hold for current contract. **NOTE** is awareness, not fail.
Values are the **2026-07-09** locked contract; update this table when the contract changes.

### C1 — `occ_state_*` production-on and explicit

| Check | Expected | Where |
|:--|:--|:--|
| `occ_state_enabled` | **true** (written in **both** s and m YAML) | presets |
| `occ_iou_thresh` / `occ_foot_gap` / `occ_ttl` / `occ_cost_weight` | **0.45 / 0.15 / 4 / 0.50** explicit | presets |
| Inject | `pipeline.py` → `set_occ_params(...)` | callpoints |
| math_model | baseline **on** (not “off if unset”) | `math_model.md` §7.6 |

**Fail if:** missing keys (silence), `enabled: false`, or inject skipped.

---

### C2 — s/m primary association thresholds still shared

Keys that must be **equal** across s and m:

```text
match_thresh
new_track_thresh
confirm_streak / confirm_score_thresh   # as present in YAML
oao_tau / oao_ramp_frames
multiplicative_cost
sinkhorn_lambda
stability_cost_w
private_continuation_enabled
private_candidate_nms_iou
private_prior_iou_threshold
# (+ other private_* if both define them)
occ_state_* family
relink_bridge_enabled
```

**Fail if:** s≠m on any of the above without a deliberate contract revision
(update README + this file in the same change).

---

### C3 — m motion / bridge deltas still intentional

| Knob | s | m |
|:--|:--|:--|
| `kalman_r_scale` | **2.8** | **3.5** |
| `relink_bridge_px` | **0.25** | **0.4** |
| `relink_bridge_h_lo` / `_h_hi` | **0.75 / 1.33** | **0.6 / 1.7** |
| `relink_bridge_dir_bonus` | **0.8** | **0.0** (explicit) |

**Intent reminder:** ↑ `kalman_r_scale` → larger R → trust **predict** more
(not measurement). See [../kalman_gmc_motion.md](../kalman_gmc_motion.md).

**Fail if:** deltas collapse accidentally (m copies s) or reverse without docs;
or m `dir_bonus` becomes silent inherit of non-zero schema default.

---

### C4 — private continuation score-clamp (continue ≠ birth)

| Check | Expected |
|:--|:--|
| `private_continuation_enabled` | **true** on s/m |
| Semantics | expands **det set** pre-track; **not** a GPUByteTracker setter |
| Score policy | private scores clamped to **`< new_track_thresh`** (`birth_ceiling = new_track_thresh − eps`) |
| Code anchor | `detection_filters._append_private_continuation_candidates` |

**Fail if:** clamp removed, ceiling ≥ birth thresh, or docs claim it is a setter.

---

### C5 — `relink_bridge_dir_bonus` s=0.8 / m=0.0 intentional

| Check | Expected |
|:--|:--|
| s YAML | `0.8` |
| m YAML | **`0.0` written explicitly** (not omitted) |
| Contract | m disables direction bonus by design |

**Fail if:** m key missing (silent default risk) or values swapped without note.

---

### C6 — dual stability still separate (until cleanup decision)

| Knob | Stage | Expected |
|:--|:--|:--|
| `stability_cost_w` | cost Π reward | **0.20** YAML s/m |
| `SACCADE_STABILITY_W` | auction bid | env default **0.1** if unset (code) |

**PASS means:** both still present as independent controls; docs do not claim
they are the same knob.

**Fail if:** one silently zeroed in production path **without** dual-stability
decision PR + eval; or docs merge them into one name without stage labels.

See [dual_stability_cleanup.md](dual_stability_cleanup.md) for A/B/C — **do not
“fix” dual stability in a cleanup PR.**

---

### C7 — NO-GO knobs not enabled on headline presets

Spot-check (full table: [no_go_guardrails.md](no_go_guardrails.md)):

| Knob | Headline |
|:--|:--|
| `fuse_score_weight` | **0** |
| `nsa_kalman` | **false** / absent |
| `gmc_fg_mask` | **false** |
| OAO spatial family | inert / off |
| `reid_mode` | **off** |
| PRESET-OFF geometry priors | **false** |

**Fail if:** any NO-GO flipped on in `mamba_whole_graph*.yaml`.

---

### C8 — inject map still correct (sanity) — **automated**

| Expectation | Path | Checker |
|:--|:--|:--|
| Primary tracker inject | `pipeline.py` calls `detector.tracker.set_params` / `set_occ_params` / `set_relink_params` | hard fail |
| Name remap | `r_scale=cfg.geometry.kalman_r_scale` | hard fail |
| occ knobs flow | `cfg.geometry.occ_*` into `set_occ_params` | hard fail |
| Multiplicative / stability | `set_multiplicative_cost` + `set_stability_cost_w` present | hard fail |
| Private continuation | `_append_private_continuation_candidates` + `birth_ceiling` in `detection_filters.py` | hard fail |
| Private not a setter | no `set_private_continuation` / `set_params(..., private_*)` on tracker facade | hard fail |
| Facade remap target | `tracker_gpu.py` `set_params` accepts `r_scale` | hard fail |

**Fail if:** production inject moves off `pipeline.py`, remap drops, or private becomes a tracker setter.

---

## Pass / fail summary template

Copy into PR description when relevant:

```text
Active contract healthcheck (manual):
- [ ] C1 occ_state explicit on
- [ ] C2 s/m shared assoc keys
- [ ] C3 m deltas (r_scale / bridge / dir_bonus)
- [ ] C4 private continuation clamp
- [ ] C5 dir_bonus s/m
- [ ] C6 dual stability still separate (or linked decision PR)
- [ ] C7 NO-GO not on headline
- [ ] C8 inject still pipeline.py (+ remap + private det-set)
- [ ] C9 surface allowlist / no forbidden ablation keys
Behavior change? no / yes → smoke → 04 → 7-seq
```

Also see `.github/pull_request_template.md` decision-layer checklist.

---

## C9 — surface allowlist / forbid ablation keys (P6)

| Check | Expected |
|:--|:--|
| Every YAML key ∈ `HEADLINE_ALLOWED_KEYS` | fail unknown (surface creep) |
| No key ∈ `HEADLINE_FORBIDDEN_KEYS` | OAO spatial, NSA, energy mode, bank relink, lifecycle merge, birth experiments, … |
| New ACTIVE key | update allowlist + README active contract in same PR |

---

## Automation status

| Item | Status |
|:--|:--|
| `scripts/tools/check_headline_decision_contract.py` | **done** (YAML C1–C7/C9 + inject C8) |
| pytest `tests/unit/test_headline_decision_contract.py` | **done** |
| CI `contracts` job | **done** |
| C8 inject map (`pipeline.py` / private det-set) | **done** (P5.1) |
| C9 surface allowlist / forbid | **done** (P6) |
| Argparse help `[NO-GO]` / `[LATENT]` tags | **done** (P6; defaults unchanged) |
| PR template decision checklist | **done** (`.github/pull_request_template.md`) |
| Env `SACCADE_STABILITY_W` default assert | **not in CI** (NOTE only; dual-stability RFC) |

**Do not** auto-merge dual stability knobs from this checker.

---

## Out of scope

```text
✗ Kernel math correctness
✗ Throughput / CUDA graph
✗ Full LATENT inventory (see config_surface)
✗ Replacing math_model or 7-seq
```

---

## Related

- Contract text: [../README.md](../README.md)
- NO-GO rules: [no_go_guardrails.md](no_go_guardrails.md)
- Dual stability: [dual_stability_cleanup.md](dual_stability_cleanup.md)
- Surface inventory: [config_surface.md](config_surface.md)
