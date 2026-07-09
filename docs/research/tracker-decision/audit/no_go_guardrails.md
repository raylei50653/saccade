# NO-GO / LATENT Guardrail Plan

**Status:** rules / process RFC — **no validator implementation yet**  
**Date:** 2026-07-09  
**Source inventory:** [config_surface.md](config_surface.md) prohibition table + ACTIVE tables  
**Not this PR:** schema delete, CLI hide, kernel changes, 7-seq.

Goal: stop NO-GO and large LATENT surfaces from **accidentally becoming
headline production policy** when someone edits presets, schema defaults,
or inject paths.

---

## Tag definitions (recap)

| Tag | Meaning | Guardrail stance |
|:--|:--|:--|
| **ACTIVE** | Production identity policy on headline path | Must stay explicit and reviewed |
| **PRESET-OFF** | Schema may default on; headline YAML forces off | Re-enable = behavior PR + score-dist study |
| **LATENT** | Wired; off; for ablation | Not a safe default; needs evidence to promote |
| **NO-GO** | Evidence already rejects as default policy | Must not land in headline without overturning evidence |
| **ENV** | Controlled outside YAML | Invisible to preset review; document + healthcheck |

**Rule of thumb:** LATENT = “wired for ablation.” NO-GO = “already rejected as headline.”

---

## Guardrail tiers (future implementation order)

| Tier | Mechanism | When | Behavior impact |
|:--|:--|:--|:--|
| **T0 Docs** | This file + config_surface + README contract | **now** | none |
| **T1 Healthcheck** | Active-contract + NO-GO absence checks ([active_contract_healthcheck.md](active_contract_healthcheck.md)) | P5 script optional | none if read-only |
| **T2 Preset validator** | Fail CI if headline presets violate rules | P5 | blocks bad YAML only |
| **T3 Schema / CLI hygiene** | Demote help text, experimental group, warn on promote | later | none if defaults unchanged |
| **T4 Surface shrink** | Hide/archive knobs, split experimental config | later | risk if defaults change |

P4 only commits to **T0** (+ checklist for T1). T2+ is P5 decision.

---

## 1. Must not appear as “on” in headline presets

Headline presets = `mamba_whole_graph.yaml` / `mamba_whole_graph_m.yaml`
(and any future “production” alias that claims the same contract).

| Knob / family | Tag | Headline expectation | Why |
|:--|:--|:--|:--|
| `fuse_score_weight` | NO-GO #45 | **0.0** (explicit ok) | Score affinity bipolar / seq-split risk |
| `nsa_kalman` | NO-GO #8 | **false** / absent | Double-comp with high `kalman_r_scale` |
| OAO spatial family (`oao_contest_thresh`, `oao_score_w`, `oao_occ_mode`, `oao_crowd_radius`, `oao_height_gate`, `oao_foot_gate`) | NO-GO | off / schema inert defaults | Failed to separate 05 harm vs 04 benefit; duration ramp is GO |
| `gmc_fg_mask` | NO-GO #20 | **false** | Does not fix PCR-dominated BG |
| Sync ReID / `relink_bridge_app_veto` as critical path | NO-GO #57 | off | Throughput vs tiny AssA |
| Fancy interpolate geometry knobs | NO-GO #44 | keep simple interpolate only | Endpoint association is FP source |
| `birth_prox_norm_thresh` / experimental birth gates | NO-GO / LATENT | off | Unvalidated birth FP |
| Bank `relink_enabled` | LATENT | off while `reid_mode=off` | Needs ReID path |
| Lifecycle / post / cheb-gr merge | LATENT | off | Offline / partial NO-GO |
| `association_scoring_mode=energy` | LATENT | `baseline` | Extra Π without λ/stability retune |
| `vel_dir_weight > 0` | LATENT | 0 | Unvalidated on current cost form |

**PRESET-OFF (must stay false on headline Mamba path):**

```text
person_geometry_prior
detection_quality_scaling
geometry_suspect_support
id_stability_filter
track_person_only
per_seq_adapt
```

Re-enable only with score-distribution study (Mamba recalibration fight risk).

### Writing rule for presets

```text
✓ ACTIVE knobs: prefer explicit YAML (occ_state_*, m dir_bonus, …)
✓ NO-GO that default-on in schema: explicit false / 0.0 in headline
✗ NO-GO experimental values “just to try” in mamba_whole_graph*.yaml
✗ Copying ablation YAML keys into headline without promotion checklist
```

Ablation / probe presets may live under research configs — **not** under
headline names.

---

## 2. Env-only / ablation-only knobs

These should **not** become first-class headline YAML without a promotion PR:

| Knob | Surface | Production stance |
|:--|:--|:--|
| `SACCADE_STABILITY_W` | env | ENV ACTIVE today (default 0.1); dual-stability debt — see [dual_stability_cleanup.md](dual_stability_cleanup.md) |
| `SACCADE_FRESHNESS_W` | env | LATENT (default 0) |
| `SACCADE_ENABLE_DDA` / `SACCADE_DDA_MAX_COST` | env | ENV ACTIVE (document; do not silently flip) |
| `SACCADE_GMC_PCR_THRESH` | env | GMC confidence; not association YAML |
| `SACCADE_ASSOC_DUMP` | env | debug only |

**Rules**

1. Sweeping env bid knobs **together** with `stability_cost_w` requires a
   written factor design (dual stability).
2. CI / docs reviews of “preset-only” diffs **must** note env defaults if
   auction behavior is claimed constant.
3. Promoting an env knob to YAML is a **visibility improvement** only if
   default and inject stay behavior-preserving.

---

## 3. Promotion bar: LATENT / NO-GO → production

### 3.1 NO-GO overturn

To enable a NO-GO as headline default:

| Requirement | Detail |
|:--|:--|
| **Paired evidence** | Cite original rejection (registry # / research note) and new counter-evidence |
| **7-seq** | Full MOT17 train train-half (or project-standard 7-seq) — not single-seq IDF1 |
| **Bipolar / split check** | Especially for score-affinity (`fuse_score`) and OAO spatial family (04 vs 05) |
| **Interaction review** | e.g. NSA only if `kalman_r_scale` retuned; OAO spatial only if duration ramp story revised |
| **Explicit preset + math_model + config_surface update** | Same PR or stacked docs PR |
| **Owner sign-off** | Decision-layer reviewer acknowledges identity risk |

Without this checklist, CI / reviewers treat enablement as **out of contract**.

### 3.2 LATENT promote (not previously NO-GO)

| Requirement | Detail |
|:--|:--|
| Mechanism doc | What decision changes (match / birth / relink) |
| Smoke → MOT17-04-SDP → 7-seq | Only if behavior can change |
| ACTIVE contract update | README + config_surface row → ACTIVE |
| Healthcheck update | New invariant if it becomes contract |

### 3.3 PRESET-OFF re-enable

| Requirement | Detail |
|:--|:--|
| Score-distribution / Mamba interaction study | Priors fought recalibration |
| Same ladder as LATENT promote | |

---

## 4. Preset validator (P5 — design only)

**Proposed name:** `scripts/tools/check_headline_decision_contract.py`  
**Inputs:** `configs/presets/mamba_whole_graph.yaml`, `mamba_whole_graph_m.yaml`  
**Mode:** fail CI on violation (strict), optional local warn.

### Suggested checks (YAML static)

```text
1. NO-GO not enabled (fuse_score_weight==0, nsa_kalman false, gmc_fg_mask false, …)
2. PRESET-OFF still false (person_geometry_prior, DQS, suspect, id_stability, …)
3. OAO spatial family at inert defaults
4. reid_mode == off on headline (unless contract deliberately changes)
5. s/m shared assoc keys equal (match_thresh, new_track_thresh, confirm_*, oao_*, …)
6. m deltas present and intentional (r_scale, bridge_px, h_lo/h_hi, dir_bonus)
7. occ_state_* explicit true + knobs
8. private_continuation_enabled true
9. multiplicative_cost true; stability_cost_w == 0.20 (until dual-stability PR)
```

**Out of validator scope (initially):** env vars, native member defaults,
runtime inject bugs (those need unit tests / callpoint tests).

### Optional soft checks

```text
- warn if headline preset gains a new key not in allowlist (surface creep)
- warn if m equals s on a documented delta key (accidental desync the other way)
```

---

## 5. CI grep / config healthcheck (P5 options)

| Check | Style | Fail? |
|:--|:--|:--|
| Headline preset NO-GO values | validator (preferred over raw grep) | yes |
| `nsa_kalman:\s*true` in headline path | grep fallback | yes |
| `fuse_score_weight:\s*[1-9]` in headline | grep fallback | yes |
| Schema default drift vs documented schema | harder; golden config tests already cover some | existing pytest |
| Doc path / freshness | existing `check_doc_*` | stale paths fail; freshness warn |

**Do not** use brittle greps on `tracker_gpu.cu` for policy — prefer preset +
schema + golden fixtures.

Existing related tests:

- `tests/fixtures/golden_config_*.json` — resolved config snapshots  
- `tests/unit/test_math_model_doc_consistency.py` — LaTeX baseline numbers  
- GPU / API layer contracts — orthogonal to decision policy

P5 may **extend** golden fixtures with an explicit “decision contract” section
rather than inventing a second source of truth.

---

## 6. Process rules (immediate, no code)

### For preset PRs

```text
1. Diff only ACTIVE + intentional m deltas unless promotion checklist attached
2. If touching NO-GO / PRESET-OFF / LATENT → link this doc §3
3. Update tracker-decision README active contract if identity policy changes
4. Behavior change ⇒ smoke → 04 → 7-seq; docs-only ⇒ static review
```

### For schema / argparse PRs

```text
1. New knobs default to OFF / inert unless replacing an ACTIVE
2. Do not flip schema default to “on” for LATENT experiments
3. Help text should not present NO-GO as recommended
```

### For inject / native setter PRs

```text
1. Name remaps documented in native_bridge.md
2. Do not pack NO-GO “helpfully on” when bridge/bank flags confuse
3. Dual stability: do not silently read new env defaults without docs
```

---

## 7. What we are not doing yet

```text
✗ Deleting schema fields or pybind args
✗ Hiding CLI flags
✗ Splitting experimental config modules
✗ Changing any production default
✗ Claiming validator already enforces this file
```

---

## Related

- Inventory: [config_surface.md](config_surface.md)
- Active checklist: [active_contract_healthcheck.md](active_contract_healthcheck.md)
- Dual env/YAML debt: [dual_stability_cleanup.md](dual_stability_cleanup.md)
- Tree contract: [../README.md](../README.md)
