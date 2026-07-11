---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-11
task-type: research-exploration-analysis
production-impact: none
default-behavior-change: forbidden
---

# Gap-conditioned probabilistic motion representation probe

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — semantic sole active；independent conditional motion representation probe |
| Research object | gap-conditioned transition density \(p(x_1,v_1 \mid x_0,v_0,\Delta t,c)\) → standardized mismatch / NLL / optional reach mass |
| Motivation | deterministic motion atoms 在 long-gap slice 出現 GT role reversal（short: high-mismatch→FP；部分 long: high-mismatch 集中 GT） |
| Relation to mainline | **parallel / independent** · 不納入 restricted global closure · 不改 \(\{dist_h,\log h_{ratio}\}\) 安全域主線 |
| Execution | **E0–E1 complete** · E0 `PARTIALLY_IDENTIFIABLE` · E1 0/20 aggregate reversal cells but clear gap degradation · substrate frozen pair table only |
| Research acceptance | **`ACCEPTED_WITH_LIMITS`** for E0–E1 packet · E0 `ACCEPT` · E1 marginal baseline `ACCEPTED_WITH_LIMITS` |
| Probabilistic verdict | **`NOT_YET_EVALUATED`** · no V1–V5 verdict · Phase B unauthorized |
| Engineering / production | **none** · no tracker / preset / online hook / baseline change |
| Research promotion | **none** until LOO + bounded V1–V5 verdict |

## Current boundary

```text
Core question:
  Gap-conditioned probabilistic transition representation
  能否保留 short-gap motion discrimination，
  同時降低 long-gap GT tail concentration / 消除 motion role reversal，
  並形成比 deterministic motion atoms 更穩定的 conditional safe region？

Representation shift (not reweighting existing atoms):
  deterministic motion mismatch
    → gap-conditioned probabilistic transition likelihood

Model ladder (first round; do not jump to PDE / learned SDE):
  M0  deterministic baseline (bridge_dist · speed_mismatch · dir_cos · resid_mean)
  M1  constant-velocity Gaussian transition  y_Δ ~ N(F_Δ y_0, Q_Δ)
  M2  integrated Ornstein–Uhlenbeck (velocity memory decay / residual OU)

Signals to derive (min):
  q_motion   Mahalanobis / standardized innovation
  E_motion   full NLL = ½q + ½log det Σ + const  (terms must be stored separately)
  P_reach    optional secondary only; not sole headline

Context (predeclared, simple, reproducible):
  global · sequence · exit-zone · image-normalized bin ·
  GMC direction cluster · frozen route-like grouping
  Prefer residual OU: v = v̄(c) + u, du = −γ u dt + σ dW
  No held-out retune of μ(c), γ, σ, or calibration

  LOO firewall (sequence context):
    sequence-conditioned context is diagnostic / in-sample only
    unless a predeclared train-only fallback or transferable
    context mapping exists; it is NOT eligible for LOO headline
    comparison by default.
    Forbidden: held-out sequence stats · ad-hoc held-out fallback ·
    mixing sequence-conditioned in-sample into LOO headline.

Scope:
  frozen pair table offline only
  independent of restricted global-closure prototype
  motion atoms remain conditional; not global solve atoms
```

## Read first

1. [gt_support_morphology thread](gt_support_morphology_20260711.md) — escape-tail / role-reversal context；global=`{dist_h, log_h_ratio}`；motion atoms = conditional only
2. Escape-tail forensic note / packet (via morphology thread) — `ROLE_REVERSAL_SUPPORTED` · L1 MOT17-10 bound
3. Boolean-atom partial-order note — `bridge_dist` + motion atoms = conditional_orderable
4. Existing frozen pair substrate used by morphology / motion studies（pool path TBD at E0；do not invent a new unlabeled table）
5. Framework / safe-region contracts only as **reference for analysis language** — this probe does **not** reopen closed A1 / R2–R4 gates

## Artifacts

**Produced / planned:**

| Artifact | Role |
|:--|:--|
| [E0 research note](../../modules/semantic/research/gap_conditioned_motion_e0_20260711.md) | substrate audit · canonical bins · `PARTIALLY_IDENTIFIABLE` gate |
| [E0 packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e0_20260711/manifest.json) | source SHA seal · schema/integrity audit · byte verification |
| [E1 research note](../../modules/semantic/research/gap_conditioned_motion_e1_m0_20260711.md) | deterministic baseline · aggregate-vs-local role-reversal boundary |
| [E1 packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e1_m0_20260711/manifest.json) | 20 gap×atom cells · AUC · frozen q90 tail · sequence attribution |
| Final research note | problem · equations · context · substrate · fit/LOO protocol · limitations · bounded verdict |
| Pair-level signal table | M0 + M1 + M2 signals · gap/context · model/parameter IDs · labels · regularization flags |
| Model artifacts / fold | γ, σ, drift/context mean, covariance def, coordinate system, time unit, fit/exclude rows, code version |
| Summary tables (12) | substrate · M0 RR baseline · cal by gap/seq · short-gap retention · escape-tail · separability · RR rate · LOO · conditional SR · M1 vs M2 · failure attribution |
| Figures | dist-by-gap · q calibration · coverage · RR heatmap · tail occupancy · escape movement · LOO · safe/productive area |
| Single reproduction entrypoint | rebuild fits · pair outputs · tables · figures · verdict from frozen pair table |

## Current step

**E0–E1 `ACCEPTED_WITH_LIMITS`; next = E2 position-only M1-P/M2-P family freeze.**

Frozen E0 boundary:

```text
identifiable:
  M0 deterministic atoms
  position-only observation = delta_foot_xy / h_ref conditioned on gap

not identifiable on the frozen table:
  velocity-only / joint position-velocity observations
  exit-zone / image-normalized / GMC-cluster / route contexts

LOO headline context:
  global only
  sequence remains diagnostic / in-sample only

Phase B remains unauthorized.
```

PR #109 acceptance limit: the E1 `AUC < 0.5 AND pooled within-bin q90 GT
enrichment > 1` rule is a frozen descriptive reporting criterion, not an
auditable predeclared or confirmatory gate. E0 is schema-identifiability only;
future vector fields require finite-value, usable-support, time-window,
coordinate-semantics, and provenance audits before joint identifiability can be
upgraded.

When opened, follow Phase A → freeze → Phase B（不得在 exploration 中途挑單一漂亮結果作結論）:

```text
Phase A — exploration
  E0 substrate / identifiability audit — DONE: PARTIALLY_IDENTIFIABLE
  E1 M0 deterministic role-reversal baseline — DONE:
     0/20 aggregate reversal cells; gap degradation retained;
     PR-C escape tail remains local/conditional
  E2 M1/M2 limited predeclared parameter family
  E3 signal generation (position-only · velocity-only · joint; terms split)
  freeze model family + analysis inputs

Phase B — analysis
  A1 gap-bin calibration (χ² coverage; under/approx/over dispersed)
  A2 role-reversal rate R_flip + sequence attribution
  A3 short-gap retention (must not collapse)
  A4 escape-tail audit (known motion GT tails)
  A5 separability (AUC descriptive only)
  A6 conditional safe-region S_old vs S_new
  A7 LOO sequence transfer (params/calibration frozen per fold)
  A8 M1 vs M2 attribution (matched coverage / log det Σ / short-gap)
  → single V1–V5 bounded verdict
```

### Model comparison logic

| Result | Interpretation |
|:--|:--|
| M1 ≻ M0 | probabilistic uncertainty representation supported |
| M2 ≻ M1 (matched uncertainty) | velocity memory decay / mean reversion has independent value |
| M2 ≯ M1 | keep simpler Gaussian CV; do not claim OU necessary |
| M1 & M2 fail | reject this form of probabilistic motion redesign |

### Suggested gap bins

Prefer existing canonical gap bins. If none: fix short / medium / long / extreme-long **before** results; report pair/GT/FP counts, sequence coverage, gap range, median gap per bin. **No post-hoc rebinning.**

## Acceptance

### Success boxes (all required to support probabilistic representation)

```text
short-gap discrimination retained
long-gap GT tail concentration reduced
effect direction across gap bins more consistent
LOO conditional safe region no thinner
improvement not explained by unrestricted diffusion
```

### Bounded verdict (exactly one)

| Code | Meaning |
|:--|:--|
| **V1** | probabilistic uncertainty supported — M1 fixes main RR; M2 no stable gain |
| **V2** | integrated OU supported — M2 beats M1 under LOO + matched uncertainty + short-gap retention |
| **V3** | position-only supported — joint velocity too noisy |
| **V4** | inconclusive — support / velocity quality / context identifiability insufficient |
| **V5** | not supported — no real RR fix or only via over-diffusion |

### Definition of Done (close only when all true)

- [x] substrate audit complete (`PARTIALLY_IDENTIFIABLE`)
- [x] M0 role-reversal baseline rebuilt
- [ ] M1 and M2 reproducible implementations
- [ ] position-only vs joint outputs separated; energy terms stored separately
- [ ] gap-bin calibration · RR rate · short-gap retention · escape-tail · conditional SR · LOO · M1 vs M2 attribution · failure/scale/regularization audit
- [ ] single reproduction entrypoint rebuilds all headline results
- [ ] exactly one V1–V5 bounded verdict
- [ ] explicit note: whether a follow-up default-off online hook task is warranted
- [ ] production preset and baseline behavior **unchanged**

### Post-verdict promotion gate

Only after V1/V2/V3 + complete evidence may a **separate** task discuss: frozen signal contract · research-only default-off hook · baseline vs hook A/B · e2e safety.  
Until then: no evidence_ledger · no production preset · no online-safe claim · no “replaces motion signals” claim.  
**PR merge ≠ research acceptance.**

## Must not

- reweight / retune `speed_mismatch` · `dir_cos` · `resid_mean` · `bridge_dist` as the primary intervention
- fold this probe into restricted **global** closure or change \(\{dist_h,\log h_{ratio}\}\) mainline
- start with Fokker–Planck PDE · neural SDE · unrestricted mixtures · particle filters · production runtime work
- large unrestricted hyperparameter search; per-sequence pick-best then aggregate; label-aware held-out selection
- retune context mean / γ / σ / calibration on held-out sequences
- use pure `sequence` context as LOO headline without predeclared train-only fallback / transferable mapping; steal held-out sequence stats; ad-hoc held-out fallback; mix sequence-conditioned in-sample into LOO headline
- silent repair: drop hard sequences, exclude long-gap failures, different regularization for GT vs FP, post-hoc model redefine without new model ID
- compare position-only / velocity-only / joint NLL as same-dimension quantities without recording \(d\) and per-term splits
- modify tracker · online association hook · production preset · baseline behavior · default-on new signals
- claim production gain from offline results alone
- treat exploration mid-run cherry-pick as verdict

## History

- 2026-07-11: Task proposed as independent research-exploration-analysis thread. Captured full program (M0–M2 ladder, E0–E3 / A1–A8, V1–V5, DoD) as navigation-only mother line. **Not executed.** Not sole active; does not authorize engineering delivery or ledger promotion.
- 2026-07-11: PR #108 review — lock LOO firewall: sequence-conditioned context is diagnostic/in-sample only by default (not LOO headline without predeclared train-only fallback / transferable mapping).
- 2026-07-11: Owner task direction activates this independent probe as semantic sole active; the not-yet-started restricted-closure continuation is parked without changing its accepted global/conditional boundary.
- 2026-07-11: E0 sealed `PARTIALLY_IDENTIFIABLE` on the existing seven-sequence pair table (SHA `0ae38967…`): M0 + position-only observation available; vector velocity and transferable contexts absent. Canonical gap bins frozen; Phase B remains unauthorized. [Note](../../modules/semantic/research/gap_conditioned_motion_e0_20260711.md) · [packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e0_20260711/manifest.json).
- 2026-07-11: E1 rebuilt the four-atom deterministic baseline across five canonical gap bins. Frozen aggregate criterion found 0/20 marginal reversal cells; `bridge_dist`/`resid_mean` AUC still erodes strongly with gap. Claim narrowed: PR-C role reversal is local/conditional, not a whole-bin sign flip. [Note](../../modules/semantic/research/gap_conditioned_motion_e1_m0_20260711.md) · [packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e1_m0_20260711/manifest.json).
- 2026-07-11: PR #109 review: engineering/reproducibility `PASS`; E0 `ACCEPT`; E1 marginal baseline **`ACCEPTED_WITH_LIMITS`**; probabilistic representation **`NOT_YET_EVALUATED`**; production/hook authorization `NONE`. Removed the unauditable claim that the exact AUC+q90 criterion was fixed before outputs; it remains frozen for descriptive E1 reporting only. E2 must add finite/support/window/coordinate/provenance gates before any vector-state identifiability upgrade.
