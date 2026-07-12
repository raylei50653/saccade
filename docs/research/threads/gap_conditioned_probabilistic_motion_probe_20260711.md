---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: engineering-follow-up
wip-role: non-wip
created: 2026-07-11
task-type: research-exploration-analysis
production-impact: none
default-behavior-change: forbidden
---

# Gap-conditioned probabilistic motion representation probe

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE — non-WIP engineering follow-up only.** Semantic sole-active is **none**; Phase B is concluded and this card retains only D0 / #112 capture closure. |
| Research object | gap-conditioned transition density \(p(x_1,v_1 \mid x_0,v_0,\Delta t,c)\) → standardized mismatch / NLL / optional reach mass |
| Motivation | deterministic motion atoms 在 long-gap slice 出現 GT role reversal（short: high-mismatch→FP；部分 long: high-mismatch 集中 GT） |
| Relation to mainline | **parallel / independent** · 不納入 restricted global closure · 不改 \(\{dist_h,\log h_{ratio}\}\) 安全域主線 |
| Execution | **E0–E2 accepted** · E2 freeze `FROZEN_ACCEPTED_WITH_LIMITS` · **E3 signals `E3_SIGNALS_SEALED`** · **Phase B `PHASE_B_EXECUTED` → `V5`** · **D0 fail-closed** (`not_fidelity_aligned` / `runtime_capture_unavailable`; Issue #112 incomplete) |
| Research acceptance | **`ACCEPTED_WITH_LIMITS`** for E0–E2 and **Phase-B `V5 ACCEPTED_WITH_LIMITS`** · E0 `ACCEPT` · E1 marginal baseline `ACCEPTED_WITH_LIMITS` · E2 family + LOO lineage `ACCEPTED_WITH_LIMITS` · E3 signals sealed |
| Probabilistic verdict | **`V5 ACCEPTED_WITH_LIMITS`** · representation + attribution contract not established · claim ceiling = representation / level 1 |
| Engineering / production | **none** · no tracker / preset / online hook / baseline change |
| Research promotion | **none** · accepted V5 authorizes no ledger, production, threshold-transfer, or hook promotion |

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
6. [Production substrate mapping](../../modules/semantic/research/production_substrate_mapping_20260711.md) — **binding** on E3/A1–A8: §5 support cuts（primary=\(S_A=[1,26]\)）· §6 D0 estimator-fidelity gate（[Issue #112](https://github.com/raylei50653/saccade/issues/112)）· §8 headline constraint · §9 claim ladder

## Artifacts

**Produced / planned:**

| Artifact | Role |
|:--|:--|
| [E0 research note](../../modules/semantic/research/gap_conditioned_motion_e0_20260711.md) | substrate audit · canonical bins · `PARTIALLY_IDENTIFIABLE` gate |
| [E0 packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e0_20260711/manifest.json) | source SHA seal · schema/integrity audit · byte verification |
| [E1 research note](../../modules/semantic/research/gap_conditioned_motion_e1_m0_20260711.md) | deterministic baseline · aggregate-vs-local role-reversal boundary |
| [E1 packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e1_m0_20260711/manifest.json) | 20 gap×atom cells · AUC · frozen q90 tail · sequence attribution |
| [E2 research note](../../modules/semantic/research/gap_conditioned_motion_e2_family_20260711.md) | reduced position-only M1-P/M2-P equations · train-only fit/LOO firewall · E3 signal contract |
| [E2 packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e2_family_20260711/manifest.json) | machine-readable four-member family · finite/support/window/provenance audit · per-fold LOO lineage hashes · parameter/selection artifact schemas · fit/scoring primitives |
| [E3 research note](../../modules/semantic/research/gap_conditioned_motion_e3_signals_20260711.md) | LOO fold signal generation · 28 parameter + 7 selection artifacts · all-four scores per pair · no A1–A8 |
| [E3 packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json) | sealed pair×fold×model score table · fold artifacts · manifest (signals only) |
| [Phase B design note](../../modules/semantic/research/gap_conditioned_motion_phase_b_design_20260711.md) | predeclared A1–A8 protocol · support layers (primary=\(S_A\)) · frozen numeric criteria · V1–V5 decision rule · D0 claim-ceiling coupling |
| [Phase B research note](../../modules/semantic/research/gap_conditioned_motion_phase_b_20260711.md) | A1–A8 execution from sealed E3 cube · **`V5 ACCEPTED_WITH_LIMITS`** · representation / level 1 only |
| [Phase B packet](../../modules/semantic/research/evidence/gap_conditioned_motion_phase_b_20260711/manifest.json) | deterministic tables / A6 train-firewall audit / verdict record / full E3→Phase-B verifier |
| [D0 fidelity note](../../modules/semantic/research/d0_bridge_estimator_fidelity_20260711.md) | fail-closed capture unavailable · terminal **`not_fidelity_aligned`** · Issue #112 incomplete |
| [D0 packet](../../modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/manifest.json) | reconstruction diagnostics · same-event join · single-factor decomp · verdict |
| Final research note | problem · equations · context · substrate · fit/LOO protocol · limitations · bounded verdict |
| Pair-level signal table | M0 + M1 + M2 signals · gap/context · model/parameter IDs · labels · regularization flags |
| Model artifacts / fold | γ, σ, drift/context mean, covariance def, coordinate system, time unit, fit/exclude rows, code version |
| Summary tables (12) | substrate · M0 RR baseline · cal by gap/seq · short-gap retention · escape-tail · separability · RR rate · LOO · conditional SR · M1 vs M2 · failure attribution |
| Figures | dist-by-gap · q calibration · coverage · RR heatmap · tail occupancy · escape movement · LOO · safe/productive area |
| Single reproduction entrypoint | rebuild fits · pair outputs · tables · figures · verdict from frozen pair table |

## Current step

**Current concrete work (non-WIP):** resolve D0 / [Issue #112](https://github.com/raylei50653/saccade/issues/112) by obtaining the default-off runtime CUDA capture required to replace `runtime_capture_unavailable`. This is an engineering fidelity follow-up only; it does **not** reopen Phase B, create a semantic mainline, or authorize threshold transfer.

**Recorded research state:** E0–E2 `ACCEPTED_WITH_LIMITS`; E2 freeze
`FROZEN_ACCEPTED_WITH_LIMITS`; E3 `E3_SIGNALS_SEALED`; Phase B
**`V5 ACCEPTED_WITH_LIMITS`** (representation / level 1); D0
`D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE` / `not_fidelity_aligned` (Issue #112
incomplete).

Frozen E0/E2/E3 boundary:

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

E3 sealed: 7 LOO folds · 28 parameter + 7 selection artifacts ·
full fold×pair×model cube 679,952 rows (24,284 × 7 × 4) with
evaluation_role=held_out|train (A6 train-side surface sealed) ·
one sealed A1–A8 run completed → **`V5 ACCEPTED_WITH_LIMITS`**; no
criterion/model/input deviation. The accepted claim ceiling is representation /
level 1 only.

Binding since PR #111 (production substrate mapping):
  E3/A1–A8 headline = E_motion on S_A=[1,26] (consumer A);
  S_C2/S_B secondary; all-gap exploratory only (§8)
  threshold transfer additionally gated by D0 (Issue #112, §6)
  D0 fail-closed: not_fidelity_aligned
    primary_fail_reason: runtime_capture_unavailable
    → reconstruction diagnostics only (not runtime CA capture)
    → offline bridge_dist numeric thresholds may NOT transfer to Consumer A
    → Issue #112 incomplete until default-off CUDA capture exists
    → E3 signals: sealed
    → Phase B: executed; representation-level V5
    → A1–A8: executed; no production threshold transfer

A1–A8 protocol predeclared (design sealed; execution still unauthorized):
  docs/modules/semantic/research/gap_conditioned_motion_phase_b_design_20260711.md
  ordering: design merge (seal) ✓ → E3 signals sealed ✓ → owner authorization ✓
  → single Phase B run ✓ → V5 recorded (V3 predeclared unreachable)
```

PR #109 acceptance limit: the E1 `AUC < 0.5 AND pooled within-bin q90 GT
enrichment > 1` rule is a frozen descriptive reporting criterion, not an
auditable predeclared or confirmatory gate. E0 is schema-identifiability only;
future vector fields require finite-value, usable-support, time-window,
coordinate-semantics, and provenance audits before joint identifiability can be
upgraded.

PR #110 acceptance limit: E2 family mathematics and LOO lineage are accepted
with limits; E3 may only generate sealed fold signals. Verifier temporary
seven-fold rebuild is not E3 evidence. No calibration, A1–A8, family change,
or V1–V5 verdict.

When opened, follow Phase A → freeze → Phase B（不得在 exploration 中途挑單一漂亮結果作結論）:

```text
Phase A — exploration
  E0 substrate / identifiability audit — DONE: PARTIALLY_IDENTIFIABLE
  E1 M0 deterministic role-reversal baseline — DONE:
     0/20 aggregate reversal cells; gap degradation retained;
     PR-C escape tail remains local/conditional
  E2 M1-P/M2-P limited predeclared parameter family — ACCEPTED_WITH_LIMITS:
     GCM-E2-POSITION-ONLY-v1; FROZEN_ACCEPTED_WITH_LIMITS
  E3 signal generation (position-only only; terms split) — DONE:
     E3_SIGNALS_SEALED (28+7 artifacts · full cube + evaluation_role · no A-tables)
  freeze model family + analysis inputs — DONE for E2

Phase B — analysis (`V5 ACCEPTED_WITH_LIMITS`; representation / level 1)
  A1 gap-bin calibration (χ² coverage; under/approx/over dispersed)
  A2 role-reversal rate R_flip + sequence attribution
  A3 short-gap retention (must not collapse)
  A4 escape-tail audit (known motion GT tails)
  A5 separability (AUC descriptive only)
  A6 conditional safe-region S_old vs S_new
  A7 LOO sequence transfer (params/calibration frozen per fold)
  A8 M1 vs M2 attribution (matched coverage / log det Σ / short-gap)
  → V5 bounded verdict (representation + attribution contract not established)
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
escape cohort not high-energy under E_motion
  (weakened per PR #113 review; no M0-relative "reduction" claim —
   not identifiable on this substrate)
no new aggregate reversal + positive held-out direction
  (weakened per PR #113 review; M0 baseline is already 0/20, so
   "more consistent than M0" is not establishable)
LOO conditional safe region no thinner
  (held-out GT-leakage ≤ ε precondition; non-vacuous FP_removed)
improvement not explained by unrestricted diffusion
```

### Bounded verdict (exactly one)

| Code | Meaning |
|:--|:--|
| **V1** | probabilistic uncertainty supported — M1 fixes main RR; M2 no stable gain |
| **V2** | integrated OU supported — M2 beats M1 under LOO + matched uncertainty + short-gap retention |
| **V3** | position-only supported — joint velocity too noisy |
| **V4** | inconclusive — support / velocity quality / context identifiability insufficient |
| **V5** | representation + attribution contract not established — no member both passes all success boxes and holds a claimable verdict slot（redefined per PR #113 second review; includes the anomaly case: a member passes boxes but fails A8 dominance while M1 fails boxes — anomaly note mandatory. The pre-partition wording "no real RR fix or only via over-diffusion" is retired） |

### Definition of Done (close only when all true)

- [x] substrate audit complete (`PARTIALLY_IDENTIFIABLE`)
- [x] M0 role-reversal baseline rebuilt
- [x] M1-P and M2-P reproducible fit/scoring primitives + machine-readable family freeze
- [x] position-only vs joint outputs separated; energy terms stored separately (E3 score table)
- [x] gap-bin calibration · RR rate · short-gap retention · escape-tail · conditional SR · LOO · M1 vs M2 attribution · failure/scale/regularization audit
- [x] single reproduction entrypoint rebuilds E3 then all headline results from the frozen pair table
- [x] exactly one V1–V5 bounded verdict (`V5 ACCEPTED_WITH_LIMITS`; representation / level 1)
- [x] explicit note: a default-off online hook follow-up is **not warranted** from V5; post-verdict hook discussion is limited to V1/V2/V3
- [x] production preset and baseline behavior **unchanged**

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
- 2026-07-11: E2 engineering packet freezes `GCM-E2-POSITION-ONLY-v1`: global random-CV marginal M1-P plus integrated-OU M2-P half-lives 30/90/270 frames, train-GT-only fitting, global LOO firewall, deterministic covariance regularization, and split `q`/`log det`/constant/NLL output contract. Frozen-source finite/support/window/provenance gate passes with 340 eligible GT rows. [Note](../../modules/semantic/research/gap_conditioned_motion_e2_family_20260711.md) · [packet](../../modules/semantic/research/evidence/gap_conditioned_motion_e2_family_20260711/manifest.json).
- 2026-07-11: PR #110 review follow-up closes the engineering-side LOO artifact gap without changing model mathematics: packet now seals GT support and training-row lineage hash for every held-out sequence (minimum train fold 183), requires explicit parameter and fold-selection artifacts, records all four training NLL values, and forbids winner-only E3 output. Tests mutate held-out observations to prove fit hash/NLL exclusion and reproduce identical artifacts from identical fold input.
- 2026-07-11: PR #110 second review acceptance: engineering/reproducibility `PASS`; E2 mathematics `ACCEPT`; LOO lineage/selection `ACCEPT`; E2 research acceptance **`ACCEPTED_WITH_LIMITS`** · freeze status `FROZEN_ACCEPTED_WITH_LIMITS`; **E3 signal generation AUTHORIZED**; Phase B / A1–A8 `NONE`; V1–V5 `NOT_YET_EVALUATED`; production/hook `NONE`.
- 2026-07-11: PR #111 merged the [production substrate mapping](../../modules/semantic/research/production_substrate_mapping_20260711.md) as canonical precondition: E3/A1–A8 headline constrained to \(S_A=[1,26]\) (consumer A) with secondary/exploratory layering (§8); threshold transfer gated by the D0 estimator-fidelity study opened as [Issue #112](https://github.com/raylei50653/saccade/issues/112) (three-verdict contract; parallel to Phase B, gates claim ceiling only). Sealed E2 family / LOO firewall / E3 output contract unchanged.
- 2026-07-11: A1–A8 Phase B protocol predeclared in the [design note](../../modules/semantic/research/gap_conditioned_motion_phase_b_design_20260711.md): frozen numeric criteria (calibration classes · retention margin 0.05 · escape-tail ≥3/4 · A6 ε=0.05 no-thinner · A8 dominance rule), support floors (LOW_SUPPORT <15 GT · qualifying fold ≥20 held-out GT · folds 04/09 diagnostic-only), success-box mapping, and V1–V5 decision rule (V3 predeclared unreachable; V4 only via named routes). Design-only: Phase B execution remains unauthorized until design review acceptance + sealed E3 signals.
- 2026-07-11: PR #113 review (5 merge blockers, all accepted): ① D0 ceiling scope-limited to bridge atoms — \(E_{motion}\) is research-only for consumer A, no D0 outcome upgrades \(S_{new}\); level 2 needs a future \(E_{motion}\)/\(d\) consumer contract. ② A6 bound frozen in-design (one-sided CP 95% on `(sequence, lost_id)` cluster containment; train-cluster-only selection; `NO_FEASIBLE_THRESHOLD`/`BOTH_EMPTY` terminal semantics; positive-productivity clause; metric renamed `FP_removed`) — morphology step-0 established no cluster-aware bound to inherit; sequence-level residual clustering stays a declared limitation. ③④ Success boxes weakened: "no new aggregate reversal + positive held-out direction" (M0 already 0/20) and "escape cohort not high-energy under \(E_{motion}\)" (min-\(d_H\) representative pairs frozen from the forensic packet; single pooled-q90 reference population; no M0-relative reduction claim). ⑤ V1–V5 rewritten as a priority partition (V4 routes → V2 → V1 → V5) with a mandatory anomaly note for the boxes-without-slot residual case.
- 2026-07-11: PR #113 second review — five original blockers confirmed closed; two semantic blockers fixed: ① A6 held-out safety precondition frozen (every qualifying fold × non-BOTH_EMPTY primary cell must show held-out cluster-level empirical GT leakage ≤ ε = 0.05 for \(S_{new}\) **before** any FP_removed comparison; \(S_{old}\) leakage descriptive), and the CP bound renamed to a **model-based bound under the track-cluster independence assumption** (not a cluster-robust population bound; sequence-level residual clustering stays a declared limitation). ② V5 redefined as "representation + attribution contract not established" (verdict table updated; the pre-partition "no real RR fix or only via over-diffusion" wording retired), so the boxes-without-slot residual case is inside V5's definition instead of contradicting it. PR body synchronized with the final frozen design.
- 2026-07-11: PR #113 final review — **review-accepted / merge-ready**; no remaining merge blockers. The PR #113 merge commit is the **Phase B predeclaration seal**. Next authorized step: **E3 signal generation only** (sealed E2 contract; no A-table may be computed in the E3 PR). A1–A8 execution still requires an explicit owner authorization recorded in this thread after E3 signals are sealed.
- 2026-07-11: **E3 signals sealed** (`E3_SIGNALS_SEALED`): rebuilt 7 LOO folds via the E2 lineage-aware fold builder; persisted 28 parameter + 7 selection artifacts matching sealed train-GT counts/hashes; emitted full fold×pair×model cube (679,952 rows = 24,284 × 7 × 4) with `evaluation_role=held_out|train` so A6 can select τ on train clusters under fold-frozen parameters; energy terms split; no winner-only filter; Phase B design seal `69b0e5be…` + content sha256 recorded in manifest. Packet [evidence/gap_conditioned_motion_e3_signals_20260711/](../../modules/semantic/research/evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json) · [note](../../modules/semantic/research/gap_conditioned_motion_e3_signals_20260711.md). **No A1–A8 tables · no V1–V5 · Phase B still unauthorized** until owner authorization in this thread.
- 2026-07-11: PR #114 review — structural gap closed: prior held-out-only surface (97,136 rows) was insufficient for A6 training-side threshold selection; cube + role tag is the seal revision (signal completion, not Phase B execution).
- 2026-07-11: **D0 packet** ([Issue #112](https://github.com/raylei50653/saccade/issues/112)): initial host-replica path reviewed as insufficient for runtime fidelity. PR #115 review required: (1) fail-closed on capture unavailable / no “exact Consumer-A” claim; (2) `--verify` rebuilds capture from pairs+substrate; (3) single-factor decomposition; (4) hash real headline preset + reachable git commit. Status `D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE`; terminal **`not_fidelity_aligned`**; primary=`runtime_capture_unavailable`; Issue #112 remains **incomplete**. [Note](../../modules/semantic/research/d0_bridge_estimator_fidelity_20260711.md) · [packet](../../modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/manifest.json).
  ```text
  D0 verdict: not_fidelity_aligned
  primary_fail_reason: runtime_capture_unavailable
  Issue #112: incomplete
  E3 signals: sealed
  Phase B: unauthorized
  A1–A8: not executed
  production/default/preset: unchanged
  ```
- 2026-07-11: **Owner authorization recorded; Phase B executed once** over the
  sealed E3 cube. The new [Phase-B packet](../../modules/semantic/research/evidence/gap_conditioned_motion_phase_b_20260711/manifest.json)
  and [result note](../../modules/semantic/research/gap_conditioned_motion_phase_b_20260711.md)
  contain A1–A8, A6 train/held-out firewall evidence, and the single bounded
  verdict **`V5`**. All members fail the non-vacuous A6/no-thinner success box;
  primary `1–10` calibration is under-dispersed. This is representation-level
  only; research acceptance remains pending, D0 stays fail-closed, and no
  production/hook/preset follow-up is authorized.
- 2026-07-11: **Phase-B review correction:** regenerated the packet without
  changing E3, the family, or predeclared criteria. A5 now covers every frozen
  support-layer/cell intersection plus all four M0 atoms; A6 pools captured FP
  counts over \(S_A\) and applies the 0.8 guard after fold pooling; A3/A4/A7/A8
  now emit their committed auxiliary outputs. Criterion and mutation tests
  cover tail, retention, A5 fields, A6 pooling, A8 dominance, and the role
  firewall. The bounded runner verdict remains `V5`; research acceptance is
  still pending.
- 2026-07-11: **Phase-B second review correction:** A5 now scores the frozen
  `1 - dir_cos` mismatch directly (no collision with the raw cosine column);
  the all-gap `11–30` M1 AUC is `0.71884494`, aligned with the E1 direction.
  Added exact synthetic M0-direction assertions and pure V4/V2/V1/V5
  priority-partition transition tests. Packet hashes were regenerated; V5 is
  unchanged and research acceptance remains pending.
- 2026-07-11: **Research-owner acceptance:** sealed Phase-B A1–A8 execution
  **`ACCEPTED_WITH_LIMITS`**; bounded verdict **`V5 ACCEPTED_WITH_LIMITS`**.
  The accepted claim ceiling is representation / level 1. This accepts neither
  production behavior nor threshold transfer: D0 remains fail-closed, and
  tracker / preset / production threshold / online hook work remains
  unauthorized.
