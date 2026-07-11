<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Boolean-domain / gap-conditioned motion — production substrate mapping

> **Step-0 audit terminal (rev. 2, post PR #111 review):** `CONSUMER_SPLIT`.
> Production relink is **three distinct consumers** with different decision
> algebra and different reachable gap supports; no single verdict covers them.
> The active headline consumer (tracker-core bridge) ranks by a **continuous
> speed-weighted motion aggregate** — the earlier claim that motion atoms
> never enter an aggregate score was wrong and is withdrawn. What survives:
> the aggregate and its components are **production-native counterparts of
> the frozen atom family** (same names/formulas/ABI, estimator-shifted
> numerically, §2.2), computed natively in the kernel, and a default-off
> Boolean OR-tail hook over that family already ships in that kernel
> (plumbing only; level-3 acceptance pending, §2.3). Three bounded gaps (consumer-
> specific support, estimator fidelity, candidate context) bind E3/A1–A8 and
> any deployment-facing claim. This document is a **binding precondition**
> for E3 analysis design.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)

Audited surfaces (2026-07-11, branch base `99c163da`):

- **Consumer A** — tracker-core bidirectional bridge (CUDA):
  `src/tracking/tracker_gpu.cu` (`relink_bidir_propose_kernel` L1952–L2170,
  commit kernel L2175); C++ mirror `src/tracking/tracker_gpu_python.cpp`
  (`midpoint_bridge_dist` L1404).
- **Consumer B** — optional Python semantic relinker:
  `src/saccade/perception/eval/relink.py` (`resolve()` L1433, gate helpers
  L889–L1007).
- **Consumer C1** — Cheb-GR semantic live claim:
  `src/saccade/perception/eval/relink.py` (`_cheb_gr_claim_best` L1236).
- **Consumer C2** — evfifo live-bank output handover:
  `src/tracking/cheb_gr_online.cpp` (`causal_handover_lines`, gap gate
  L246–L247), `src/saccade/perception/eval/cheb_gr_online.py`,
  `include/tracking/cheb_gr_online.hpp`; live-bank switch
  `src/saccade/perception/eval/pipeline.py` L908–L928.
- Wiring/presets: `configs/presets/mamba_whole_graph_m.yaml`
  (`reid_mode: off`, `relink_bridge_enabled: true`, `relink_bridge_px: 0.4`),
  `configs/presets/mamba_whole_graph_m_extract_ho_live.yaml`,
  `src/saccade/perception/eval/pipeline.py` (L714–L717, L900),
  `src/saccade/perception/eval/config.py` (`track_buffer: 30`, L1355).
- Offline research substrate builder:
  `scripts/tools/build_relink_candidates.py` (source of the frozen
  `pairs.csv`, SHA `0ae38967…`, see E0).

## 1. Consumer split

| Consumer | Active in headline `m`? | Decision quantity | Motion algebra |
|:--|:--|:--|:--|
| **A** tracker-core bridge | **yes** (`relink_bridge_enabled: true`, `reid_mode: off`) | continuous `bdist` vs `bridge_px = 0.4` + distance ranking + margin | **continuous speed-weighted aggregate** of motion atoms |
| **B** Python semantic relinker | no (reid presets only) | Boolean gate chain, then appearance/unified scoring | Boolean pre-gates; motion enters soft scoring only in non-default configs |
| **C1** Cheb-GR semantic live claim | no (lives inside B) | pooled-low-mean appearance cost + max-cost + runner-up margin | inherits B's Boolean motion/spatial pre-gates |
| **C2** evfifo live-bank output handover | separate ho-live preset track | pooled-low-mean appearance cost + max-cost + runner-up margin | **gap window only** (`1 ≤ gap ≤ max_gap`); no motion pre-gates |

**Consumer A governs E3/A1–A8 deployment claims**: it is the default-on
headline surface, owns `bridge_px = 0.4`, and already contains the research
hook plumbing. B, C1, and C2 are named secondary surfaces with their own
supports; results for one consumer must not be reported under another's
support.

## 2. Consumer A — active tracker-core bridge

### 2.1 Decision algebra

Per newborn track, fired **once** at `hit_streak == bridge_at (= 4)`, over
live lost tracks (`CONFIRMED`, unmatched, `bridge_min_lost ≤ la ≤ bridge_ttl`):

Hard pre-gates (each reject-only): scale ratio (EMA `h` ratio clipped to
`[0.6, 1.7]`), physical max-speed, optional spatial gate (off in headline),
optional gap-occupancy gate (off in headline), appearance veto
(`cos < app_veto_cos`, only when both embeddings exist), and the default-off
portable OR-tail hook (§2.3). Survivors are scored by

\[
\text{bdist}
= w\cdot\tfrac{1}{2}(\text{fwd}_r+\text{bwd}_r) + (1-w)\cdot\text{dist}_h,
\qquad
w=\sqrt{\operatorname{clip}(s_{lost}/0.12,\,0,\,1)}
\]

(`tracker_gpu.cu` L2058–L2065; optional directional blend `bridge_dir_bonus`
is `0.0` in headline). Accept requires `bdist ≤ bridge_px (0.4)` **and**
winning the distance ranking with runner-up margin
`second − best ≥ bridge_margin (0.05)` (L2098, L2157–L2161); conflicting
claims resolve by detection score (atomicMax, L2167–L2169).

So the active consumer **does** aggregate motion atoms into a continuous
scalar that serves as both threshold and ranking key, with a
state-conditioned weight (exit speed). The compensating structural fact:
`bdist` **is itself a frozen research atom** (`score_m_bridge`), and its
components `fwd_r`, `bwd_r`, `dist_h` are computed natively in the same
kernel. Boolean-domain research on this consumer therefore operates on
production-native quantities, but must treat the aggregate as an atom and
must not describe the surface as gate-only.

### 2.2 Estimator deltas vs the offline builder (D0 targets)

Same-named atoms differ between `pairs.csv` and the kernel:

1. **velocity**: kernel `bridge_anchor4` anchored regression on the foot
   ring (lost last-4 / cand head-4) vs offline window-mean `_velocity`;
2. **extrapolation horizon**: kernel extrapolates over `la` (= pair gap
   `+ bridge_at − 1`) to the current frame vs offline over the true pair
   gap;
3. **`h_ref`**: kernel `max((ema_lost + ema_cand)/2, 1)` vs offline mean of
   raw endpoint heights. (The C++ mirror `midpoint_bridge_dist` uses
   lost-side EMA only and a midpoint formula — a third estimator; D0 must
   name which path it certifies.)

### 2.3 Native hook (claim-ladder level-3 plumbing available; acceptance pending)

`relink_bidir_propose_kernel` contains a default-off **portable OR-tail**
hook (L2130–L2156): after all baseline gates pass, a frozen threshold vector
over atoms `[score_m_bridge, abs_log_h, dist_h, abs_ratio_m1, resid_mean]`
rejects the pair if any tail fires; disabled it is a bit-identical no-op.

This establishes **level-3 plumbing** for consumer A — no new wiring is
needed. It does **not** establish the level-3 verdict: level 3 requires the
predicate to be *exactly replayed* on the live path, and the full
candidate-event online audit is explicitly not implemented
(`ONLINE_BAUDIT_IMPLEMENTED = False`, `portable_or_tail.py` L67). Level-3
acceptance remains pending activation, disabled-arm no-op verification, and
live atom/predicate parity evidence. The D0 gate (§6) resolves estimator
fidelity; it does not by itself prove hook runtime replay parity.

### 2.4 Reachable support \(S_A\)

Lifecycle derivation (headline `m`):

- lost tracks: `predict` increments `age` then deactivates at
  `age ≥ max_age = track_buffer = 30` (`tracker_gpu.cu` L76), so the bridge
  sees `la ∈ [bridge_min_lost, 29] = [2, 29]`; `bridge_ttl = 120` is **not
  binding** under the headline lifecycle;
- newborn fires at `hit_streak = 4`, so the pair gap in `pairs.csv`
  semantics is `gap = la − bridge_at + 1 = la − 3` (L2099);
- therefore

\[
S_A=\{1\le\text{gap}\le 26\},
\]

with `la ∈ {2, 3}` producing `gap ≤ 0` temporally-overlapping pairs that lie
**outside** the frozen substrate (builder gates `gap ≥ 1`). The boundary 26
derives from `track_buffer − bridge_at + 1` minus the predict-order
off-by-one; if `track_buffer` changes, \(S_A\) moves with it.

Substrate mass inside \(S_A\): canonical bin 1–10 fully (1,037 pairs / 53
GT) and bin 11–30 partially (row-level `gap ≤ 26` of 1,895 pairs / 76 GT) —
at most 2,932 / 21,789 `gt_valid` pairs (**≤ 13.5%**) and ≤ 129 / 340 GT
(**≤ 38%**). Bins 31–60, 61–150, 151–300 are entirely unreachable for the
active consumer.

## 3. Consumer B — optional Python semantic relinker

Active only under reid presets (`reid_mode` ≠ off). Decision structure: a
Boolean hard-gate conjunction (age window, max-speed, direction, Kalman chi²
— position-only `dims=2` when bidirectional —, scale ratio, midpoint bridge
distance, spatial, min-sim) followed by appearance-led scoring
(`relink.py` L1433–L1814). In its default configuration motion atoms enter
as reject-only Booleans; unified/legacy weighted scoring paths exist but are
config-gated. Support: `semantic_min_lost_frames = 2`, `semantic_ttl = 45`
(pipeline.py L714/L717; no preset overrides) →

\[
S_B=\{2\le\text{gap}\le 45\}.
\]

\(S_B\) applies **only** when this consumer is enabled; it is not the
headline support and must not be reported as such.

## 4. Consumers C1 / C2 — the two Cheb-GR paths

Both rank by the same appearance-only pooled cost —

\[
c_{\mathrm{app}}(i)
=\operatorname{mean}\bigl(\text{lowest }\lceil\text{pool\_frac}\cdot n_i\rceil
\text{ pairwise head×bank distances}\bigr),
\]

accepted iff \(c_{\mathrm{app}}(i^*)\le\) `max_cost (0.45)` and runner-up
margin ≥ `margin (0.05)` — but their upstream screens and lifecycles differ
and must not be conflated:

- **C1 — semantic live claim** (`_cheb_gr_claim_best`, relink.py L1236):
  runs inside consumer B's `resolve()`, so it **inherits B's Boolean
  motion/spatial pre-gates** and B's candidate lifecycle; its reachable
  support is \(S_B\) (§3), conditional on that consumer being enabled.
- **C2 — evfifo live-bank output handover**: under the ho-live preset
  (`cheb_gr_online_live_bank: true`), the pipeline **disables the
  per-frame handover** (`enabled = not _live_bank`, pipeline.py L908–L928)
  and runs `causal_handover_lines` over output tracklets at sequence end.
  Its only motion screen is the tracklet gap window
  (`gap >= 1 && gap <= max_gap`, `cheb_gr_online.cpp` L246–L247); there
  are **no inherited motion pre-gates**. Support with `max_gap = 60`:

\[
S_{C2}=\{1\le\text{gap}\le 60\}.
\]

The \(\mu-\lambda\sigma\) Chebyshev threshold belongs to the **separate
birth-bank relink path** (`cheb_lambda`, off in headline) and must not be
written as either Cheb-GR objective.

## 5. Support rules (binding)

| Cut | Consumer | Role | Claim level |
|:--|:--|:--|:--|
| \(S_A\): gap ∈ [1, 26] (row-level) | A (active bridge, `bridge_px = 0.4`) | **headline deployable surface** | **primary** |
| \(S_{C2}\): gap ∈ [1, 60] (row-level) | C2 (evfifo output handover) | ho-live consumable surface | secondary deployment |
| \(S_B\): gap ∈ [2, 45] (row-level) | B (semantic relinker; also bounds C1) | conditional (reid presets only) | secondary, conditional |
| gap beyond the named consumer | — | future TTL/lifecycle extension | exploratory only |

- All cuts are **row-level** gap filters. Canonical bins remain frozen for
  reporting; bin 11–30 straddles the \(S_A\) boundary (26) and bin 31–60
  straddles the \(S_B\) boundary (45) — bin-level slicing cannot express
  either cut.
- **Standalone-support rule:** every deployment claim must hold on the
  named consumer's reachable support standalone. All-gap results are a
  scientific supplement and may not drive a deployment narrative (≥ 86% of
  substrate pairs are unreachable for the active consumer).
- Supports are lifecycle-derived, not free parameters: \(S_A\) moves with
  `track_buffer`/`bridge_at`; changing them is a production change, not a
  reporting choice.

## 6. D0 gate: bridge-score estimator fidelity

Threshold-level conclusions may not transfer until a bounded D0 study
computes, on the same relink events, the offline-builder atoms and the
**consumer-A kernel** quantities (`bdist`/`score_m_bridge`, `dist_h`,
`fwd_r`/`bwd_r`; naming the kernel path, not the C++ mirror), and reports:

| Metric | overall | GT | FP | gap bins |
|:--|--:|--:|--:|:--|
| Spearman rank correlation | | | | |
| quantile alignment error (e.g. q85↔q85) | | | | |
| predicate agreement @ production threshold (0.4 on `bdist`) | | | | |
| offline-safe / online-unsafe count | | | | |

The named estimator deltas of §2.2 are the prior suspects; disagreement must
be localized (gap / height / velocity regime), not just counted. GT/FP-
conditional agreement is mandatory: high overall correlation with a
distorted GT boundary still blocks threshold transfer.

Verdict vocabulary (exactly one):

```text
threshold_transfer_supported
rank_only_transfer_supported
not_fidelity_aligned
```

`rank_only_transfer_supported` keeps morphology/rank research valid but
forbids porting numeric thresholds.

## 7. Atom classification (binding vocabulary)

Classification is **per consumer**; claims must name both atom class and
consumer. For consumer A:

| Class | Atoms | Transfer rule |
|:--|:--|:--|
| **production-native** (computed in the active kernel) | `gap` (`gap_len`), `score_m_bridge` (= `bdist`), `dist_h`, `fwd_r`/`bwd_r`, `resid_mean`, `abs_log_h`, `abs_ratio_m1` | numerically **estimator-shifted** vs the offline builder (§2.2); threshold transfer requires the D0 gate |
| **reconstructable** | E2 observation \(d=\Delta\text{foot}/h_{ref}\) as a 2-D vector | ingredients (foot ring, EMA `h`) in scope; a hook must compute it before any claim binds |
| **research-only** | \(E_{motion}\) (E3 NLL), `dir_cos` as a retained scalar | no consumer-A counterpart; deployment claims require naming a concrete consumer first |

For consumer B, the earlier proxy-aligned vocabulary applies to
`_midpoint_bridge_dist`, `log_h_ratio`, `speed_mismatch`, `dir_cos`
(Boolean gates over near-analogs; C1 inherits these). For consumer C2,
motion atoms beyond the gap window have no counterpart.

## 8. E3 headline constraint

E3/A1–A8 must not use the full 1–300 gap population as the primary
statistic. The layered design is:

- **Primary:** \(E_{motion}\) on \(S_A\) (the governing consumer);
- **Secondary:** \(E_{motion}\) on \(S_{C2}\) (and \(S_B\) when consumer
  B/C1 is under study);
- **Exploratory:** \(E_{motion}\) on long-gap support.

The acceptance question is fixed as:

> On the governing consumer's production-reachable support, does the
> population-level motion prior provide stable conditional information
> beyond the existing decision quantity (`bdist` ranking for A; per-track
> Kalman chi² for B)?

"Can \(E_{motion}\) describe the motion landscape across all gaps?" is
retained as theoretical exploration only and cannot decide the production
research line. This constrains analysis/claims only; the sealed E2 family,
LOO firewall, and E3 output contract (all four members' scores on all
pairs) are unchanged.

## 9. Claim ladder (binding)

1. **Signal-level regularity** — holds on the static pair table.
2. **Production-aligned predicate candidate** — support (§5) and fidelity
   (§6) both pass for a named consumer.
3. **Online intervention candidate** — a default-off hook replays the
   predicate exactly on the live path. For consumer A the plumbing exists
   (portable OR-tail, §2.3) but acceptance is pending activation,
   disabled-arm no-op, and live parity evidence; for B/C1/C2 it requires
   new plumbing kept bit-faithful across Python/GPU/C++ mirrors or
   explicitly fallback-gated.
4. **Pipeline-safe candidate** — A/B shows no unacceptable harm from greedy
   claims, score-keyed conflict resolution, or downstream candidate-set
   rewriting (live feedback compounding is a known, quantified effect).

Levels must not be skipped; candidate-context effects can only be
discharged at level 4 by a default-off intervention (D stage). This does
not devalue lower-level results — it names them.

## 10. Deployment story anchor

The strongest math→policy closure available on this line:

\[
\max_D P_{\mathrm{FP}}(D)\ \text{s.t.}\ \operatorname{UCB}[P_{GT}(D)]\le\epsilon
\]

maps onto consumer A's gap-independent `relink_bridge_px = 0.4` as a
**gap-conditioned threshold schedule** for `bdist` over \(S_A=[1,26]\).
Any such proposal inherits every constraint above (consumer naming, support
rule, D0 gate, claim ladder).
