<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Boolean-domain / gap-conditioned motion — production substrate mapping

> **Step-0 audit terminal:** `BOOLEAN_NATIVE_ALIGNED_WITH_GAPS`. The online
> relink decision surface is a conjunction of Boolean hard gates over
> near-raw motion scalars followed by an appearance-only continuous ranking.
> No weighted motion energy, clip, or rank transform exists between the
> research atoms and the production decision. Boolean-domain research is
> therefore describing the production-native decision language, not an
> external surrogate. Three verified gaps bound what may be claimed:
> consumer-specific support, estimator fidelity, and candidate context.
> This document is a **binding precondition** for E3 analysis design
> (A1–A8) and for any deployment-facing claim from the Boolean atom /
> GT-support morphology lines.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)

Audited surfaces (2026-07-11, `main` @ `99c163da`):

- Python relinker decision loop: `src/saccade/perception/eval/relink.py`
  (`resolve()`, gate helpers at L889–L1007), bit-faithful GPU gate-kernel
  A/B path and C++ mirrors included by reference.
- Cheb-GR online handover: `include/tracking/cheb_gr_online.hpp`,
  `src/saccade/perception/eval/cheb_gr_online.py`.
- Wiring/preset values: `src/saccade/perception/eval/pipeline.py`
  (L714–L717, L900), `configs/presets/mamba_whole_graph_m.yaml`,
  `configs/presets/mamba_whole_graph_m_extract_ho_live.yaml`.
- Offline research substrate builder:
  `scripts/tools/build_relink_candidates.py` (source of the frozen
  `pairs.csv`, SHA `0ae38967…`, see E0).

## 1. Production decision equation

For a newborn claim against lost candidates \(z=(\text{signals},s)\):

\[
\mathcal C_{\text{survive}}
=\bigcap_{j=1}^{m}\{z: f_j(z,s)=1\},
\qquad
i^{*}=\arg\min_{i\in\mathcal C_{\text{survive}}}(\mu_i-\lambda\sigma_i)
\]

with an acceptance margin against the runner-up. The \(f_j\) are
sequential hard gates (age window, max-speed, direction, Kalman chi²,
scale ratio, midpoint bridge distance, spatial, min-sim); the continuous
objective is **appearance-only** (Cheb-GR cost). Motion atoms never enter
a weighted sum and are never converted to a retained continuous score.
State conditioning is limited to: `is_clean` switching the appearance
sim threshold, and the direction gate being skipped for near-stationary
tracks. No lifecycle state rewrites motion-signal semantics.

Consequences:

- Boolean atoms, necessary/sufficient conditions, OR-tail rejects, and
  GT-support morphology are structurally isomorphic to the production
  gate chain (framework-level isomorphism).
- Framework isomorphism does **not** imply the offline atoms are
  numerically identical to the production quantities (§3, §5).

## 2. Abstraction chain per research atom

| Research atom | Online existence | Transform vs offline builder | Decision use | Intervention point |
|:--|:--|:--|:--|:--|
| `gap` | `age = frame_id − last_seen` | none, but hard-truncated by support (§4) | candidate filter + time input to bridge/Kalman/speed | in scope pre-gate |
| `bridge_dist` | `_midpoint_bridge_dist` (relink.py L901) / GPU gate col 1 | online: OLS-4 velocity + EMA heights; offline: window-mean velocity + raw endpoint heights | hard threshold `relink_bridge_px = 0.4`, reject-only, **gap-independent** | the existing threshold itself |
| `dist_h` / E2 obs \(d=\Delta\text{foot}/h_{ref}\) | **absent as a decision quantity** | ingredients (foot history, EMA h) in scope | none | reconstructable at gate layer |
| `log_h_ratio` | `_bridge_scale_gate_ok` (L889) | EMA-height ratio, Booleanized by clip to `[0.6, 1.7]`; continuous value not retained | hard gate | ratio in scope inside gate |
| `speed_mismatch` | `_exceeds_max_speed` (L925) | different units (m/s via person height + fps), center- not foot-based | hard gate | in scope inside gate |
| `dir_cos` | `_direction_behind` (L948) | Kalman velocity (not trajectory regression); skipped near-stationary | hard gate | in scope inside gate |
| `fwd/bwd_resid`, `resid_mean`, `score_m_bridge` | none | — | none | research-only |
| \(E_{motion}\) (E3 NLL) | none; nearest relative `_kalman_gate_dist` chi² (L973), already position-only (`dims=2`) when bidirectional | per-track Kalman + covariance inflation vs population marginal | chi² hard gate (`kalman_chi2`) | population prior beside/replacing chi² gate; or gap-conditioned bridge threshold |

## 3. Atom classification (binding vocabulary)

Claims must name which class an atom belongs to; "Boolean structure is
isomorphic" must never be written as "all signals are equivalent".

| Class | Atoms | Transfer rule |
|:--|:--|:--|
| **production-native exact** | `gap` | thresholds transfer directly |
| **reconstructable** | `dist_h`, E2 observation \(d\) | computable from in-scope state; a hook must compute it before any claim binds |
| **proxy-aligned** | `bridge_dist`, `log_h_ratio`, `speed_mismatch`, `dir_cos` | same-name/near-analog with a different estimator; threshold-level transfer requires the D0 fidelity gate (§6); `bridge_dist` upgrades to exact only on `threshold_transfer_supported` |
| **research-only** | `fwd_resid`, `bwd_resid`, `resid_mean`, `score_m_bridge`, \(E_{motion}\) | no production counterpart; deployment claims require naming a concrete consumer first |

## 4. Consumer-specific production support

Two distinct supports exist and must not be merged:

\[
S_{\text{relink}}=\{2\le\Delta t\le 45\},
\qquad
S_{\text{handover}}=\{\Delta t\le 60\}
\]

- `S_relink`: `semantic_min_lost_frames = 2`, `semantic_ttl = 45`
  (pipeline.py L714/L717 defaults; no preset overrides them).
- `S_handover`: `max_gap = 60` (cheb_gr_online.py L143 default;
  `cheb_gr_merge_max_gap`, no preset override).
- Frozen substrate mass beyond both supports (canonical bins 61–150 and
  151–300): 16,308 / 21,789 `gt_valid` pairs (74.8%) and 148 / 340 GT
  (43.5%).
- **Bin caveat:** the canonical bin `31–60` straddles the
  `S_relink` boundary at 45. Any `S_relink` cut must be a **row-level**
  gap filter; bin-level slicing cannot express it. The canonical bins
  remain frozen for reporting; the support cut is an additional
  row-level restriction, not a rebinning.

Required reporting cuts for A1–A8:

| Cut | Role | Claim level |
|:--|:--|:--|
| `S_relink` (gap ≤ 45, row-level) | current relink deployable surface | **headline** |
| `S_handover` (gap ≤ 60, row-level) | Cheb-GR handover consumable surface | secondary deployment |
| gap 61–300 | future TTL-extension | exploratory only |

**Support rule:** every deployment claim must hold on the relevant
production-reachable support **standalone**. All-gap results are a
scientific supplement and may not drive a deployment narrative
(74.8% of pairs are otherwise undeployable mass).

## 5. E3 headline constraint

E3/A1–A8 must not use the full 1–300 gap population as the primary
statistic. The layered design is:

- **Primary:** \(E_{motion}\) on \(S_{\text{relink}}\);
- **Secondary:** \(E_{motion}\) on \(S_{\text{handover}}\);
- **Exploratory:** \(E_{motion}\) on long-gap support.

The acceptance question is fixed as:

> On the current production-reachable support, does the
> population-level motion prior provide stable conditional information
> beyond the existing per-track Kalman gate?

"Can \(E_{motion}\) describe the motion landscape across all gaps?" is
retained as theoretical exploration only and cannot decide the
production research line. If E3 is ultimately hooked beside the chi²
gate, its consumer is the reachable event set by construction. This
constrains analysis/claims only; the sealed E2 family, LOO firewall, and
E3 output contract (all four members' scores on all pairs) are unchanged.

## 6. D0 gate: `bridge_dist` estimator fidelity

Threshold-level conclusions on proxy-aligned atoms may not transfer until
a bounded D0 study computes, on the same relink events, both the
offline-builder and production-path values and reports:

| Metric | overall | GT | FP | gap bins |
|:--|--:|--:|--:|:--|
| Spearman rank correlation | | | | |
| quantile alignment error (e.g. q85↔q85) | | | | |
| predicate agreement @ production threshold (0.4) | | | | |
| offline-safe / online-unsafe count | | | | |

The last row is the transfer-risk quantity:
\(C_{\text{offline}}(x)=\text{safe}\wedge C_{\text{online}}(x)=\text{unsafe}\).
GT/FP-conditional agreement is mandatory: high overall correlation with a
distorted GT boundary still blocks threshold transfer. Disagreements must
be localized (gap / height / velocity regime), not just counted.

Verdict vocabulary (exactly one):

```text
threshold_transfer_supported
rank_only_transfer_supported
not_fidelity_aligned
```

`rank_only_transfer_supported` keeps morphology/rank research valid but
forbids porting numeric thresholds.

## 7. Claim ladder (binding)

1. **Signal-level regularity** — holds on the static pair table.
2. **Production-aligned predicate candidate** — support (§4) and
   fidelity (§6) both pass.
3. **Online intervention candidate** — a default-off hook replays the
   predicate exactly on the live path (Python + GPU gate kernel + C++
   mirror kept bit-faithful, or explicitly fallback-gated).
4. **Pipeline-safe candidate** — A/B shows no unacceptable harm from
   greedy claims, `assigned` mutex, or downstream candidate-set
   rewriting (live feedback compounding is a known, quantified effect).

Levels must not be skipped; candidate-context effects can only be
discharged at level 4 by a default-off intervention (D stage). This does
not devalue lower-level results — it names them.

## 8. Deployment story anchor

The strongest math→policy closure available on this line:

\[
\max_D P_{\mathrm{FP}}(D)\ \text{s.t.}\ \operatorname{UCB}[P_{GT}(D)]\le\epsilon
\]

maps directly onto today's gap-independent `relink_bridge_px = 0.4` as a
**gap-conditioned threshold schedule** for the existing bridge gate. Any
such proposal inherits every constraint above (support rule, D0 gate,
claim ladder).
