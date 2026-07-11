<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — E1 deterministic baseline

> **E1 terminal:** the aggregate pair-level baseline does **not** show a
> marginal sign flip under the frozen definition. All 20 `(gap bin, M0 atom)`
> cells retain GT AUC above 0.5 and none meet the frozen descriptive reporting
> reversal criterion. `bridge_dist` and `resid_mean` nevertheless lose marked
> discrimination as gap grows. The earlier four-track escape-tail finding
> remains a local conditional phenomenon, not an aggregate gap-bin reversal.
> Research acceptance = **`ACCEPTED_WITH_LIMITS`**; probabilistic
> representation verdict = **`NOT_YET_EVALUATED`**.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)  
E0: [substrate audit](gap_conditioned_motion_e0_20260711.md)  
Packet: [evidence/gap_conditioned_motion_e1_m0_20260711/](evidence/gap_conditioned_motion_e1_m0_20260711/manifest.json)

## 1. Frozen E1 protocol

Universe is the E0-sealed `gt_valid` `U_relink_pair` pool. Gap bins remain
`1–10`, `11–30`, `31–60`, `61–150`, `151–300`.

| Stored atom | Mismatch used by E1 | Expected GT direction |
|:--|:--|:--|
| `bridge_dist` | identity | lower |
| `speed_mismatch` | `abs(lost_exit_speed - cand_entry_speed)` | lower |
| `dir_cos` | `1 - dir_cos` | lower mismatch / higher cosine |
| `resid_mean` | `0.5 * (fwd_resid + bwd_resid)` | lower |

Per cell E1 records:

- tie-aware GT AUC using `-mismatch`;
- a within-bin pooled q90 high-mismatch tail, inclusive on ties;
- tail GT enrichment relative to that bin's GT base rate;
- descriptive reversal only if `AUC < 0.5` **and** q90 GT enrichment `> 1`.

This frozen descriptive criterion is used for E1 reporting; it is not a
hypothesis-test threshold or model-family acceptance gate.

## 2. Aggregate result

GT AUC (`-mismatch`; higher is better):

| Signal | 1–10 | 11–30 | 31–60 | 61–150 | 151–300 |
|:--|--:|--:|--:|--:|--:|
| `bridge_dist` | 0.963 | 0.920 | 0.862 | 0.807 | 0.754 |
| `speed_mismatch` | 0.519 | 0.568 | 0.663 | 0.636 | 0.624 |
| `dir_cos` | 0.719 | 0.719 | 0.693 | 0.654 | 0.655 |
| `resid_mean` | 0.957 | 0.912 | 0.851 | 0.798 | 0.752 |

Result summary:

```text
cells evaluated:                 20
cells with GT AUC < 0.5:         0
q90-tail GT enrichment > 1:      0
descriptive reversal cells:      0
Phase B authorized:              false
```

The strong gap trend is in the geometric/residual composites: `bridge_dist`
drops by about 0.21 AUC from shortest to longest bin, and `resid_mean` by about
0.20. Direction also weakens. `speed_mismatch` is near-random at short gap and
does not exhibit the same monotone pattern.

## 3. Interpretation and claim boundary

E1 rejects the broad reading “the whole long-gap pair distribution reverses
sign.” It does **not** retract the sealed PR-C observation: the four
far-Hamming GT tracks were a selected conditional escape tail, all in
MOT17-10-SDP, and need not occupy the pooled within-bin q90 after 21,449 FP
pairs are included.

The sharper working statement is therefore:

```text
deterministic motion discrimination degrades with gap;
the supported role reversal is localized/conditional rather than marginal;
a probabilistic model must preserve aggregate short-gap ordering while
addressing the declared escape-tail cohort without unrestricted diffusion.
```

No V1–V5 verdict follows from E1. A probabilistic family that merely improves
pooled AUC while leaving the escape cohort untouched does not satisfy the
thread's success boxes.

### 3.1 Review acceptance boundary

PR #109 research review records:

```text
Engineering / reproducibility: PASS
E0 substrate verdict: ACCEPT
E1 marginal baseline: ACCEPTED_WITH_LIMITS
Probabilistic representation verdict: NOT_YET_EVALUATED
Production / hook authorization: NONE
Next authorized task: E2 position-only M1-P / M2-P family freeze
```

The limit is procedural as well as statistical: the exact AUC+q90 reporting
criterion was introduced in the same PR as its results and has no earlier
immutable task seal. It is reproducible and frozen for this E1 packet, but it
must not be described as predeclared or used as a confirmatory acceptance gate.

E0 is also a schema-identifiability audit, not a future vector-state acceptance
test. Before any added velocity columns can upgrade velocity/joint
identifiability, E2 or its substrate prerequisite must audit finite values,
usable row and gap-bin support, velocity time windows, coordinate semantics,
and observation provenance. Column names alone are insufficient.

## 4. Reproduction

```bash
uv run python \
  docs/modules/semantic/research/evidence/gap_conditioned_motion_e1_m0_20260711/run_e1_m0.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv \
  --verify

uv run pytest tests/unit/test_gap_conditioned_motion_e1_m0.py -q
```

The packet stores the full 20-cell table and per-tail GT sequence attribution.
It is D1 descriptive research only: no ledger, production, preset, or online
hook promotion is authorized.
