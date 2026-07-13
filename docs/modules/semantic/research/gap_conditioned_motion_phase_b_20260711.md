<!-- doc-status: active -->
<!-- doc-promotion: none; bounded Phase-B result, not ledger evidence -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — Phase B A1–A8 result

> **Execution status:** `PHASE_B_EXECUTED` under the sealed E3 cube. The
> predeclared decision rule emits **`V5`**: the representation + attribution
> contract is **not established**. This is a representation-level offline
> result only. **Research acceptance:** `V5 ACCEPTED_WITH_LIMITS`; claim
> ceiling = representation / level 1. Production, preset, tracker, threshold
> transfer, and hook state remain unchanged and unauthorized.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/closed/gap_conditioned_probabilistic_motion_probe_20260711.md)  
Protocol: [frozen A1–A8 design](gap_conditioned_motion_phase_b_design_20260711.md)  
E3 input: [sealed signals](gap_conditioned_motion_e3_signals_20260711.md) ·
[packet](evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json)  
Packet: [Phase-B evidence](evidence/gap_conditioned_motion_phase_b_20260711/manifest.json)

## Result

All four members retain the two short-gap discrimination cells, have no new
aggregate `E_motion` reversal cell, pass the four-pair exploratory escape
cohort check, and have positive held-out direction on every qualifying fold.
They do **not** establish the five-box contract: `1–10` is under-dispersed
for every member (`c90=0.604` for M1/H270/H90; `0.623` for H30), the native
escape-cohort bins include over-dispersed cases, and A6 cannot establish a
non-vacuous no-thinner result under its frozen train-only CP selection. The
resulting no-thinner box therefore fails for every member.

The packet records the complete predeclared output surface: A3 includes both
`E_motion` and `q_motion` AUC; A4 includes the frozen `bridge_dist` and
`resid_mean` pooled-q90 diagnostics; A5 compares both motion scores with all
four M0 atoms on every support-layer/canonical-cell intersection. A6 pools
captured FP counts over \(S_A\) and applies its 0.8 guard to each fold after
pooling its primary cells. A7 links each held-out report to the frozen A6
threshold state, while A8 records train-NLL selection, held-out NLL, log-det
growth curves, and primary-cell calibration classes.

The three OU members satisfy the predeclared A8 dominance comparison against
M1 on retention, held-out primary GT NLL, and matched-uncertainty condition,
but that cannot yield V2 because none passes all five success boxes. M1 also
does not pass all boxes, so the priority partition selects `V5`, with no
anomaly-note residual.

## Boundaries and limitations

- Analysis begins from `gt_valid=1`, exactly the E0/E1 frozen pair universe.
- A6 selects every candidate threshold exclusively on the E3 `train` role;
  held-out rows are evaluated only after selection. `BOTH_EMPTY` cells never
  support a pass.
- D0 remains `not_fidelity_aligned`: numeric `bridge_dist` threshold transfer
  is not supported. `E_motion` has no Consumer-A counterpart, so this result
  is claim-level 1 representation evidence only.
- `V5` does not mean “only over-diffusion” or a production NO-GO. It means the
  sealed position-only representation and attribution contract did not meet
  its complete predeclared success box.

## Reproduction

```bash
uv run python \
  docs/modules/semantic/research/evidence/gap_conditioned_motion_phase_b_20260711/run_phase_b.py \
  --verify
```

`--verify` first rebuilds and byte-verifies E3 from the frozen pair table in a
temporary directory, then reproduces the Phase-B packet byte-for-byte.
