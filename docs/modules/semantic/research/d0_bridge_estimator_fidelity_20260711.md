<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# D0 — Consumer-A bridge estimator fidelity audit

> **Status:** `D0_SEALED`
> **Terminal verdict:** `not_fidelity_aligned`
> **Issue:** [#112](https://github.com/raylei50653/saccade/issues/112)
> **Packet:** [evidence/d0_bridge_estimator_fidelity_20260711/](evidence/d0_bridge_estimator_fidelity_20260711/)

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)

---

## Goal

Determine whether numeric threshold conclusions learned from the frozen offline
relink substrate (`bridge_dist` and residual atoms) transfer to **Consumer A**,
the active tracker-core CUDA bridge (`bdist` in
`relink_bidir_propose_kernel`).

D0 evaluates **estimator fidelity only**. It does **not** certify E_motion,
runtime hook replay parity, online intervention safety, preset changes, or
Phase B / A1–A8 conclusions.

```text
E3_SIGNALS_SEALED
Phase B authorization: NONE
A1–A8 execution: UNAUTHORIZED
D0 execution: AUTHORIZED → SEALED
```

---

## Frozen inputs

| Input | Value |
|:--|:--|
| pairs | `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv` |
| SHA-256 | `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17` |
| substrate MOT | `results/MOT17_eval_m_b1_substrate_20260709T092543Z` (relink off) |
| production threshold | `0.4` |
| primary support \(S_A\) | `gt_valid && 1 ≤ gap ≤ 26` |
| primary gap cells | `1–10`, `11–26` |

SHA mismatch fails closed. No pair rebuild, no label edits, no threshold search.

---

## Consumer-A exact surface

| Item | Value |
|:--|:--|
| Kernel | `src/tracking/tracker_gpu.cu` |
| Entry | `relink_bidir_propose_kernel` |
| Primary quantity | `bdist` |
| Also compared | `dist_h`, `fwd_r`, `bwd_r` |
| Anchor | adaptive (`bridge_anchor=2`, `rate_gate=0.03`) |
| Horizon | `la = gap + bridge_at − 1` (`bridge_at=4`) |
| Normalization | bilateral EMA `h_ref = max((ema_lost+ema_cand)/2, 1)` |
| Aggregation | \(w\cdot\tfrac12(\mathrm{fwd}_r+\mathrm{bwd}_r)+(1-w)\cdot\mathrm{dist}_h\), \(w=\sqrt{\mathrm{clip}(s_{lost}/0.12,0,1)}\) |

**Not used as substitutes:** Python semantic relinker, C++ `midpoint_bridge_dist`
mirror (lost-only EMA / different history).

---

## Capture contract

| Field | Value |
|:--|:--|
| Research audit default | **off** (`RESEARCH_BRIDGE_FIDELITY_AUDIT_DEFAULT=False`) |
| Live CUDA event ring | **not implemented** (reserved; default-off) |
| Sealed capture mode | `kernel_formula_substrate_replay` |
| Event key | `(seq, lost_id, cand_id, lost_last_frame, cand_first_frame)` — exact, no fuzzy join |
| Decision path | audit flag does not change accept/reject, ordering, lifecycle, or preset |

**Why substrate replay (not live online dump):** the offline pair universe is a
combinatorial tracklet table from a no-relink MOT dump. Live propose only
evaluates lifecycle-timed newborns (`hit_streak==bridge_at`) against live lost
slots — a different candidate set. Same-event join to frozen `pairs.csv` keys
requires applying the **exact CUDA estimator equations** to the sealed
substrate tracklets that define those keys.

Host replica formulas are line-cited from `bridge_vel4` / `bridge_anchor4` /
speed-weighted `bdist` in `tracker_gpu.cu`. Cand requires ≥4 frames (kernel
early return). Short lost tracks use the kernel's last-point / zero-velocity
branch.

Module: `src/saccade/perception/eval/consumer_a_bridge_fidelity.py`

---

## Join coverage

| Metric | Value |
|:--|--:|
| Offline eligible rows | 24,284 |
| \(S_A\) rows | 2,561 (GT 117 / FP 2,444) |
| Consumer-A captured | 22,465 |
| Exact matched | 22,465 |
| Offline-only | 1,819 |
| Online-only | 0 |
| Duplicate keys | 0 |
| Ambiguous keys | 0 |
| GT coverage on \(S_A\) | **95.73%** (gate: 100%) |
| Overall coverage on \(S_A\) | **95.90%** (gate: ≥98%) |
| Coverage gates | **FAIL** |

Uncaptured \(S_A\) rows are almost entirely candidates with `cand_len < 4`,
which Consumer A never scores (`foot_len[cand] < 4` early return). Under the
frozen coverage gate this is a hard fail → terminal
`not_fidelity_aligned` even before rank/threshold checks.

---

## Overall / GT / FP metrics (matched \(S_A\), `bdist` vs `bridge_dist`)

| Slice | n | Spearman ρ | 95% cluster CI | q85 \|err\| | pred. agree @0.4 | GT offline-safe/online-unsafe |
|:--|--:|--:|:--|--:|--:|--:|
| overall | 2,456 | 0.965 | [0.955, 0.972] | 0.213 | 0.950 | 84 (3.4%) |
| GT | 112 | 0.714 | — | 0.469 | **0.759** | **24 (21.4%)** |
| FP | 2,344 | 0.966 | — | 0.317 | 0.959 | 60 (2.6%) |

Cluster unit for CI: `(sequence, lost_id)`. Bootstrap seed `20260711`, 400 reps.

**Reading:** overall rank correlation is high, but the **GT boundary is
distorted** — exactly the failure mode named in the production substrate
mapping §6. Numeric threshold transfer is blocked by both coverage failure and
GT predicate / q85 / safe→unsafe counts.

Full tables: `metrics_overall.csv`, `metrics_by_gap.csv`,
`metrics_by_sequence.csv`, `quantile_alignment.csv`, `predicate_confusion.csv`.

---

## Gap-cell metrics

Primary cells on matched \(S_A\):

| Cell | GT n (matched) | Notes |
|:--|--:|:--|
| gap 1–10 | see packet | GT Spearman / predicate below threshold-transfer floors |
| gap 11–26 | see packet | same |

Supplementary long-gap diagnostics are out of verdict scope.

---

## Predicate confusion at 0.4

Fixed production predicate only: `safe := value ≤ 0.4`.

Headline (matched \(S_A\)):

| | online-safe | online-unsafe |
|:--|--:|--:|
| offline-safe | 118 | **84** |
| offline-unsafe | 40 | 2,214 |

GT offline-safe / online-unsafe = **24** (rate **0.214**) — far above the
threshold-transfer ceiling (count ≤ 1, rate ≤ 0.02).

Boundary band `[0.35, 0.45]`: see `boundary_diagnostics.csv` (localization
only; does not alter the frozen verdict rule).

---

## Estimator decomposition

Progressive rebuild on matched \(S_A\) (speed-weighted surface vs D4):

| Step | Spearman vs D4 | q85 \|err\| vs D4 | pred. agree @0.4 | GT step-safe / D4-unsafe |
|:--|--:|--:|--:|--:|
| D0 offline `bridge_dist` (midpoint) | 0.965 | 0.213 | 0.950 | 24 |
| D0 offline speed-weighted | 0.987 | 0.174 | 0.972 | 8 |
| D1 CA velocity only | 0.989 | 0.200 | 0.982 | 7 |
| D2 + CA horizon (`la`) | 0.993 | 0.169 | 0.995 | 2 |
| D3 + CA normalization (EMA `h_ref`) | 1.000 | 0.000 | 1.000 | 0 |
| D4 exact captured Consumer-A | 1.000 | 0.000 | 1.000 | 0 |

**Attribution:**

1. Aggregation formula (midpoint offline `bridge_dist` vs speed-weighted) is a
   non-trivial but secondary gap (D0 mid → D0 sw).
2. **Horizon** (`gap` → `la = gap+3`) is a major remaining driver (D1 → D2).
3. **Normalization** (raw endpoint mean → bilateral EMA) finishes alignment
   (D2 → D3); D3 ≡ D4 on this sealed path.
4. Velocity (window-mean → adaptive anchor-4) helps rank slightly; alone it
   does not repair the GT threshold boundary.

Errors are **not purely additive** — residual GT safe→unsafe collapses only
when velocity, horizon, and normalization are combined.

---

## Disagreement localization

`offline-safe / online-unsafe` rows on matched \(S_A\) are listed in
`disagreement_localization.csv` with:

- gap / sequence / GT vs FP
- offline `h_ref` vs Consumer-A EMA `h_ref`
- true gap vs `la`
- `fwd_r` / `bwd_r` / `dist_h`
- pooled quantile bins (`≤q25` … `>q75`) for descriptive regimes only

**Top regimes:** multi-sequence; both short (1–10) and long (11–26) cells;
GT disagreements individually keyed. No single sequence owns the GT
distortion exclusively. Horizon + EMA together dominate residual threshold
disagreement after velocity swap.

---

## Exact terminal verdict

```text
not_fidelity_aligned
```

Frozen rule (predeclared; not retuned after seeing data):

1. Coverage gates **FAIL** (GT coverage on \(S_A\) < 100%, overall < 98%) →
   mandatory `not_fidelity_aligned`.
2. Independently, rank-only floors also fail on GT Spearman (0.714 < 0.75) and
   threshold-transfer floors fail on q85 error, GT predicate agreement, and GT
   safe→unsafe count/rate.

**Interpretation:** the frozen offline `bridge_dist` estimator is **not** an
adequate production proxy for Consumer-A `bdist` within \(S_A\). Ordering
morphology must not be treated as threshold-transferable; D0 does not authorize
porting numeric offline bridge thresholds to Consumer A.

---

## Claim ceiling

| Claim level | Allowed after this D0? |
|:--|:--|
| 1. Signal-level regularity on offline table | yes (unchanged) |
| 2. Production-aligned numeric threshold on Consumer A | **no** |
| 3. Online intervention acceptance | **no** (not in scope) |
| 4. Pipeline-safe intervention | **no** (not in scope) |

Phase B / A1–A8 remain unauthorized. E3 signals remain sealed.

---

## Limitations

1. Capture is kernel-formula substrate replay, not a live CUDA event ring.
   Live ring plumbing remains default-off / unimplemented; level-3 online
   parity is a separate gate.
2. EMA heights are reconstructed by causal tracklet replay
   (`α=0.05`), not dumped from a live GPU buffer. D3≡D4 validates internal
   consistency of that sealed path, not bit-identity with a particular online
   float32 run.
3. Coverage is incomplete for offline pairs with `cand_len < 4` (Consumer A
   never evaluates them). That incompleteness fails the frozen coverage gate by
   design.
4. Offline `bridge_dist` is the midpoint builder atom; Consumer A uses
   speed-weighted aggregation. Decomposition isolates that as one of several
   non-transfer factors.
5. No threshold search, quantile mapping search, or offline estimator repair
   was performed.

---

## Reproduction

```bash
uv run python \
  docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/run_d0_bridge_fidelity.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv

uv run python \
  docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/run_d0_bridge_fidelity.py \
  --capture docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/consumer_a_capture.csv.gz \
  --verify

uv run pytest tests/unit/test_d0_bridge_estimator_fidelity.py -q

uv run ruff check \
  docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711 \
  src/saccade/perception/eval/consumer_a_bridge_fidelity.py \
  tests/unit/test_d0_bridge_estimator_fidelity.py

uv run mypy \
  docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/run_d0_bridge_fidelity.py \
  src/saccade/perception/eval/consumer_a_bridge_fidelity.py
```

---

## Expected final state

```text
D0: SEALED
D0 verdict: not_fidelity_aligned
E3: SEALED
Phase B: UNAUTHORIZED
A1–A8: NOT_EXECUTED
production/default/preset: UNCHANGED
```
