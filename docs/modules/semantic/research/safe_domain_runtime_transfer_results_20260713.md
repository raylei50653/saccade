# S0 — safe-domain axis transfer to runtime coordinates: results

<!-- doc-status: active -->
<!-- doc-promotion: evidence packet; terminal output pending owner acceptance -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

> **Terminal output: `S0_UNDECIDABLE` (owner acceptance pending).** None of the
> 228 frozen grid points is safe on the offline substrate, so the required active
> offline-safe set is empty. Per declaration §8, this is a fail-closed
> non-portability result: it makes **no** claim that the axes hold, break, or
> degrade at runtime.

Declaration: [sealed Amendment 1](safe_domain_runtime_transfer_declaration_20260712.md) ·
Thread: [runtime-faithful safe domain](../../../research/threads/runtime_faithful_safe_domain_20260712.md) ·
Canonical packet: [manifest](evidence/s0_safe_domain_runtime_transfer_20260713/manifest.json)

---

## 1. Frozen execution and validity

The runner evaluated every frozen point
\(\theta_d \in \{0.2,0.3,\ldots,2.0\}\) ×
\(\theta_r \in \{0.05,0.10,\ldots,0.60\}\), with the exact reviewed
Amendment 1 head `70a40cf9d61eb6512b9b5096049ca59efd58aa95` recorded in the
packet. It selected no point and changed no production setting.

| Gate | Result | Evidence |
| --- | --- | --- |
| V1 provenance | **PASS** | all four declared input SHA256s reproduced |
| V2 partition conservation | **PASS** | `1,684 matched + 539 cohort_gap + 354 unemitted = 2,577` |
| V3 join integrity | **PASS** | unique matched keys, both coordinate systems, and GT flag present |
| V4 exposure floor | **PASS** | 116 GT lost tracks (≥59); 1,684 matched pairs (≥1,000) |
| V5 adversarial unjoined coverage | **NOT EVALUATED** | V7 supplied no active offline-safe point at which to evaluate `M(θ)` |
| V6 no GT leakage | **PASS** | labels used only after coordinate and mask construction |
| V7 non-empty active offline-safe set | **FAIL** | 0 offline-safe / 228 points; therefore 0 active offline-safe points |

The best offline result still has 3 hurt lost tracks out of 116:
\(L_{\mathrm{GT}}=3/116=0.02586\), but its one-sided 95% Clopper–Pearson
upper bound is **0.0654833**, above the frozen \(\varepsilon=0.05\) bar. No
other grid point improves that bound. This is why V7 fails even though the
observed hurt rate at those points is below 5%.

## 2. Ordered terminal application

V1–V4 and V6 pass. V7 then fails, which maps directly to
`S0_UNDECIDABLE` before any runtime safety inversion or agreement diagnostic can
classify the axes as BROKEN or DEGRADED. V5 is intentionally `not evaluated`, not
failed: its raw 893 unjoined events are a coverage diagnostic only and are
defined only at active offline-safe points.

Consequently, the required next condition is a **wider runtime join** before any
closure solve can be considered. This S0 execution does not authorize that join,
the closure solve, threshold selection, a registry/ledger transition, or a
production change.

## 3. Diagnostics (non-terminal)

| Readout | Observed | Frozen bar |
| --- | ---: | ---: |
| `dist_h` offline↔runtime Spearman \(\rho\) | 0.990856 | ≥ 0.98 |
| `abs(log_h_ratio)` offline↔runtime Spearman \(\rho\) | 0.710222 | ≥ 0.98 |
| matched pairs | 1,684 | reporting exposure |
| GT lost tracks | 116 | track-level independence unit |
| FP pairs | 1,373 | pair-level reporting unit |
| unjoined runtime events | 893 | V5 coverage input, never a CP trial count |

The ratio-axis rank diagnostic would fail the agreement bar, but it is not used
to promote this terminal to `AXES_TRANSFER_DEGRADED`: the declaration's ordered
mapping gives V7's `S0_UNDECIDABLE` precedence, and no offline-safe point exists
to test transfer.

## 4. Reproduction record

The canonical packet contains the complete [grid](evidence/s0_safe_domain_runtime_transfer_20260713/grid.csv),
[metrics](evidence/s0_safe_domain_runtime_transfer_20260713/metrics.json), and
[manifest](evidence/s0_safe_domain_runtime_transfer_20260713/manifest.json).
Its runner SHA256 is `07d4eb40433e915a4335f44756ab2e0a3e4b8d5b5f86bfe190193070867da4e2`.

Before publishing the packet, the runner's representation of V5 in the
V7-empty case was corrected from `false` to `not evaluated`. This is a reporting
correction only: it changes no frozen input, grid row, safety statistic, or
terminal. The packet linked above is the canonical regenerated output.

Owner acceptance is required before any accepted research state changes.
