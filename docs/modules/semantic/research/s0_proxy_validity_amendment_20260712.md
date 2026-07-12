---
doc-status: active
doc-promotion: append-only validity amendment; normative on scope, silent on findings
owner-module: semantic
created: 2026-07-12
origin: Issue #112 (T2 — PROXY_UNFAITHFUL)
---

# Validity amendment — `score_m_bridge` / `s0` is an offline quantity

> **One-line:** Issue #112 measured `score_m_bridge` against the live CUDA
> `bdist` it was assumed to represent, and it **failed all three fidelity boxes**.
> Every conclusion previously drawn from `s0` is hereby **scoped to the offline
> ordering**. Nothing is retracted; nothing sealed is modified. What changes is
> what those conclusions are permitted to *say about production*.

Evidence: [d0_runtime_shadow_fidelity_results_20260712](d0_runtime_shadow_fidelity_results_20260712.md) ·
Protocol: [runtime-quantity fidelity protocol](../../../research/eval/runtime_quantity_fidelity_protocol.md)

## 1. The measurement

On 1,684 runtime bridge proposals joined exactly to the offline cohort, with
every validity gate passing:

| Box | Bar | Observed | |
| --- | --- | --- | --- |
| B1 decision agreement @ `≤ 0.4` | ≥ 99 % | **95.07 %** | FAIL |
| B2 \|Δ\| q95 | ≤ 0.05 | **1.4171** | FAIL |
| B3 Spearman ρ | ≥ 0.98 | **0.9558** | FAIL |

Terminal: **T2 — PROXY_UNFAITHFUL** (non-compensatory boxes; all three failed).

Coverage **passed** (accept-region composition −1.68 pp from overall), so the
failure is not an artifact of an unrepresentative slice. The proxy failed on the
slice that counts.

## 2. Normative scope limit (binding)

1. **`score_m_bridge` / `s0` is an offline quantity.** It must no longer be
   used, cited, or reasoned about as an equivalent of production `bdist`.
2. **Mainline consumes the real captured `bdist`** from the runtime shadow
   capture. Offline enumeration is justified only where a study needs candidate
   pairs the runtime never proposed — and then only via a **faithful replay**
   that shares or reproduces the production estimator line by line, and that
   re-passes the fidelity gate. Until it does, it is not production-equivalent.
3. **No new score or gate study opens on `s0`** as a production stand-in.
4. Quantities defined purely in the offline domain and used only for capability
   maps or morphology **may remain offline quantities** — but their documents
   must say so explicitly. Unlabelled transferability is no longer assumed.

## 3. Affected conclusions — scope, not retraction

### 3.1 Door-0 ranking-power probe / step ⑤ class closure (#136) — **owner decision required**

The probe's baseline ordering was `s0 = score_m_bridge`. With
ρ(`s0`, `bdist`) = 0.9558 < 0.98, the terminal
`T2 = NO_USABLE_RANKING_POWER_IN_CLASS` is established **against the offline
ordering, not production's ordering**.

* **Still true:** the 12-member Door-0 candidate class contains no candidate that
  usably improves ranking **over `s0`**. The class remains closed with respect
  to `s0`.
* **No longer supported:** reading that closure as a statement about the
  production `bdist` ordering. It does not establish the absence of ranking
  power over production's actual score.
* **This is a scoping limit, not a reversal.** The study is valid; its reach is
  narrower than it was recorded as being.

Whether step ⑤ retains its mainline-transition status under this narrower
reading is an **owner call**, not a research finding, and is deliberately left
open here.

### 3.2 `m_b1` production-shaped gate-coverage claims

Gate coverage computed on `s0` at the 0.4 threshold **mislabels ≈ 5 % of
accept/reject decisions**, and the error concentrates in the marginal band that
a gate is precisely about. Coverage numbers stated in `s0` terms are offline
coverage numbers. They are not statements about what the production gate reaches.

### 3.3 Sealed D0 reconstruction packet (2026-07-11)

**Unaffected and unchanged.** It already fails closed
(`D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE`) and never claimed runtime capture. It
stays frozen, checksums intact.

> Note: `b43772b7` had mutated that packet's runner in place without updating its
> checksum inventory, tripping the fail-closed contract checker. The sealed bytes
> have been **restored**, not re-checksummed. Historical evidence is not
> redefined by later code.

### 3.4 Other `s0`-referencing notes

Documents that reference `s0` for capability-map or morphology purposes are not
invalidated, but they inherit clause §2.4: they must label `s0` as an offline
quantity and must not assume transfer to production.

## 4. Mechanism (why it failed — so the failure is not mysterious)

The formula shape was right; the **estimators feeding it** were not.

| Component | ρ vs runtime |
| --- | ---: |
| `dist_h` (static geometry) | **0.9909** |
| `bwd_r` | 0.9551 |
| `fwd_r` | 0.9292 |
| `w` (speed weight) | **0.8656** |

Static geometry transfers. Everything involving velocity extrapolation and the
speed weight does not — precisely the terms the kernel computes with
`bridge_anchor4` (foot-ring OLS, adaptive edge-weighted anchor, EMA heights) and
the offline builder approximated with window-mean velocity and raw endpoint
heights. The two also extrapolate over **different horizons** (kernel `la` vs
offline frame gap; `gap = la − bridge_at + 1`).

This is what a faithful replay (§2.2) must align.
