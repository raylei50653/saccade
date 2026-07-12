---
doc-status: active
doc-promotion: normative protocol; binding on any claim that a quantity represents production runtime
owner-module: research/eval
created: 2026-07-12
origin: Issue #112 (D0 runtime shadow bridge fidelity)
---

# Runtime-quantity fidelity protocol

> **One-line:** a formula that *looks like* a production quantity is not that
> quantity. Any offline signal claiming to represent a production runtime value
> must earn that claim by measurement, through this protocol, before any
> conclusion drawn from it may be read as a statement about production.

## 0. Why this exists

Issue #112 measured the offline proxy `score_m_bridge` against the live CUDA
`bdist` it was believed to represent. The two share the *same algebraic form*.
They are not the same quantity:

| | Bar | Observed |
| --- | --- | --- |
| decision agreement @ `≤ 0.4` | ≥ 99 % | **95.07 %** |
| \|Δ\| q95 | ≤ 0.05 | **1.417** (3.54× the whole threshold) |
| Spearman ρ | ≥ 0.98 | **0.9558** |

The formula shape was identical; the **estimators feeding it** were not
(`bridge_anchor4` foot-ring OLS + adaptive edge weighting + EMA heights, versus
window-mean velocity + raw endpoint heights), and the divergence concentrated
exactly where those inputs entered. The static term transferred (ρ = 0.991);
everything touching velocity extrapolation did not (`w`: ρ = 0.866).

**The lesson generalizes.** Semantic identity is never inherited from a shared
name or a shared formula. It is established by measurement or not at all.

## 1. Scope — what this protocol binds

**Binding** on any parameter, score, threshold, or derived quantity where a
document or a research conclusion claims it *represents a production runtime
quantity* — i.e. where a conclusion about the offline quantity is intended to
say something about what production does.

**Not binding** on signals defined purely in the offline domain and used only
for capability maps or morphology. Those may remain offline quantities — **but
the document must say so explicitly**, and they may no longer be assumed
transferable to production by default. An unlabelled quantity is treated as
claiming runtime representativeness and therefore falls under this protocol.

## 2. The protocol

### 2.1 Declare (before any measurement)

Name, as separate axes:

* the **runtime target** (the exact production quantity, by source location);
* the **proxy** (the offline quantity, by construction);
* the **use**: `threshold` / `ranking` / `morphology-only`.

The use determines which terminals in §2.7 are even reachable. A proxy used to
rank does not need threshold fidelity; a proxy used to gate does.

### 2.2 Capture the real quantity without changing production behaviour

Obtain the runtime value through a **shadow capture**: observe the production
path without altering its output. The capture must be default-off and provably
decision-neutral.

> **The hazard that makes this non-trivial.** In #112 the capture lived inside a
> kernel that could only run when the bridge was *enabled* — but an enabled
> bridge *committed*, rewriting the very track identities the join key lived in.
> Observation and mutation were entangled, so no capture on any git revision
> could have been joined. The fix was to separate them
> (`set_research_bridge_shadow`: propose and capture, skip commit).

**Requirement:** prove decision-neutrality by **byte-identical output** against
the unmodified path, on the full evaluation set, as a regression test — not by
inspection.

### 2.3 Fix the join contract (versioned)

* **Version the event key.** Never redefine an existing key in place: that
  retroactively reinterprets sealed evidence. Add `v2`, freeze `v1`.
* **Pin the ID universe explicitly.** Internal/local ids and emitted/global ids
  are different universes. In #112 the tracker recorded *local* ids while the
  cohort was built from *global* ids — and a raw-id join silently produced
  **202 false matches**, confined to the sequences where the remap happened to
  be the identity. A join that "works" on a subset is the most dangerous
  failure mode here.
* **Fail closed, never fall back.** An unresolvable id is a partition, not a
  best-effort match.
* **Assert key uniqueness**; do not rely on current kernel behaviour to
  guarantee it.
* **Drop unsound fields** rather than carrying them as deprecated. #112's
  `lost_last_frame` was derived from a capture-local counter minus a track age
  and underflowed to *negative frame indices*; it had passed a smoke test only
  because nothing asserted `frame >= 1`.

### 2.4 Partition exhaustively, and conserve

Every captured runtime event lands in exactly one class:

| Class | Meaning |
| --- | --- |
| `matched` | joins the offline cohort — **the only fidelity analysis set** |
| `cohort_gap` | ids emitted, but the pair was never enumerated offline |
| `unemitted` | the id never reached production output at all |

**Conservation (`matched + cohort_gap + unemitted == total captured`) is a
contract test, not a comment.**

`cohort_gap` and `unemitted` are **not missing values.** They must never be
dropped, imputed, or admitted to an agreement denominator. They are the bound on
how far the fidelity conclusion extrapolates.

### 2.5 Seal before looking

Freeze — before any terminal quantity is computed — the estimators, the decision
boundary and its inclusivity, the tie policy, the quantile method, and every
box threshold. Disclose anything already observed during plumbing that could
have informed a bar. No re-fitting the proxy to close a gap: that measures a
*new* proxy, not the one the conclusions were built on.

### 2.6 Measure all five

1. **Threshold decision agreement** at the production boundary, reported as a
   full confusion matrix. The off-diagonal cells are **never netted**: a proxy
   that accepts what production rejects is a different failure from the converse.
2. **Numeric error distribution** — a tail quantile (q95), not a mean or median,
   which would let a mass of mid-sized offsets average away.
3. **Rank correlation** — this is what ranking-class studies actually rest on;
   it is primary for them, not supporting.
4. **Coverage of the decision-relevant region** — the partition composition
   restricted to the region where production *acts*. A proxy validated on a
   region production never enters proves nothing. (In #112 coverage passed,
   which *strengthened* the negative verdict: the proxy failed on the slice that
   counts.)
5. **Estimator-shift mechanism** — per-component attribution, so a failure is
   explained rather than merely recorded.

### 2.7 Terminals (must be distinguished — no collapsing)

| Terminal | Meaning |
| --- | --- |
| **FAITHFUL** | threshold **and** rank fidelity; the proxy may represent the runtime quantity within its measured coverage |
| **THRESHOLD-ONLY FAITHFUL** | gate/threshold conclusions transfer; **ranking conclusions do not** |
| **RANK-ONLY FAITHFUL** | ordering conclusions transfer; **threshold/gate conclusions do not** |
| **PROXY-UNFAITHFUL** | neither; the proxy is an offline quantity only |

Threshold fidelity and rank fidelity are **non-compensatory**. A strong rank
correlation does not rescue a failed decision agreement, and vice versa. There
is no weighted total and no grey band.

Coverage is reported **separately** and constrains which proposal universe any
terminal may be extrapolated to. It never converts one terminal into another.

### 2.8 On failure: amend, never rewrite

A failed fidelity check does **not** license editing historical sealed evidence.
Issue an **append-only validity amendment** that narrows the *scope* of the
affected conclusions, leaving the sealed artifacts and their checksums intact.

> Precedent: `b43772b7` mutated a sealed packet's runner in place without
> updating its checksum inventory, and the fail-closed contract checker caught
> it. The correct remedy was to **restore the sealed bytes** — not to regenerate
> the checksum, which would have laundered the mutation.

A scope limit is not a retraction: a closure established against an offline
ordering remains true *of that ordering*. It simply may no longer be read as a
statement about production.

## 3. Faithful replay (when offline enumeration is genuinely needed)

The mainline should consume the **real captured runtime quantity** directly.
Offline enumeration is justified only when a study needs candidate pairs the
runtime never proposed.

Such a replay must **share or line-by-line reproduce the production estimator** —
never re-fit a similar-looking proxy. At minimum it must align:

* `bridge_anchor4`;
* foot-ring OLS and adaptive edge weighting;
* EMA height;
* the runtime horizon / `la` (note: the kernel's `gap = la − bridge_at + 1` is a
  *different convention* from the offline frame gap — a horizon that differs by
  one frame compounds against an already-divergent velocity);
* clipping, epsilon, precision, and boundary conventions.

A replay must then **re-pass this protocol's fidelity gate**. Until it does, it
must not be described as production-equivalent.

## 4. Checklist

```text
[ ] runtime target, proxy, and use (threshold/ranking/morphology) declared
[ ] shadow capture exists; byte-identical output proven by regression test
[ ] event key versioned; old key frozen, not redefined
[ ] ID universe pinned; no local/global confusion; no raw-id fallback
[ ] key uniqueness asserted; unsound fields dropped
[ ] partition exhaustive + conservation enforced in a contract test
[ ] unmatched/unemitted excluded from every agreement denominator
[ ] estimators, boundary, tie policy, quantile method, boxes sealed pre-terminal
[ ] pre-seal disclosure of anything already observed
[ ] threshold agreement (confusion, un-netted)
[ ] numeric error tail quantile
[ ] rank correlation
[ ] decision-relevant-region coverage
[ ] estimator-shift mechanism attributed
[ ] terminal one of: faithful / threshold-only / rank-only / unfaithful
[ ] on failure: append-only amendment; sealed evidence untouched
```

## 5. Reference execution

Issue #112 is the worked example, end to end:

* Declaration (sealed): [d0_runtime_shadow_fidelity_declaration_20260712](../../modules/semantic/research/d0_runtime_shadow_fidelity_declaration_20260712.md)
* Results (T2 — PROXY_UNFAITHFUL): [d0_runtime_shadow_fidelity_results_20260712](../../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md)
* Amendment: [s0_proxy_validity_amendment_20260712](../../modules/semantic/research/s0_proxy_validity_amendment_20260712.md)
