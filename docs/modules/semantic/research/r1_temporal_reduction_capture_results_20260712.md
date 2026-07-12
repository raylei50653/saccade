# R1 temporal-reduction capture — owner terminal results

<!-- doc-status: active -->
<!-- doc-promotion: evidence packet; executed under a sealed preflight declaration -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

> **Terminal: `R1_FAITHFUL`.** Owner review accepts that the sealed
> `r1_temporal_reduction_capture_v1` contract faithfully reconstructs the
> frozen runtime Consumer-A quantity and decision surface **within the
> declared support**. This is **not** a general claim over all bridge
> reductions, anchor modes, or substrates.

Declaration (sealed pre-outcome): [r1_temporal_reduction_capture_declaration_20260712.md](r1_temporal_reduction_capture_declaration_20260712.md)  
Thread: [score temporal-to-stable-domain](../../../research/threads/score_temporal_to_stable_domain_20260712.md)  
Evidence packet: [evidence/r1_temporal_reduction_capture_20260712/](evidence/r1_temporal_reduction_capture_20260712/manifest.json)

---

## 0. Scope of the terminal (binding)

`R1_FAITHFUL` applies **only** to:

| Axis | Sealed value |
|---|---|
| Capture contract | `r1_temporal_reduction_capture_v1` |
| Consumer / quantity | Consumer-A CUDA `bridge_anchor4` temporal reduction → production `bdist` |
| Configuration | headline preset adaptive anchor (mode 2), rate 0.03, bridge_at=4, threshold 0.4 |
| Support | MOT17 train SDP baseline seven sequences |
| Authority calculator | fail-closed device backend `device_bridge_anchor4` (`--require-device`) |
| Float budget | presealed abs tolerance `1e-5` (unchanged after data) |

It is **not** a fidelity claim for center/foot anchors, non-headline presets,
other detectors, or arbitrary runtime substrates. That restriction does not
weaken the terminal; it prevents over-reading.

---

## 1. Owner terminal (accepted)

```text
R1 packet owner review
→ accept R1_FAITHFUL
→ close R1 capture/replay unit
→ authorize separately declared discrete-M capability study
```

### Owner terminal — R1_FAITHFUL

The owner review accepts `R1_FAITHFUL` for the sealed
`r1_temporal_reduction_capture_v1` contract under the declared Consumer-A
headline configuration and seven-sequence MOT17 support.

The authority replay used the fail-closed device backend
`device_bridge_anchor4`. V1–V6, replay-backend provenance, per-field
`1e-5` fidelity, production-predicate preservation, event-local ordering,
serialization determinism, and causal-sensitivity interpretability all passed.

Packet:

`out/r1_temporal_20260712T115004Z/`

Audit record:

`out/r1_temporal_20260712T115004Z/owner_terminal_review.json`

(repo-sealed copy under [evidence/r1_temporal_reduction_capture_20260712/](evidence/r1_temporal_reduction_capture_20260712/))

This terminal establishes that the captured temporal reduction faithfully
reconstructs the frozen runtime Consumer-A quantity and decision surface within
the declared support. It does not establish score utility, outcome
separability, optimality, or production-policy value.

**Authorized next transition:** a separately predeclared discrete-\(M\)
representation-capability study may be opened.

**Not authorized by this terminal:**

- score fitting;
- gate or threshold sweeps;
- preset changes;
- production-policy changes;
- generalization beyond the sealed configuration and declared support.

---

## 2. Validity gates (non-compensatory) — all PASS

| Gate | Result | Notes |
|---|---|---|
| V1 shadow neutrality | **PASS** | 7/7 MOT outputs byte-identical to bridge-off substrate |
| V2 complete capture | **PASS** | every sequence `complete=true`, `overflow_events=0`, count match |
| V3 version / provenance | **PASS** | contract `r1_temporal_reduction_capture_v1`; provenance identical; source/payload/id-map hashes agree |
| V4 causal completeness | **PASS** | windows, branches, finite scalars, `la = gap + bridge_at - 1` |
| V5 native identity | **PASS** | 2577 unique keys; slots/ordinals ≥ 0; 354 missing global IDs reported, not dropped |
| V6 production support | **PASS** | seven sequences each contribute events; accept 311 / reject 2266 |

Any single failure would have been `R0_INVALID`. None failed.

---

## 3. Authority backend provenance

Fail-closed device path only (`require_device=true`):

| Field | Value (authority run) |
|---|---|
| `replay_backend` | `device_bridge_anchor4` |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU (CC 12.0) |
| Architectures | `-arch=native` |
| Helper source SHA | present; matches build meta and live recompute |
| Helper binary SHA | present; matches build meta and live recompute |
| Consumer module SHA | present |
| Production `tracker_gpu.cu` SHA | present |

Missing or host-fallback authority would have been **`R0_INVALID`** (unauditable
calculator), not `R2_UNFAITHFUL`.

---

## 4. Coverage (not only total N)

| Dimension | Value |
|---|---|
| Events | 2577 |
| Per-sequence | 02:355 · 04:121 · 05:657 · 09:78 · 10:565 · 11:90 · 13:711 |
| Accept / reject (`bdist ≤ / > thr`) | **311 / 2266** (both sides present on every sequence) |
| Lost branch | full last-4: 2491 · short-lost: 86 |
| Anchor mode | adaptive (2) on all events |
| `la` range | 2 … 29 |
| Cand-local groups | 618 total · 517 with ≥2 events |
| Event-local order | comparable pairs **6566** · near ties 0 · disagreements **0** |
| Missing global id | 354 (reported by sequence; retained under native identity) |

**False-completeness checks:** minimum sequence mass is 78 (not a single
token event); reject mass is present on all seven sequences; order is measured
on thousands of comparable pairs (not an empty ranking test).

---

## 5. R0 field maxima (sealed `1e-5`)

| Field | max \|error\| | ≤ 1e-5 |
|---|---:|:---:|
| ax, ay, cx0, cy0, h_ref, v_lost_*, v_cand_* | 0 | PASS |
| s_lost | 5.96e-8 | PASS |
| w | 1.19e-7 | PASS |
| bdist, dist_h | 1.91e-6 | PASS |
| bwd_r | 2.86e-6 | PASS |
| **fwd_r** (closest) | **3.81e-6** | PASS (margin ≈ 6.19e-6) |

- `predicate_disagreements = 0`
- `order_disagreements = 0`
- Tolerance was **not** relaxed after reading data.

---

## 6. Stability

| Check | Result |
|---|---|
| `causal_sensitivity_interpretable` | **true** |
| Mutations reported | cyclic shift · omit oldest candidate · omit oldest lost (each 2577 events) |
| Unavailable counts explicit | omit-cand 2577 (kernel &lt;4 early-return); omit-lost 86 (short-lost); cyclic 0 |
| Serialization re-run | byte-identical verifier JSON |
| Labels / score fit | false / false |

Large cyclic-shift predicate flips are **expected**: they show window order is
causally informative, not that serialization is free to permute.

---

## 7. Engineering lineage (not the terminal)

| Step | Record |
|---|---|
| Contract seal | PR #142 |
| Host float32 / FMA + device helper | PR #143 (engineering; no terminal) |
| Owner review evidence | this document + evidence packet |

---

## 8. Next unit boundary

Open only via a **new** sealed declaration (separate PR from this promotion):

```text
discrete-M representation-capability study
  on R1-faithful local state z
  short-horizon z_{t+1} ≈ M z_t + c  (and optional multi-step)
  against identity / CV / diagonal / full / regime baselines
```

Until that declaration exists, no discrete-\(M\) fitting, scoring, or policy
work is authorized.
