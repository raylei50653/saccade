<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# D0 — Consumer-A bridge estimator fidelity audit

## 2026-07-12 — SUPERSEDED (historical pointer; this document is unchanged below)

**This is the legacy v1 reconstruction packet. It stays frozen and its semantics
are not redefined.** Two statuses must be kept apart:

| | Status |
| --- | --- |
| **This legacy packet** | `D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE` — primary reason `runtime_capture_unavailable`. Correct, and unchanged. |
| **Issue #112 (current)** | **COMPLETE.** Certified by the v2 runtime-shadow packet against real CUDA `bdist`. |

This packet reached `not_fidelity_aligned` as a **reconstruction diagnostic** — it
*suspected* the answer but could not certify it, because no runtime capture
existed. That gap is now closed: a shadow bridge (propose + capture, commit
skipped) yields output byte-identical to bridge-off while emitting real float32
kernel values, and 1,684 exactly-joined pairs **confirm** the verdict.

* Results (v2, authoritative for Issue #112): [d0_runtime_shadow_fidelity_results_20260712.md](d0_runtime_shadow_fidelity_results_20260712.md)
* Sealed declaration: [d0_runtime_shadow_fidelity_declaration_20260712.md](d0_runtime_shadow_fidelity_declaration_20260712.md)
* Scope amendment: [s0_proxy_validity_amendment_20260712.md](s0_proxy_validity_amendment_20260712.md)

Everything below is the original 2026-07-11 content, retained verbatim.

> **Status:** `D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE`  
> **Terminal verdict:** `not_fidelity_aligned`  
> **Primary fail reason:** `runtime_capture_unavailable`  
> **Issue #112:** **incomplete** (native capture buffer exists; no captured
> evidence packet has yet been executed and accepted)
> **Packet:** [evidence/d0_bridge_estimator_fidelity_20260711/](evidence/d0_bridge_estimator_fidelity_20260711/)

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/closed/gap_conditioned_probabilistic_motion_probe_20260711.md)

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
D0 execution: AUTHORIZED → FAIL-CLOSED (capture unavailable)
Issue #112: incomplete until runtime CUDA capture exists
```

---

## Frozen inputs

| Input | Value |
|:--|:--|
| pairs | `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv` |
| SHA-256 | `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17` |
| substrate MOT | `results/MOT17_eval_m_b1_substrate_20260709T092543Z` (relink off) |
| headline preset | `configs/presets/mamba_whole_graph_m.yaml` (file-byte hash in verdict) |
| production threshold | `0.4` (asserted from preset) |
| primary support \(S_A\) | `gt_valid && 1 ≤ gap ≤ 26` |
| primary gap cells | `1–10`, `11–26` |

SHA mismatch fails closed. No pair rebuild, no label edits, no threshold search.

---

## Evidence identity (binding)

| Claim | Allowed? |
|:--|:--|
| kernel-formula reconstruction audit | **yes** (this packet) |
| Consumer-A exact surface / D4 exact captured | **no** |
| runtime fidelity sealed | **no** |
| Issue #112 complete | **no** |

Production `foot_ring` / `ema_h` are updated in CUDA from live tracker state.
This packet reconstructs estimators from MOT text tracklets — a **different
observation substrate**. Metrics are **reconstruction diagnostics only**.

```text
capture_mode = kernel_formula_reconstruction
evidence_role = reconstruction_diagnostic_not_runtime_consumer_a_capture
LIVE_CUDA_EVENT_RING_IMPLEMENTED = false
NATIVE_CUDA_BRIDGE_FIDELITY_CAPTURE_IMPLEMENTED = true
primary_fail_reason = runtime_capture_unavailable
```

### Capture contract

| Field | Value |
|:--|:--|
| Research audit default | **off** |
| Native CUDA event buffer | **implemented, default-off**; no D0 packet yet |
| Sealed artifact | reconstruction from pairs + substrate |
| Event key | `(seq, lost_id, cand_id, lost_last_frame, cand_first_frame)` exact |
| Decision path | audit flag does not change accept/reject |

Module: `src/saccade/perception/eval/consumer_a_bridge_fidelity.py`

---

## Consumer-A formula surface (reference algebra)

Named production surface under **future** runtime capture (not dumped here):

| Item | Value |
|:--|:--|
| Kernel | `src/tracking/tracker_gpu.cu` |
| Entry | `relink_bidir_propose_kernel` |
| Primary quantity | `bdist` |
| Anchor | adaptive (schema default; preset omits → adaptive / rate 0.03) |
| Horizon | `la = gap + bridge_at − 1` (`bridge_at=4`) |
| Normalization | bilateral EMA `h_ref` |
| Aggregation | speed-weighted blend; `dir_bonus=0.0` in m preset |

**Not used as substitutes:** Python semantic relinker, C++ `midpoint_bridge_dist`.

---

## Join coverage (reconstruction)

| Metric | Value |
|:--|--:|
| Offline eligible rows | 24,284 |
| \(S_A\) rows | 2,561 (GT 117 / FP 2,444) |
| Reconstruction captured | 22,465 |
| Exact matched | 22,465 |
| Offline-only | 1,819 |
| Duplicate / ambiguous keys | 0 / 0 |
| GT coverage on \(S_A\) | **95.73%** (gate: 100%) |
| Overall coverage on \(S_A\) | **95.90%** (gate: ≥98%) |
| Coverage gates | **FAIL** (secondary) |

Uncaptured \(S_A\) rows are almost entirely `cand_len < 4` (kernel never scores).

**Terminal verdict is forced by runtime capture unavailability first.** Coverage
failure is an additional independent fail reason under the frozen metric gates.

---

## Reconstruction diagnostics (matched \(S_A\))

These numbers compare offline atoms to **kernel-formula reconstruction**, not
to live CUDA `bdist`. They **cannot** support threshold transfer.

| Slice | n | Spearman ρ | q85 \|err\| | pred. agree @0.4 | GT offline-safe / recon-unsafe |
|:--|--:|--:|--:|--:|--:|
| overall | 2,456 | 0.965 | 0.213 | 0.950 | 84 (3.4%) |
| GT | 112 | 0.714 | 0.469 | 0.759 | **24 (21.4%)** |
| FP | 2,344 | 0.966 | 0.317 | 0.959 | 60 (2.6%) |

Cluster unit: `(sequence, lost_id)`. Bootstrap seed `20260711`.

`metric_based_verdict_diagnostic_only` (if runtime capture existed): also
`not_fidelity_aligned` (GT boundary / coverage floors fail independently).

---

## Estimator decomposition (single-factor steps)

Progressive reconstruction surface (descriptive step deltas only):

| Step | Factor changed | Spearman vs recon | q85 \|err\| | pred. agree | GT step-safe/recon-unsafe |
|:--|:--|--:|--:|--:|--:|
| S0 offline midpoint | (reference) | 0.965 | 0.213 | 0.950 | 24 |
| S1 aggregation only | mid → SW | 0.987 | 0.174 | 0.972 | 8 |
| S2 anchor endpoints only | offline feet → CA positions | 0.988 | 0.219 | 0.978 | 7 |
| S3 velocity only | offline vel → CA vel | 0.989 | 0.200 | 0.982 | 7 |
| S4 horizon only | gap → la | 0.993 | 0.169 | 0.995 | 2 |
| S5 normalization only | raw h → EMA h_ref | 1.000 | 0.000 | 1.000 | 0 |
| S6 reconstruction | (= S5) | 1.000 | 0.000 | 1.000 | 0 |

**Attribution language (revised):** each step changes one declared factor.
Sequential residual reductions after S1 are visible for horizon (S4) and
normalization (S5); these are **descriptive step deltas**, not a claim that a
single factor “dominates” independently of order or interactions.

---

## Exact terminal verdict

```text
not_fidelity_aligned
```

**Primary reason:** `runtime_capture_unavailable`  
(the sealed packet contains no native CUDA capture; host reconstruction is not
Consumer-A capture)

**Secondary reasons (diagnostics):** coverage gates fail; reconstruction GT
boundary metrics fail threshold and rank-only floors.

**Interpretation:** offline `bridge_dist` numeric thresholds must **not** transfer
to Consumer A. Issue #112 remains open for runtime capture plumbing.

---

## Claim ceiling

| Claim level | Allowed after this packet? |
|:--|:--|
| 1. Signal-level regularity on offline table | yes |
| 2. Production-aligned numeric threshold on Consumer A | **no** |
| 3–4. Online / pipeline intervention | **no** |

Phase B / A1–A8 remain unauthorized. E3 signals remain sealed.

---

## Provenance

| Artifact | Binding |
|:--|:--|
| pairs SHA | `0ae38967…` |
| kernel source | `src/tracking/tracker_gpu.cu` SHA in verdict |
| fidelity module | `consumer_a_bridge_fidelity.py` SHA in verdict |
| headline preset | **file bytes** of `configs/presets/mamba_whole_graph_m.yaml` |
| git commit | seal-time `git rev-parse HEAD` (must be reachable on the PR branch) |

Preset asserts `relink_bridge_px=0.4` and `relink_bridge_dir_bonus=0.0`.

---

## Limitations

1. **No accepted live CUDA packet** — the default-off native buffer is present,
   but the archived frozen substrate contains MOT outputs, not the detector and
   frame-level inputs needed to reproduce the same live kernel events. A
   current-main smoke capture therefore cannot be substituted for the frozen
   exact-key cohort.
2. MOT-tracklet EMA/foot reconstruction ≠ live `d_foot_ring_` / `d_ema_h_`.
3. Coverage incomplete for `cand_len < 4`.
4. Decomposition is ordered single-factor; interactions remain possible.
5. No threshold search, quantile mapping search, or offline estimator repair.

---

## Reproduction

```bash
uv run python \
  docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/run_d0_bridge_fidelity.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv

# Rebuilds capture from pairs + substrate (not sealed capture as input)
uv run python \
  docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/run_d0_bridge_fidelity.py \
  --verify

# To collect a new native capture, set this only on the frozen evaluator run
# with relink_bridge_enabled=true. It is default-off and aborts on overflow.
export SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_DIR=out/d0_runtime_capture
export SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_CAPACITY=65536

# After the evaluator writes one JSON file per sequence, merge only complete
# native captures into the verifier's exact-key CSV contract.
uv run python scripts/tools/export_d0_runtime_capture.py \
  --capture-dir out/d0_runtime_capture \
  --output out/d0_runtime_capture/consumer_a_capture.csv.gz

# This accepts runtime_cuda_event_ring rows for diagnostics. It does not
# replace the sealed historical reconstruction packet or its fail-closed status.
uv run python \
  docs/modules/semantic/research/evidence/d0_bridge_estimator_fidelity_20260711/run_d0_bridge_fidelity.py \
  --capture out/d0_runtime_capture/consumer_a_capture.csv.gz \
  --output-dir out/d0_runtime_capture/packet

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
D0 packet: D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE
D0 verdict: not_fidelity_aligned
primary_fail_reason: runtime_capture_unavailable
Issue #112: incomplete
E3: SEALED
Phase B: UNAUTHORIZED
A1–A8: NOT_EXECUTED
production/default/preset: UNCHANGED
```
