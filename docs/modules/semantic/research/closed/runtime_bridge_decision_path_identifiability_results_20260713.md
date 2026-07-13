# P0 — runtime bridge decision-path identifiability: results

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

> **Terminal output: `P0_CAPTURE_SEMANTICS_INVALID` (owner accepted).** The frozen D0/R1/S0
> evidence cannot be attributed to the sole permitted headline bridge policy.
> P4 and P5 were not entered; no GT or FP label was read.

Declaration: [P0 sealed execution scope](../runtime_bridge_decision_path_identifiability_declaration_20260713.md) ·
Canonical packet: [manifest](../evidence/p0_runtime_bridge_decision_path_20260713/manifest.json)

## 1. Ordered terminal application

The headline policy is s: `px=0.25`, directional bonus `0.8`, and an enabled
EMA ratio gate `[0.75, 1.33]`.  D0's sealed capture provenance instead stamps
`px=0.4` and directional bonus `0.0`; it does not stamp either headline height
bound or the disabled speed/spatial knobs.  R1 repeats `px=0.4`, bonus `0.0`,
and explicitly names `configs/presets/mamba_whole_graph_m.yaml`.

S0's frozen `capture.csv.gz` hash is D0's same capture hash, so it cannot cure
that mismatch.  Neither packet records a capture-time `tracker_gpu.cu` file
hash, therefore a commit label cannot prove identity with the source DAG audited
here.  This maps to `P0_CAPTURE_SEMANTICS_INVALID` before any replay-level claim
or attribution statistic.

## 2. Canonical decision path

The source DAG is frozen in the declaration.  The controlling order is:

1. Candidate and lost-track eligibility.
2. Height-ratio gate, then optional speed and centre gates.
3. `fwd_r`, `bwd_r`, `dist_h`, speed weight, optional directional adjustment,
   and final `bdist` construction.
4. `bdist <= bridge_px`; then optional occupancy, appearance, and portable-tail
   vetoes.
5. Per-candidate lowest and second-lowest `bdist`, followed by margin rejection.
6. Per-lost-slot `atomicMax` claim competition on quantized detection score and
   candidate index.
7. The winner adopts the lost ID and deactivates the lost slot.

The native event is appended after the pre-score gates and score construction,
but before the hard cutoff.  It is therefore neither a raw eligible-pair census
nor a proposal/commit log.

## 3. Field sufficiency and replay level

| Stage | Current evidence | Consequence |
| --- | --- | --- |
| Raw eligibility / height gate | emitted survivor pairs only | cannot observe height-gate rejects or raw attrition |
| Score construction | scalar terms are present | observable only for the foreign m-config survivor population |
| Pair cutoff | `bdist` is present | mechanically evaluable, but not for headline s |
| Candidate ranking / margin | D0 v2 export removes `frame` and slots | no complete competitor groups; no best/second-best replay |
| Claim competition | no detection score, quantized key, or proposer groups | `atomicMax` winner unreconstructable |
| Commit | capture is shadow mode | final ID adoption and lost-slot deactivation unobserved |

Thus the observed replay level is deliberately **not assigned** under the
invalid provenance.  Even if headline alignment were somehow independently
proven, the recorded fields would cap this packet at **L1 pair-cutoff replay**;
they cannot reach L2 or L3.

The complete machine-readable [field matrix](../evidence/p0_runtime_bridge_decision_path_20260713/field_sufficiency.json)
and explicitly unobserved [funnel](../evidence/p0_runtime_bridge_decision_path_20260713/decision_funnel.csv)
record the proof without inferring missing rows.

## 4. Owner acceptance and closure

The owner accepted `P0_CAPTURE_SEMANTICS_INVALID` on 2026-07-13. This closes
P0 and releases its sole-active WIP lock. The terminal itself authorizes no new
capture, threshold sweep or selection, B1, registry/ledger change, or
production modification. Any H0 work remains governed only by its separately
sealed declaration.
