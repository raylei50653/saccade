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

---

## 5. Correction 1 — the terminal's stated cause does not hold (2026-07-13; append-only)

**Nothing above is rewritten**, and the sealed evidence packet is untouched.
Full record: [declaration § Correction 1](../runtime_bridge_decision_path_identifiability_declaration_20260713.md#correction-1--the-frozen-policy-is-not-the-audited-evidences-policy-2026-07-13-append-only).

§ 1 above judges D0's provenance (`px=0.4`, directional bonus `0.0`) against
"the headline policy … `px=0.25` … bonus `0.8`". That yardstick is the **`s`**
preset. The audited D0/R1/S0 evidence is sealed on **`m`**, whose preset
*correctly* resolves to `px=0.4` and `dir_bonus=0.0` — as
`HEADLINE_PRESET_REL` (`consumer_a_bridge_fidelity.py`), the Step-0 substrate
audit, and D0 itself all state, and as this study's own
`tests/unit/tracking/test_runtime_bridge_decision_path.py` asserts.

> The delta is a **scope mismatch, not capture corruption.** The capture
> stamped its policy faithfully; P0 compared it against a different preset.

**The terminal nevertheless stands — it was over-determined.** Re-running the
audit against `m` (the runner now takes the policy target as a required
parameter) returns the *same* terminal: `px` and `dir_bonus` now match, but
`h_lo`, `h_hi`, `spatial_gate` and `max_speed` are **never stamped into capture
provenance at all**, so configuration alignment fails under *every* preset. The
`px` / `dir_bonus` comparison reported above never carried the decision.

> **The terminal is right; the reason published for it is wrong.** The capture
> is not foreign — it is **under-stamped**. That distinction decides the next
> move: a foreign capture would mean the D0/R1/S0 line is corrupt and must be
> re-captured; an under-stamped one means the evidence is sound but
> under-documented, and the fields simply need stamping (which H0 does).

**Surviving, preset-independent findings** (§ 3 stands on these, not on the
foreign-config framing): the field-sufficiency cap at **L1 replay**;
`h_lo`/`h_hi` (and `spatial_gate`/`max_speed`) unstamped; no capture-time
`tracker_gpu.cu` hash.

**Terminal status (re-issued cause, owner accepted 2026-07-14):**
`P0_CAPTURE_SEMANTICS_INVALID` stands. Its cause of record is now
**capture-provenance incompleteness** — `h_lo`, `h_hi`, `spatial_gate`,
`max_speed` unstamped and no capture-time `tracker_gpu.cu` hash, so no packet can
self-certify its policy under *any* preset. The foreign-capture cause published
in § 1 is **withdrawn** (declaration § C1.5–C1.6). No unit may cite this terminal
as evidence that the D0/R1/S0 capture **semantics** are invalid, nor as grounds
to downgrade D0/R1/S0 — they never claimed the `s` policy.
