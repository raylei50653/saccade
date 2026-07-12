<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

# R1 temporal-reduction capture — sealed preflight declaration

> **One-line:** `r1_temporal_reduction_capture_v1` records the exact causal
> windows consumed by Consumer-A `bridge_anchor4` (or its one-point short-lost
> fallback), then replays the existing `bdist`; it is not a score model, gate
> study, or production-path change.

Parent: [score temporal-to-stable-domain charter](score_temporal_to_stable_domain_20260712.md) ·
Navigation: [active thread](../../../research/threads/score_temporal_to_stable_domain_20260712.md) ·
Prior boundary: [D0 runtime-shadow fidelity](d0_runtime_shadow_fidelity_results_20260712.md) ·
Fidelity vocabulary: [runtime-quantity protocol](../../../research/eval/runtime_quantity_fidelity_protocol.md)

## 1. Seal and scope

**Seal time:** 2026-07-12, before any R1 capture output, replay measurement,
or outcome label is read. This declaration may only be amended append-only;
the D0 CSV and its sealed evidence packet are not inputs that can be rewritten.

| Field | Sealed value |
|---|---|
| Study unit | R1 capture-contract preflight |
| Target | Consumer-A CUDA bridge `bdist`, emitted before threshold/policy gates |
| Capture contract | `r1_temporal_reduction_capture_v1` |
| Payload schema | `r1_temporal_reduction_payload_v1` / `TemporalReductionContract` |
| Capture mode | native CUDA observation, shadow (`propose + capture`, no bridge commit) |
| Default | disabled unless `SACCADE_RESEARCH_R1_TEMPORAL_REDUCTION_CAPTURE_DIR` is set |
| Required shadow guard | `SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_SHADOW=1` |
| Required event identity | native `(seq, native_capture_ordinal, lost_slot, cand_slot)`; local/global ids are audit provenance, and an evaluator `_global_id_map.txt` is optional rather than a permission to retain state |
| Labels / score fitting | forbidden; verifier reads neither |
| Policy / preset / ledger | unchanged and unauthorized |

The R1 directory is mutually exclusive with the legacy D0 capture directory in
one run. This prevents a new nested-window payload from being silently emitted
as D0's frozen flat CSV contract.

## 2. Causal payload contract

One JSONL event contains these source-owned fields:

| Source | Serialized R1 state | Rule |
|---|---|---|
| candidate head | four chronological `(cx, cy, h)` samples consumed by `bridge_anchor4` | exactly four; index 0 is the kernel's candidate endpoint |
| lost exit | last four chronological samples, or one final sample for short-lost | branch is explicit: `bridge_anchor4_last4` or `short_lost_last_point_zero_velocity`; unused tail must be zero |
| reduction configuration | `gap`, `bridge_at`, `la`, anchor mode/rate, `bridge_dir_bonus` | require `la = gap + bridge_at - 1`; row bonus must equal provenance configuration |
| causal normalizer | `ema_lost`, `ema_cand`, derived `h_ref` | replay uses recorded pre-score EMA state, never a row/global reconstruction |
| native terms | anchors, OLS velocities, `dist_h`, `fwd_r`, `bwd_r`, `s_lost`, `w`, `bdist`, production threshold | capture before any threshold / occupancy / appearance policy action |
| provenance | shadow flag, bridge/detector configuration, source JSON SHA256, optional id-map SHA256, payload SHA256 | mixed provenance, overflow, invalid native identity, or duplicate event keys fail closed |

The payload deliberately contains only the **effective** window. It does not
claim that unconsumed ring entries are causal state. The bridge's stride and
event order remain configuration/provenance; a future change to either is a
new capture-contract version, not a reinterpretation of v1.

## 3. Frozen validity gates

All gates are non-compensatory. A failed gate yields `R0_INVALID`; no score or
transition study may consume the payload.

| Gate | Must pass before replay/stability is interpreted |
|---|---|
| V1 shadow neutrality | every captured run is shadow mode; the existing byte-identity comparison against the corresponding bridge-off output passes |
| V2 bounded complete capture | every per-sequence native file has `complete=true`, `overflow_events=0`, and `total_events == len(events)` |
| V3 version / provenance | all files declare exactly `r1_temporal_reduction_capture_v1` with identical provenance; source and payload hashes are present (an optional id-map hash may annotate emitted identities) |
| V4 causal completeness | each event has both valid windows, declared lost branch, finite scalar state, valid `la` identity, and configuration-consistent direction bonus |
| V5 native identity integrity | every event has a nonnegative native append ordinal plus nonnegative lost/candidate slots; all native R1 event keys are nonempty and unique. A missing output-layer global id is reported, not dropped, because it does not invalidate the native causal state |
| V6 declared production support | run the seven MOT17 baseline sequences; each sequence must contribute at least one keyed event, and the packet must contain both `bdist <= production_threshold` and `bdist > production_threshold` events |

## 4. Frozen replay and temporal-stability criteria

`verify_r1_temporal_reduction_replay.py` is the only first-pass calculator. It
reconstructs the estimator from the serialized R1 state; it does not join GT,
fit a proxy, select a threshold, or modify a tracker decision.

| Check | Frozen calculation / criterion |
|---|---|
| R0 term replay | recompute `bdist`, `dist_h`, `fwd_r`, `bwd_r`, four velocity components, both anchors, `h_ref`, `s_lost`, and `w`; each maximum absolute error must be `<= 1e-5` |
| R0 structural replay | `gap`, `bridge_at`, and `la` must match exactly for every event |
| Predicate preservation | `replayed_bdist <= captured_production_threshold` agrees for every event |
| Event-local order | group only by `(seq, cand_local_id)` within the native capture; for every pair with captured separation `> 2e-5`, replay order must agree; pairs at or below that separation are reported as near ties, never counted as agreement |
| Serialization stability | canonical JSONL parse/replay is deterministic; a byte-identical payload rerun must produce byte-identical verifier JSON under the same tool revision and tolerance |
| Causal sensitivity report | process every event under (a) omission of the oldest available lost/candidate sample and (b) cyclic shift of a four-sample window; aggregate the resulting effects by lost branch and `la`, with per-term maxima, predicate flips, and explicit unavailable branches. These are **non-equivalent inputs**, not fidelity substitutions; do not pool them into a score claim |
| Stability disposition | absent mutation reporting, an order/predicate change under equivalent serialization, or an unprovenanced window-order transformation yields `R_STABILITY_UNRESOLVED` even if untouched R0 replay passes |

The (10^{-5}) tolerance is an a-priori float32 reconstruction budget, not a
calibration target. It is intentionally much smaller than the production
`bdist` threshold and must not be relaxed after data are read.

## 5. Commands and artifacts

```bash
# Explicitly opt in; production/default runs leave both allocation and writes off.
SACCADE_RESEARCH_R1_TEMPORAL_REDUCTION_CAPTURE_DIR=out/r1-native \
SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_SHADOW=1 \
uv run scripts/eval/mot17.py --relink-bridge-enabled ...

uv run python scripts/tools/export_r1_temporal_reduction_capture.py \
  --capture-dir out/r1-native \
  --id-map results/<run>/_global_id_map.txt \
  --output out/r1-temporal/events.jsonl

uv run python scripts/tools/verify_r1_temporal_reduction_replay.py \
  --payload out/r1-temporal/events.jsonl \
  --output out/r1-temporal/replay.json
```

The exporter's manifest is the pre-outcome evidence index. A future capture
must add its own packet directory and hashes; this declaration contains no
results and is not evidence promotion.

## 6. Terminal mapping

| Terminal | Condition | Consequence |
|---|---|---|
| `R0_INVALID` | any V1–V6 gate fails | instrumentation repair only; no proxy, score, or dynamics conclusion |
| `R1_FAITHFUL` | all validity, R0 replay, predicate/order, and stability requirements pass | may request a separately declared discrete-​M representation-capability study |
| `R_STABILITY_UNRESOLVED` | R0 replay passes but required sensitivity / serialization evidence does not | refine the capture contract; no score-model selection |
| `R2_UNFAITHFUL` | any future reduced state changes this quantity or its decision surface | diagnostic-only; no score substitute |

No terminal authorizes a score fit, gate sweep, preset change, ledger entry, or
online policy evaluation. Research acceptance remains owner-side after the
future packet is reviewed.
