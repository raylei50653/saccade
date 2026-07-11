# Phase-2 disposition inventory

Produced during the 2026-07-11 pytest reorganization (phase 1: quarantine).
Phase 2 (#119) executes terminal states in small reviewable batches.

**Progress**
- Batch 1: `external_fp/test_external_fp_filter_runtime.py` → T2 promote to `tests/unit/eval/` (done, PR #120)
- Batch 2: gap_conditioned_motion e0–e3 → T4 delete (done, PR #121); phase_b retained (line still warm)
- Batch 3: `misc/test_boolean_atom_partial_order.py` → T3/T4 delete (done, PR #122); sealed packet remains under contract checkers
- Batch 4: near_miss trio → T2 promote to `tests/unit/eval/diagnostics/` (done, PR #124); scripts remain supported eval diagnostics

Each research test file must eventually reach one of four terminal states:

- **T1 promote** — durable engineering contract → `tests/contract/`
- **T2 consolidate** — generic module behavior → merge into a parameterized behavior test in `tests/unit/`
- **T3 generic-covered** — only verified packet completeness → delete; `tests/contract/test_research_packet_{schema,manifest}.py` covers it
- **T4 delete** — one-shot research reproduction → delete; recipe/artifacts/git history preserve it

Nothing was deleted in phase 1. "Seal status" reflects 2026-07-11 knowledge.

## Moved to tests/research/ (23 files)

| File | Packet / study | Seal status | Proposed | Rationale |
|---|---|---|---|---|
| safe_region/test_safe_region_a1_audit.py | conversion pack `1a180620bc…` (safe-region A1) | CLOSED (A1_ACCEPTED_WITH_LIMITS) | T3 | Audits pack presence/consistency; generic checkers + sealed pack supersede it |
| safe_region/test_safe_region_asset_r1_conversion.py | R1 conversion pack | CLOSED | T3/T4 | Pack-shape checks → T3; frozen-number replays → T4 |
| safe_region/test_safe_region_assetization_r1.py | assetization apparatus (live src module) | study CLOSED, module retained | T2 | Synthetic tests of `saccade.perception.eval.safe_region_assetization_r1`; keep only if the apparatus serves the active gt_support_morphology line, else delete with the module |
| safe_region/test_safe_region_assetization_r11.py | same (R1.1 attribution) | study CLOSED | T2 | Same as above |
| ~~gap_conditioned_motion/test_gap_conditioned_motion_e0.py~~ | gap_conditioned_motion_e0_20260711 | sealed | **done — T4** (#119 batch 2) | Packet + runner in evidence dir; contract tests cover integrity |
| ~~gap_conditioned_motion/test_gap_conditioned_motion_e1_m0.py~~ | …e1_m0_20260711 | sealed | **done — T4** (#119 batch 2) | Same |
| ~~gap_conditioned_motion/test_gap_conditioned_motion_e2_family.py~~ | …e2_family_20260711 | sealed | **done — T4** (#119 batch 2) | Same |
| ~~gap_conditioned_motion/test_gap_conditioned_motion_e3_signals.py~~ | …e3_signals_20260711 | sealed | **done — T4** (#119 batch 2) | Same |
| gap_conditioned_motion/test_gap_conditioned_motion_phase_b.py | …phase_b_20260711 | sealed (PR #116) | T4 | Line still warm — keep until the gap-motion line closes, then delete |
| d_online_stage2/test_d_online_stage2_q1q3.py | out/signal_study m_b1_5 …_20260710 | sealed | T4 | Pins dated out/ artifacts + MOT17 GT; skips when absent |
| d_online_stage2/test_d_online_stage2_q4.py | same | sealed | T4 | Same |
| d_online_stage2/test_d_online_stage2_q45_atlas.py | m_b1_5_stage2_q45_20260710 | sealed | T4 | Same; see "known gaps" for external artifact_hashes |
| ~~misc/test_boolean_atom_partial_order.py~~ | boolean_atom_partial_order_20260711 (#106) | sealed | **done — T3/T4** (#119 batch 3) | Packet integrity → generic checkers (T3); atom roles/terminal/PRC binding → sealed verdict (T4); no live `src/saccade` imports |
| misc/test_d0_bridge_estimator_fidelity.py | d0_bridge_estimator_fidelity_20260711 (PR #115) | sealed | T4 | D0 certifies bridge atoms only; packet + recipe preserve it |
| misc/test_portable_or_tail.py | frozen portable_policy.json (M-B1 OR-tail) | frozen; kernel acceptance PENDING (ONLINE_BAUDIT_IMPLEMENTED=False) | keep in research | Still active: hook acceptance outstanding. On acceptance, promote the acceptance-relevant assertions (T1), delete the rest (T4) |
| misc/test_cheb_gr_offline_handover_report.py | chebgr handover applicability study | study sealed | T4 | 315 lines / 1 test driving a diagnostics report script |
| misc/test_synthesize_handover_applicability.py | same | study sealed | T4 | Subprocess smoke of the synthesis script |
| ~~near_miss/test_analyze_near_miss_offsets.py~~ | scripts/eval/diagnostics (supported tooling) | study concluded | **done — T2** → `tests/unit/eval/diagnostics/` (#119 batch 4) | Live script behavior; listed in scripts/eval README as current helpers |
| ~~near_miss/test_analyze_near_miss_final_output.py~~ | same | concluded | **done — T2** (#119 batch 4) | Same |
| ~~near_miss/test_analyze_near_miss_stage_attribution.py~~ | same | concluded | **done — T2** (#119 batch 4) | Same |
| external_fp/test_external_fp_model.py | external-FP model (live src module) | study line inactive | T4 | Fits logistic/softmax classifiers; research apparatus test |
| ~~external_fp/test_external_fp_filter_runtime.py~~ | evaluator `_apply_external_fp_filter` hook | hook wired | **done — T2** → `tests/unit/eval/test_external_fp_filter_runtime.py` (#119 batch 1) | Live evaluator wiring; default-collection coverage restored |
| external_fp/test_external_fp_rows.py | external_fp_rows module | inactive | T4 | Row-extraction apparatus for the study |

## Considered and left in place (rubric: tests live module behavior)

| File | Why it stayed | Phase-2 note |
|---|---|---|
| tests/unit/test_check_association_tools.py | Unit test of live `check_association_tools.py` (DEFAULT_REGISTRY, `--list`/`--print-recipe` CLI, synthetic schema/error paths); no dated packet read. Initially misfiled into research (PR #118 review), moved back | Core; keep |
| tests/unit/reid/test_occ_audit.py | Numeric core of live `occ_audit` module | If the occ-exit line is retired, delete module + tests together |
| tests/unit/eval/test_occ_audit_bank_reference.py | Live module + bank behavior | Same retirement coupling |
| tests/unit/eval/test_occ_audit_chebgr_probe.py | Default-off but wired probe | Same |
| tests/unit/eval/test_occ_audit_chebgr_wiring.py | Tests evaluator/run_eval config wiring | Same |
| tests/unit/eval/test_occ_audit_seq_conditioning.py | Live analysis module behavior | Same |
| tests/unit/eval/test_decimal_hash.py | Determinism-chain tooling used by pre_push | Core; keep |
| tests/unit/eval/test_decimal_matrix_2x2.py | Same tooling | Core; keep |
| tests/unit/eval/test_decimal_chain_routine.py | Same tooling (routine sentinel) | Core; keep |

## Known gaps of the generic packet checkers

- `m_b1_5_stage2_q45_20260710` `manifest.json:artifact_hashes` uses logical
  names for artifacts stored outside the packet (out/…); the generic checker
  verifies the packet via `SHA256SUMS.json` only. External artifacts are not
  integrity-checked by pytest. This is machine-readable: the packet is listed
  in `EXTERNAL_ARTIFACT_HASH_EXCEPTIONS`
  (`tests/contract/packet_inventory.py`), and
  `test_unverified_hash_fields_are_declared_exceptions` fails on any packet
  with unresolvable hash fields that is not declared there.
- Runner scripts frozen inside packets (`run_*.py`) are hash-verified as
  files but never executed by the contract tests — by design (replay is a
  recipe action, not a pytest).
