<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — E3 LOO fold signals

> **E3 status:** `E3_SIGNALS_SEALED`. This packet rebuilds the seven sealed
> LOO folds under `GCM-E2-POSITION-ONLY-v1`, persists 28 parameter artifacts
> and 7 selection artifacts, and emits the **full fold × pair × model score
> cube** with an explicit `evaluation_role` tag so A6 can select thresholds
> on training clusters under fold-frozen parameters without recomputing
> signals. It does **not** compute A1–A8 tables, select a V1–V5 verdict,
> calibrate, change the family, or touch production defaults. Phase B remains
> unauthorized until the research owner records an explicit authorization
> after this seal.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/closed/gap_conditioned_probabilistic_motion_probe_20260711.md)

E2 family: [note](gap_conditioned_motion_e2_family_20260711.md) ·
[packet](evidence/gap_conditioned_motion_e2_family_20260711/manifest.json)

Phase B design (execution unauthorized; predeclaration seal `69b0e5be…`):
[A1–A8 protocol](gap_conditioned_motion_phase_b_design_20260711.md)

Packet: [evidence/gap_conditioned_motion_e3_signals_20260711/](evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json)

## 1. Claim boundary

Authorized E3 scope (E2 §4 / §5.1 · thread · Phase B design §1 step 2,
extended for A6 train-side surface):

```text
rebuild 7 LOO folds
persist 28 parameter artifacts
persist 7 selection artifacts
emit full fold × pair × model score cube
  evaluation_role=held_out  — pair.seq == fold held-out sequence
  evaluation_role=train     — pair.seq ∈ six train sequences
                              under the SAME fold-frozen parameters
```

Forbidden in this packet:

```text
A1–A8 tables · V1–V5 verdict · held-out calibration · family redefinition
winner-only score filtering · criterion edits · production/hook/preset change
using held_out rows for A6 threshold selection (Phase B must filter role=train)
```

### Why the train surface is required

Frozen A6 chooses τ for each held-out fold \(f\) from **training clusters
scored with fold-\(f\) parameters**. Own-sequence LOO scores for those
training sequences use different parameters and, for fold \(f\), their fits
generally include \(f\)'s held-out sequence — contaminating the LOO firewall.
Phase B must not create an unsealed score surface after E3. Therefore the
sealed cube includes both roles under each fold's parameters.

Phase B consumption:

| Analysis | Filter |
|:--|:--|
| A1–A5, A7, A8 held-out metrics | `evaluation_role=held_out` |
| A6 training-side τ selection | `evaluation_role=train` |
| A6 held-out safety / FP_removed | `evaluation_role=held_out` after τ frozen |

`selected_model_id` is a fold marker only; non-winner scores are retained
because A8 needs the complete matched M1/M2 surface.

## 2. Inputs and lineage

| Item | Frozen value |
|:--|:--|
| Source pairs | `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv` |
| Source SHA256 | `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17` |
| Family freeze | `GCM-E2-POSITION-ONLY-v1` (`FROZEN_ACCEPTED_WITH_LIMITS`) |
| Fit/score primitives | E2 runner `build_fold_artifacts` · `score_model` (no redefinition) |
| Observation | \(d=\Delta\text{foot}/h_{ref}\in\mathbb{R}^2\), gap ∈ [1, 300] |
| Headline context | global only (sequence remains diagnostic for LOO headline) |
| Phase B design seal | commit `69b0e5be0c26d6fa9f460f90aef37e891555da67` (PR #113 merge) |
| Phase B design content SHA256 | recorded in `manifest.json` → `phase_b_design.content_sha256` |

Per-fold train-GT counts and fit-row lineage hashes match the sealed E2 map
exactly (minimum train fold = 183 rows on MOT17-10).

## 3. Packet contents

| Artifact | Count / role |
|:--|:--|
| `parameters/LOO__<seq>__<model>.json` | **28** parameter artifacts (7 folds × 4 members) |
| `selections/LOO__<seq>.json` | **7** selection artifacts (train NLL for all four + winner) |
| `pair_fold_model_scores.csv.gz` | **679,952** rows = 24,284 pairs × 7 folds × 4 models (gzip; mtime=0 for byte-stable seal) |
| ↳ held_out role | **97,136** (= 24,284 × 4) |
| ↳ train role | **582,816** (= 6/7 of cube; A6 τ surface) |
| `fold_summary.json` | fold index · per-fold train/held-out counts · design seal · claim ceiling |
| `recorded_output.txt` | human-readable seal summary |
| `manifest.json` | SHA map · Phase B design provenance · counts |

### Parameter / selection contract (from E2)

Every parameter artifact records:

```text
freeze_id · model_id · parameter_artifact_id · fold_id
held_out_sequence · train_sequences · fit_row_count · fit_row_key_sha256
source_pairs_sha256 · dimension · drift_per_frame · base_covariance
regularization_applied · eigenvalue_floor · training_total_nll
```

Every selection artifact records:

```text
selection_artifact_id · fold_id · held_out_sequence · train_sequences
fit_row_count · fit_row_key_sha256 · source_pairs_sha256
training_nll_by_model · selected_model_id · selection_tolerance · model_order
```

### Score-row contract

```text
freeze_id · fold_id · held_out_sequence · evaluation_role
model_id · parameter_artifact_id · selection_artifact_id
selected_model_id · is_selected_model
seq · lost_id · cand_id · gap · gt_match · gt_valid · dx_h · dy_h
q_motion · log_det_covariance · gaussian_constant · nll_motion
source_pairs_sha256
```

Energy identity (unchanged):

\[
E_{\mathrm{motion}} = \tfrac12\bigl(q_{\mathrm{motion}}+\log\det\Sigma(t)+2\log(2\pi)\bigr)
= \texttt{nll\_motion}.
\]

Terms stay split; Phase B must not recombine them before the frozen criteria.
Every score row binds to its fold's parameter and selection artifact IDs
(lineage linkage for A6/A7 audit).

## 4. Fold summary (descriptive only — not A1–A8)

| Held-out | Train GT | Held-out pairs | Train pairs | Held-out scores | Train scores | Selected (train-NLL) | Hash prefix |
|:--|--:|--:|--:|--:|--:|:--|:--|
| MOT17-02-SDP | 268 | 3,434 | 20,850 | 13,736 | 83,400 | `M2P-GLOBAL-OU-H30` | `9e7fe454…` |
| MOT17-04-SDP | 328 | 687 | 23,597 | 2,748 | 94,388 | `M2P-GLOBAL-OU-H30` | `9caca30e…` |
| MOT17-05-SDP | 298 | 6,519 | 17,765 | 26,076 | 71,060 | `M2P-GLOBAL-OU-H30` | `ada1a106…` |
| MOT17-09-SDP | 326 | 411 | 23,873 | 1,644 | 95,492 | `M2P-GLOBAL-OU-H30` | `164034ec…` |
| MOT17-10-SDP | 183 | 6,793 | 17,491 | 27,172 | 69,964 | `M2P-GLOBAL-OU-H30` | `8b1e67dd…` |
| MOT17-11-SDP | 320 | 1,485 | 22,799 | 5,940 | 91,196 | `M2P-GLOBAL-OU-H30` | `0aa3abb4…` |
| MOT17-13-SDP | 317 | 4,955 | 19,329 | 19,820 | 77,316 | `M2P-GLOBAL-OU-H30` | `a88c8ed7…` |

Train-only selection chose `M2P-GLOBAL-OU-H30` on every fold in this sealed
rebuild. That fact is recorded for lineage only; it is **not** a Phase B
attribution or a V1–V5 verdict.

## 5. Gates and result

```text
source SHA:              PASS (0ae38967…)
E2 LOO lineage hashes:   PASS (7/7 match sealed map)
parameter artifacts:     28
selection artifacts:     7
score cube:              679,952 = 24,284 × 7 × 4
  held_out rows:         97,136
  train rows:            582,816
four-member completeness: PASS (every fold×pair)
energy identity:         PASS (nll = ½(q+logdet+const))
lineage linkage:         PASS (parameter/selection IDs per row)
role firewall:           PASS (train never includes held-out seq)
Phase B design seal:     recorded (69b0e5be… + content sha256)
A1–A8 tables:            not computed
V1–V5 verdict:           NOT_YET_EVALUATED
Phase B:                 unauthorized
production/hook/preset:  unchanged
```

## 6. Reproduction

```bash
uv run python \
  docs/modules/semantic/research/evidence/gap_conditioned_motion_e3_signals_20260711/run_e3_signals.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv \
  --verify

uv run pytest tests/unit/test_gap_conditioned_motion_e3_signals.py -q
```

The verifier rebuilds the full packet in a temporary directory and compares
`fold_summary.json`, `recorded_output.txt`, `pair_fold_model_scores.csv.gz`,
`manifest.json`, and all 28+7 fold JSON artifacts byte-for-byte. The score
cube is sealed as gzip (`mtime=0`) so the artifact stays under GitHub's
100 MB limit while remaining byte-reproducible.

## 7. Next authorized step

```text
E3 signals sealed (A6-complete cube)
  → owner records Phase B authorization in the thread
  → single A1–A8 reproduction entrypoint over this table
  → exactly one V1–V5 verdict
```

Do not compute any A-table inside an E3 follow-up without that authorization.
Do not edit the frozen Phase B numeric criteria after this seal without a
design revision + deviation note.
