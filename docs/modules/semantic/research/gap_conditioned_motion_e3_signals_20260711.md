<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — E3 LOO fold signals

> **E3 status:** `E3_SIGNALS_SEALED`. This packet rebuilds the seven sealed
> LOO folds under `GCM-E2-POSITION-ONLY-v1`, persists 28 parameter artifacts
> and 7 selection artifacts, and emits all four frozen-member scores for every
> pair under its held-out fold. It does **not** compute A1–A8 tables, select a
> V1–V5 verdict, calibrate, change the family, or touch production defaults.
> Phase B remains unauthorized until the research owner records an explicit
> authorization after this seal.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)

E2 family: [note](gap_conditioned_motion_e2_family_20260711.md) ·
[packet](evidence/gap_conditioned_motion_e2_family_20260711/manifest.json)

Phase B design (execution unauthorized):
[A1–A8 protocol](gap_conditioned_motion_phase_b_design_20260711.md)

Packet: [evidence/gap_conditioned_motion_e3_signals_20260711/](evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json)

## 1. Claim boundary

Authorized E3 scope (E2 §4 / §5.1 · thread · Phase B design §1 step 2):

```text
rebuild 7 LOO folds
persist 28 parameter artifacts
persist 7 selection artifacts
emit all 4 model scores per pair × held-out fold
```

Forbidden in this packet:

```text
A1–A8 tables · V1–V5 verdict · held-out calibration · family redefinition
winner-only score filtering · criterion edits · production/hook/preset change
```

The sealed score unit is: every pair in sequence \(s\) is scored under fold
`LOO::s` (parameters fit on the other six sequences only). Full-pool fits are
not emitted. `selected_model_id` is a fold marker only; non-winner scores are
retained because A8 later needs the complete matched M1/M2 surface.

## 2. Inputs and lineage

| Item | Frozen value |
|:--|:--|
| Source pairs | `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv` |
| Source SHA256 | `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17` |
| Family freeze | `GCM-E2-POSITION-ONLY-v1` (`FROZEN_ACCEPTED_WITH_LIMITS`) |
| Fit/score primitives | E2 runner `build_fold_artifacts` · `score_model` (no redefinition) |
| Observation | \(d=\Delta\text{foot}/h_{ref}\in\mathbb{R}^2\), gap ∈ [1, 300] |
| Headline context | global only (sequence remains diagnostic for LOO headline) |

Per-fold train-GT counts and fit-row lineage hashes match the sealed E2 map
exactly (minimum train fold = 183 rows on MOT17-10).

## 3. Packet contents

| Artifact | Count / role |
|:--|:--|
| `parameters/LOO__<seq>__<model>.json` | **28** parameter artifacts (7 folds × 4 members) |
| `selections/LOO__<seq>.json` | **7** selection artifacts (train NLL for all four + winner) |
| `pair_fold_model_scores.csv` | **97,136** rows = 24,284 pairs × 4 models |
| `fold_summary.json` | fold index · counts · claim ceiling |
| `recorded_output.txt` | human-readable seal summary |
| `manifest.json` | SHA map over every sealed artifact |

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

Every score row retains the E2 required signal fields plus pair identity and
the fold selection marker:

```text
freeze_id · fold_id · held_out_sequence · model_id
parameter_artifact_id · selection_artifact_id
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

## 4. Fold summary (descriptive only — not A1–A8)

| Held-out | Train GT | Held-out pairs | Selected model (train-NLL) | Fit-row hash prefix |
|:--|--:|--:|:--|:--|
| MOT17-02-SDP | 268 | 3,434 | `M2P-GLOBAL-OU-H30` | `9e7fe454…` |
| MOT17-04-SDP | 328 | 687 | `M2P-GLOBAL-OU-H30` | `9caca30e…` |
| MOT17-05-SDP | 298 | 6,519 | `M2P-GLOBAL-OU-H30` | `ada1a106…` |
| MOT17-09-SDP | 326 | 411 | `M2P-GLOBAL-OU-H30` | `164034ec…` |
| MOT17-10-SDP | 183 | 6,793 | `M2P-GLOBAL-OU-H30` | `8b1e67dd…` |
| MOT17-11-SDP | 320 | 1,485 | `M2P-GLOBAL-OU-H30` | `0aa3abb4…` |
| MOT17-13-SDP | 317 | 4,955 | `M2P-GLOBAL-OU-H30` | `a88c8ed7…` |

Train-only selection chose `M2P-GLOBAL-OU-H30` on every fold in this sealed
rebuild. That fact is recorded for lineage only; it is **not** a Phase B
attribution or a V1–V5 verdict. A8 must still consume the full four-member
score surface.

## 5. Gates and result

```text
source SHA:              PASS (0ae38967…)
E2 LOO lineage hashes:   PASS (7/7 match sealed map)
parameter artifacts:     28
selection artifacts:     7
score rows:              97,136 = 24,284 pairs × 4 models
winner-only filtering:   none (every pair retains all four members)
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
`fold_summary.json`, `recorded_output.txt`, `pair_fold_model_scores.csv`,
`manifest.json`, and all 28+7 fold JSON artifacts byte-for-byte.

## 7. Next authorized step

```text
E3 signals sealed  →  owner records Phase B authorization in the thread
                   →  single A1–A8 reproduction entrypoint over this table
                   →  exactly one V1–V5 verdict
```

Do not compute any A-table inside an E3 follow-up without that authorization.
Do not edit the frozen Phase B numeric criteria after this seal without a
design revision + deviation note.
