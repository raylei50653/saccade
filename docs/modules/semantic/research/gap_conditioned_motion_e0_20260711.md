<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — E0 substrate audit

> **E0 terminal:** `PARTIALLY_IDENTIFIABLE`. The frozen seven-sequence
> relink-pair table supports M0 and a position-only displacement observation,
> but it does not contain 2D endpoint velocity vectors or transferable context
> fields. Joint/velocity likelihoods and sequence-conditioned LOO headlines are
> fail-closed.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/closed/gap_conditioned_probabilistic_motion_probe_20260711.md)  
Packet: [evidence/gap_conditioned_motion_e0_20260711/](evidence/gap_conditioned_motion_e0_20260711/manifest.json)

## 1. Frozen substrate

| Item | Frozen value |
|:--|:--|
| Pair universe | `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv` |
| SHA256 | `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17` |
| Raw / `gt_valid` rows | 24,284 / 21,789 |
| GT / FP in `gt_valid` | 340 / 21,449 |
| Sequence coverage | 7 MOT17 SDP sequences |
| Integrity | 0 gap/frame mismatches · 0 invalid position rows · 0 duplicate `(seq,lost_id,cand_id)` keys |

This is the existing labeled `U_relink_pair` substrate. E0 does not create a
new unlabeled universe, replay tracker output, change labels, or change the
hard-pool definition.

## 2. Canonical gap bins are frozen

The repository already defines canonical relink bins in
[`signal_table_schema.md`](../../../research/contracts/signal_table_schema.md), so
this probe adopts them without rebinning:

| Gap | Pairs | GT | FP |
|:--|--:|--:|--:|
| 1–10 | 1,037 | 53 | 984 |
| 11–30 | 1,895 | 76 | 1,819 |
| 31–60 | 2,549 | 63 | 2,486 |
| 61–150 | 6,979 | 100 | 6,879 |
| 151–300 | 9,329 | 48 | 9,281 |

These bins are analysis inputs, not tunable model parameters. E1–E3 and A1–A8
must retain them; any alternative binning requires a new model/analysis ID and
cannot replace the headline table post hoc.

## 3. Identifiability result

| Object | E0 result | Reason / boundary |
|:--|:--|:--|
| M0 deterministic atoms | **identifiable** | `bridge_dist`, both residuals, direction cosine, and endpoint speed magnitudes are present; `resid_mean` and `speed_mismatch` are derivable without new labels |
| Position-only observation | **identifiable** | Δfoot `(cand-lost) / h_ref`, `gap`, label, and sequence are present |
| Velocity-only observation | **not identifiable** | no `(lost_exit_vx, lost_exit_vy, cand_entry_vx, cand_entry_vy)` |
| Joint position+velocity observation | **not identifiable** | scalar speeds/residual norms do not uniquely recover vector direction; `dir_cos` still leaves a 2D reflection ambiguity |
| Global context | **eligible** | transferable and independent of held-out sequence statistics |
| Sequence context | **diagnostic only** | allowed in-sample; forbidden in LOO headline by the thread firewall |
| Exit-zone / image-normalized / GMC cluster / route group | **not identifiable** | required transferable context fields are absent from the frozen table |

The position-only result authorizes only an explicitly named reduced family
whose observation is Δfoot/`h_ref`. It does **not** make the originally written
joint transition density identifiable, and it does not permit reconstructing
velocity direction from residual norms.

## 4. Gate and next step

```text
E0 = PARTIALLY_IDENTIFIABLE

allowed next:
  E1  rebuild M0 role-reversal baseline on the frozen canonical bins
  E2  specify/freeze position-only M1-P and M2-P before fitting

still blocked:
  velocity-only or joint q_motion / NLL
  sequence-conditioned LOO headline
  exit-zone / GMC / route-conditioned claims
  Phase B verdict
```

E2 must keep model terms split (`q`, `log det Σ`, dimension, regularization)
and must not relabel a position-only marginal as the full joint M1/M2 model. A
future vector-state table would be a separately authorized substrate change,
not an E0 silent repair.

## 5. Reproduction

```bash
uv run python \
  docs/modules/semantic/research/evidence/gap_conditioned_motion_e0_20260711/run_e0_audit.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv \
  --verify

uv run pytest tests/unit/test_gap_conditioned_motion_e0.py -q
```

The verifier checks the frozen source SHA, regenerates the audit in a temporary
directory, and compares `manifest.json`, `substrate_audit.json`, and
`recorded_output.txt` byte-for-byte. This is an offline D1 research packet; no
ledger promotion or production claim is authorized.
