# CleanFifoBank — Reusable Clean-FIFO Embedding Substrate

> Created: 2026-07-04. Corresponds to registry [#58](../../../reference/no_go_registry.md),
> semantic TODO "Sparse key-embedding bank" and "occ-exit audit (#55)".

## 1. What

`CleanFifoBank` (`src/saccade/perception/eval/clean_fifo_bank.py`) is a
per-track container of the most recent `fifo_n` (default 20)
visclean-gated clean raw embeddings, with optional stride-scheduled
extraction and event-aware birth window.

It is the probe-validated sparse equivalent of dense-50 bank for identity
relink, extracted as a reusable substrate so multiple consumers share the
same bank instead of each re-extracting reference embeddings.

## 2. Hard Constraints (probe 2026-07-04, must not be violated)

1. **No mean / prototype for Cheb-GR** — store raw embeddings only.
   `mean1` single-prototype collapsed intra-track variance and made
   Cheb-GR k-reciprocal distances meaningless (IDF1 −2.61 / −1.14).
   `representative()` returns a mean but is labelled **GRAPH-EXTERNAL
   cosine only** (direct cosine for assoc cost), never for Cheb-GR
   k-reciprocal.

2. **No duplicate embeddings in Cheb-GR graph** — `dupfill` copies are
   self-nearest-neighbours that squeeze out k2 neighbourhood, making the
   mechanism conservative (accepts 51→31, IDs up). Copies may be used for
   graph-external purposes (per-frame assoc cost) but never fed into the
   k-reciprocal gallery. The bank stores unique extractions only.

3. **No quality-signal selection** — after visclean front-occlusion gating,
   det-score / box-height / neighbour-IoU have no residual correlation with
   embedding self-consistency (Spearman ≈ 0 / +0.15 / +0.1, m/s reproduced).
   Quality-gated top-K selection loses to same-budget temporal sampling.

## 3. API

```
CleanFifoBank(fifo_n=20, stride=1, decide_n=5)
  .store(tid, embedding, frame_id)          # raw store + FIFO eviction
  .update(tid, emb, frame, is_clean=, frame_offset_from_birth=)  # online: schedule + store
  .should_extract(tid, is_clean=, frame_offset_from_birth=)     # decision (side-effects counter)
  .samples(tid) -> Tensor | None            # raw FIFO embeddings (Cheb-GR graph)
  .samples_before(tid, frame) -> Tensor | None  # occ-audit pre-episode reference
  .representative(tid) -> Tensor | None     # mean — GRAPH-EXTERNAL cosine ONLY
  .clean_ids() -> set[int]
  .count(tid) -> int
  .frames(tid) -> list[int]
  .prune(alive_ids)
  .fill_from(crops, embeddings)             # batch fill from extraction

plan_clean_fifo_crops(records, *, cov, fifo_n, stride, decide_n, neighbor_iou_max)
  -> CleanFifoPlan(head_crops, bank_crops, bank)

build_filled_bank(results_lines, seq_dir, extractor, *, cov, fifo_n, ...)
  -> CleanFifoBank  # ready for queries
```

## 4. Consumers

| Consumer | Status | Usage | Graph-internal? |
|---|---|---|---|
| `cheb_gr_online.extract_handover_embeddings` | ✅ wired (Phase 1) | Bank = dead track's FIFO; head = newborn's first `decide_n` clean frames | Yes (k-reciprocal) |
| `occ_audit.occ_exit_audit_lines_from_bank` | ✅ wired (Phase 2a) | Reference = `bank.samples_before(tid, occ_start)`; audit = post-exit crops | No (direct cosine) |
| `TrackAppearanceBank` FIFO replacement | 🔬 probe (Phase 2b) | Replace score-gated top-K + EMA with visclean-gated FIFO-20 + raw | No (consistency gate, C++ feed) |
| `OutputAppearanceBank` FIFO replacement | 🔬 probe (Phase 2b) | Replace score-gated top-K + mean with visclean-gated FIFO-20 + raw | No (post-merge appearance gate) |
| Online assoc cost via forwarded embedding | 🔬 probe (Phase 2c) | `representative()` cosine in post-merge cost (graph-external copy use) | No (direct cosine) |
| C++ async sidecar ring buffer | 📋 spec only (#57) | Per-track clean-FIFO-20 + stride-3 + event-aware birth on GPU | TBD (async, #57 gates sync) |

## 5. Scheduling: stride + birth window

- **Birth window** (first `decide_n` frames): dense extraction (every clean
  frame). `min_head=2` needs evidence; #56 lesson was budget starvation of
  head.
- **Steady state**: every `stride`-th clean frame (stride=3 → cost ÷3,
  probe-validated equivalent at −0.05/−0.03 dIDF1).
- **FIFO eviction**: oldest sample evicted when `len > fifo_n`. The FIFO
  naturally holds the most recent `fifo_n` clean samples — death during
  occlusion is handled because visclean gate skips dirty frames, so the
  FIFO contains samples from before the dirty period.

## 6. Boundary with `OutputAppearanceBank` / `TrackAppearanceBank`

| Property | CleanFifoBank | TrackAppearanceBank | OutputAppearanceBank |
|---|---|---|---|
| Selection | Time-gated FIFO-N | Score-gated top-K | Score-gated top-K |
| Gate | visclean (full-frame) | `geometry_clean` (per-track) | `min_score` (det score) |
| Representative | Mean (graph-external only) | EMA (alpha=0.8) | Mean |
| Consistency | Not built-in | Mean pairwise cosine | Mean pairwise cosine |
| Store | Raw embeddings | Raw embeddings + metadata | Raw embeddings |
| Consumers | Handover, occ-audit, assoc cost | Consistency gate, C++ clean_ids feed, relink | Post-merge appearance gate |
| Scope | Reusable substrate | Live tracker (per-frame) | Post-lifecycle merge |

**Do not mix**: `TrackAppearanceBank` serves the live tracker's per-frame
assoc loop (EMA, velocity-aligned, high-quality gate). `CleanFifoBank`
serves identity relink (handover, occ-audit, post-merge). They have
different gating contracts and different consumers. A future unification
(Phase 2b GO) would replace `TrackAppearanceBank` internals with FIFO
scheduling while keeping its consumer-facing API.

## 7. Probes

| Probe | Script | Question |
|---|---|---|
| occ-audit bank reference | `probe_occ_audit_bank_reference.py` | Does bank-sourced ref match post-hoc re-extract? |
| TrackAppearanceBank FIFO replacement | `probe_track_bank_fifo_replacement.py` | Is FIFO-20 equivalent to top-K-EMA for consistency/rep? |
| Forwarded embedding assoc cost | `probe_forwarded_embedding_assoc_cost.py` | Does FIFO rep cosine improve post-merge assoc? |

## 8. Config

| Flag | Default | Effect |
|---|---|---|
| `--cheb-gr-offline-bank-mode {spread,recent}` | `spread` | Handover bank mode (`recent` = CleanFifoBank) |
| `--cheb-gr-offline-bank-n` | `0` (= n_samples) | FIFO size |
| `--occ-audit-bank-reference` | `False` | Use CleanFifoBank for occ-audit reference |
| `--occ-audit-bank-n` | `20` | occ-audit bank FIFO size |

## 9. Files

- Substrate: `src/saccade/perception/eval/clean_fifo_bank.py`
- Handover adapter: `src/saccade/perception/eval/cheb_gr_online.py` (`extract_handover_embeddings`)
- Occ-audit bank path: `src/saccade/perception/eval/occ_audit.py` (`occ_exit_audit_lines_from_bank`, `extract_audit_embeddings_post_exit`)
- Config: `src/saccade/perception/eval/config.py` (`occ_audit_bank_reference`, `occ_audit_bank_n`)
- Evaluator wiring: `src/saccade/perception/eval/evaluator.py` (both paths)
- Config file: `configs/modules/cheb_gr_offline_mnv4_fifo20.yaml`
- Tests: `tests/unit/eval/test_clean_fifo_bank.py`, `tests/unit/eval/test_occ_audit_bank_reference.py`
- Probes: `scripts/eval/diagnostics/probe_{occ_audit_bank_reference,track_bank_fifo_replacement,forwarded_embedding_assoc_cost}.py`
