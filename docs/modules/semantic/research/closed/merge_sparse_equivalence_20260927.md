# Merge-only F2: sparse scalability and numerical qualification (2026-09-27)

<!-- doc-status: closed -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-27 -->
<!-- doc-module: semantic -->

## Decision and scope

Owner review on 2026-09-27 accepts the fixed merge-only operating point as
**retained opt-in offline repair; not production eligible**. This supplements
[#459](https://github.com/raylei50653/saccade/issues/459) and the
[original report](../merge_only_cross_dataset_20260924.md). That report and its
sealed dense/OOM packet are unchanged. #463 was merged independently before
this follow-up.

F2 replaces dense sample intermediates with row-blocked distances, sparse
reciprocal/query-expanded weights, and Jaccard evaluation for temporally
eligible tracklet pairs. All samples remain in the same doubled query/gallery
graph: temporal pre-gating excludes pooled pair costs, **not graph nodes**.
`max_cost=0.45`, `max_gap=60`, `n_samples=50`, embedding/gallery definition,
candidate ordering/conflict resolution and interpolation policy are unchanged.
F4 frame-name portability is excluded; DanceTrack uses the original symlink
mirror.

## Numerical execution contract

Qualification is specific to the recorded GPU/software stack and source hashes;
it is not a theorem of equivalence on arbitrary inputs or other GPUs.

- CUDA input is FP32, autocast is off and FP32 matmul precision is `highest`
  (TF32 off). The sparse entry point rejects other modes rather than silently
  casting embeddings or altering precision.
- Default distance row blocks use `max_block_elems=2^28`, with row count
  `max(1, floor(2^28 / (2S)))` and an unpadded final block. Qualification also
  exercises `2^20` forced blocking. Neither mode adapts to free VRAM.
- Sparse normalization sums reciprocal entries in ascending column order.
  Query expansion visits source nodes in ascending node order, matching the
  dense COO coalescing convention. Stable grouping followed by sequential
  elementwise additions replaces both floating `index_add_` accumulations.
  Jaccard support is ascending; its existing reduction and pooled `topk/mean`
  remain unchanged. The block/reduction implementation is bound by source SHA.
- `torch.topk` is unchanged from dense. Its ties have no portable stable-index
  guarantee. We do not introduce a sparse-only tie rule, epsilon, rounding or
  threshold tolerance. Dense/sparse decision equivalence is a mandatory empirical
  gate, including the forced block shape.
- Qualification runs under `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The packet records
  PyTorch/CUDA versions, GPU name, matmul flags, block parameters and embedding
  hashes. Repeating sparse with the same embeddings must reproduce **all scored
  pair costs and output hashes exactly**.

Changing GPU/software stack, precision, block policy, reduction order or top-k
implementation invalidates this qualification and requires the full reference
replay. Deterministic execution alone does not imply equality with dense:
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
and [top-k tie semantics](https://docs.pytorch.org/docs/2.11/generated/torch.topk.html).

## Acceptance and evidence provenance

The [supplemental packet](../evidence/merge_sparse_equivalence_20260927/manifest.json)
separates two implementation epochs:

1. Original sparse `e7960930`: historical MOT17/DanceTrack replay, newly completed
   MOT20 4/4 scored replay (three repeats per arm), and interpolation-off replay.
2. Ordered-reduction qualification `31dfcaee`: fresh dense/sparse/forced-blocked
   comparison on the 49 reference sequences, sparse repeated-cost checks, and
   MOT20-03/05 completion/memory/time measurement.

Final MOT byte identity binds the qualified implementation to the independently
scored replay rows. Full-precision metrics are reused only after exact final
hash equality; no metric tolerance or rounded table comparison is used.
The qualifier also checks historical dense and pre-interpolation output hashes.
Partition equivalence follows from identical pre-interpolation MOT records and
is independently reconstructed from accepted edges in the prior acceptance
check. Accepted-pair equality is checked directly against dense by the qualifier.

The 49-sequence reference set is MOT17 7 + MOT20-01/02 + DanceTrack 40.
Accepted merge sets, merged track partitions and final MOT outputs are identical;
full-precision IDF1/HOTA/AssA/MOTA agree. Both default and forced-blocked sparse
pass; default sparse also repeats all scored costs exactly within process.
Temporally pre-gated rejected logs may differ as declared. Costs need not equal
dense bitwise; decisions and outputs must.

MOT20 now covers **4/4 sequences**, including previously missing 03/05. There is
no dense decision oracle for 03/05: feasibility and metrics are measured with
the qualified replacement; their historical dense coverage gap remains recorded.

| MOT20 4/4 arm comparison | ΔIDF1 | ΔHOTA | ΔAssA | ΔMOTA | ΔIDs | ΔFP |
|---|---:|---:|---:|---:|---:|---:|
| Shipping interpolation, n=3 | +1.572534 | +0.733476 | +1.220984 | +0.272339 | -338 | +1590 |
| Interpolation off, n=1 | +1.119531 | +0.459957 | +0.961029 | +0.031641 | -373 | +7 |

Shipping-interpolation IDF1 increases on each of the four sequences. All three
full-set repeats have byte-identical final outputs and zero full-precision metric
variation. These results remove the tested dense-scene feasibility limitation;
they do not resolve F1 or confer production eligibility.

| Sequence | Samples | Accepted merges | Core wall (s) | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 4026 | 12 | 1.154 | 0.489 | 0.496 |
| MOT20-02 | 22718 | 85 | 6.133 | 2.031 | 2.051 |
| MOT20-03 | 36871 | 131 | 11.091 | 2.369 | 2.604 |
| MOT20-05 | 65348 | 326 | 32.255 | 4.117 | 4.459 |

Memory is PyTorch allocation accounting during merge, including its allocated
baseline; it excludes TensorRT/driver allocations and is not total-device NVML
peak. The packet also records incremental allocated memory. Core wall time
excludes extraction, interpolation and scoring. MOT20-02 in the same qualification
run takes 277.029 s for dense versus 6.1 s for sparse (full values in the packet).

The original #459 report grouped 03/05 as missing OOM coverage; its manifest
specifically records 05 as not attempted after 03's OOM. This supplement preserves
that distinction instead of retroactively inventing an observed dense-05 failure.


## Remaining promotion conditions and handoff

F1 remains a separate merge × interpolation study. The proposed candidates are
(1) interpolation within each original tracklet, excluding merge-introduced
seams, and (2) interpolation before identity merge. GT-labelled merge precision
is diagnostic, not a deployable interpolation gate. Neither candidate has been
implemented or promoted here.

The owner-requested next review is after F1, using this supplemental evidence.
F4 `%06d` portability remains an independent small fix. Do not change the
operating point, recapture tracker substrates, reopen #459's original evidence,
or silently extend this numerical qualification to a new execution stack.

Replay commands, exact source identities, checksums and the standalone acceptance
verifier are in the packet. The narrow next entry point is F1 design review, not another F2 search.
Use `tools/resctl.py handoff-show --here` for local takeover metadata; GitHub
remains authoritative for the PR head and CI.

## Hash-domain clarification

Captured qualification schema v2 calls its normalized-lines digest
`substrate_sha256` (blank lines excluded, one trailing newline). Original #459
manifests instead hash raw file bytes. `substrate_bindings.json` binds both
representations to the unchanged original substrate checksums. Captured v2 JSON
is preserved byte-for-byte. The current qualifier's v3 schema records raw bytes
as `substrate_sha256` and normalized lines as `substrate_lines_sha256`; this is
an output-metadata correction after qualification, with no numerical code change.

## Qualification-boundary repair after PR review

The shared merge API again defaults to the historical dense implementation.
The offline repair harness and qualifier explicitly select sparse. Sparse merge
validates each supplied embedding as FP32 before concatenation can promote a
mixed-dtype input. No thresholds, datasets, numerical reductions or interpolation
rules changed in this repair.

The maintained acceptance verifier pins the immutable supplemental and original
packet inventories, checks the full captured runtime/source identity, and directly
compares dense/default/forced output hashes. Its default also rejects current
source drift; `--archive-only` explicitly audits the historical capture without
qualifying current code or runtime. The original captures and `acceptance.json`
remain historical. Boundary regression tests do not replace full reference
requalification after a source change. See the packet command guide for the
updated archival check. F1 interpolation research remains separate and unresolved;
status remains **not production eligible**.
