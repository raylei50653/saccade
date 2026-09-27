# Reproduction and takeover

Run commands from the repository root. Dataset images, frozen local substrates
and the engine are local prerequisites; no tracker recapture is needed.

## Qualified implementation

Source head: `31dfcaee69173841dd62323bb53dc1cd327a838c`, clean during capture.
Numerical implementation is `55f4d82b` plus #463's main merge ancestry.

```bash
python tools/resctl.py run machine-bench -- bash docs/modules/semantic/research/evidence/merge_sparse_equivalence_20260927/qualify.sh
```

The script runs MOT17 7, MOT20 4 and DanceTrack 40, using the unchanged 6-digit
DanceTrack symlink mirror from #459. Each sequence extracts embeddings once;
dense is run only for the 49 reference sequences. Default sparse runs twice
with identical embeddings, and forced-blocked sparse is compared with dense.
Any failed decision/output/repeated-cost comparison returns nonzero.
`--dense-max-samples=27000` excludes only MOT20-03/05 from dense execution.

Qualification output: `results/xval459_20260924/qualified_ordered_20260927/`.
The sealed copies are under `qualification/`. Each JSON records source hashes,
embedding/substrate hashes, numeric settings, full pair-comparison summaries,
pre-interpolation and final MOT hashes, and sparse core time/memory.
Timing is merge core wall time with CUDA synchronization, **excluding** image
loading, embedding extraction, interpolation and metric scoring. Memory reports
PyTorch peak allocated/reserved and incremental allocated bytes. It is not an
NVML measurement of total device use or TensorRT's allocator.

## Scored replay epoch

MOT17 and DanceTrack scored replays were already produced at clean `e7960930`
on 2026-09-24. MOT20 2/4 end-to-end equivalence and full 4/4 scored replay were
completed at that same clean head on 2026-09-27. The command below was run
at that head; run it from an isolated checkout of `e7960930` with the local
prerequisites linked to reproduce that epoch. Running it on a newer head
produces new measurements, not a reproduction of its source identity:

```bash
python tools/resctl.py run machine-bench -- bash docs/modules/semantic/research/evidence/merge_sparse_equivalence_20260927/replay_mot20.sh
```

The full shipping-interpolation matrix has three in-process repeats per arm;
the no-interpolation matrix has one. Exact arguments and runtime identities are
inside each `replay/*/results.json`. These rows are never relabelled as measured
at the ordered-reduction head: the qualification's exact final MOT hashes bind
the replacement to these scored rows.

## Acceptance checks

```bash
python docs/modules/semantic/research/evidence/merge_sparse_equivalence_20260927/verify_acceptance.py --archive-only
# On the capture host, also read original local MOT files and substrates:
python docs/modules/semantic/research/evidence/merge_sparse_equivalence_20260927/verify_acceptance.py --archive-only --local-mot
python tools/resctl.py run machine-bench -- .venv/bin/python -m pytest tests/unit/reid/test_cheb_gr_merge_sparse.py tests/unit/reid/test_cheb_gr_merge.py tests/unit/eval/test_output_layer_repair_chaining.py -q
```

`prior_equivalence/` and `prior_acceptance.json` retain the pre-ordered evidence.
`acceptance.json` is the final verifier result. `SHA256SUMS.json` seals every
other packet file. The original #459 packet is referenced and unchanged.

The boundary repair pins immutable evidence and its inventory to reviewed head
`31c36d78a5f8197300127052b54a6a6e62b8a1cb`. Only this command guide and the verifier
are maintained; their current bytes remain covered by `SHA256SUMS.json`.
The verifier compares the entire captured runtime/source identity and the actual
dense/default/forced output hashes, independently of comparison booleans.
The pinned Git objects must be available; missing objects fail closed.

Without `--archive-only`, the verifier additionally rejects current source hash
drift. The input-validation/default repair changes the merge source, and the v3
hash-domain correction changes the qualifier source; neither is silently
relabelled as the captured qualification. `--archive-only` verifies historical
evidence and explicitly reports source drift and `current_runtime_qualified=false`.
Even a source match alone does not qualify a different execution stack. A current
qualification requires the full frozen reference replay under the declared
contract, with separately reviewed new evidence. Do not overwrite these captures
or reuse `acceptance.json` as a verdict for the repaired source.

## Narrow next step

F2 qualification is bounded to this recorded execution stack. Changes to GPU,
CUDA/PyTorch, precision, block shape policy or reduction/top-k implementation
require requalification. Continue with the separately scoped F1 design review
(merge-seam-aware interpolation or interpolation before merge), then promotion
review. F4 frame naming is a separate portability task. Production remains
ineligible; no operating point or interpolation policy was changed here.

## Hash-domain clarification

Captured qualification schema v2 calls its normalized-lines digest
`substrate_sha256` (blank lines excluded, one trailing newline). Original #459
manifests instead hash raw file bytes. `substrate_bindings.json` binds both
representations to the unchanged original substrate checksums. Captured v2 JSON
is preserved byte-for-byte. The current qualifier's v3 schema records raw bytes
as `substrate_sha256` and normalized lines as `substrate_lines_sha256`; this is
an output-metadata correction after qualification, with no numerical code change.
