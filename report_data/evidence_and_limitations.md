# Evidence, Provenance, and Limitations

## What can be claimed now

- The historical v14 behavior is materially reproducible.
- The reconstructed plain recipe is stable within about 0.4 IDF1 across three
  student-chain seeds.
- T3-to-T1 temporal shaping has two valid same-seed comparisons:
  one strong positive and one small positive.
- The best T3-to-T1 run improves association metrics without increasing
  deployment temporal state.
- CUDA graph and pointwise compilation improve runtime without changing the
  recorded tracking metrics.

These are development and replication claims.

## What cannot be claimed now

- No existing Mamba table is a clean held-out MOT17 generalization result.
- The current evidence does not establish superiority over an equal-budget CNN
  or MLP head.
- Two valid paired seeds are insufficient for a strong confidence interval.
- The original T3-to-T1 result is not paired with the original plain replica.
- Training-loss-selected checkpoints are not publication-grade model
  selection.

## Data leakage

The v14 replication intentionally reproduces the historical lineage:

- teacher training used all seven MOT17-SDP train sequences;
- cache generation used all seven sequences;
- distillation used all seven sequences;
- tracking evaluation is run on the same seven sequence identities.

The results are useful for mechanism development and historical
reproducibility, but they must not be labeled validation, test, or held-out
performance.

The strict-clean requirements are documented in
`docs/modules/detection/mamba-v14r-training-protocol.md`.

## Seed provenance correction

Checkpoint metadata shows:

| Experiment | Actual seed |
|---|---:|
| plain replica | 20260612 |
| original T3-to-T1 | 42 |
| plain / T3-to-T1 pair 2 | 20260613 / 20260613 |
| plain / T3-to-T1 pair 3 | 20260614 / 20260614 |

The original `run_v14replica_t3t1.sh` omitted `--seed`, so the training script
used its default seed 42. Therefore:

- `replica_20260612 -> t3t1_seed42` is an unpaired comparison;
- only 20260613 and 20260614 are valid paired curriculum ablations.

Use the `paired` column in `tables/mamba_t3t1_pairs.csv`.

## Metric protocol

- Dataset: MOT17 train, SDP detector variant, seven sequences.
- IDF1/MOTA/IDs/FP/FN: `motmetrics`, IoU distance threshold 0.5.
- HOTA/DetA/AssA: vendored TrackEval with MOTChallenge preprocessing.
- Runtime: existing `_fps_summary.txt`, 4,966 evaluated frames.
- Tail latency: `_latency_profile.json`; timer starts after decoded-frame fetch
  and ends after tracking output. Current files contain the final evaluated
  sequence profile rather than a pooled seven-sequence distribution.
- Metrics are recomputed from result txt files by
  `report_data/build_paper_assets.py`.

## Remaining threats to validity

1. **Checkpoint selection:** current replica uses training loss.
2. **Small sample count:** only two valid paired seeds.
3. **Single dataset/domain:** all core training evidence is MOT17-SDP.
4. **Hardware specificity:** runtime is from an RTX 5070 Ti Laptop setup.
5. **Confounded architecture comparison:** no equal-budget non-Mamba control.
6. **Historical reconstruction:** legacy seed and exact early scheduler are
   unavailable.
7. **Result-directory provenance:** MOT txt files do not embed git SHA or full
   config; checkpoint hashes are available, but inference config provenance is
   partly documentary.
8. **Tail-latency aggregation:** current root-level latency JSON is overwritten
   per sequence, so its P99 describes MOT17-13-SDP only. Publication runs
   should retain and pool samples from all seven sequences.

## Required cleanup before publication

1. Run the complete strict-clean teacher/cache/distill/GT chain.
2. Pass an explicit seed to every T3 and T1 command.
3. Use at least three valid paired seeds.
4. Select checkpoints on a held-out sequence using recall plus HOTA/IDF1.
5. Add CNN/MLP and full-gradient Mamba controls.
6. Repeat on a second tracking dataset.
7. Store command, git SHA, dirty-tree hash, checkpoint SHA, and resolved config
   beside every result directory.
