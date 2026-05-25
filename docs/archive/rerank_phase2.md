# MOT17 Rerank Phase 2 Experiments

Date: 2026-04-29

## Goal

Validate whether Phase 2 controls improve identity stability after Phase 1 showed that multi-sample rerank scoring was not the main bottleneck.

Focus:

- `C`: semantic bank inject
- `D`: reciprocal margin
- `CD`: combined variants

Success criteria:

- `IDs` down
- `IDF1` flat or up
- `FP` not materially worse
- no regression on `MOT17-04 / MOT17-10 / MOT17-13`

## Run

```bash
uv run python scripts/eval/ablation_rerank.py --fast --output-root scripts/eval/output/ablation_rerank
```

Recorded summary output:

```text
scripts/eval/output/ablation_rerank.txt
```

Expected runs:

- `base`
- `c_bank_inject`
- `d_margin002`
- `d_margin005`
- `d_margin010`
- `cd_margin002`
- `cd_margin005`

## Phase 1 Context

Conclusion recorded on `2026-04-29`:

- Increasing `semantic_buffer_size` and changing `rerank_mode` did not fix the real problem.
- `MOT17-11` showed systematic `IDF1` regression.
- `MOT17-02` did not show meaningful `FP` / `IDs` recovery.
- The main bottleneck appears to be reference quality and false-accept filtering, not multi-sample appearance scoring.

## Results

| Config | IDF1 | MOTA | IDs | FP | FN | Notes |
|---|---:|---:|---:|---:|---:|---|
| `base` | 43.5% | 33.8% | 1301 | 16888 | 56203 | buf=1 EMA baseline |
| `c_bank_inject` | 44.9% | 34.6% | 1034 | 16724 | 55738 | Strong global gain, especially on 04 |
| `d_margin002` | 44.1% | 34.1% | 1190 | 16761 | 56020 | Mild gain, conservative |
| `d_margin005` | 43.9% | 33.8% | 1244 | 16734 | 56357 | Near-tied with base |
| `d_margin010` | 45.0% | 34.4% | 1071 | 16786 | 55780 | Stronger than D low margins, slight FP cost |
| `cd_margin002` | 45.0% | 34.2% | 1037 | 16652 | 56172 | Better IDF1 and lowest FP, but weaker than best CD |
| `cd_margin005` | 45.3% | 34.7% | 842 | 16719 | 55723 | Best overall Phase 2 result |

## Sequence Notes

Key sequence observations from `ablation_rerank.txt`:

- `MOT17-02`: bank inject alone regresses slightly; reciprocal-only `0.02` is the cleanest target-sequence variant.
- `MOT17-11`: all Phase 2 variants avoid the large regression seen in Phase 1; differences are small but non-negative.
- `MOT17-04`: this is the main win source. `C` and especially `CD 0.05` sharply reduce IDs and improve IDF1/MOTA.
- `MOT17-10`: all variants are neutral-to-slightly-better, mostly via small IDs reductions.
- `MOT17-13`: essentially flat; tiny `IDs +1` on margin variants is not material.

## Decision

Adopt / reject criteria:

- Adopt `C` if it reduces `IDs` without noticeable `IDF1` loss.
- Adopt `D` if reciprocal filtering cuts false accepts without raising fragmentation.
- Adopt `CD` only if the combination is strictly better than the best single control.

Final decision:

- Adopt `C+D: inject+margin=0.05` as the leading candidate.
- `C` alone is also a viable simpler fallback if we want lower complexity while keeping most of the gain.
- Pure reciprocal margin without inject is not the best tradeoff.

## Next Step

- Promote `C+D: inject+margin=0.05` into the eval default candidate and verify against the current documented base in `tracking/fp_fn_recovery_and_gmc.md`.
- If wider verification reproduces the gain, update the tracking experiment doc and CLI defaults.
