# math_model.md Drift Audit (2026-08-30)

Static cross-check of [`docs/reference/math_model.md`](../../../reference/math_model.md)
(source audit banner **2026-07-09**) against every source anchor the banner names, over the
commit range `8b2f4e05..0e869fea`.

**Baseline ref:** `8b2f4e05` (`2026-07-09`, last commit on the audit date) — the tree state the
2026-07-09 audit examined.
**Head ref:** `0e869fea` (`2026-08-29`, `origin/main`).

**Scope:** documentation drift only. No production code, preset value, or kernel changes in this
audit.
**Does not trigger P4** (no behavior change introduced here).

**Method:** commit-range diff of the anchored files only; no MOT17 runs, no build. Predicate-level
comparison for every gate the model transcribes.

Status legend is inherited from
[math_model_drift_2026-07-09](math_model_drift_2026-07-09.md):
`MATCH` / `DRIFT` / `STALE` / `NO-GO` / `UNKNOWN`.

---

## Executive summary

| Bucket | Count | Headline |
|:--|:--|:--|
| `MATCH` | 7 of 9 checked anchors | 所有被轉寫的 gate 述詞逐條存活,浮點形式相同 |
| `DRIFT` | **0** | 沒有任何 math_model 的句子變成假的 |
| `STALE` | **2** | §4.2 compaction 取得新的 ordering 性質未記載;`pipeline.py` 行錨過期 ~153 行 |

**沒有 `DRIFT`。** 這次的結論是模型的**式子**仍然忠實,過期的是兩處框架性描述。

Three of the six anchored sources have **zero diff** across the whole range, so the sections they
back are not merely re-verified — they are untouched:

| Anchor | Diff vs baseline | Sections backed |
|:--|:--|:--|
| `include/tracking/kalman_gpu.cuh` | **0 commits / 0 lines** | §6 Kalman(prediction、update、Mahalanobis gate) |
| `src/tracking/gmc_kernel.cu` | **0 commits / 0 lines** | §5 GMC(downscale、cross-power spectrum、confidence/warp) |
| `src/tracking/relink_gate.cu` | **0 commits / 0 lines** | §10.5 gates and commit |
| `configs/presets/mamba_whole_graph.yaml` | **0 commits / 0 lines** | §1 baseline contract(s) |
| `configs/presets/mamba_whole_graph_m.yaml` | **0 commits / 0 lines** | §1.1 m delta |
| `src/tracking/tracker_gpu.cu` | 9 commits / +1435 −50 | §4.2、§7、§8、§9、§10 |
| `src/saccade/perception/eval/pipeline.py` | 7 commits / +153 −0 | inject site |
| `src/saccade/perception/eval/evaluator.py` | 5 commits / +141 −12 | stage orchestration |

---

## A. `tracker_gpu.cu` — +1435/−50,但被轉寫的述詞零漂移

The bulk of the range on this file is H0/D0 trace-record plumbing (`H0Bridge*Record`, cursors,
overflow counters) inserted **around** the decision points, not into them. Every gate the model
transcribes was located in the current file and compared to the deleted line:

| Model claim | Baseline form | Current form | Status |
|:--|:--|:--|:--|
| §10.5 height gate | `if (hr < bridge_h_lo \|\| hr > bridge_h_hi) continue;` | `tracker_gpu.cu:2240` — same predicate, trace record written at `:2237` **before** it | MATCH |
| §10.5 speed gate | `if (speed > bridge_max_speed) continue;` | `:2264`, verdict recorded at `:2262` | MATCH |
| §10.5 spatial gate | `if (cdist > bridge_spatial_gate) continue;` | `:2283`, verdict at `:2281` | MATCH |
| §10.5 bdist cutoff | `bool ok = bdist <= bridge_px;` | `:2420`, identical | MATCH |
| §7.6 / occ cover gate | `if (occ_gate_cover > 0.0f && occ >= 0.0f && occ < occ_gate_cover) continue;` | `:2445`, identical predicate; the surrounding `if (occ_gate_cover > 0.0f \|\| expandable)` at `:2433` is the pre-existing expand path (`occ_expand_px` / `occ_expand_cover` were already launch args at baseline) | MATCH |
| appearance veto | `if (n2l > 1e-6f && dot * rsqrtf(n2l) * cand_inv_norm < app_veto_cos)` | `:2478` reads `if (n2l > 1e-6f && cosine < app_veto_cos)` where `cosine` is defined at `:2472` as `n2l > 1e-6f ? dot * rsqrtf(n2l) * cand_inv_norm : 0.0f` — **same operations in the same order**, hoisted so the trace can record the scalar | MATCH |
| §10.5 margin | `if (bridge_margin > 0.0f && (second_dist - best_dist) < bridge_margin) return;` | `:2587`, identical | MATCH |
| §10.5 commit claim | `if ((bridge_claim[lost] & 0xFFFF) != (cand & 0xFFFF)) return;` | `:2651` hoists `winning_cand`, `:2667` returns on the same comparison | MATCH |

The `−50` deletions are kernel-signature extensions (trace pointers appended to argument lists),
launch-site re-indentation, and the hoists above. None removes a modeled term.

### A1. `STALE` — §4.2 compaction 現在有一個模型沒記的性質

`dc1691b0` (2026-07-10, *Fix native filter nondeterminism and add decimal consistency guards*,
one day after the baseline audit) replaced the atomic-compaction detection filter with a
three-phase path:

```text
filter_keep_mask_kernel -> CUB exclusive scan -> filter_stable_scatter_kernel
```

The **predicate is unchanged** — `tracking::detection_keep(box4, score, class, params, ...)` still
decides membership, so §4.2's `score/class/geometry filtering` line stays true. What changed is a
property §4.2 never stated: the surviving detections are now emitted in **stable index-ascending
order** rather than in atomic-arrival order. The legacy kernel is retained default-off for A/B, and
a single-thread serial oracle is gated behind `SACCADE_DETERMINISTIC_FILTER_COMPACTION=1`.

This matters to the model beyond §4.2 because §8 (sparse top-K and auction assignment) consumes
that order. The model's `compact/gather` box is not false; it is silent on a guarantee the
implementation now provides.

**Action:** record the ordering property in §4.2. Not a `DRIFT` — no sentence is inverted.

---

## B. `pipeline.py` / `evaluator.py` — 佈線增加,注入值未變

All 7 commits on `pipeline.py` are additive research-capture wiring: D0 bridge-fidelity capture,
R1 temporal-reduction capture, the portable OR-tail hook, and their env-var / kwargs gates. Every
one is default-off, and the mutual-exclusion guard between the D0 and R1 capture dirs raises rather
than silently picking one. None of them changes a value passed to `set_params` /
`set_occ_params` / `set_relink_params`.

`evaluator.py`'s deletions in this range are debug-print formatting for the relink counters, not
stage scheduling.

`a4981b65`'s portable OR-tail hook does not appear in either headline preset, so the headline path
takes the default (`False`).

### B1. `STALE` — inject 行錨過期

§4.2 cites the inject site as *「約 `pipeline.py:951+`」* and frames the three setters as one block.
Current file:

| Setter | Line |
|:--|:--|
| `set_relink_params` | `pipeline.py:648` |
| `set_params` | `pipeline.py:1104` |
| `set_occ_params` | `pipeline.py:1132` |

`set_relink_params` is ~450 lines **before** the other two, not adjacent to them. The 2026-07-09
audit already carried line anchors as `STALE`; the +153 lines in this range widened the gap.

**Action:** replace the single `951+` anchor with the three symbol-level anchors above.

---

## C. 不在錨點清單但值得記的

`9191ea1a` (*fix(h2): separate probe from runtime equivalence*) is the most recent commit under
`src/` in the range, but it touches CI workflow, runtime-identity tooling, generated inventories and
research contracts — no modeled kernel or inject value. No model impact.

---

## D. 本次不做的事

- 沒有跑 MOT17,沒有 build。所有結論是靜態比對。
- 沒有動 production code、preset 值、kernel。
- 沒有回頭重驗 2026-07-09 audit 自己的判斷;本文件只覆蓋 `8b2f4e05..0e869fea` 這一段。
- §11 semantic relink gate 的 anchor(`relink_gate.cu`)零 diff,但 semantic 線在此範圍另有研究活動;本文件不對那條線的結論表態。
