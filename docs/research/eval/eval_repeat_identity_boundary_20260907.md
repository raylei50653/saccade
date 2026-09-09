# eval 靜默 MOT 分歧的 observability bound 與 fail-closed harness (#363, 2026-09-07)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-07 -->
<!-- doc-module: cross -->

> **本文回答的問題.** 固定組態下重複執行 eval 會靜默寫出不同的 MOT 檔
> ([#363](https://github.com/raylei50653/saccade/issues/363)).
> 在 MOT 檔裡,最早可觀測的 divergence 落在哪一欄? 是不是 ID 指派?
> 既有的 decimal-hash pre-push sentinel 看不看得到它?
> 如何讓這個症狀再出現時檢查失敗,而不是 exit 0 通過?
> 若開啟 opt-in per-stage fingerprint,首次不同值落在哪個 producer-facing stage?
>
> **本文不主張什麼.** 不主張根因是 #340、decode race 或 CUDA graph.
> 不把樣本分歧比例解讀為底層機率. 不判定哪一個 MOT 輸出是正確的.
> **不主張** `fast_emit_mot_lines` / `emit_tracks_unified` 是 producer —
> 它們是 MOT 行的寫出口. 定位到某個 stage 是首次可觀測 divergence,不是
> causal mechanism. ingest / detect-internal / Kalman 仍可能包在同一個 stage 裡.

量測本體與判讀規則仍以
[nogpudecode_reproducibility_20260907.md](nogpudecode_reproducibility_20260907.md)
為準. 本文只對那批已存 MOT 做第一個分歧幀的分類,並交付 harness.

---

## 1. Observability bound

對 `~/.local/state/saccade/perf/nogpudecode-reproducibility-20260907/` 與
`gpu-decode-nondeterminism-recheck-20260907/` 的 tracker 輸出,用既有
`decimal_hash` (去掉 track ID 的 box+score canonical 記錄) 對第一條 raw 差異行分類.

**單一序列 (block S, `--preset baseline`, MOT17-02-SDP, `--no-gpu-decode`, n=40 中的 4 次分歧).**
每一次的第一個差異都已經在 MOT 序列化的 box 和/或 score 上,而不是 ID 指派:

| pair | first frame | 該幀 records | 第一個改到的 track | 分類 |
|---|---:|---|---|---|
| S_r1 vs S_r3 | 63 | 17/17, 16 條完全相同 | id 15, box 與 score 都變 | `geometry_or_score` |
| S_r1 vs S_r15 | 15 | 17/16 | id 7 box+score, ref 多 id 17 | `geometry_or_score` |
| S_r1 vs S_r30 | 154 | 18/18, 15 條相同 | id 43 **box 字串相同**, score `0.3617` vs `0.3784` | `geometry_or_score` |
| S_r1 vs S_r40 | 291 | 26/26, 25 條相同 | id 102, y 差 0.04px, score 也變 | `geometry_or_score` |

S_r1 vs S_r2 位元相同 (負對照).

GPU-decode 路徑 (arm G, 同 preset / 同序列, n=8 中 1 次分歧) 同一個形狀:
frame 443, 14/14 records, 11 條相同, 3 條 box+score 都變, track id 仍對得上.

因此,至少在單序列重複執行上:

**最早可觀測的 divergence 已在 `track_results` → MOT 這個邊界存在:**
serialization 時 box 和/或 score 已經不同,不是 identity assignment.
`fast_emit_mot_lines` / `emit_tracks_unified` (`helpers.py`, `stages.py`)
是把當時的 `track_results` 寫成 MOT 行的出口,本文**沒有**證明它們製造差異.
那一筆分數是 tracker score,箱子是 tracker box,所以 producer 仍可以是
detect / NMS,或 Kalman / GMC 之後的更新 — MOT 檔分不開.
S_r30 的 score-only 差異(箱子序列化相同)削弱「只是 GMC 把幾何 warp 了」
這種單因解釋,但還不能定位 producer.

**七序列 (block H) 的第二種 MOT 第一行形狀不是獨立的可觀測事件.**
MOT17-02 幾乎都與 reference 相同;某個較晚序列先出現 `geometry_or_score`,
隨後同一 process 的後面序列常在第一個輸出幀 (frame 4) 變成 `identity_only`.
例: H_r1 MOT17-05 vs reference, frame 4 是同一個箱子與 score,ID `464` vs `463`.
這與 `GlobalTrackIdMapper` (`tracking.py`) 跨序列單調分配 global ID 相符:
前面序列若因幾何分歧少生/多生一條 track,後面序列的 MOT ID 整體平移.
後面序列仍可能另有幾何差異;raw 檔的「第一行差異」會先被 ID 平移佔住.
這解釋了為什麼 7-seq 的分歧比例看起來比較高,不是「序列越多就越常 race」的機制判斷.

**block D (`mamba_whole_graph_m --double-buffer`, 7-seq, n=20) 仍是 0 次分歧.**
本文不把這個對照讀成「double-buffer 修復了它」:組態與 block 順序混在一起
(見 nogpudecode 文 §6).

---

## 2. 既有 sentinel 為什麼靜默通過

| 工具 | 它在問的問題 | 對 #363 |
|---|---|---|
| `scripts/eval/mot17.py` | 這次 eval 有沒有 crash | 分歧時仍 exit 0 |
| `scratch/determinism_harness.sh` | 印 DISTINCT,但只在空檔 / crash 時 exit 1 | DISTINCT>1 仍成功 |
| `check_decimal_chain_routine.py` (pre-push) | 同一 process 連續跑 A,A,B,A,B,B,ID-free decimal hash 是否自洽 | 預設組態是 block D (`mamba_whole_graph_m --double-buffer`);而且去掉 ID,看不到 global-ID 平移 |

所以「症狀再出現時檢查失敗」這件事,先前沒有任何一條路徑在做.

---

## 3. Harness

`scripts/tools/check_eval_repeat_identity.py`,核心在
`scripts/tools/eval_repeat_identity.py`(在 ``scripts/tools/``,不在 ``src/``:
這是檢查器,不是 production eval,不能移動 published implementation digest).

- **pass/fail = raw MOT 位元** (含 track ID). 這是 issue 寫的症狀.
- `geometry_or_score` / `identity_only` 只是 forensic 標籤,不是免責.
- 空檔、缺序列、eval 非 0(即使寫出完整且相同的 MOT),一律失敗.
  不能分辨「是否相同」時不得當通過.
- `compare DIR...` 比既有 run 目錄. 對存檔的 S_r1 vs S_r3 必須失敗,S_r1 vs S_r2 必須通過.
- `run -n N [mot17 flags...]` 開 N 個獨立 process 再 compare. 預設是 block S 組態.
  N 次全同**不是**確定性證明;出現一次相異 hash 就是失敗.
- **不進 pre-push.** 現在的 `baseline` 路徑已知會分歧,預設 CI 會在 main 上紅燈.
  修復之後 `run` 才適合當 regression gate.

```bash
# 正對照 (存檔證據,不重跑 GPU)
uv run python scripts/tools/check_eval_repeat_identity.py compare \
  ~/.local/state/saccade/perf/nogpudecode-reproducibility-20260907/S_r1 \
  ~/.local/state/saccade/perf/nogpudecode-reproducibility-20260907/S_r3

# 現場重現 (獨立 process; N 次全同不是證明)
uv run python scripts/tools/check_eval_repeat_identity.py run -n 8
```

裝置無關的契約測試在 `tests/unit/eval/test_eval_repeat_identity.py`
(MOT 層)與 `tests/unit/eval/test_eval_stage_fingerprint.py`(per-stage).
本機若有上述證據目錄,同一檔會再跑一對存檔正/負對照.存檔 MOT **沒有**
stage fingerprint,不能用來回溯定位.

---

## 4. Closure 對照 (#363)

#363 原文條件不改:(1 可重現機制 **或** 2 指出是哪一段執行路徑**產生** divergence)
**並且** 3 fail-closed harness.

| 條件 | 狀態 | 說明 |
|---|---|---|
| 1 機制 | 未做 | 不指認 buffer / stream / capture;`mechanism_claim=false` |
| 2 產生路徑 | **滿足.** GPU-decode `mot`→`mot_file`;凍結表 `sufficient`(§10 / §11) | 見 §6 / `CONDITION2_RULES`.localization run **不**設 `issue_close`;owner 已接受並 close |
| 3 harness | #367 交付 | `check_eval_repeat_identity.py`:相異 MOT、空檔、缺序列、eval 非 0(含 MOT 位元相同)都 exit 1 |

因此 #367 是 **(3) + MOT observability bound** 的 milestone,不是 close.#363 依 (2) AND (3) **closed**;Condition 1 未做、非必需.
七序列後續的 ID-only 第一行仍是 `GlobalTrackIdMapper` 對前面序列不同出生數的下游 — 那是 MOT 第一行差異的形狀,同樣不是 producer 指認.

定位 live #363 組態的首次 stage 需要帶 `--stage-fingerprint` 的新 runs;2026-09-07 已存 MOT 無法回溯.不另開 MOT-level repetition study.

---

## 5. Opt-in per-stage fingerprint

預設 production eval **不**收集 fingerprint.開啟方式:

```bash
# 重現 block S,並在每個獨立 process 寫入 per-stage hash
uv run python scripts/tools/check_eval_repeat_identity.py run -n 8 --stage-fingerprint

# 只比既有目錄裡的 fingerprint(缺檔 / 不完整 → fail-closed)
uv run python scripts/tools/check_eval_repeat_identity.py compare DIR DIR --stage-fingerprint
```

**Hash-first.** 各 stage 對 canonicalized data 做 deterministic `ordered_bit_hash`.
compact integer rows(MOT 序列化尺度:centipixel / 1e-4 score)一併寫進
`stage_fingerprint/fingerprints.jsonl`,**不**寫 GPU tensor / `.pt`.
comparator 只在首次 mismatch 抽出 `first_divergence.json`(必要 payload),
不是常態 dump 每一幀的 tensor.

**Producer-facing stages**(流水線順序,也是「第一個」的定義):

| stage | 觀測面 | 仍包在裡面、切不開的 |
|---|---|---|
| `detector_output` | `detect_fn` 回傳的 boxes/scores/classes | decode / preprocess / detector raw vs detector postprocess |
| `post_nms` | evaluator NMS 之後的 detections | NMS 與 NMS 前的 tail filter 內部步驟 |
| `tracker_input` | `_run_track` 前的 fused detections | ReID / GMC / 其餘 pre-update 變換 |
| `tracker_output` | MOT emit 當下的 host `track_results` | Kalman / association / GPU buffer / materialize D2H |
| `mot` | `_fast_emit_mot_lines` 的 per-frame 行 | global-id mapping 與 `.2f`/`.4f` 序列化 |
| `mot_file` | 序列結束後、merge/interpolate 之後的 MOT 行 | post-lifecycle merge / interpolate / quality filter |

`post-decode/input` **沒有**儀器化:對整張 decoded frame 做 hash 會每幀多一次
full-image D2H,observer effect 太大.若首次 divergence 落在 `detector_output`,
現有 instrumentation **不足以**再把 decode 與 detect 切開 — comparator 會如實
報告那個 stage,不把它說成 decode race 或 detector kernel.

**Observer effect.** 預設不做 per-frame CUDA fence.GPU stage 用當前 stream 上的
D2D clone,hash 延到 eval 結束後一次 `synchronize`,然後 D2H 的是 detection 尺寸
的 tensor,不是整幀影像.Emit 路徑只 hash MOT 本來就會拷到 host 的
`track_results`.即使如此,額外 D2D alloc 仍可能改變 allocator / timing;這不是
zero-observer 儀器,限制寫在 `manifest.json` 的 `observer_effect`.

**如何讀結果.** 只有各 eval exit 0、MOT 檔完整非空且至少有兩種 raw hash 的
pair,才會產生 `kind=first_observable_divergence` 並套 §6 的凍結表.空檔、缺檔、
eval failure 不算 pair,也不會讓 session 提早停止.若最終 MOT 相同、只有 stage
fingerprint 不同,會記成 `stage_divergence_without_mot_divergence`,Condition 2
不適用. `mechanism_claim` 與 `issue_close` 恆為 false.`sufficient` /
`candidate_sufficient` 只表示 producing-path 邊界夠不夠具體,不是 close,也不是
機制.`localization_session.json` 另存有效 flags、return codes 與
`mot_pair_valid`,避免 configuration / process failure 脫離 artifact.

目前 collector schema 是 `eval_stage_fingerprint_v2`:early-exit frame 的下游空
stage 會明確記成 empty fingerprint;若 relink/write 在 background future 完成,
`mot` 取 future drain 後的實際 lines,不取 enqueue 時的空 placeholder.歷史 v1
artifact 仍可同 schema 互比,但 v1/v2 不混比;key coverage 不同或 duplicate key
一律判 incomplete,不當作 stage boundary.

裝置無關契約測試:`tests/unit/eval/test_eval_stage_fingerprint.py`
(identical inputs → identical hashes;單 stage mutation → 該 stage;缺檔 /
不完整 log → fail;condition 2 表凍結).

---

## 6. Condition 2 判讀規則 (live run 前凍結)

權威表是 `CONDITION2_RULES` (`scripts/tools/eval_stage_fingerprint.py`).
本節是給人看的副本.改表必須改測試;看到 live 數據後不得移動門檻.

這不是「可滿足 ⇒ 立刻 close」.條件 2 要的是 **concrete producing-path
boundary**.表上的 `sufficient` **就是** condition 2 的 evidence,不能事後說成
「condition 2 沒有前進」.`mechanism_claim` 仍假.條件 1 未做.
`issue_close` 由 issue-level review 決定,comparator 不關 issue.

| first divergence | last identical | 可聲稱 | `producing_path_verdict` | #363 condition 2 |
|---|---|---|---|---|
| `detector_output` | (none instrumented) | divergence 已在 detect_fn output 出現；decode / preprocess / detector internals 未切開 | `insufficient` | **不足** |
| `post_nms` | `detector_output` | divergence 被界定在 detector output → evaluator NMS | `candidate_sufficient` | **候選可滿足** |
| `tracker_input` | `post_nms` | divergence 在 post-NMS → _run_track 前產生；ReID / GMC 路徑仍多義 | `insufficient` | **目前偏不足** |
| `tracker_output` | `tracker_input` | divergence 在 tracker execution 內產生；Kalman / association / buffer 未切開 | `insufficient` | **目前仍偏不足** |
| `mot` | `tracker_output` | divergence 在 global-ID mapping / serialization 路徑產生 | `sufficient` | **可滿足** |
| `mot_file` | `mot` | divergence 在 sequence-level postprocess 路徑產生 | `sufficient` | **可滿足** |
| MOT diverged, all stages identical | `mot_file` | MOT diverged but every instrumented stage hash matched; instrumentation insufficient | `insufficient` | **不足** |

`post_nms` / `tracker_input` / `tracker_output` / `mot` / `mot_file` 列都預設
「該 stage 之前的 instrumented stages 相同」——這就是 first-divergence 的定義,
comparator 不會在更早的 stage 已不同時寫後面的 stage.

若第一輪 live 落在 `detector_output` 或 `tracker_output` 這種仍過粗的位置,
下一刀只在那個 span 裡再插一層 fingerprint,不增加 MOT-level repetition.

**Live localization (Block S).** 第一輪用歷史上確實有 divergence 的 fixed
config (`--preset baseline --detector SDP --no-gpu-decode --sequences MOT17-02-SDP`)
加 `--stage-fingerprint`.這是 localization experiment,不是重新估計 divergence
rate:不統計 x/40 機率,只需要至少一組 divergent pair,回答

**兩次 run 最後一次相同的 stage 是哪裡,第一個不同的 stage 是哪裡?**

(`last_identical_stage` / `first_divergent_stage`,再套上表.)

### Live session 2026-09-07

Artifact:
`~/.local/state/saccade/perf/block-s-stage-fingerprint-20260907T151445Z/`
(plus `-wave2` `-wave3` `-wave4`; compare at `-compare32`).

| 收尾項 | 結果 |
|---|---|
| 1. divergent pair | **未取得.** 本 session 的 instrumented Block S runs 的 MOT 與 per-stage hash 均相同 |
| 2. `last_identical_stage → first_divergent_stage` | **未定義.** 沒有 first divergence |
| 3. `CONDITION2_RULES` | **未套用.** 沒有 first_divergent_stage 就沒有 allowed_claim / producing_path_verdict |

Fingerprints 完整 (`complete=true`, 六個 stage 都在).這不是 incomplete instrumentation.
Live MOT bytes 與歷史 S_r1 / S_r2 相同 (`md5=36cb0d59904efbced12df64011bce62f`),與歷史 divergent 檔 (S_r3 / S_r15 / S_r30 / S_r40) 不同.不把這個 session 讀成 rate,也不讀成 mechanism,也不把「沒抓到 pair」改寫成 condition 2 的新門檻.

該 hunt 無 preregistered budget,不計入 §7.budgeted session 見 §8.

---

## 7. Localization session budget (下一輪 live 前凍結)

權威常數是 `LOCALIZATION_BUDGET_RUNS` (`scripts/tools/eval_stage_fingerprint.py`).
本節是給人看的副本.改 budget 必須改測試.

**Budget: 16 instrumented Block S runs** (`--preset baseline --detector SDP
--no-gpu-decode --sequences MOT17-02-SDP --stage-fingerprint`).
這是終止條件,不是 rate sample.2026-09-07 無 budget 的 hunt 不計入本 budget.
`--stage-fingerprint` 的 `run -n` 不得超過 16.

這不是 fingerprint 變更,也不是再切 stage.

| session 結果 | 何時 | 做什麼 | 不表示什麼 |
|---|---|---|---|
| `pair_found` | budget 內出現 divergent pair | 記錄 `last_identical_stage → first_divergent_stage`,原封不動套 `CONDITION2_RULES` | 不自動 close issue;condition 1 未做 |
| `in_progress` | 尚未用完 budget 且尚無 pair | 可繼續同一 session 直到 budget | 不是「divergence 消失」 |
| `budget_exhausted_identical` | 16 次全同 | 收斂為下面這句;condition 2 表**不執行** | 不表示 divergence 消失;下一步改查 observer effect / 其他既有 live config,而不是無限加 Block S repetitions |

凍結收斂句 (`BUDGET_EXHAUSTED_CLAIM`):

> divergence was not observed under the instrumented Block S executions within the preregistered localization budget

`run --stage-fingerprint -n 16` 是一次完整 budget session.中途出現 pair 可提前停.

---

## 8. Budgeted localization session 2026-09-07T234956Z

Command:

```bash
uv run python scripts/tools/check_eval_repeat_identity.py run -n 16 --stage-fingerprint
```

Artifact:
`~/.local/state/saccade/perf/block-s-stage-fingerprint-20260907T234956Z-budget16/`

**Session kind: `budget_exhausted_identical`.** 這是三種既定狀態之一.不是 rate,不是 mechanism.

| 鎖住的欄位 | 值 |
|---|---|
| `n_runs` / `budget_runs` | 16 / 16 |
| `divergent_pair` | false |
| `apply_condition2_rules` | false |
| `first_divergent_stage` | 無 (`first_divergence.json` 未寫出) |
| `complete` | true (16/16 manifest; 六 stage; 3596 records/run) |
| eval exits | 16 × 0 |
| `issue_close` | false |
| `mechanism_claim` | false |
| `condition_1_advanced` | false |
| `condition_2_advanced` | false |
| MOT md5 | `36cb0d59904efbced12df64011bce62f` (與歷史 matching `S_r1`/`S_r2` 相同;只當 artifact identity) |

允許的唯一收尾句 (`BUDGET_EXHAUSTED_CLAIM`):

> divergence was not observed under the instrumented Block S executions within the preregistered localization budget

這不表示 divergence 消失,也不表示 instrumentation 修掉 divergence,也不表示 matching mode 比較穩.`CONDITION2_RULES` 未執行.producing-path boundary 仍 unresolved.

下一步不是再加 Block S repetitions,而是查 instrumentation observer effect,或改用其他既有 live configuration.

---

## 9. Instrumentation observer-effect check (Block S repetitions 之後凍結)

權威讀法是 `read_observer_effect` (`scripts/tools/eval_stage_fingerprint.py`).
本節是給人看的副本.改判讀必須改測試.看到 live 量測後不得移動門檻.
`CONDITION2_RULES` 與 `LOCALIZATION_BUDGET_RUNS` 本刀不動.

**問題.** fingerprint 開啟後新增的 D2D clone、allocation、end-of-eval
synchronize、host-side hashing,有沒有改變與歷史 Block S divergence 相關的
stream ordering / allocator state / timing / 其他 runtime boundary.

**不是.** 不是 divergence rate.不是 localization session.不是 condition 1/2.
1+1 MOT identity 只當 artifact,不當 verdict 輸入.

**靜態盤點** (`OBSERVER_EFFECT_SITES`):

| site | 何時 | 在 MOT write 前? | device-wide sync? |
|---|---|---|---|
| `detector_output` D2D clone | `_run_detect` 之後;Block S 此處已有 full-device sync | 是 | 否 |
| `post_nms` D2D clone | NMS→track,沒有 full-device barrier | 是 | 否 |
| `tracker_input` D2D clone | `_run_track` 前 | 是 | 否 |
| GPU snapshot retention | clone 只留到 copy-stream D2H 完成 | 是(in-flight occupancy) | 否 |
| `tracker_output` / `mot` host hash | emit 已把 `track_results` 拷到 host 之後 | 是(拖到下一幀 launch) | 否 |
| `mot_file` host hash | `sequence_result_callback`,檔已寫完 | 否 | 否 |
| `finalize` | `mot17.py` 返回後等 copy-stream event,host hash | 否 | 否 |

**量測.** `check_eval_observer_effect.py run`:同一 Block S,n=1 uninstrumented + n=1 instrumented.
比的是 `torch.cuda.synchronize` 次數(在 `run_eval` 內)與 caching-allocator `reserved_bytes`.
不為了看 divergence 加 runs.

**凍結門檻.** 下列任一即 `observer_effect_identified`:

- instrumented 在 `run_eval` 內的 device-wide sync 次數比 uninstrumented 多
- clone 當下 `reserved_bytes` 上升,或兩臂 `run_eval` 結束時 reserved 差 ≥ 2MiB(一個 caching-allocator block)

否則若仍有 producing-path GPU clone 或 post-eval synchronize → `observer_effect_bounded`
(額外工作存在,但沒有新的 in-loop device join,也沒有 snapshot 造成的 reserved 成長).
兩臂在受檢 boundary 上無差異 → `no_material_perturbation`.

| kind | 允許的句子 | 下一步 | 不表示什麼 |
|---|---|---|---|
| `observer_effect_identified` | enabling per-stage fingerprinting materially perturbs producing-path execution conditions at the inspected runtime boundaries | 調 instrumentation,不改 Condition 2 表 | 不是 mechanism;condition 1/2 不前進 |
| `observer_effect_bounded` | fingerprint extra work exists but is bounded: no extra device-wide sync during the frame loop and no snapshot-attributable caching-allocator reserved growth; remaining ops are detection-sized D2D clones without fence, async D2H on a copy stream, and host hashing of already-copied MOT data | 儀器可信,可轉其他既有 live config | 不是 divergence 消失 |
| `no_material_perturbation` | no material perturbation found at the inspected runtime boundaries | 同上,轉其他既有 live config | 不是 zero-observer 的一般證明 |

`issue_close=false`,`mechanism_claim=false`,localization budget 不重開.

### Live 1+1 2026-09-08T002317Z (retune 前)

Artifact:
`~/.local/state/saccade/perf/block-s-observer-effect-20260908T002317Z/`

| 受檢 boundary | 量測 |
|---|---|
| `run_eval` 內 extra `torch.cuda.synchronize` | 0 (1204 vs 1204) |
| reserved_bytes after `run_eval` | +10485760 (10 MiB) |
| `n_reserved_increases_on_clone` | 6 |
| GPU clones held at finalize | 1800 |
| clone payload | 8957664 bytes |
| post-eval extra sync | 1 |

**kind: `observer_effect_identified`.** 不是 in-loop `torch.cuda.synchronize`,是 snapshot retention 把 caching-allocator reserved 撐大,clone 當下 6 次 reserved 上升(cudaMalloc 級 device join,不走 `torch.cuda.synchronize`).MOT identity 不當 verdict.

Retune(不改 stage / hash / Condition 2):D2D clone 仍在 current stream;接著 copy stream 做 detection-sized async D2H;GPU clone 只留到 D2H 完成;finalize 等 copy-stream event,不再 `torch.cuda.synchronize()`.驗證仍是同一 1+1 契約,不是 localization budget.

### Live 1+1 2026-09-08T002939Z (retune 後)

Artifact:
`~/.local/state/saccade/perf/block-s-observer-effect-20260908T002939Z-retune2/`

| 受檢 boundary | 量測 |
|---|---|
| `run_eval` 內 extra `torch.cuda.synchronize` | 0 (1204 vs 1204) |
| reserved_bytes after `run_eval` | 0 (兩臂都是 150994944) |
| allocated_bytes after `run_eval` | 0 (兩臂都是 75977728) |
| `n_reserved_increases_on_clone` | 0 |
| GPU clones pending at finalize | 0 |
| post-eval extra device-wide sync | 0 |
| fingerprints | `complete=true`, 3596 records |

**kind: `observer_effect_bounded`.**

> fingerprint extra work exists but is bounded: no extra device-wide sync during the frame loop and no snapshot-attributable caching-allocator reserved growth; remaining ops are detection-sized D2D clones without fence, async D2H on a copy stream, and host hashing of already-copied MOT data

1+1 MOT md5 不同 (`36cb0d59904efbced12df64011bce62f` vs `b37cac5784a1c1d6c71db4da135fb542`)只當 artifact identity,不是 observer-effect verdict,也不是 rate.condition 1/2 不前進.localization budget 不重開.

下一步:儀器在受檢 boundary 上可信,轉**既有、歷史上已知會 diverge 的另一個 live configuration**,沿用同一 frozen fingerprint / Condition 2 contract.不再加 Block S repetitions.

---

## 10. GPU-decode localization session budget (live 前凍結)

歷史 arm G (`~/.local/state/saccade/perf/gpu-decode-nondeterminism-recheck-20260907/`):
`--preset baseline --detector SDP --sequences MOT17-02-SDP`,**沒有** `--no-gpu-decode`,n=8 中 G_r1 與 G_r2–r8 分歧.

本 session 用 retuned fingerprint,不變:

* stage ordering / canonicalization
* `CONDITION2_RULES`
* `mechanism_claim` / `issue_close`
* observer-effect acceptance wording

**Budget: 8 instrumented GPU-decode runs** (`GPU_DECODE_LOCALIZATION_BUDGET_RUNS`).
這是這組 configuration 自己的終止條件,不是 rate sample,也**不是** Block S 的 16.
歷史 arm G 的 n=8 已產生過 pair;本 budget 取同一上限,中途出現 pair 即停.

凍結收斂句 (`GPU_DECODE_BUDGET_EXHAUSTED_CLAIM`):

> divergence was not observed under the instrumented GPU-decode executions within the preregistered localization budget

```bash
uv run python scripts/tools/check_eval_repeat_identity.py run -n 8 \
  --stage-fingerprint --localization-config gpu_decode
```

pair → 只報 `last_identical_stage → first_divergent_stage`,原封不動套 `CONDITION2_RULES`.
表上 `sufficient` 是 condition 2 evidence,不是 issue closure.
budget 內全同 → 上句;不表示 divergence 消失.

### Live session 2026-09-08T003732Z

Command:

```bash
uv run python scripts/tools/check_eval_repeat_identity.py run -n 8 \
  --stage-fingerprint --localization-config gpu_decode
```

Artifact:
`~/.local/state/saccade/perf/gpu-decode-stage-fingerprint-20260908T003732Z-budget8/`

**Session kind: `pair_found`.** n=6/8, MOT pair at r6, early stop. fingerprints `complete=true` (6/6, 3596 records). eval exits 6 × 0.

| 收尾項 | 結果 |
|---|---|
| `last_identical_stage → first_divergent_stage` | **`mot` → `mot_file`** |
| sequence / frame | MOT17-02-SDP / 228 |
| `CONDITION2_RULES` | `producing_path_verdict=sufficient` |
| allowed_claim | divergence 在 sequence-level postprocess 路徑產生 |
| `mechanism_claim` | false |
| `issue_close` | false |
| condition 1 | 未前進 |
| condition 2 | **已有凍結表 `sufficient` evidence**;owner 已接受並 close #363 |

> Condition 1 remains unresolved. Condition 2 has produced a frozen-table `sufficient` producing-path result and awaits issue-level closure review; the localization run itself does not close #363.

Reference MOT md5 `b7bc17c2f1f3ba4cffaf441dece37ef0` 與歷史 arm G matching group (`G_r2`–`G_r8`) 相同,只當 artifact identity.不把 first stage 讀成 interpolation / merge 機制.不再細切 `mot`→`mot_file`,也不再做 localization.

---

## 11. #363 closure review

Owner-level 判斷:**已具備關閉條件,不需要再追 Condition 1.** Issue closed.本節是關閉前的三項審查紀錄,不是再跑 localization,也不是改凍結表.

Closure contract 原文不變:(**1 可重現機制 OR 2 具體產生路徑**) **AND** 3 fail-closed harness.
本節是 issue-level review,不是再跑 localization,也不是改凍結表.

| 問題 | 結論 | 依據 |
|---|---|---|
| 1. GPU-decode 是否在 #363 原始 evidence/scope 內? | **Yes.** 不是另一個問題,也不是 scope mismatch | Issue 正文:「GPU decode 路徑上同樣觀察到分歧。」證據目錄含 `gpu-decode-nondeterminism-recheck-20260907/`(arm G).組態是既有 live config(`--preset baseline --detector SDP --sequences MOT17-02-SDP`,無 `--no-gpu-decode`),不是新發明的刺激 |
| 2. `mot`→`mot_file` 是否構成原文 condition 2「明確指出是哪一段執行路徑**產生**的」? | **Yes, per frozen `CONDITION2_RULES`.** 這不是把 MOT 第一行差異當成 producer | 原文要產生路徑,且明講「只證明 serialization 時 box/score 已經不同,還不夠」.本 pair 的 `mot`(per-frame emit)相同,`mot_file`(sequence-level postprocess)不同.凍結表預先把 `mot_file` 列為 `sufficient` / 「divergence 在 sequence-level postprocess 路徑產生」.看到 live 後不得把 `sufficient` 降級.若拒絕,只能主張 **scope mismatch**(Q1=no),不能改表 |
| 3. fail-closed harness 是否已達 condition 3? | **Yes.** #367 已交付 | `check_eval_repeat_identity.py`:相異 MOT / 空檔 / 缺序列 / eval 非 0 都 exit 1.原文要求「症狀再出現時檢查失敗,而不是靜默通過」;不進 pre-push 是 issue 已接受的現況(`baseline` 已知會分歧) |

三項皆 yes ⇒ 依 (1 OR 2) AND 3,**#363 已具備關閉條件**.不需要為 Condition 1 繼續挖,也不用再切 `mot`→`mot_file`.

Comparator / localization run **不**關 issue(`issue_close=false`).Owner 已接受這份 evidence 並 close.Closing comment 指向本節與 https://github.com/raylei50653/saccade/issues/363#issuecomment-5577344845 .
