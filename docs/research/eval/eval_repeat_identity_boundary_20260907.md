# eval 靜默 MOT 分歧的 runtime 邊界與 fail-closed harness (#363, 2026-09-07)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-07 -->
<!-- doc-module: cross -->

> **本文回答的問題.** 固定組態下重複執行 eval 會靜默寫出不同的 MOT 檔
> ([#363](https://github.com/raylei50653/saccade/issues/363)).
> 分歧從 MOT 的哪一欄開始? 這是不是 tracker ID 指派的第一事件?
> 既有的 decimal-hash pre-push sentinel 看不看得到它?
> 如何讓這個症狀再出現時檢查失敗,而不是 exit 0 通過?
>
> **本文不主張什麼.** 不主張根因是 #340、decode race 或 CUDA graph.
> 不把樣本分歧比例解讀為底層機率. 不判定哪一個 MOT 輸出是正確的.
> 不把 ingest / detect / NMS / GMC / Kalman 再往下拆 — MOT 檔沒有那些 stage dump.

量測本體與判讀規則仍以
[nogpudecode_reproducibility_20260907.md](nogpudecode_reproducibility_20260907.md)
為準. 本文只對那批已存 MOT 做第一個分歧幀的分類,並交付 harness.

---

## 1. Runtime 邊界

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

**第一個靜默差異發生在 tracker 寫進 MOT 的 box / score,不是 identity assignment.**
對應程式邊界是 `fast_emit_mot_lines` / `emit_tracks_unified` 寫出的
`track_results` 記錄 (`src/saccade/perception/eval/helpers.py`,
`stages.py`). 那一筆分數是 tracker score,箱子是 tracker box,所以還可以
是 detect / NMS 輸出,或 Kalman / GMC 之後的更新 — MOT 檔分不開.
S_r30 的「箱子序列化相同、只有 score 變」與「只是 GMC 把幾何 warp 了」不相容,
但這仍不是對 detect 的機制量測.

**七序列 (block H) 的第二種形狀不是獨立的第一事件.**
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
`src/saccade/perception/eval/repeat_identity.py`.

- **pass/fail = raw MOT 位元** (含 track ID). 這是 issue 寫的症狀.
- `geometry_or_score` / `identity_only` 只是 forensic 標籤,不是免責.
- 空檔、缺序列、eval 非 0,一律失敗. 不能分辨「是否相同」時不得當通過.
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

裝置無關的契約測試在 `tests/unit/eval/test_eval_repeat_identity.py`.
本機若有上述證據目錄,同一檔會再跑一對存檔正/負對照.

---

## 4. Closure 對照 (#363)

1. **可重現機制** — 未做. 本文不指認 buffer / stream / capture.
2. **runtime 邊界** — 單序列第一事件: MOT 序列化的 box/score
   (`track_results` → MOT 行),不是 ID 指派. 七序列後續的 ID-only 第一行
   是 `GlobalTrackIdMapper` 對前面序列不同出生數的下游.
3. **fail-closed harness** — `check_eval_repeat_identity.py`. 症狀出現時 exit 1.

下一步若要再往下拆 ingest / detect / NMS,需要 per-stage dump,不是更多只看 MOT 的重複跑.
