# `--no-gpu-decode` 的重複執行變異 — 重現性不能由旗標單獨保證,必須逐組態驗證 (2026-09-07)

<!-- doc-status: active -->
<!-- doc-promotion: report_data -->
<!-- doc-date: 2026-09-07 -->
<!-- doc-module: cross -->

> **本文回答的問題**:過去的判讀規則說「加 `--no-gpu-decode` 即得 bit-exact,故小 delta A/B
> 用 N=1 就夠」。本文在**固定版本、固定執行條件**下量這條規則的前提:重複執行會不會分歧、
> 分歧從哪裡開始、規模多大,以及 IDF1／HOTA／IDs 等決策指標會動多少。
>
> **本文不主張什麼。** 不主張任何 tracking 改動、不把樣本分歧比例解讀為底層分歧機率的估計、
> 不指認機制(哪個 buffer 被 race 完全未量),也不判定哪一個輸出是正確的。
>
> **核心結果先講。** 同一支 flag、同一台機器、同一個 session:
> `mamba_whole_graph_m --double-buffer` 20 跑輸出完全相同;`baseline` 7-seq 20 跑裡有 14 跑
> 與參考輸出不同。**⇒ `--no-gpu-decode` 本身不足以保證重現性,必須逐 configuration 驗證。**
> (本文的 block order 與 configuration 混在一起,見 §6,故**不反過來宣稱「configuration 決定了
> 重現性」**;per-configuration 是**證據的適用範圍**,不是機制判斷。)
>
> 受本文影響的既有結論分級見 §5。

---

## 1. 方法

**預註冊在先。** 樣本數、reference output 定義、要報告的量與判讀規則,都在跑之前寫定於
`~/.local/state/saccade/perf/nogpudecode-reproducibility-20260907/preregistration.md`。
block D 以 amendment 加入,該 amendment **寫在讀取 block S/H 任何結果之前**,理由是 §5 的文件稽核
發現受影響最深的兩份結論用的是另一個組態。看到結果後未再追加任何 block。

| 項目 | 值 |
|---|---|
| 版本 | `feat/math-model-two-arm` `c9b3f72c`(runtime 與 main `0fe937b4` 相同);全程未切 branch |
| host | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| 共同旗標 | `--no-gpu-decode --detector SDP` |
| 執行方式 | 每個 block 內連續執行,期間無其他 GPU 工作 |
| 每跑的觀測 | tracker 輸出檔的 md5(多序列 block 取七個 md5 的組合) |

**reference output** = 該 block 內出現次數最多的輸出。**它只是量離散程度的參考點;出現次數多
不是正確的證據,本文不判定何者正確。** 「分歧」= 與該 block 的 reference output 不同。

指標由存下來的 MOT 輸出**全精度重算**(每個相異輸出算一次),不取 console 印出的 0.1pp 值。

---

## 2. 結果

| block | 組態 | n | 分歧 | 相異輸出 |
|---|---|---:|---:|---:|
| S | `--preset baseline`,MOT17-02-SDP | 40 | 4 | 5 |
| H | `--preset baseline`,7-seq SDP | 20 | 14 | 15 |
| D | `--preset mamba_whole_graph_m --double-buffer`,7-seq SDP | 20 | **0** | 1 |

三個 block 是三種執行條件,**逐 block 報告,不合併成單一比率**。

### 2.1 觀測範圍(observed range = max − min,全精度)

| metric | S(單序列,n=40) | H(7-seq,n=20) | D(7-seq,n=20) |
|---|---:|---:|---:|
| IDF1 | 0.4019 | 0.1298 | 0 |
| HOTA | 0.1306 | 0.1050 | 0 |
| MOTA | 0.1507 | 0.1416 | 0 |
| DetA | 0.0595 | 0.0532 | 0 |
| AssA | 0.3390 | 0.1731 | 0 |
| IDs | 2 | 2 | 0 |
| FP | 25 | 84 | 0 |
| FN | 14 | 126 | 0 |

**observed range 是這批樣本上看到的最大與最小之差,不是界。** 對**同一批累積樣本追加觀測**時,
range 不會縮小;但**獨立重跑一個新 block 不保證**得到同樣或更大的 range。

### 2.2 分歧的形狀

- block S 的四次分歧,首個相異幀分別在 frame 15／63／154／291,相異行數 5,917–10,701
  (檔案共約 10,868 行)。一旦分歧就不再收斂回參考軌跡。
- block H 的分歧散布在七條序列上;同一跑可能只有一條序列不同(如 r2 只有 MOT17-05 差 164 行),
  也可能六條都不同(r1、r9、r16)。
- block D 二十跑輸出完全相同,無可描述的分歧形狀。

### 2.3 三點描述(不含原因判斷)

1. **H 的整體輸出分歧比例較高(14/20),但聚合指標的觀測範圍較小**(IDF1 0.1298 對 S 的 0.4019)。
   七序列提供更多分歧機會、且聚合會攤平單序列的偏移,是可能的解釋;**僅憑這兩個 block 不能斷定
   「序列越多就越常分歧」,也不足以概括「單序列是較差的儀器」。**
2. **本次 S/H 觀察到的所有分歧輸出,都至少改變一個指標**(S 4/4、H 14/14 皆非全指標相同)。
   本次資料不支持「位元不同但指標不動」這個免責說法;這是對本次觀測的陳述,不是普遍命題。
3. **同旗標、同 host、同 session 下,不同組態的分歧比例差異很大**:D 0/20 對 H 14/20。
   可直接推得的是「這支旗標本身不足以保證重現性」;**推不得「configuration 決定重現性」**,
   因為 block order 與 configuration 在本設計中混在一起(§6)。

---

## 3. 判讀規則(取代「加旗標 ⇒ bit-exact ⇒ N=1」)

**observed range 是判讀的參照,不是判決門檻。**

- Δ **落在**該組態的 observed range 內 ⇒ 寫成:**「單次 A/B 尚不足以區分處置效果與已觀察到的
  run-to-run 變異」**。這不是「無效果」的證據,而是這個儀器解析不了 ⇒ 需要更多跑數,不是下判決。
- Δ **超過** range ⇒ **不自動證明處置有效**。可寫「超出本組態在 n = N 下觀察到的變異範圍」,
  不是 p 值,也不是顯著性宣稱。
- **判讀一律用全精度值**,不要先四捨五入再比較(例:H 的 MOTA range 是 `0.1416`,不是 `0.14`)。

各組態的參照值即 §2.1 該欄。**其他組態未量,不得繼承上表任何一欄。**

### 3.1 `mamba_whole_graph_m --double-buffer` 7-seq 的 N=1

20 跑未見分歧,**僅支持受測組態的有限重現性**:

- 可以保留 **N=1 作探索性比較**。
- **不因此背書微小 delta 的正式歸因**。
- **處置臂修改後不一定繼承 base 的重現性** —— base 的 0/20 不是處置臂的重現性證據。
- 零事件的量化意義:**在獨立、固定分歧機率的假設下,單側 95% 上限約為 13.9%**。
  連續同 session 執行未必滿足獨立與固定機率的假設,故此上限只是粗略參照。
- 要寫的句子是「n=20 未見分歧」,**不是「bit-exact」**。

---

## 4. 被撤回的規則

**撤回**:「`--no-gpu-decode` 保證 eval 確定性 / bit-exact,故小 delta A/B 用 N=1 即可」。

撤回的理由不只是本文量到分歧,也在於**原本的證據強度**:該規則的量測基礎是每臂三跑
(2026-08-08 兩份 benchmark 的 §0)或兩跑(2026-06-20 的定位、D0 declaration §2)。
**本文已直接觀察到同一旗標下存在 run-to-run 分歧,因此先前僅 2–3 次一致的觀測不足以支撐
該規則的普遍形式** —— 即使該規則在 config D 上恰好沒有被本文推翻。

(此處刻意不談「三跑全同有多常見」:那需要把樣本上的分歧比例當成底層機率模型,與 §1、§6
的口徑衝突。少量一致觀測不足以排除分歧,這件事不需要機率推論就成立。)

---

## 5. 受影響結論的分級

分級用語:**「本次未提供推翻證據」**,不是「不受影響」。大 delta、config D 類組態與跨 dataset 結論
**都不能直接繼承 block D 的結果**;block D 只說它自己那個組態在 20 跑下未見分歧。

| 結論 | 組態 | 依據的量 | 分級 |
|---|---|---|---|
| [bridge_gate_stability_20260808](../../reference/benchmarks/bridge_gate_stability_20260808.md) §3 的 `−0.021` / `−0.018` 判為「可重現的小負 delta」 | D 類 | 0.021 IDF1 | **本次未提供推翻證據**;其原始理由(bit-exact)已改寫 |
| 同上的 `+0.714` / `+0.200` / `+0.052` 集中度敘述 | D 類 | 0.052↑ | **本次未提供推翻證據** |
| [reid_handover_ablation_20260808](../../reference/benchmarks/reid_handover_ablation_20260808.md) §1「ReID 買到 0.0 IDF1,差異只有 1 ID / 7 FN」 | D 類 | 1 ID、7 FN | **本次未提供推翻證據**;其原始理由已改寫 |
| [bridge_gate_cross_dataset_20260808](../../reference/benchmarks/bridge_gate_cross_dataset_20260808.md) 的 pooled `−0.753` 與否決 | 跨 dataset | 0.753 | **本次未提供推翻證據**;跨 dataset 組態未量,不繼承 D |
| `scripts/eval/diagnostics/bridge_gate_breakpoints.py` 以「輸出相等」為謂詞的二分法 | D 類 | 相等 | **本次未提供推翻證據**;#364 已在掃描前加 session identity 自檢(預設 2 次 fresh eval,分歧 fail-closed)。匹配的自檢不是確定性證明,只是本 session 未觀察到矛盾 |
| [d0_runtime_shadow_fidelity_declaration_20260712](../../modules/semantic/research/d0_runtime_shadow_fidelity_declaration_20260712.md) §2 「The pipeline is deterministic under these flags」(跑兩次 byte-identical) | D 類 + 兩個額外旗標 | byte 相等 | **兩次 byte-identical 的歷史觀測本身仍成立**;由該觀測推出的「確定性」一般化**撤回**。該句位於 §2 `(frozen)`,**故不改動原文**,更正記於本表 |
| [tracker_lane_dose_response_20260907](../../reference/benchmarks/tracker_lane_dose_response_20260907.md) §2 dose inertness 的 35 跑同值 | 另一 preset,decode 走 preset 預設 | 三個指標相等 | 觀測成立;由其推出的「the pipeline is deterministic at this preset」已改寫為零事件敘述 |
| [saccade_module_reference](../../reference/saccade_module_reference.md) §5 「多次重複 stdev = 0 / byte-identical / A/B 可用 N=1」 | 未載明 | 未載明 | **未定 — delta 未載明**;措辭已改寫 |
| `docs/TODO_history.md` 等歷史紀錄中標註「(確定性)」的 ablation | D 類 | −6.2 IDF1 等 | **本次未提供推翻證據**;歷史紀錄不改寫,引用時以本文為準 |
| 任何在 `--preset baseline` 下、Δ 落在 §2.1 S／H 欄內的 A/B | S／H | 隨案 | **population 為空**(2026-09-08 盤點,見 §5.1) |

### 5.1 Revalidation inventory 收束(2026-09-08)

**#363 已依 condition 2 + condition 3 關閉**(producing path 定到 `mot`→`mot_file`;fail-closed
harness 交付;condition 1 機制未做、非必需)。**該 closure 沒有縮小 §2.1 的任何一欄,也沒有任何
runtime mechanism fix。** #367 是 observability bound,不是 runtime boundary:`--preset baseline`
的 run-to-run 分歧仍然存在,規模仍只由 §2.1 的 observed range 描述。
**不得把「#363 closed」讀成「reproducibility uncertainty 已解決」。**

在此前提下逐列盤點上表:

- **0 項進 revalidation,0 個 tracker 新開。**
- 上表最後一列(`--preset baseline` 下 Δ 落在 S／H 欄內的 A/B)**population 為空**:所有 live
  benchmark 都跑在 `mamba_whole_graph_m`([frozen_v2_ablation](../../reference/benchmarks/frozen_v2_ablation.md)
  為 `mamba_whole_graph`);文件中僅存的 `--preset baseline` 數字是
  [PIPELINE_REFERENCE](../../reference/PIPELINE_REFERENCE.md) 的 2026-05-10 P3 表,而
  [mot17_default_config](../../reference/mot17_default_config.md) 已把 `baseline` / `speed`
  標為 legacy comparison、非 production baseline。`--preset baseline` 目前唯一的活消費者是
  `scripts/tools/check_eval_repeat_identity.py` 自己的預設組態。
- 其餘各列維持上表既有分級。**「維持現況」不表示那些 delta 被證實**,只表示重新量一次不會改變
  任何現行結論:D 類與跨 dataset 各列未因 closure 取得新證據,`saccade_module_reference` 列改寫後
  已不帶 delta。

因此本輪的收束句是:

> reproducibility uncertainty remains bounded only observationally; revalidation inventory is
> exhausted because no currently decision-relevant delta falls inside the unresolved range.

**Conditional watch(不是現在的工作)。** `PIPELINE_REFERENCE` 的 P3 sweep 跑在 `--preset speed`
(yolo26s),相鄰列差 0.1–0.4 pp 卻據以選出「Pareto 最優點 `match=0.66, ntt=0.28`」。該 ranking
沒有重現性支撐,也**不得繼承** §2.1 任何一欄(不同 preset、不同版本)。它只落在 legacy preset 內,
出貨的 mamba presets 用 `match_thresh: 0.50`,故本輪不動。**若 `baseline` / `speed` 再度成為
決策面,該 ranking 必須重推。**

**同輪一併降階的三處措辭**(上表未涵蓋;均為措辭問題,非重驗項,結論不動):
[frozen_v2_ablation](../../reference/benchmarks/frozen_v2_ablation.md) 前言與 §4、
[ADR 018](../../decisions/018-project-main-line-direction.md) 的 canonical headline、
[no_go_registry_details](../../reference/no_go_registry_details.md) #49 / #50 / #51。
三者的 NO-GO / 累積表都由遠大於任何 observed range 的臂承載;被撤回的只是把單次輸出相等寫成
determinism property 的措辭。

`80.4471` / IDs `344`:block D 的參考值與 2026-08-08 兩份 benchmark 記載的出貨 base 相符,
**是組態相符的佐證**;完整的組態身分仍須以版本、輸入與執行設定逐項核對,不以單一指標值認定。

---

## 6. 限制

- 單一 host、單一 session、單一資料集(MOT17 train SDP)。range 不轉移到其他 preset、host、
  資料集,也不轉移到 GPU decode 路徑。
- n 是為了量指標離散度而定。文中的 4/40、14/20、0/20 是**樣本上的分歧比例**,
  **不解讀為底層分歧機率的估計**,也不跨 block 合成。
- 未做任何機制量測。本文不說明哪個 buffer 被 race,也不說明 block D 為何沒出現。
- block 順序(S → H → D)未隨機化,**block order 與 configuration confounded**;session 內若有漂移,
  會與組態效應混在一起,且**偏差方向無法判定**(除非已知漂移如何隨時間改變分歧,而本文未量)。
- 另一條假說(專用 decode stream 是否讓 race window 變寬)另案,未在本文測試。

---

## 7. 證據索引

`~/.local/state/saccade/perf/nogpudecode-reproducibility-20260907/`(80 跑,約 207 MB,非 repo 內):

| 檔案 | 內容 |
|---|---|
| `preregistration.md` | 預註冊 + amendment 1(block D) |
| `manifest.md` | 執行條件、結果、判讀規則、分級表 |
| `summary.log` | 每一跑的 exit code 與輸出 md5 |
| `analyze_repro.py` · `analysis_output.txt` · `ranges.json` | 分析腳本與其原始輸出 |
| `S_r*/` `H_r*/` `D_r*/` | 每跑的 stdout 與 MOT 輸出檔 |

同日的前置量測(GPU decode 側,以及 PR #344 的 A/B)另存
`~/.local/state/saccade/perf/gpu-decode-nondeterminism-recheck-20260907/` 與
`~/.local/state/saccade/perf/decode-stream-ab-20260907/`。

對這批 MOT 的第一個分歧幀分類、以及 fail-closed 重複執行檢查,見
[eval_repeat_identity_boundary_20260907.md](eval_repeat_identity_boundary_20260907.md)
(#363;不改寫本文的量測或判讀規則)。
