<!-- doc-status: active -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-09 -->

# #340 capture-race incidence — preregistration (2026-09-09)

本文件在**第一個 rate run 之前**凍結。凍結後只能以 append-only amendment 修訂
(§11),不得 inline 編輯。權威 seal bar 是
[experiment contract §20.8](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md),
declaration 欄位是同一份契約的 §20.2;本文件引用它們,不複述、不分叉。

結構參考
[closed/capture_race_incidence_preregistration_20260908.md](closed/capture_race_incidence_preregistration_20260908.md)
(以下稱「20260908 預註冊」)。**不得修改那份已封閉文件。** 本文件是下一輪
campaign 的獨立宣告,不是它的 amendment。

配套 seal 記錄:
[capture_race_incidence_preregistration_20260909_seal.md](capture_race_incidence_preregistration_20260909_seal.md)。

## 0. 這回答什麼、不回答什麼

研究問題只回答:

> 在凍結的 target / environment / workload 上,#340 capture-race failure
> incidence 的上界是多少?

| | 狀態 |
|---|---|
| 2026-09-08 CUDA 901 的 root cause | **不回答** |
| #379 是否「修好了」歷史 901 | **不回答** |
| mechanism attribution 的歷史反推 | **不回答** |
| performance effect | **不回答** |
| incidence | **本研究**。在下方凍結的 production target / coordinate / 兩條 path / 完整 7-sequence workload 上,量 #340 capture-race failure 的發生率上界 |

20260908 campaign 的 terminal 是 `FAILURE_OBSERVED_B`(target
`b649de68e36ad530ed883f579478ab656a238158`)。那一輪沒有留下可用的
attribution 證據。本輪 production target 已前進到包含 #374 / #375 / #379 的
`4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f`,並附上已 qualified 的 observer
overlay,使得**若**再出現第一個 900/901/906,同一事件可以做第二層
attribution。observer 不改寫 incidence 判定,也不把 CUPTI 多看到的東西送進
numerator。

本研究不因 0/100 而宣稱 #340 的 root cause 已知、不宣稱 race 已消失、不把
#379 寫成歷史 901 的修復證明。它只回答:**在這個 production target、這個
coordinate、這兩條 path、這個 workload、這個凍結的 observer overlay 上,
#340 failure 的發生率上界是多少。**

## 1. §20.2 declarations

```text
Target decision layer   none (cross-layer substrate work)
Study intent            boundary diagnostic
                        (secondary: 若出現第一個 §3 failure,為同一事件提供
                         attribution evidence;這是第二層分析,不是 incidence
                         判定的輸入)
Design objective        n/a — 非 design evaluation。本研究不挑候選、不調參數、
                        不產生 design candidate
Selection rule          n/a — 無候選集合可排序
Validity gate           見 §4:sequence execution 開始**前**的 infrastructure
                        failure 標 invalid 並補跑;開始**後**該 run 永久進
                        denominator,且若因非 §3 原因無法跑完 7 條 sequence,
                        整個 campaign 立即 EXECUTION_INVALID。
                        任一 path 的有效 run 數未達 100 即為 validity failure
Stop condition          見 §5:sufficiency = 該 path 100 個有效 run 且 0 failure;
                        early-terminate = 第一個符合 §3 定義的 failure
Output class            diagnostic result(§20.4)。不得升格為 design candidate,
                        也不得當作 performance upper-bound candidate
Mainline transition     見 §7 terminal 表。CLOSED_BOUNDED 關閉「#340 在本凍結
                        coordinate 上的發生率未知」這個 core unknown;
                        FAILURE_OBSERVED 不關閉任何東西,直接轉入該次 failure
                        的 attribution。mechanism 仍保持未解
Type κ                  quantification space: 一個 path 上的完整 7-sequence run
                                             (trial unit = run,不是 sequence、
                                              不是 frame、不是 capture)
                        comparison relation: 該 run 是否命中 §3 的 failure 述詞
                                             (布林,無 tolerance)
                        decision rule: 該 path 的 failure count == 0 ⇒ 該 path pass;
                                       >= 1 ⇒ 該 path fail(§5 立即終止)
```

## 2. 凍結的執行組態

Campaign 分成兩個 worktree,外加一套**不得進入 production target** 的
observer/control 實作:

- **target worktree** — clean、detached、釘死在下表的 production target source
  SHA。campaign 執行的每一個 `mot17.py` 都從這個 checkout 啟動。
- **control worktree** — 本文件、harness、campaign 記錄、analyzer、observer
  binary 住在這裡,可以自由前進。controller 只能 spawn target checkout 裡的
  `scripts/eval/mot17.py`,不得執行 control worktree 裡的任何 production
  程式碼。
- **observer/control implementation** — 釘在下表的 observer/control source
  commit。這是 analyzer / observer / qualification 的凍結點,不是新的
  production target。control repository 之後可以再前進(例如寫 harness);
  production target **不得**因此移動。

**⚠️ worktree 分離不等於 import 分離。** `.venv` 是 editable install。用共用
`.venv` 去跑另一個目錄的 `mot17.py`,`import saccade` 仍然解析到**那個 `.pth`
指的目錄**,pin 就只是裝飾。所以本研究要求的是 import 層的不變量:

> **target worktree 必須就是 `.venv` 的 editable install 解析到的那個目錄。**

滿足這條最省的作法是把既有的主目錄 `/home/ray/developer/ai/saccade` 本身當
target worktree(detached 到 production target source SHA;`build/` 與 `.venv/`
都被 gitignore,checkout 不會動到,attested 的 `build/h2_layer_p` 原地可用),
control worktree 另開一個。反過來擺(target 另開目錄)則必須為它建**獨立的
venv 與 build**,並先證明 environment 軸 digest 未因此移動 —— 未證明前不得開跑。

### 2.1 Production target 與 coordinate

量測對象是一個固定的 **production source SHA**,不是 campaign 當下的 `main`
HEAD,也不是 observer/control freeze commit。本文件自己 merge 就會讓 `main`
前進,harness 的 commit 也會。那些前進**不得**進入被量測的 production 程式碼。

完整值從 target SHA 上的 `docs/reference/runtime_identity.generated.json`
機械抄入(該檔在 target 與 observer/control freeze 之間未改),不得手打縮寫。

| | 值 |
|---|---|
| production target source SHA | `4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f` |
| #379 included in target | yes — 該 SHA 即 `Merge pull request #379 from raylei50653/fix/340-capture-safety`;patch `b7bdb57378d04adfbf3310b5faa8aeca41ba6e1d` 是 ancestor |
| coordinate.decision_surface | `9b7faeb0f76a43483a924ac4028361bb155373027e9fe44e4cffbd5f1a2b0369` |
| coordinate.environment | `df4c89b6aae2c555aaac108bb8fbaf835aa21f1057c03e25ea7e26d009d5b091` |
| coordinate.identity_semantics | `8fc9bd85dd651791ec552dd2d6ab0244e849d213dcee7a2b77dfb9246a7969ca` |
| coordinate.implementation | `2f69ac56e8cbfeb41300d479b2910ad6d750391b7099bb1af649956fc209d05d` |
| coordinate.runtime_inputs | `0b839df0b89141959a4ae4762c727d446a2292016832ab468ce48334fff1a3d5` |
| probe.digest | `2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5` |
| build dir | `build/h2_layer_p`(target worktree 內) |

`runtime_inputs` 即本 campaign 凍結的 assets 軸。assets 中途移動 ⇒ 與
coordinate 中途移動相同,已完成的 run 全部作廢。

### 2.2 Observer / control implementation(不是 production target)

| | 值 |
|---|---|
| observer/control source commit | `276d8d744d7050ad272f9b69cf3c60b0c31333a5` |
| analyzer source SHA256 | `6d72a6107fedc739370489b13f6699f08e06a9e2ea4655e6836e127b9fe1fb42` |
| observer source SHA256(`observer.cpp`) | `4a20104185394a565014f83fd2f6993e33aa4c95a6775ebe15e91017140c592d` |
| observer binary SHA256(`observer.so`) | `e8b83ec4866c82f4e5a6ddbb64e5b207ff65dfc031db0101078f72083da30f29` |
| control_owner source SHA256 | `a2e97433d16c9cb516f0d8c808d6afd595ad5f1e4f26533e174c714f9cb471c5` |
| control_owner binary SHA256 | `a7c9f01c3500ed1d08e6aed9b5c2bd29ffb8f1f6ff23eeb669f3e5eeb5288d0e` |
| qualification tests | `tests/unit/eval/test_capture_attribution_nested_driver.py`; `tests/unit/eval/test_capture_attribution_harness.py`; `scripts/tools/capture_attribution/qualify.py` 六案 |
| readiness artifacts | `~/.local/state/saccade/perf/capture-attribution-340-20260909-parentage/` |

Freeze commit 相對 smoke-tested worktree 的唯一差異是 pre-commit `ruff format`
對 `analyze.py` 與 nested-driver 測試的換行。AST / `ast.unparse` 與
smoke-tested bytes 相同;observer.cpp / observer.so 未改。因並非
byte-identical,freeze 後重跑 synthetic qualification 與兩條 bounded topology
smoke(仍不是 incidence / rate run),記錄在同一 readiness 目錄的
`requalify-ruff-276d8d74/`。

**每個 run 啟動前,harness 必須機械驗證下列四條,任一不符即 fail-closed 中止
campaign(§7 `EXECUTION_INVALID`),不得只記 warning、不得續跑:**

1. target worktree 的 `git rev-parse HEAD` == production target source SHA
2. target worktree 的 `git status --porcelain` 為空
3. 在**即將用來跑 workload 的那個 interpreter** 裡解析 `saccade.__file__`,
   其路徑必須落在 target worktree 之下
4. 本 run 即將載入的 `observer.so` SHA256 == 上表凍結的 observer binary SHA256

第 3 條的觀測值逐 run 記錄(§10 `saccade_import_root`)。第 4 條的觀測值逐 run
記錄(§10 `observer_sha256_observed`)。

任一 coordinate 軸在 campaign 中途移動 ⇒ 已完成的 run 全部作廢,重新開始;
不得把兩個 coordinate 的 run 併進同一個分母。把 production target 釘死正是為了
讓這件事不會**因為 control repository 自己前進**而發生。

### 2.3 Path 組態(除 decode switch 外逐字一致)

Workload argv 與 20260908 預註冊相同。一個 trial unit = 完整 7-sequence MOT17
SDP run,不是 sequence / frame / capture。

**Path A(production GPU decode).**

```text
<target>/.venv/bin/python \
  <control>/scripts/tools/capture_attribution/run.py \
    --observer <frozen observer.so> \
    --output <trace_dir> \
    -- \
  <target>/scripts/eval/mot17.py \
    --preset mamba_whole_graph_m \
    --detector SDP --double-buffer --output <run_dir>
```

**Path B(`--no-gpu-decode` / DALI 不參與 GPU decode)。**

```text
<target>/.venv/bin/python \
  <control>/scripts/tools/capture_attribution/run.py \
    --observer <frozen observer.so> \
    --output <trace_dir> \
    -- \
  <target>/scripts/eval/mot17.py \
    --preset mamba_whole_graph_m \
    --detector SDP --double-buffer --no-gpu-decode --output <run_dir>
```

`<target>` = target worktree 根目錄,cwd 亦設為它。兩條 path 除
`--no-gpu-decode` 外逐字相同。`--detector SDP` 且不給 `--sequences` ⇒ 展開為
train split 全部 7 條 SDP sequence
(`MOT17-02/04/05/09/10/11/13-SDP`,合計 5,316 frames)。**一個 run = 這 7 條全跑完。**

`run.py` 是 control 側 spawn wrapper,用來掛凍結的 observer 並把 failure-time
trace 落到 `<trace_dir>`。它**不是** production 執行來源。workload 本體仍是
target 的 `mot17.py` + target 的 `import saccade`。

環境變數:`PYTHONUNBUFFERED=1`(繼承 20260908 A1:stdout 導向 pipe 時是
block-buffered,硬崩潰會丟掉 §3 要比對的那行)。這不是 §8 意義下對
numerator 的擴張:它只改 I/O buffering。

### 2.4 Interleaving 與 N

交錯執行 `A,B,A,B,…`。不得先跑完 100 個 A 再跑 B。交錯順序本身凍結為 A 先。

`A,B` × 100 是 **effective slot**,不是 attempt(繼承 20260908 A2,理由不變:
否則一個合法 setup-invalid 會讓該 path 永遠到不了 100 個有效 run)。

- 每個 slot 由「該 path 第一個 verdict **不是** `invalid` 的 attempt」填滿。
- setup-invalid **不推進 slot**:同一個 path 原地重新 attempt,直到填滿或觸及
  連續上限。
- `seq_index` 維持 **attempt ordinal**(跨所有 attempt 單調遞增,含 invalid)。
- `slot_index` 記錄該 attempt 想填的 slot。
- 同一個 slot 上**連續** setup-invalid attempt 上限 **5 次**。超過 ⇒ terminal
  `UNRESOLVED_INVALID_STUDY`,**不是** `EXECUTION_INVALID`。

**N.** 每 path 100 個有效 run。**看到中途結果後不得加 N。** 兩條 bounded
topology smoke 的 0 capture error **不是**改 N 的理由;本文件不因那些
diagnostic smoke 縮小或放大樣本。

## 3. Primary failure 定義(事前列舉)

一個 run 命中 **#340 capture-race failure**,若且唯若該 run 的 workload
stdout/stderr(spawn wrapper 保留的那份 log,不是 `cuda.jsonl`、不是 analyzer
report)出現下列任一 CUDA capture 錯誤,或其已知 wrapper 形式:

| CUDA code | 名稱 | 已知 wrapper 形式 |
|---|---|---|
| 900 | `cudaErrorStreamCaptureUnsupported` | `operation not permitted when stream is capturing`;torch `RuntimeError: CUDA error: operation not permitted when stream is capturing` |
| 901 | `cudaErrorStreamCaptureInvalidated` | `operation failed due to a previous error during capture`;torch 同名 `RuntimeError` |
| 906 | `cudaErrorStreamCaptureImplicit` | `legacy stream depend on a capturing blocking stream`;torch `RuntimeError: CUDA error: operation would make the legacy stream depend on a capturing blocking stream`;`currentStreamCaptureStatusMayInitCtx` 的 `c10` traceback |

述詞是**機械可判定**的:對上述字串集合做比對,命中即為 failure。判定不看
exit code,也不看 CUPTI rc、不看 analyzer 的 `capture_errors`、不看
ownership gap。exit code 仍逐 run 記錄(§10),但不進述詞。

**不算 #340 failure 的東西**(即使 run 失敗):OOM、dataset/IO 錯誤、TensorRT
engine build 失敗、driver 未載入、磁碟滿、被外部訊號中止、observer/analyzer
自己失敗。這些一律不進 numerator。它們**進不進 denominator 取決於發生時點**,
規則在 §4。

不得因為本次 attribution observer 能看到更多 CUDA/CUPTI 資訊而擴張
numerator。事前沒有寫進上表的字串,即使 CUPTI 記到了,也不是 §3 failure。

## 4. 分母規則

分界點是 **sequence execution 是否已開始**。20260908 A1 已證明「第一個
sequence 的第一個 frame 已進 pipeline」沒有 frame-1 觀測,因此本文件把 A1 的
fail-closed 讀法寫成正文,不再當 amendment。

`sequence_execution_started` 的**預設值是 `true`**。只有在下列兩條**同時**
成立時才能記為 `false`:

1. 該 run 的 log 中無任何 `🎬 ` 進度行,且 run_dir 中無任何 `<seq>.txt`;**且**
2. 該 run 的 log 正面命中 **表 4-1** 列舉的 setup-phase failure signature 之一。

兩條缺一 ⇒ `sequence_execution_started = true`,走本節第二/三條。因此**一個
沒有進度行、也不符任何已列舉 setup signature 的崩潰,會讓整個 campaign
`EXECUTION_INVALID`,而不是被補跑掉。**

- **開始前的純 infrastructure failure**(表 4-1)⇒ 標 `invalid`,不進
  denominator,**補跑一個 run 遞補**。這是**唯一**可補跑的 invalid 類別。
- **開始後,run 跑完完整 7 條 sequence** ⇒ 進 denominator。命中 §3 述詞為
  `failure`,否則為 `ok`。
- **開始後,run 因非 §3 原因無法跑完 7 條 sequence**(mid-run OOM、IO 錯誤、
  observer 崩潰、被外部訊號砍掉、host 重開機……)⇒ **整個 campaign 立即
  `EXECUTION_INVALID`(§7),停止,不補跑、不續跑。** 該 run **不得**被當成
  failure-free observation 計入分母,**也不得**被單獨移出分母。

**不得增加新的 setup-failure fuzzy 類別。** observer 載入失敗、hash 不符、
spawn wrapper 寫不出 trace,都不進表 4-1;它們走 §2 第四條或 §7
`EXECUTION_INVALID`。

**表 4-1 相對 20260908 A1-1 的唯一事前修改:** context term 與 failure term
必須出現在**同一行**(大小寫不敏感),不是「同一份 log 各出現一次」。
理由已事前存在:#375 證明全 log 合取是 fail-open —— 每個 production run 都會
印 `.engine` banner,之後任何 `error` 都會被標成 `tensorrt` setup signature,
正好擴大唯一可補跑的 invalid 類別。target `4afb57c3` 已包含 #375。表的類別與
用詞集合不擴張。

**表 4-1:setup-phase failure signatures(事前列舉,同一行、大小寫不敏感子字串)**

每一列是 **(context term) ∧ (failure term)** 的合取:兩側各取一個子字串,
**都**出現在同一行才算命中。單邊詞不構成 signature。

| 類別 | context term(任一) | ∧ failure term(任一) |
|---|---|---|
| `dataset` | `seqinfo.ini`、`data_root`、`MOT17-` | `No such file`、`FileNotFoundError`、`not found`、`does not exist` |
| `weights` | `checkpoint`、`state_dict`、`.ckpt`、`.pth` | `No such file`、`FileNotFoundError`、`not found` |
| `tensorrt` | `TensorRT`、`trtexec`、`engine build`、`.engine` | `failed`、`Error`、`Exception` |
| `cuda_device` | (無 —— 下列自成 signature) | `no CUDA-capable device is detected`、`CUDA driver version is insufficient`、`CUDA unknown error`、`Found no NVIDIA driver` |
| `output_dir` | 本 run 的 `--output` 路徑字串 | `Permission denied`、`Read-only file system` |

`cuda_device` 一列的 failure term 已經自帶足夠 context,不需要合取。
表 4-1 視為 failure 定義的一部分(§11 適用)。在觀察到一個它「新涵蓋」的
failure **之後**才擴充它 ⇒ 已完成的 run 全部作廢。

**記錄義務.** 每個 `invalid` 都要記錄原因與判定時點(§10),closure report
必須列出 invalid 總數。invalid 數量本身不是 verdict 輸入,但隱藏它是 report 缺陷。

## 5. 停止條件

- **Sufficiency(該 path pass).** 該 path 累積 100 個有效 run 且 failure count == 0。
- **Early terminate(該 path fail).** 出現**第一個**符合 §3 的 failure 即宣告該
  path closure fail,**立刻停止該 path 與另一條 path 的 incidence execution**。
  不繼續跑到 100 去稀釋它。已完成的部分不得改寫成 rate。隨後只允許 §8 對
  **該次** failure-time trace 做 attribution。
- **Execution abort(整個 campaign).** §4 第三條,或 §2 的四條 pre-run 驗證任一
  不符 ⇒ 立刻停止兩條 path,terminal = `EXECUTION_INVALID`。
- **Validity failure.** 因外部原因(硬體更換、driver 升級、coordinate 移動、
  連續 5 次 setup-invalid)無法補到 100 個有效 run ⇒
  `UNRESOLVED_INVALID_STUDY`(§20.7)。

## 6. 統計與措辭

0/100 的 one-sided 95% exact binomial upper bound:

```text
1 - 0.05^(1/100) = 0.029513  →  2.95%
```

(等價於 Clopper–Pearson 的 `Beta.ppf(0.95, 1, 100) = 0.0295130`。)

**允許的措辭,逐字:**

> Path A: 0/100 observed; one-sided 95% upper bound ≈ 2.95% per 7-sequence run.

**禁止的措辭**(任一出現即為 report 缺陷):

- 「true failure rate = 0」、「發生率為零」、「不再發生」、「race 已消失」
- 「#340 已修復」、「capture race 已消除」、「#379 修好了歷史 901」
- 把兩條 path 的 200 個 run 併成一個分母去算更緊的 bound
- 把 upper bound 說成 point estimate
- 從 0/100 反推 mechanism,或宣稱 blocking capturing stream 已辨識
- 把本輪 bound 說成未掛 observer 的 production incidence,或說成 20260908
  campaign 的延續樣本

bound 的單位是 **per 7-sequence run**,不是 per frame、不是 per sequence、
不是 per capture。任何換算都要重新宣告分母。本輪 incidence 是**凍結
production target 加上凍結 observer overlay** 的 incidence。

## 7. Terminal 對照(§20.8 item 3:窮盡且各自具名 transition)

| Terminal | 條件 | Mainline transition |
|---|---|---|
| `CLOSED_BOUNDED` | A 與 B 皆 100 有效 run、皆 0 failure | **關閉 core unknown**「#340 在本凍結 coordinate / observer overlay 上的發生率未知」。#340 進 closure review。**不**關閉 mechanism,不關閉歷史 901 |
| `FAILURE_OBSERVED_A` | Path A 出現第一個 §3 failure | 無 transition。#340 維持 open。該 failure 先封存為 incidence `FAILURE_OBSERVED`,再對同一事件做 §8 attribution。**不得**改門檻、改定義、改組態後重跑,也不得因 attribution 成敗改寫該 failure |
| `FAILURE_OBSERVED_B` | Path B 出現第一個 §3 failure | 同上。額外意義:failure 不需要 GPU decode |
| `UNRESOLVED_INVALID_STUDY` | 無法補足 100 有效 run,或 coordinate 中途移動 | 無 transition。不得報成 incidence 結果,也不得報成 signal-family exhaustion |
| `EXECUTION_INVALID` | (a) harness 自身失效(log 未寫、schema 不合、campaign 記錄不完整);(b) §2 四條 pre-run 驗證任一不符;(c) §4 第三條:sequence execution 開始後因非 §3 原因未跑完 7 條 sequence | 無 transition,fail-closed。已產生的 run 不得部分採信,**不得**從中抽出「至少這幾次沒 fail」當弱結論 |

沒有第六種結果。沒有「再多描述一點然後繼續」這個出口。
attribution 不是第六個 terminal:它是 `FAILURE_OBSERVED_*` 之後對同一事件的
第二層分析,成功或失敗都不產生新的 incidence terminal。

Harness filesystem setup 失敗(run_dir / log / trace 目錄建立、log 寫入、
`runs.jsonl` append、workload 無法 spawn)一律 normalize 成
`EXECUTION_INVALID`。manifest 一旦建立成功,任何 campaign-ending 條件都必須
best-effort 回寫 `terminal` 與 `detail`。中止當下那個 attempt 也要
best-effort 寫出一筆 `runs.jsonl` 記錄。

## 8. Attribution-on-failure(本輪相對 20260908 必須寫死的差異)

20260908 §8 禁止 primary rate measurement 開 CUPTI / LD_PRELOAD observer。
本輪**不繼承那條禁令**。理由:那一輪 `FAILURE_OBSERVED_B` 只留下 log,沒有
stream flags / capture state / owner,使 attribution 無法開始。本輪把凍結的
observer overlay 寫進執行組態,使第一個 §3 failure 帶有 failure-time trace。

這不是把 observer 證據送進 incidence 判定。規則如下。

若任一 path 出現第一個 900/901/906:

1. 依 §5 立即停止 incidence execution。
2. failure 本身先按 §3 primary predicate 計為 `FAILURE_OBSERVED_*`。
   此時 incidence 判定已封存。
3. preservation 完成後,用凍結的 analyzer 分析**該 failure-time trace**
   (同一個 run 的 `<trace_dir>`,不得另開一個「再跑一次看看」的 reproduction
   來替代)。
4. attribution report 至少保留:
   * capture site
   * Python/native thread
   * current/origin stream
   * flags
   * capture state/ID
   * overlapping capture errors
   * stream lifetime/generation
   * official owner + resolution provenance
   * event join history
   * raw runtime/driver stacks
5. 若 owner 走 `nested_driver_parentage`,必須另外保留:
   * `runtime_api_identity`
   * nested driver API
   * temporal / thread / handle correlation evidence
   * caller module/symbol
6. attribution 成敗不得改寫該 run 已命中的 incidence failure。
   analyzer gap、符號剝離、CUPTI 缺欄,一律記成 named evidence gap,不得把
   該 run 改成 `ok`、`invalid`、或 `EXECUTION_INVALID`。

**incidence 判定先封存;attribution 是之後對同一事件的第二層分析。**

Observer 未掛上、binary hash 不符、或 spawn wrapper 在 sequence execution
開始前就失敗:這不是表 4-1 的 setup-invalid,而是 §2 第四條 /
`EXECUTION_INVALID`。不得為此新增 fuzzy 類別。

健康 run(未命中 §3)不要求把 analyzer report 寫進 incidence verdict;harness
仍須保留該 run 的 trace 目錄,直到 campaign terminal 落盤。不得在看到
interim 0 error 之後丟掉 trace。

## 9. Scope 之外(明列,避免事後擴張)

不量 FPS / throughput。不比較 IDF1 / HOTA / MOTA。**MOT identity 不進 verdict**。
不重跑、不重新比較 F3c candidate。不把 2026-09-08 的 901 拿來做機制反推。
不因 smoke / qualification 的 0 error 修改 N。不關閉 #340,除非本文件的
`CLOSED_BOUNDED` 真的被跑完且 closure review 另做。

production 程式碼的凍結範圍限定在 **target worktree**。observer/control
commit 與後續 harness commit 都是 control,只要它們不是 execution source。

## 10. 凍結的 per-run 記錄 schema

每個 run 一筆 JSON,append 進 campaign 的 `runs.jsonl`。欄位凍結如下。
schema 名維持 `capture_race_incidence_run_v1`;本輪欄位是加性純觀測,不改
既有欄位語義。

```json
{
  "schema": "capture_race_incidence_run_v1",
  "campaign_id": "<UTC timestamp of campaign start>",
  "seq_index": 0,
  "slot_index": 0,
  "path": "A|B",
  "run_dir": "<absolute path>",
  "trace_dir": "<absolute path>",
  "started_utc": "<ISO8601>",
  "finished_utc": "<ISO8601>",
  "argv": ["..."],
  "exit_code": 0,
  "sequence_execution_started": true,
  "sequences_completed": ["MOT17-02-SDP", "..."],
  "capture_error_hits": [],
  "progress_marker_seen": true,
  "progress_markers": 53,
  "setup_failure_signature": null,
  "verdict": "ok|failure|invalid|execution_invalid",
  "invalid_reason": null,
  "log_sha256": "<sha256 of the captured stdout+stderr>",
  "target_source_sha": "4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f",
  "target_head_observed": "<git rev-parse HEAD of the target worktree>",
  "target_worktree_clean": true,
  "saccade_import_root": "<dirname of saccade.__file__ in the spawned interpreter>",
  "coordinate_decision_surface": "9b7faeb0f76a43483a924ac4028361bb155373027e9fe44e4cffbd5f1a2b0369",
  "coordinate_environment": "df4c89b6aae2c555aaac108bb8fbaf835aa21f1057c03e25ea7e26d009d5b091",
  "coordinate_identity_semantics": "8fc9bd85dd651791ec552dd2d6ab0244e849d213dcee7a2b77dfb9246a7969ca",
  "coordinate_implementation": "2f69ac56e8cbfeb41300d479b2910ad6d750391b7099bb1af649956fc209d05d",
  "coordinate_runtime_inputs": "0b839df0b89141959a4ae4762c727d446a2292016832ab468ce48334fff1a3d5",
  "probe_digest": "2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5",
  "observer_control_commit": "276d8d744d7050ad272f9b69cf3c60b0c31333a5",
  "observer_sha256_observed": "e8b83ec4866c82f4e5a6ddbb64e5b207ff65dfc031db0101078f72083da30f29",
  "analyzer_source_sha256": "6d72a6107fedc739370489b13f6699f08e06a9e2ea4655e6836e127b9fe1fb42",
  "attribution_status": "not_applicable|preserved|analyzed|analysis_incomplete",
  "attribution_report": null
}
```

`capture_error_hits` 記錄命中的字串與其所在行號;非空 ⇒ `verdict = "failure"`。
`sequence_execution_started = false` 且非 ok ⇒ `verdict = "invalid"`(§4 第一條)。
`sequence_execution_started = true` 且未跑完 7 條 sequence 且 `capture_error_hits`
為空 ⇒ `verdict = "execution_invalid"`,campaign 停止(§4 第三條)。

`attribution_status` / `attribution_report` **不得**回寫 `verdict`。
未命中 §3 時 `attribution_status = "not_applicable"`。命中後先
`preserved`,analyzer 跑完 `analyzed` 或 `analysis_incomplete`。

原始 stdout+stderr 與 observer trace 逐 run 落到非 scratch 的 timestamped
artifact 目錄
(`~/.local/state/saccade/perf/capture-race-incidence-<campaign_id>/`),
連同 `runs.jsonl` 與一份 manifest。**raw log 與 failure-time trace 不得只留在
scratchpad。**

## 11. Amendment 規則

凍結後發現任何影響 terminal 的自由度沒被釘住 ⇒ 那是 declaration defect,
以 **append-only amendment**(本文件末尾新增 `## A1`、`## A2` …,註明日期與
當時已完成的 run 數)修補,絕不 inline 改寫上文。amendment 若改變 failure
定義、N、path 組態、observer overlay、或 attribution-on-failure 規則,已完成
的 run 全部作廢。

## 12. §20.8 自檢

| 條 | 本文件 |
|---|---|
| 1 凍結自由度 | §2 production target SHA、observer/control commit、artifact hashes、四條 pre-run 檢查、path 組態逐字、交錯與 effective slot、N、§3 字串集合、§4 分母三分與表 4-1、§5 停止點、§6 bound 公式與措辭、§8 attribution-on-failure、§10 schema |
| 2 機械可判定 | §3 是字串比對述詞;§5/§7 是計數比較;§8 不得改寫已封存的 incidence bit。無「明顯較好」類語言 |
| 3 窮盡 terminal | §7 五個 terminal 含 validity failure 與 execution-invalid,各自具名 transition。attribution 不是第六個 terminal。§4 的三種時點各自對應唯一 terminal,無未定義情形 |
| 4 scoped exhaustion | §6 禁止「rate = 0」與「race 已消失」;§0 明示不回答歷史 901 / #379 修復 / mechanism;bound 單位限定 per 7-sequence run,且明示是 observer overlay 上的 incidence |
| 5 joint headroom | n/a — 單一決策量(failure count),無多量權衡 |
| 6 blind→reveal | n/a — 本研究無 blind phase |

## 13. 執行順序

1. observer/control freeze commit(已完成:`276d8d744d7050ad272f9b69cf3c60b0c31333a5`)
2. 本文件 commit(= 凍結點)
3. 備妥 target worktree:detached 於 production target source SHA、clean、且
   滿足 §2 的 import 不變量。control worktree 另立,持有凍結的 observer.so 與
   analyzer
4. 寫 harness:交錯排程、§2 四條 pre-run 檢查、§3 述詞、§4/§8、§10 schema、
   fail-closed(log 寫不出來即中止)
5. harness ↔ 本文件的 spec equivalence check:逐條對照述詞字串集合、分母三分、
   schema 欄位、terminal 對照、attribution-on-failure 不回寫 verdict
6. 第一個 rate run
7. closure report

**到步驟 2 為止。harness 尚未寫。** 在 harness 存在且其述詞與 schema 經對照
本文件驗證之前,不得開始 run 1。不得偷跑「先看看一兩次」,不得根據 smoke 的
0 error 修改 N,不得關閉 #340。

## 14. Pre-execution gate(seal 前確認)

| 項 | 狀態 | 證據 |
|---|---|---|
| #379 已包含在 target | 通過 | `4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f` = merge of #379 |
| target SHA frozen | 通過 | §2.1 |
| production coordinate frozen | 通過 | §2.1 六軸完整值,抄自 target 的 `runtime_identity.generated.json` |
| observer/control implementation frozen | 通過 | §2.2 commit `276d8d744d7050ad272f9b69cf3c60b0c31333a5` |
| observer qualification passed | 通過 | readiness `qualification/qualification.json` 六案 `passed: true`;freeze 後重跑見 `requalify-ruff-276d8d74/` |
| production topology smoke `trace_structure_ok=true` | 通過 | readiness `smoke-gpu-decode-summary.json` |
| DALI topology smoke `trace_structure_ok=true` | 通過 | readiness `smoke-no-gpu-decode-summary.json` |
| `ownership_evidence_ok=true` | 通過 | 兩條 smoke 皆 true |
| `evidence_gaps=[]` | 通過 | 兩條 smoke 皆 `[]` |
| no unexplained participant class | 通過 | 兩條 smoke 的 `owner_classes` 皆 `resolved:*`;gpu-decode 見 nvinfer / c10 / GMC / nvjpeg,DALI 見 nvinfer / c10 / GMC / `libdali_core.so:dali::CUDAStream::Create` |
| assets frozen | 通過 | `coordinate.runtime_inputs` = `0b839df0b89141959a4ae4762c727d446a2292016832ab468ce48334fff1a3d5` |
| primary predicate frozen | 通過 | §3 |
| denominator/validity rules frozen | 通過 | §4 |
| stopping rule frozen | 通過 | §5 |
| N frozen | 通過 | 100 effective runs / path;0/100 bound = `1 - 0.05^(1/100) = 0.029513 ≈ 2.95%` |
| incidence campaign started | **否** | 停在 first rate run 之前 |
| #340 closed | **否** | 本文件不關閉 issue |
