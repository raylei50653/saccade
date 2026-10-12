<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 S2-2 N03：離線 admission 提案（P-N03 rev 3d）

**Status: ACCEPTED（owner，session 內，2026-10-12）。O-N03-1／2／3 均接受；授權 source
implementation、local evidence、push 與 Draft PR。merge 未授權。**

把 N03 的 ELF／`$ORIGIN` 檢查移到 admission，runtime 改為只接受編入程式的唯一 N03
digest。保留 sealed-memfd 載入與固定來源檢查。不批准 package re-pin 或公開發行。

盤點基準：`feat/549-s2-2-n03-sealed-load@1d0ffa1ee809777642215b3443e92a40fa046380`。
[Phase 2 實作與證據](model_bundle_s2_2_n03_phase2_549.md) 依本稿修訂。原設計在
[PR #583](https://github.com/raylei50653/saccade/pull/583) 的 `docs/549-s2-2-n03-proposal`
分支，路徑為 `docs/reference/model_bundle_s2_2_n03_549.md`。

## 1. 問題與取捨

r2–r4 持續發現的 ELF findings 涉及檢查器與 loader 的不同視圖：dynamic table 的
file／virtual 位置、後續 `PT_LOAD` 覆寫，以及零大小 segment 的映射。
它們是同一類設計缺口的不同變體。保留 old-accept／new-reject controls 有助回歸，
但正常 fixture 或 3,798 個本機 shared objects 的 survey 不能證明惡意 ELF closure。

P1「loader input 等於已驗證 bytes」與 D-N03-5「N03 屬 runtime package 的固定來源」
仍有必要。可縮減的是在每次 runtime 載入時，對 caller 選定的 ELF 重新解析。
若 runtime 的接受集合只有一份經 admission 的 N03，這項屬性可以綁在該 digest 上。

**目前還沒有這個可信 singleton pin。** `shipping/src/detector_plan.cpp` 的
`plan_detector` 從 caller attestation 取得 operator SHA；`tests/native/test_shipping_preflight.cpp`
的 `pass_attestation_operator_override` 明確允許替換 operator bytes 並同步修改
attestation SHA。這是既有 checksum 自洽契約，不是來源認證失效。
[model-bundle 契約 §4.2](../architecture/model_bundle_contract_549.md#42---lineageattestation-可以證明什麼)
也明說 attestation 本身不構成可信來源。因此，只刪 runtime parser 不足以成立此方案。

## 2. 可信 digest 與更新規則

新增 source-controlled singleton record `shipping/n03_operator_pin.json`，由 runtime source
的審查／建置 authority 維護；configure 時產生 private C++ constant。它只有一筆 N03 pin，
不引入通用 allowlist、簽章、動態核准或撤銷服務。runtime 不讀 installed JSON、caller
attestation 或環境變數來決定 approved digest。

**為何另立一份紀錄（A2）。** tracked
[realization attestation](../../configs/shipping/mamba_head_realization.attestation.json)
記錄的是「這份 build 重現 lineage」的事實，也會被安裝進 model root、在 runtime 以 caller
輸入身分讀入；pin 記錄的是「這份 digest 被接受為 runtime 可載入的 N03」。兩者角色不同，
沿用 `shipping/entrypoint_pin.json` 的先例分開。為避免兩份漂移：configure 與 install
都要求 pin 的 SHA／bytes 等於 tracked attestation 的 `op_library`，不一致即失敗。

接受的值：

| 項目 | 值 |
|:--|:--|
| SHA-256 | `aa84cccd5c5094b33d63217eeb377452ae19c69cd1382ec7de8ba96d0d8d31c4` |
| bytes | `3517632` |
| ELF admission policy | `n03-no-origin-v1`，第 3 節列出的保守 layout 與 strings 規則 |
| runtime 來源 | `<prefix>/share/saccade/build/libsaccade_scan_torchop.so` |
| 適用範圍 | manifest／legacy、shipping／measurement／developer probe 的共用 N03 loader |

2026-10-12 對本機 `build/libsaccade_scan_torchop.so` 做同一 buffer 的 size／SHA 觀察，
與上表及 tracked attestation 一致。

建置工具只核對 source pin；不得從任意現存 `.so` 自動取得 hash 並當作批准值。新的 N03
digest 必須重新離線 admission、審查接受、更新 source pin 並重建 runtime；部署該新 runtime
仍需其 package re-pin。這次不改 N03 bytes。信任邊界包含已審 source 和 runtime binary；
替換 runtime executable、kernel／root 或已在 process 內執行的惡意程式仍沿用原設計的
out-of-scope。

## 3. 離線 admission：移用現有 checker，不重寫 parser

把目前 `operator_loader.cpp` 的 `require_no_origin_search_path` 及其 ELF fixture controls
移入 CPU-only 的 admission library 與工具 `saccade_n03_admission`；shipping runtime 不再
編譯、連結或呼叫該 parser。

**parser 的角色（A3）。** runtime 只接受審查過的 digest，所以 parser 對惡意 ELF 的誤判
接受不再是 runtime 安全性質；它只決定 reviewer 是否看到一份通過紀錄。parser 的完備性
不再是 review blocker。保守規則與既有 controls 保留作回歸。

admission 工具一次完整讀取到 owned buffer，對同一 buffer 計算 size／SHA，要求等於 source
pin，再檢查以下保守規則：

- 64-bit little-endian ELF；program-header 範圍與 arithmetic 安全。
- 每個 `PT_LOAD` 都驗證：非零 `p_memsz`、`p_filesz <= p_memsz`、4096-byte page
  congruence、ascending／page-disjoint，並檢查 addition／rounding overflow。
- 唯一 `PT_DYNAMIC`；以 `p_vaddr` 對應 file-backed segment，要求 stated offset 一致
  且在界內找到 `DT_NULL`。
- 依既有 last-tag 規則選 `DT_STRTAB`／`DT_STRSZ`；string table 完整落在單一
  file-backed segment，使用有界 string offsets／terminators。
- `DT_RPATH`、`DT_RUNPATH`、`DT_NEEDED`、`DT_AUXILIARY`、`DT_FILTER` 的字串均拒絕
  literal `$ORIGIN` 或 `${ORIGIN}`。不宣稱與所有 glibc 接受行為等價。

範圍是既有 x86-64／4096-byte page layout 與 N03 deployment，不擴展成完整 ELF validator
或跨 loader／ABI 的相容性認證。新的 target loader／ABI 需重新檢視 admission 適用性。

本機精確 N03 bytes 的兩種 ORIGIN literal 都是零次；whole-file byte scan 只作附加觀察，
不以 `grep`／`readelf -d` 單獨取代 loader-view／layout 檢查。

**admission 不等於依賴可信。** N03 的 `DT_RUNPATH` 是指向 build tree 與 `.venv` 的絕對路徑，
是 [§16](native_runtime_resolved_config.md) 列舉的 RUNPATH 例外；pin 與 admission 只涵蓋
N03 本身的 bytes，不涵蓋其 `DT_NEEDED` 依賴如何解析。

**檢查時點（A2）。**

| 時點 | 做什麼 |
|:--|:--|
| 接受新 digest | 執行 `saccade_n03_admission`；紀錄保存 artifact digest／size、pin digest、policy、checker source digest、target loader／ABI 與結果。紀錄是證據，不能自行核准另一份 bytes。 |
| CI／CPU ctest | parser 的 fixture controls；artifact 存在時，對真實 N03 執行 admission 工具 |
| configure | 由 pin 產生 constant；pin 與 tracked attestation 的 `op_library` 不一致即失敗（`shipping/cmake/n03_pin.cmake`，configure 與 install 共用；shipping CMake 最低版本因 `string(JSON)` 升為 3.19）；pin 變更觸發重新 configure／重建 |
| install／package | 只核對 digest：既有 [install-time copy check](../../shipping/cmake/install_model_root.cmake) 對照本次 build 編入的 pin（source pin 在 build 後改過、或要安裝的本次 build `saccade_track` 不帶該 digest，就要求重建），且 install 所用的 attestation snapshot 必須等於它；`scripts/native/check_shipping_package.py` 的 `check_n03_pin` 核對 packaged operator 等於該 commit 的 pin；該 commit 的 shipping CMake 已編入 pin 時，pin 讀不到即拒絕（`read_n03_pin`） |

同一份 bytes 的 parser 結果不會改變，因此 install／package 不重跑 parser，也不讀 caller
提供的 `PASS` 紀錄。pin 核對不放在 saccade_track 的 compile 步驟：重編 op library 只會在
install／package 被拒，不阻擋開發 build。封裝的 memfd evidence reader 適配仍屬既有
package re-pin 工作。

## 4. Runtime 的精確變更

共用 `VerifiedOperator::prepare` 在 registry reuse 或 `dlopen` 前要求 caller 的 expected SHA
等於 compiled N03 pin、已知的 expected size 等於 pin bytes；初次讀取時還要求 source 的
size 與 buffer SHA 等於 pin。未知 digest 的 caller pairing 即使自洽，也在 constructors
前拒絕。reuse 仍只回傳先前完成驗證的同 digest／同 source handle。

**pin 的連結方式（A1）。** loader object 只有一份。pin 以一個函式
`n03_operator_pin()` 提供，定義在獨立的 object：

- production：由 `shipping/n03_operator_pin.json` 在 configure 時生成，經
  `saccade_shipping_detector`／`_measurement` 進入所有 runtime、probe 與 GPU 測試；
- CPU fixtures：測試 executable 直接連結 fixture pin object，在 test main 設定 fixture
  digest；
- 沒有 pin 的 executable 若用到 loader 會 link 失敗（fail closed）；production pin object
  沒有 setter、CLI 或 env 入口。

另加兩項檢查：`saccade_shipping_n03_pin_test` 斷言生成的 production pin 等於 source pin；
`saccade_track` 的 POST_BUILD 檢查其 binary 內含 pin digest。

保留以下已接受的機制：

- D-N03-5 的 package-root 推導、固定來源、拒絕 symlink／magic-link，以及 auditor 的
  mapped-object/source 檢查。
- open-once、完整 read／write、`F_SEAL_WRITE | GROW | SHRINK | SEAL`、seal readback、
  sealed buffer 的 size／SHA／bytes readback。
- returned handle 的 memfd mapping identity、FD lifetime、process-wide registry、
  post-load failure poisoning 與 no-pathname-fallback。

runtime 不解析 ELF 或提供特定 ORIGIN 診斷；未知 bytes 會得到 N03 pin refusal。Gate B host
refusal 仍為 catchable `DetectorLoadError`（exit 2／failed／null byte scope）；auditor
refusal／SIGKILL 的 uncatchable 語義沿用原契約。Gate A 可以先記錄 `checksum_matched`，
但新的 Gate B 拒絕不得改寫 Gate A identity。N03 pin 不自動提升整個 model bundle 的
`identity.level` 或 publisher authentication。per-object N03 `loaded_buffer` 與 aggregate
`hashed_before_load` 的界線沿用既有決策。

**接受集合的實際變更：**

- caller 換成另一份 operator 並同步改 SHA，先前可能自洽通過；修訂後在 Gate B 拒絕
  （Gate A 的 `pass_attestation_operator_override` 不變）。
- developer probe 也適用 singleton pin，不提供 CLI／env 例外。
- legacy 模式先前沒有 size 期望值（−1）；現在由 pin 提供，legacy 也要求 bytes 相符。
- 沒有 attestation 的 lineage-only plan 取 lineage 的 `cfea78…`，不等於 pin，一律在 Gate B
  拒絕。那份 build 已不存在，實際行為不變，只是明文化。
- 目前已釘死的 shipping entrypoint（`shipping/entrypoint_pin.json`，`fbab8b79…`）不含此
  pin；安裝版要等 entrypoint re-pin 後才受此保護。之後每次改 N03 都連動 entrypoint pin
  與 runtime-identity coordinate。

## 5. 替換 rev 3c 的條文

原 Phase 2 要求（proposal 分支第 406–407 行）：

> Refuse an operator whose RUNPATH or RPATH contains `$ORIGIN` (required, since a memfd load would change its expansion).

替換為：

> N03 在 admission 必須以同一份完整 bytes 驗證 loader 可見的
> RPATH／RUNPATH／NEEDED／AUXILIARY／FILTER 字串，拒絕 `$ORIGIN`、`${ORIGIN}`
> 及不能證明 file／loader 視圖一致的 layout。只有通過 admission 並經接受的精確
> digest 可編入 runtime。runtime 在 constructors 前拒絕任何未被編入的 N03 digest，
> 保留固定來源與 sealed-memfd 的 byte identity，不再執行 ELF parser。

tag 範圍由 RPATH／RUNPATH 擴到五種，是 r3 review 後實作已採用的擴大，在此明文接受。
原 acceptance matrix 的「runtime ORIGIN refusal」改成兩條驗收：admission checker 的
ORIGIN／layout 拒絕，以及 runtime 對 unknown digest 的 pre-constructor 拒絕。
P1、P2、P3、D-N03-2／3／4／5 與 D-N03-6 NOT SELECTED 沿用既有決策。

## 6. 最小驗證與執行順序

| 驗證層 | 必須成立的控制 |
|:--|:--|
| Admission | 精確 N03 通過；ORIGIN／braced ORIGIN／五種 tags 拒絕；split dynamic、overlap、zero-size、overflow、unterminated／out-of-range controls 保留；pin 不符或 parser 拒絕都不產生成功紀錄 |
| Binding／build | pin／attestation 不一致在 configure 與 install 拒絕；production pin 等於 source pin；saccade_track 內含 pin digest；build 不自動採納 observed hash |
| Runtime／CLI | approved digest 正常載入；unknown bytes 加同步 caller SHA、錯／缺 pairing 都拒絕；constructor sentinel 證明拒絕前未執行；manifest／legacy、shipping／measurement／probe 共用規則 |
| Sealed load／source | source race、seals、readback、FD reuse、mapping identity、registry／poison、foreign source 與同 bytes foreign path controls 沿用 |
| Final acceptance | 凍結後做既有 manifest／legacy、serial／double-buffer／graph／七序列、載入觀測及 negative controls；qualification 另列 |
| Coordinate／landing | 最後 fresh bounded probe、scratch candidate、獨立 coordinate review、append-only archive、attested check、完整 pre-push／CI；implementation 與 coordinate atomic co-land |

ELF fixture controls 直接驗 admission parser helper，要求對應拒絕原因；production
admission 入口另測 pin pairing。

先完成 CPU／真實 loader controls 與 code review，處理或明確接受所有會改執行碼的 findings，
再凍結 source，之後才做 GPU 與 coordinate。不能只以「沒有 major」開始出版。沿用
[development／promotion 分層與 stacked landing](runbooks/runtime_identity_republication.md#32-review-boundary-與-landing-的關係)，
保留 main 的完整 final gate，不新增全 repo gate 框架。每次 republication 的 bounded probe
必須 fresh；七序列 acceptance 是另一項驗證。

## 7. 既有證據與決策

`1d0ffa1e` 的既有 r4 chain 是 **PARTIAL**：CPU 30 × 55 checks、GPU native ctest 5/5、
CLI 517 passed／0 skipped 已記錄。七序列 replay 的四個正向 run（shipping trace／no-trace、
measurement none／serial）有 report/raw，mot_txt／graph sections EXACT，但 parity 判定為
UNRESOLVED（driver 617.42 ≠ 616.92，anchor A_L 不相同；no-trace detector NOT_RUN）。
後續 negative-control run 收到 signal 15，缺 final summary、after-capture 與 manifest／launcher
載入觀測。它不是本稿實作的 final evidence；舊 runs／失敗／reverted coordinate 紀錄都保留。

決策（owner，session 內，2026-10-12，於本稿 A1–A3 與次要修訂後）：

1. **O-N03-1 ACCEPTED**：以 admission 和 runtime pre-constructor pin refusal 替換 rev 3c 的
   runtime ELF／ORIGIN parser。
2. **O-N03-2 ACCEPTED**：第 2 節的唯一 digest／size，pin 放在 `shipping/n03_operator_pin.json`
   並與 tracked attestation 一致；unknown self-consistent operator、developer probe 的其他
   operator 都拒絕。
3. **O-N03-3 ACCEPTED**：先 CPU／review 凍結、再最終 GPU／coordinate；code 收斂前不做
   canonical republication。

接受此提案授權上述 source implementation、local evidence、push 與 Draft PR。N03 MB-41 仍
OPEN、S2-2 仍 PARTIAL、installed package 仍 UNVERIFIED，V5 qualification 仍 UNRESOLVED；
GPU probe equality 不構成 equivalence。merge、package re-pin、#547 licensing／public release、
signing、S2-3 與 model-bundle allowlist approvals 另行決策。
`distribution.status=local-only`、`owner_confirmation=null`。
