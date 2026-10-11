<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-11 -->
<!-- doc-module: cross -->

# #549 S2-1：Model-bundle manifest mode 候選實作與驗證紀錄

本頁記錄 #549 S2-1 的候選實作（manifest mode Gate A、TR-1b allowlist、resolved bindings、Gate B `load_verification`）與逐層驗證邊界。
Source baseline 是 `main@f06c90627e2a1dfa48615e9643f1d0a2c97b5a44`。
實作授權是 [#549 S2-1 owner implementation authorization](https://github.com/raylei50653/saccade/issues/549#issuecomment-6104733886)，
設計依據是 [ADR 028 D1–D7](../decisions/028-model-bundle-runtime-separation.md#decision-549-s1) 與
[S1 model-bundle 契約 §2–§7](../architecture/model_bundle_contract_549.md)；identity 欄位與 Gate A／B 位置沿用
[CC-536-01-02](../architecture/ship_export_contracts_536.md#cc-536-01-02)。

**狀態：候選分支（一條 Draft PR，未 merge）。** merge 需要另一則釘在最終審查 head 的 owner 授權。
實作狀態的唯一寫入處仍是 [CC-536-01-02 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-n-t3)
與 ADR 028 決策段；本頁只保留候選的逐列證據，在授權 merge 且 main CI 成功之前不回寫那兩處。
qualified GPU V5 仍為 **UNRESOLVED**，installed package 仍為 **UNVERIFIED**，`distribution.status` 維持 `local-only`。

## 1. Implemented

### 1.1 介面（[track_driver.hpp](../../shipping/tools/track_driver.hpp)）

- `--model-bundle DIR` 與四個 legacy 參數（`--config`／`--lineage`／`--attestation`／`--model-root`）互斥：參數解析完、讀任何檔案與產生 run id 之前 exit 2（MB-54）。
- `--require-identity {none,checksum_matched,expected_source_verified}`，兩種模式、兩個入口預設都是 `none`。不認得的值在 run id 之前 exit 2。
- legacy CLI、既有 fail-closed 規則、serial／double buffer、run completion 與 `saccade_track_measurement` 的量測面不變；legacy 模式的 level 上限是 `checksum_matched`。

### 1.2 Gate A manifest mode（CUDA-free；[preflight.cpp](../../shipping/src/preflight.cpp)、[model_bundle.cpp](../../shipping/src/model_bundle.cpp)）

- **兩個 root，各解析一次**：bundle 目錄，以及 runtime package 的 `share/saccade/`。後者由 entrypoint 自己的位置（`/proc/self/exe` 的 `../share/saccade`）決定，不讀環境變數、不搜尋。N03 與 allowlist 在 runtime root，N01／N02／N04／N05／N06 在 bundle root（R-07，ADR 028 D3）。
- **VL0**：`saccade.model_bundle/v1` schema 逐 keyword 手寫（每個 issue 帶 instance path 與 keyword），加上 R-01..R-08。唯一刻意的差異：pattern 的 `$` 只匹配字串結尾（Python `re.search` 也接受結尾換行前），runtime 較嚴。
- **VL1**：每個 member 以 `openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS|RESOLVE_NO_MAGICLINKS)` 開檔並要求一般檔案；kernel 不支援 `openat2` 時 exit 2，沒有較弱的 fallback。size 在 hash 之前比對；sha256 取自實際讀到的 bytes（config、lineage、attestation 的 bytes 同時交給既有 parser）。attestation 必填（D6）。lineage／attestation 綁定的三個載入檔（path 與 sha256）必須等於 manifest pairing 指向的 member。
- **VL2（TR-1b）**：`share/saccade/trusted_model_bundles.json` 的 sha256 必須等於 entrypoint 建置時編入的值（`SACCADE_TRUSTED_MODEL_BUNDLES_SHA256`，[CMakeLists.txt](../../shipping/CMakeLists.txt)），否則不論 policy 一律 exit 2（MB-32）。之後驗 allowlist schema、R-09 與 detector contract。只有 manifest 精確 sha256 對應 `state: approved` 的 entry 才把 level 提升為 `expected_source_verified`（`expected_source: "runtime_allowlist"`）；`example`、`revoked` 或未列入一律停在 `checksum_matched`。
- **Production allowlist 是空的**（[trusted_model_bundles.json](../../shipping/trusted_model_bundles.json)）：本切片沒有批准任何 N01–N06 manifest。approved entry 只出現在測試 fixture（測試自己的 pin），不隨產品出貨。
- **Policy**：Gate A 封存 identity 後才比對 `--require-identity`；不足時 exit 2（不呼叫 CUDA），identity 保留已證明的 level 並記錄 `required`。

### 1.3 Resolved bindings 與 Gate B（[detector_plan.hpp](../../shipping/include/saccade_shipping/detector_plan.hpp)、[detector_host.cpp](../../shipping/src/detector_host.cpp)）

- Gate A 為三個載入檔建立一組不可變的 `ResolvedLoadBindings`（每個 `{role, root_kind, absolute_path, bytes, sha256}`），經既有 `DetectorPlan` handoff 交給 Gate B。plan 帶 resolved bindings 時，`DetectorHost` 只用這些路徑；同時給了 model root 就拒絕，不重新解析 manifest、model root 或 caller 參數。
- 載入方式（`dlopen`／`jit::load`／TRT）、載入前的 rehash 與 loader auditor 都不變。載入期間的任何失敗都包成 `DetectorLoadError`，訊息保留原文。
- `RunCompletion` 仍是唯一的 journal writer。Gate B 在 Gate A 通過後寫一次 `load_verification`：runtime 建好（三個檔案載入、檢查都成功）寫 `{status: verified, byte_scope: hashed_before_load}`；`DetectorLoadError` 寫 `{status: failed, byte_scope: null}`；其他錯誤、SIGKILL、abort 或 auditor `_exit(127)` 讓它維持 `null`（journal 可能停在 `running`）。沒有任何 API 能寫入 `loaded_buffer`。Gate B 不改 identity。

### 1.4 Journal v3／report v5 與 reader

- `saccade.native_track_journal/v3`：identity 多出 `mode`、`required`、`allowlist_sha256`、`allowlist_entry`（`null|absent|example|revoked|approved`）、`bundle_manifest_sha256`；manifest mode 多一個 `bundle_manifest` binding（共七個）；binding status 多了 `size_mismatch`、`unsafe_path`；另有頂層 `load_verification`。
- `saccade.native_track_report/v5`：同一份 identity 與 `load_verification`，加上 `mode`、`model_bundle`，以及 plan 的 `resolved`（legacy mode 為 `null`）。manifest mode 的 `config`／`lineage`／`attestation`／`model_root` 為 `null`。
- [native_track_parity reader](../../scripts/eval/diagnostics/native_track_parity.py) 接受 v3／v5 的 legacy-mode 紀錄（parity harness 只跑 legacy CLI；manifest-mode identity 判為問題）。complete run 必須是 `verified／hashed_before_load`。historical v2／v4（S2-1a）與 v1／v3 照原樣讀；帶有 S2-1 欄位的歷史格式判為問題，不升級、不重新解讀。

### 1.5 Package 定義

install 把 allowlist 放在 `share/saccade/trusted_model_bundles.json`，[check_shipping_bundle.py](../../scripts/native/check_shipping_bundle.py) 的 exact layout 跟著加入這個檔案（`RUNTIME_DATA_FILES`，不是 model-root role）。installed package **沒有** re-pin：`entrypoint_pin.json` 仍指向 `e0eab2f7` 建置的 entrypoint，它不認得 `--model-bundle`。

## 2. Verification matrix 對應

MB 列到實際測試的對應在 [model_bundle_s2_1_coverage_549.json](model_bundle_s2_1_coverage_549.json)，由
[contract test](../../tests/contract/test_model_bundle_contract_549.py) 核對：S2-1 擁有的 14 列（MB-30…MB-33、MB-35…MB-40、MB-54、MB-55、MB-57、MB-58）都列出，每個測試檔與 case 都存在。
schema／semantic 的 25 列由 [C++ reader ctest](../../tests/native/test_shipping_model_bundle.cpp) 逐列讀 matrix，在 runtime reader 上重跑，必須在同一個 instance path＋keyword 或同一條 rule 拒絕。

## 3. Verified／unverified：逐層證據

以下每一列獨立驗收。pending 不是 PASS，也不能由另一列的成功代替。

| 層 | 本候選的實際結果 | 狀態 | 證據（local-only，gitignored） |
|:--|:--|:--|:--|
| CPU ctest（CI `shipping-config-loader` 同一組） | standalone `-DCMAKE_COMPILE_WARNING_AS_ERROR=ON` CTest 10/10：reader 101 checks（matrix 的 23 列 schema／semantic 全數拒絕於正確 path＋keyword 或 rule）、manifest Gate A 39 cases／5310 checks、legacy Gate A 4024 checks（含 MB-57）、completion 314 checks（含 load_verification、SIGKILL during load）；0 failures | local verified；CI 以 PR checks 為準 | `results/549_s2_1/`（本機 build log） |
| 真實 frozen N01–N06（manifest Gate A） | example manifest 原樣、真實 members：`checksum_matched`，`allowlist_entry=example`（example 不是批准） | local verified | `test_shipping_preflight_manifest --frozen-model-root` |
| 真 binary CLI | 4 支 suite 58 passed／0 skipped（manifest 新增 29）：MB-54 在 run id 前拒絕；manifest 拒絕控制皆無 `cuInit`；stand-in bundle Gate B dlopen 失敗寫 `load_verification=failed`；真 bundle 兩 root 完整執行 `verified/hashed_before_load`；SIGKILL during load 留 `running`＋`null` | local verified（CI 不 build binary，這些測試在 CI 會 skip） | `tests/unit/test_saccade_track_*_cli.py` |
| Native GPU ctest | `build/shipping` CTest 17/17（serial、double buffer、detector S2、post-detector、ingest、native build 無退化）；新 Gate B ctest 16 checks：cross-root load 映射的是 runtime root 的 operator 副本、MB-55 置換在 dlopen 前拒絕、帶 model root 與 bindings 不符皆拒絕 | local verified | `test_shipping_model_bundle_gate_b.cpp` |
| GPU 七序列（V5） | 新 release binary（`saccade_track` `e36b133f…`、measurement `a58096e9…`）4 個正向模式與 7 個負控制皆實際重播 7 序列／5316 frames。qualified parity **UNRESOLVED**（V5 driver `617.42 != 616.92`），負控制標準 `caught=false`。raw：正向 detector／MOT／graph 比較 EXACT（無 trace 為 NOT_RUN），7 個負控制在預期 section 為 DIFFERS；正向 journal v3／report v5 為 `checksum_matched`＋`verified` | **UNRESOLVED**（qualification）；raw 為 local diagnostic | `results/549_s2_1/gpu_0d78e59720ca/`，SHA256SUMS 的 SHA256 `3399a14a5794817b9e97e902a87d63bc77d772a58b21f8df02a536a1a8c529b5`，source manifest `157a54c5…` before／after 相同 |
| Manifest mode 七序列 | 同一 release `saccade_track` 以 `--model-bundle`（operator 在 runtime root，其他在 bundle root）跑 7 序列／5316 frames：complete、`verified`，MOT 文字逐序列與 legacy shipping run 相同；`--require-identity expected_source_verified` 在 `checksum_matched` 被拒 | local diagnostic（不是 parity 宣稱） | 同上 `manifest_mode/` |
| Surface／link | shipping measurement surface 與 link surface 通過；measurement binary 如預期被拒 | local verified | 同上 |
| 本機 pre_push | PASS（exit 0）：pytest 5636 passed、52 skipped、105 deselected、5 xfailed；S2-1 相關 suites 另跑 464 passed／0 skipped | local verified | `results/549_s2_1/pre_push.log` |
| Runtime coordinate | fresh 完整候選，只有 implementation 軸移動（294→297 檔，`0fcbdb09…`→`a1959ff2…`）；probe 重跑 digest 等於封存值（fixture change detector，`equivalence=unproven`）；attested 通過；舊 canonical 原樣封存。獨立審查面為 stacked [#575](https://github.com/raylei50653/saccade/pull/575) | stacked review 待審 | `results/549_s2_1/runtime_identity_capture.log`、`coordinate_attested.log` |
| PR CI／C++ build | 以 Draft PR 指定 head 的 checks 為準 | pending | — |
| 獨立 Codex review | read-only review 結果記在 PR；不是 GitHub APPROVED review | pending | — |
| Installed package | 未 re-pin、未驗證（包含 auditor `_exit(127)` 路徑） | **UNVERIFIED** | — |

## 4. Deferred／known limits

| 義務 | 本切片邊界 |
|:--|:--|
| S2-2 | 從已 hash 的 buffer 載入（loaded_buffer）、TOCTOU 修復、memfd／auditor、SM／TRT／LibTorch／CUDA 版本 gate、TRT I/O dtype 與 manifest `io` 的比對（MB-41、MB-43…MB-45），都未實作 |
| S2-3 | model-free runtime／private bundle 拆包、installer、launcher 預設 policy 改成 `expected_source_verified`，未實作 |
| allowlist 內容 | production allowlist 為空；任何 N01–N06 approved entry 需要 owner 另外批准，並經 reviewed PR 與重建 entrypoint |
| installed package | 未 re-pin、未驗證；auditor `_exit(127)` 的 uncatchable 路徑只在 installed launcher 下發生，本切片 **UNVERIFIED** |
| V5 | frozen `616.92` driver 的 qualified parity 仍為 **UNRESOLVED**（本機 `617.42`）；raw 輸出相等不是 EXACT qualification |
| TR-2 | runtime 驗簽章延後；`publisher_authentication` 固定 `not_checked_by_runtime` |
| #537／#541／#547 | state machine、requirement↔check 與 as-built、權利與發行各自獨立；engineering expected identity 不是 publisher authentication，也不是 #547 權利批准 |

Gate A 的 hash 與 Gate B 的開檔仍是兩次開檔（`byte_scope=hashed_before_load`）。hardlink 不偵測（契約 §5）。
runtime root 取自 entrypoint 位置：能寫 runtime prefix 的 caller 可以連 entrypoint 一起替換，這只能靠 package digest／簽章發現（ADR 028 §3 的仍成立限制）。
