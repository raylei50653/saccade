<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-10 -->
<!-- doc-module: cross -->

# Model bundle 契約與 G01–G07 處置（#549 S1）

本稿是 [#549](https://github.com/raylei50653/saccade/issues/549) S1 的介面契約，對應 [ADR 028](../decisions/028-model-bundle-runtime-separation.md)。架構圖、信任邊界、責任分工、A／B／C 比較，以及待裁決的 D1–D7，都只放在 ADR 028，本稿只引用。

**狀態**：全文是 `proposed`，只有本段負責記錄狀態。owner 批准之前，本稿描述的介面一律不得寫成 implemented、verified 或 accepted。第 6 節「已存在的 enforcement」一欄記的是 source baseline `46ab79c1785b85f8afbef1d43909f1a7cb2eff23` 讀原始碼的現況，不是對本設計的驗證。跨 Issue 排序只在 [#550](https://github.com/raylei50653/saccade/issues/550)。

**本稿不擁有的內容**（只引用）：N01–N06 的 bytes、producer、權利未知項歸 [S0 inventory](../reference/model_runtime_inventory_549.md)；identity 輸出欄位和 Gate A／B 的位置歸 [CC-536-01-02](ship_export_contracts_536.md#cc-536-01-02)；狀態轉換語義歸 #537；requirement↔check 對應歸 #541；權利和 channel 歸 [#547](https://github.com/raylei50653/saccade/issues/547)。

## 1. 交付檔案

| 檔案 | 內容 |
|:--|:--|
| [saccade.model_bundle.v1.schema.json](model_bundle_549/saccade.model_bundle.v1.schema.json) | bundle manifest 的 JSON Schema（draft 2020-12） |
| [saccade.trusted_model_bundles.v1.schema.json](model_bundle_549/saccade.trusted_model_bundles.v1.schema.json) | release 端 expected identity（allowlist）的 schema |
| [headline_s_n01_n06.model_bundle.example.json](model_bundle_549/examples/headline_s_n01_n06.model_bundle.example.json) | 用 v1 描述現有 N01–N06。path、bytes、sha256 取自 S0 JSON，由測試核對兩者一致 |
| [trusted_model_bundles.example.json](model_bundle_549/examples/trusted_model_bundles.example.json) | 唯一一筆 entry 是 `state: example`，綁定上面 example 的 sha256，**不是 trust anchor** |
| [verification_matrix.json](model_bundle_549/verification_matrix.json) | MB-01…MB-62 正負控制。每一列有 gap、層級、預期結果、現況、既有 enforcement 的引用和切片 |
| [test_model_bundle_contract_549.py](../../tests/contract/test_model_bundle_contract_549.py) | 驗 schema 本身合法、examples 合法並符合 R-01..R-09、example 與 S0 一致、allowlist example 解析不出可信等級；矩陣中所有 schema 和 semantic 列都實際執行，而且必須在矩陣指定的位置（schema 的 instance path 與 keyword）或規則（R-xx）失敗；引用的檔案和行號必須存在 |

這個測試只檢查設計稿彼此一致，**不測 runtime**，因為 runtime 沒有讀這些檔案。

## 2. Bundle 目錄與兩種 root

一個 bundle 是一個不可變的目錄，裡面有 `model_bundle.json` 和 `carried_by: model_bundle` 的 members。`carried_by: runtime_package` 的 member（v1 只有 N03）由 installed runtime 的 `share/saccade/` 提供，manifest 只用 sha256 綁住它。這樣 bundle 就不攜帶 native code（ADR 028 D3）。每個 member 的 `path` 都是相對於自己那個 root 的路徑。

<a id="resolved-bindings"></a>**Resolved bindings（Gate A → Gate B 的唯一路徑來源）**：今天的 `DetectorHost` 只收一個 `model_root`，再對 plan 內的相對路徑各自 `resolve_model_path`（[detector_host.cpp](../../shipping/src/detector_host.cpp) 189–191 行）。拆成兩種 root 之後，這個做法不能再用。manifest 模式下：

1. Gate A 解析兩個 root（runtime 的 `share/saccade/`、bundle 目錄）各一次，用第 5 節的限制開檔、檢查 size 和 sha256，然後把每個載入檔寫成 resolved binding：`{role, root_kind, absolute_path, bytes, sha256}`，absolute_path 是 Gate A 實際 hash 過的那個路徑。
2. 這組 bindings 經由既有的 Gate A → Gate B handoff 傳下去（#565 已讓 Gate B 收 Gate A 建出的同一個 `DetectorPlan`，見 [CC-536-01-02 實作狀態](ship_export_contracts_536.md#cc-536-01-02-status)）。Gate B **只能**從 resolved bindings 取路徑，不得再從 `model_root`、manifest 或 caller 參數重新解析；manifest 模式下不存在 `model_root`。
3. Gate B 載入前照舊重新 hash（同一個 expected sha256），不符就 exit 2。

因此在 S2-2 之前，`identity` 能說的是「Gate A 與 Gate B 在載入前 hash 過的 bytes 等於 approved manifest 的值」，**不能**說「已載入的 bytes」：Gate B 的 hash 與 `dlopen`／`jit::load`／TRT 開檔之間仍是兩次開檔（TOCTOU）。這個範圍寫在 `identity.byte_scope`（4.1）。

## 3. Manifest v1

### 3.1 欄位

| 欄位群 | 內容 | runtime 如何使用（S2 之後） |
|:--|:--|:--|
| `schema` | 固定為 `saccade.model_bundle/v1` | 必須完全相同，不認得的版本 exit 2 |
| `bundle` | name、semver version、producer tool 與 commit | 只寫進 report；identity 以 manifest sha256 為準，不看 version |
| `runtime_contract` | `detector_contract`、`requires_operator`（interface 和 member） | runtime 宣告自己支援的集合，不在集合內就 exit 2 |
| `members[]` | id、role、carried_by、path、bytes、sha256、JSON member 的 `json_schema`、`inventory_id`、`rights_audit_item` | 路徑限制，加上 size 和 sha256 的檢查（第 5 節） |
| `pairing` | 六個 slot 各指向一個 member。attestation 必填（D6） | Gate A 的配對檢查（R-03） |
| `io` | backbone 和 head 的 inputs／outputs：ordinal、name、shape、dtype、device | Gate B 比對 TRT I/O 的 shape、**dtype** 和順序，以及 head 的輸出 |
| `preprocessing` | nvJPEG RGB8、`uint8_div_255`、stretch bilinear `align_corners=false`、NCHW、不做 normalization | v1 只接受現有 native 路徑的值。其他值由 schema 拒絕，不會默默改用另一條路徑 |
| `postprocessing` | sigmoid、class max、LTRB anchor decode、xyxy、num_classes | 同上 |
| `compatibility.targets[]` | platform、gpu_sm、CUDA、TRT、LibTorch、cuDNN、glibc、engine precision、`qualification`、evidence | TRT、LibTorch、CUDA 版本在 Gate A 比對，SM 在 Gate B 開頭比對。`parity_recorded` 必須附 evidence |
| `provenance` | 固定的 `reading`、inventory／#421 引用、sources（`recorded`／`unresolved`） | **runtime 不讀**。這是 producer 的 claim，只有透過 allowlist 才成為可信 identity 的一部分 |
| `rights` | 固定的 `reading`、decision owner、review status、private／public channel 的決策 | **runtime 不讀**。供 #547 審查；`approved`／`denied` 必須附 owner decision comment |
| `migration` | `supersedes`、`rollback_targets`（manifest sha256） | 給 release 和 rollback 工具用；runtime 不據此放行 |

### 3.2 Schema 無法表達的規則（R-01..R-09）

測試內有 R-01..R-09 的參考實作，S2 的 C++ 實作必須逐條對應。

- **R-01**：member id 不可重複。
- **R-02**：同一個 root 內，path 經 case-fold 之後也不可重複。這是為了不區分大小寫的檔案系統。
- **R-03**：每個 pairing slot 都要指向 role 正確的 member。
- **R-04**：v1 的每個 role 剛好出現一次。
- **R-05**：`requires_operator.member` 必須等於 `pairing.operator`。
- **R-06**：JSON role 要帶對應的 `json_schema`，binary role 不可以帶。
- **R-07**：依 D3，backbone、head、config、lineage、attestation 只能由 `model_bundle` 攜帶，operator 只能由 `runtime_package` 攜帶。D3 改變時這條跟著改。
- **R-08**：每個 tensor list 的 ordinal 是 0..n-1；head 的 inputs 與 backbone 的 outputs 逐一相同（shape **和 dtype**）；head 的 outputs 是每一層一個 cls 加一個 reg（數量是 inputs 的兩倍），cls 的 channel 等於 `postprocessing.num_classes`，reg 的 channel 等於 `postprocessing.box_channels`，兩者的 batch 與 H／W 等於對應的 head input；resize 的大小等於 backbone input 的 H／W。
- **R-09**：allowlist 中每份 manifest 只出現一次。

### 3.3 版本遷移與 rollback

- **Schema 版本**：reader 只接受完全相同的 `schema` 字串，沒有「忽略不認得的欄位」這回事（`additionalProperties: false`），新增欄位也要升版。舊 runtime 遇到 v2 一律 exit 2。
- **Bundle 版本**：可攜的 identity 是 manifest 檔的 sha256。任何 member 的 bytes 改變，或 manifest 的任何 byte 改變，都是新的 bundle，semver 只是給人看的。bundle 目錄是不可變的，安裝沿用 `install.sh` 的 digest → staging → `RENAME_NOREPLACE`，不會覆寫舊的 bundle。
- **Rollback**：新舊 bundle 目錄並存，rollback 就是把 `--model-bundle` 指回舊目錄。舊 manifest 的 allowlist entry 只要沒有 `revoked`，就仍然是 `expected_source_verified`。revoke 和 approve 一樣是 release authority 的行為：在 allowlist 加上 `revocation`（附 decision ref），再重建 runtime（TR-1b 的代價）。
- **Frozen 檔**：N05 lineage v1 和 N06 attestation v1 保持原樣，manifest 只用 sha 綁定它們。既有的 A_L 和七序列證據不重寫。
- **CLI 共存**：見 3.4。

### <a id="modes"></a>3.4 兩種模式與 policy

| | Legacy 模式 | Manifest 模式 |
|:--|:--|:--|
| 參數 | `--config --lineage [--attestation] [--model-root]`（現行） | `--model-bundle DIR`，與四個 legacy 參數**互斥**：同時出現就在讀任何檔案之前 exit 2（MB-54） |
| 期望值來源 | caller 檔案 | manifest；manifest 本身的期望值來自 allowlist |
| 最高 level | `checksum_matched` | `expected_source_verified` |
| 載入路徑 | 現行 `model_root` 解析 | 只用 resolved bindings（第 2 節） |

Policy 只有一個旗標：`--require-identity {none,checksum_matched,expected_source_verified}`，兩種模式都適用。Gate A 寫完 identity 後比對，level 低於要求就 exit 2（journal `failed`，不呼叫 CUDA），並把要求值記成 `identity.required`。沒有其他放寬 VL0、VL1、VL3 的旗標；legacy 模式下要求 `expected_source_verified` 一定失敗（MB-57）。

預設值分兩步改，避免 S2-1 改變現行行為：

1. **S2-1**：所有 entrypoint 預設 `none`。legacy 模式的行為只多寫出 `checksum_matched`，其他不變；manifest 模式是新增的。
2. **S2-3**（package 拆分，同時有 allowlist entry 可用時）：installed launcher 的預設改成 `expected_source_verified`。這是版本化的 CLI 變更，寫進 package README 與 CLI 說明。此後 legacy 模式在 installed launcher 下需要明確加 `--require-identity checksum_matched` 才能執行，結果不得引用已發表數字。legacy 模式何時移除另外決定，不在本稿。

## 4. Identity、信任根與驗證層級

### 4.1 Identity level（接 CC-536-01-02 N-T3）

level 和 binding 的格式沿用 CC-536-01-02：`null` → `checksum_matched` → `expected_source_verified`，並固定寫 `publisher_authentication: not_checked_by_runtime`。本稿補上 #536 留給 S1 的部分：

- **Bindings 加入 manifest**：manifest 模式下，`bindings` 除了 config、lineage、attestation、op library、head、engine，還要加上 `bundle_manifest`。六個 members 的 expected 值全部來自 manifest。manifest 自己的 sha256 是觀察到的值，它的 expected 值來自 allowlist。
- **`checksum_matched`**：manifest 通過 schema 和 R-01..R-09，路徑都在 root 內，每個 member 的 size 和 sha256 都相符。
- **`expected_source_verified`**：在 `checksum_matched` 之上，manifest sha256 是 allowlist 中某筆 `state: approved` 的 entry，而 allowlist 檔的 sha256 等於 entrypoint 建置時記錄的值。此時寫 `identity.expected_source = "runtime_allowlist"`，並記錄 `allowlist_sha256` 和 `bundle_manifest_sha256`。`example`、`revoked` 或沒列在 allowlist 的 manifest，最高只到 `checksum_matched`。
- **`byte_scope`**：manifest 模式固定寫出 identity 涵蓋到哪些 bytes。S2-1 是 `hashed_before_load`（Gate A 與 Gate B 載入前的 hash，見 [resolved bindings](#resolved-bindings)）；S2-2 讓 engine、head（以及依 D5 的 operator）從 hash 過的 buffer 載入之後，才改成 `loaded_buffer`。legacy 模式一律 `hashed_before_load`。
- **寫入者**：identity 只由 Gate A 寫入，之後不再改變（沿用 #536 的 State writer 規則）。Gate B 失敗只影響 run state。

### 4.2 `--lineage`／attestation 可以證明什麼

lineage 是 producer 對 checkpoint、export 工具、structural check 和 runtime flags 的**記錄**。attestation 是對 operator realization 和 A_L 重現的**記錄**。runtime 只檢查它們彼此一致，以及和 config 一致（現有的 N-R4）。它們本身不構成可信來源。只有當它們的 sha256 列在某份 approved manifest 中，它們的內容才算經過 release authority 審查。即使如此，`a_l_reproduction.identical` 這類欄位仍然只是歷史宣稱，runtime 不會重跑重現。

### 4.3 驗證層級

| 層級 | 檢查什麼 | 在哪裡 | 結果寫到哪裡 |
|:--|:--|:--|:--|
| VL0 | manifest 的 schema 和 R-01..R-09 | Gate A | 失敗就 exit 2，level 為 null |
| VL1 | 路徑限制，以及每個 member 的 size 和 sha256 | Gate A | `checksum_matched` |
| VL2 | allowlist（TR-1b） | Gate A | `expected_source_verified` |
| VL3 | 載入相容性：hash 過的 bytes 就是載入的 bytes；SM、TRT、dtype、shape、graph、operator | Gate B | run state，不改變 identity |
| VL4 | 行為資格：A_L parity、七序列 EXACT | 離線證據（S3），不在 runtime | 證據文件；決定能不能引用 benchmark |
| VL5 | 發行者認證：release 檔案的 digest 和 minisign | 安裝時，在 runtime 之外 | 使用者自行驗證；公開發行時必須簽章 |

權利（#547）不是任何一個驗證層級，runtime 一律不讀。

### 4.4 Policy

見 [3.4](#modes)。D4 的推薦是 installed launcher 最終預設要求 VL2，生效時點是 S2-3，不是 S2-1。

## 5. Fail-closed 規則

以下所有失敗都是 exit 2。Gate A 失敗時不呼叫 CUDA，journal 記 `failed`。runtime **永遠不**下載、不改找其他路徑、不改用預設模型，也不改用另一個 detector 路徑。

| 情況 | 規則 | 層級 |
|:--|:--|:--|
| `--model-bundle` 和 legacy 參數同時出現 | 讀任何檔案之前 exit 2 | 參數解析 |
| bundle 目錄或 `model_bundle.json` 不存在 | exit 2，不 fallback 到 bundled default | Gate A |
| level 低於 `--require-identity` | exit 2 | Gate A |
| manifest 不認得的 schema 或 contract | exit 2 | VL0 |
| member 缺少 | exit 2 | VL1 |
| size 不符 | 在 hash 之前就 exit 2 | VL1 |
| sha256 不符（被替換） | exit 2 | VL1 |
| 自洽的替換（manifest、lineage、attestation、artifacts 一起換） | 最高只到 `checksum_matched`；在 `--require-identity expected_source_verified` 下 exit 2 | VL2 |
| 整個 bundle 目錄搬到別處 | 允許：path 是相對的，level 不變 | VL1 |
| `..`、絕對路徑、空段或 `.` 段 | schema 拒絕 | VL0 |
| symlink（member 本身或中間的目錄） | root 只解析一次（`realpath`）並記錄；每個 member 用 `openat2(RESOLVE_BENEATH｜RESOLVE_NO_SYMLINKS)` 開檔；不支援 `openat2` 時 exit 2，不退回較弱的開檔方式 | VL1 |
| hardlink | 不嘗試偵測 runtime 檔案的 link count。完整性由「載入的 bytes 就是 hash 過的 bytes」保證；bundle archive 中的 link entry 由 package check 拒絕 | VL3／安裝 |
| Gate A 之後、Gate B 重新 hash 之前檔案被換 | Gate B 以 resolved bindings 的同一路徑、同一 expected sha256 重新 hash，不符就 exit 2 | VL3 |
| Gate B hash 之後、開檔載入之前被換（TOCTOU） | S2-2：engine 和 head 讀一次進 buffer、hash 這個 buffer、從同一個 buffer 反序列化；operator 依 D5。在那之前 `byte_scope` 標明 `hashed_before_load` | VL3 |
| TRT、LibTorch、CUDA build 不在 targets 內 | exit 2 | Gate A |
| SM 不在 targets 內 | 在反序列化之前 exit 2 | Gate B |
| I/O 的 dtype、shape 或順序不符 | exit 2 | Gate B |

## 6. G01–G07 處置

控制欄引用 [verification matrix](model_bundle_549/verification_matrix.json) 的 ID。`現況` 一欄是 baseline 的 source 事實，不是對本設計的驗證。

| Gap | 契約決策（提案） | 正控制 | 負控制 | 已存在的 enforcement | 缺口 | Owner |
|:--|:--|:--|:--|:--|:--|:--|
| G01 expected identity | TR-1b allowlist（D2）。lineage 和 attestation 只是 claim。level 依 4.1；policy 依 D4 | MB-02、MB-30 | MB-19、MB-27、MB-31、MB-32、MB-33、MB-50、MB-54、MB-57 | Gate A 依 caller 檔案比對三檔 sha256（[preflight.cpp](../../shipping/src/preflight.cpp)）；attestation 綁 lineage（[detector_plan.cpp](../../shipping/src/detector_plan.cpp)）；package 的 digest 和選用 minisign | runtime 沒有可信來源，也沒有 allowlist；`identity.level` 維持 null | 設計：#549（D2 由 owner 裁決）；實作：CC-536-01-02 N-T3，S2-1；allowlist 內容：release authority |
| G02 bundle／pairing | manifest v1、R-01..R-08；config 的 bytes 也納入 sha 綁定；frozen lineage 和 attestation 只包住、不改寫 | MB-01 | MB-13、MB-14、MB-20、MB-21、MB-22、MB-25、MB-28、MB-35、MB-36、MB-37、MB-55 | lineage 和 config 的欄位一致；attestation 綁 lineage（MB-47 enforced）；三檔被替換時拒絕（MB-34 enforced） | 沒有 manifest；config bytes 沒有綁定；size 不檢查；沒有遷移規則 | #549 S2-1 |
| G03 路徑 | 只接受相對路徑；`openat2` 禁止 symlink 和越界；root 只解析一次；載入的 bytes 就是 hash 過的 bytes | MB-40 | MB-10、MB-11、MB-12、MB-21、MB-38、MB-39、MB-41、MB-42 | package check 拒絕 link、`..` 和越界的 member；installer 要求一般檔案、不可是 symlink | `resolve_model_path` 接受絕對路徑和 `..`；`is_regular_file` 會跟隨 symlink；Gate B 以路徑重新開檔（TOCTOU） | #549 S2-1（路徑）、S2-2（TOCTOU）；D5 由 owner 裁決 |
| G04 相容性 | `detector_contract` 和 targets；Gate A 比對版本，Gate B 比對 SM 和 dtype；不設「硬跑」的旗標 | MB-01 | MB-14、MB-15、MB-18、MB-23、MB-24、MB-29、MB-43、MB-44、MB-45、MB-60、MB-61、MB-62 | TRT 反序列化；I/O 數量、順序、shape（MB-46 enforced）；head 輸出的 shape、dtype、device | 沒有查 TRT I/O dtype；沒有 SM、版本 gate；能反序列化不等於受支持 | #549 S2-2 |
| G05 model-free 拆分 | 依 D1 和 D3：runtime package 不帶 weights；bundle 有獨立的 MANIFEST、digest 和簽章；雙向綁定（allowlist 和 `requires_operator`） | MB-49、MB-56 | MB-26、MB-42、MB-48、MB-50 | `--model-root`；`install.sh` 的原子安裝語義；package check | CMake 和 package 都預期六檔在 model root；沒有 bundle 的 build 或 installer；沒有檢查 runtime tarball 是否混入模型 | #549 S2-3；channel 由 #547 決定 |
| G06 provenance／權利 | `provenance` 和 `rights` 是 release review 的輸入，runtime 不讀；allowlist 不代表權利批准；channel 決策需要 #547 的 decision ref | MB-01 | MB-16、MB-17、MB-51 | `license_audit.json` 的 public gate（[license_audit.py](../../scripts/native/license_audit.py)）；M-1 是 OPEN | bundle 沒有自己的 public gate；N04、N06、embedding 等沒有 #547 稽核項目 | #547 release owner；#421 提供血統證據 |
| G07 證據／identity | 模型座標是 manifest sha256；原 A_L 證據只在 bytes 相同且新 loader 重跑 parity 後沿用；未列入 allowlist 的 bundle 不得引用已發表數字 | MB-52 | MB-18、MB-53 | runtime identity publication、entrypoint pin、#536 package 的 EXACT 證據 | 沒有模型座標；新 loader 的 parity 尚未重跑；引用時沒有檢查 | #549 S3；檢查對應交 #541 |

**特別關注的四種情況**：

- **自洽但未授權的替換**（MB-31）：今天的 Gate A 會通過，[preflight 測試](../../tests/native/test_shipping_preflight.cpp)中的自洽替身 bundle 就是反證。目標設計下，這種替換最高只到 `checksum_matched`；在 `--require-identity expected_source_verified` 下（S2-3 起是 installed launcher 的預設）exit 2。
- **路徑逃逸與 symlink**：MB-10..12 由 schema 擋下；MB-38、MB-39 要到 S2-1 才有。
- **TOCTOU**：MB-41 要到 S2-2；operator 的處理方式依 D5。
- **版本不相容**：MB-14、MB-15 由 schema 擋下；MB-43..45 要到 S2-2。

## 7. S2 切片建議（每一片都需要另外授權）

1. **S2-1：manifest 模式 Gate A＋resolved bindings（推薦作為第一片，只需 CPU）**。加入 `--model-bundle` 和 `--require-identity`（預設 `none`）；做 VL0、VL1、VL2；用 `openat2` 限制路徑；所有 member（含 config、lineage、attestation）的 size 和 sha256；allowlist 加上建置時寫入的 sha256；N-T3 的 level、per-binding status、`byte_scope=hashed_before_load`。Gate B 唯一的改變是路徑來源：manifest 模式下只吃 Gate A 的 resolved bindings（第 2 節），`dlopen`／`jit::load`／TRT 的載入方式與載入前重新 hash 都不變。legacy 模式的行為不變，只多寫出 `checksum_matched`。負控制 MB-30..MB-40、MB-54、MB-55、MB-57 放進 ctest，用 fork 出的 process 執行，不需要 GPU；resolved bindings 讓 operator 與模型來自不同 root 時，Gate B 的實際載入至少要有一次本機 GPU smoke（不是 parity 宣稱）。allowlist 一開始是空的；真正的 N01–N06 entry 要由 owner 另外批准後才加入。entrypoint re-pin 和 coordinate republication 依 runbook 以 stacked chore 一併處理。如果 owner 想要更小的一片，可以先只做 S2-1a：legacy 模式的 N-T3 `checksum_matched` 加 per-binding status，不加 manifest 模式。這部分已在 #536 第一批設計中批准，也不碰 Gate B。
2. **S2-2：Gate B 強化（需要 GPU）**。從已 hash 的 buffer 載入（D5）、SM 和版本 gate、TRT dtype。完成後重跑 A_L parity（MB-52）。
3. **S2-3：package 拆分**。model-free runtime tarball 和 private bundle tarball 分開建置；各自的 MANIFEST、digest 和簽章；檢查 runtime tarball 不含 `carried_by: model_bundle` 的 member，而 N03 照常存在（MB-48、MB-56）；並存安裝和 rollback（MB-49）；installed launcher 的預設 policy 改成 VL2（3.4）。`distribution.status` 維持 `local-only`。
4. **S2-4（延後）**：TR-2，runtime 驗 bundle 簽章。前提是 #547 先審過新的 crypto 依賴。

## 8. 本稿沒有做的事

沒有改 runtime、CLI、CMake、package、模型 bytes、frozen evidence、runtime identity、CI required gates 或 `distribution.status`。沒有重跑 GPU 或 parity。沒有認證任何來源，也沒有做權利結論。examples 都不是 trust anchor。
