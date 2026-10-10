<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-10 -->
<!-- doc-module: cross -->

# #536：安裝後 package 的 Completion 與 Gate A

本切片把 [entrypoint pin](../../shipping/entrypoint_pin.json) 從 PR-C2 換到已含
[#562 Completion](../architecture/ship_export_contracts_536.md#cc-536-01-01-status)
與 [#565 Gate A](../architecture/ship_export_contracts_536.md#cc-536-01-02-status)
的 binary，並把失敗重跑行為寫入 [package README](../../shipping/package/README.txt)。
授權來源是本次 owner 的「#536 package re-pin」指示。只驗證 local-only package；
不更動模型、tracker numerics、launcher、auditor、installer 或第三方集合，
不完成 identity integration、#535 A3、#541 as-built 或 #547 公開發行裁決。

## 1. 固定產物與來源

Binary 的 source baseline 是乾淨的 `e0eab2f7caa3c495c74b8cf00e45788498d573cc`。
`git diff 7c24b9e3 e0eab2f7 -- shipping/src shipping/include shipping/tools src include`
為空：#565 的 runtime source 未再改動。
`build-release/` 用 [#465 §16.3](native_runtime_resolved_config.md#163-測量契約正式-run-之前寫定)
的 configure 參數重建 `saccade_track`，`SACCADE_SHIPPING_ENTRYPOINT` 為空；
build 的 link／measurement surface 檢查通過，再以 `cmake --install` 取得
RUNPATH `$ORIGIN/../lib` 的 installed binary。使用既有 build directory，不宣稱 fresh configure。

| 項目 | 固定值 |
|:--|:--|
| 保留位置 | `results/536_package_repin/entrypoint_e0eab2f7/saccade_track` |
| SHA-256 | `fbab8b79710f52a6f6c668b956c42e1650fe59e3f0d1eeeb525e914e47a07dde` |
| bytes | 8862856 |
| operator SHA-256 | `aa84cccd5c5094b33d63217eeb377452ae19c69cd1382ec7de8ba96d0d8d31c4`（既有 attestation） |

之後安裝必須以 `SACCADE_SHIPPING_ENTRYPOINT` 指向這份固定產物，不能用另一次
build 的 bytes 代替。舊 PR-C2 pin 與正式證據仍保留；重建 binary 並不可重現。

## 2. 驗證契約（正式執行前提交）

正式 run 使用乾淨的已提交工作樹，依
[coordinate runbook](runbooks/runtime_identity_republication.md) fresh probe、scratch
candidate、獨立審查及 archive 後更新出版。`--mode attested` 必須 exit 0；
probe equality 不構成行為等價主張，`equivalence` 維持 `unproven`。
全部 GPU 步驟經 `tools/resctl.py run --wait gpu0` 依序執行。

| Gate | 必須成立的正／負控制 |
|:--|:--|
| Package | builder 不帶 `--trial`，`tree_clean`／`identity_current` true；tarball 七項 PASS；同一 tree／commit 第二次打包的三個檔案 bytes 相同 |
| Install | 既有 pinned Ubuntu 24.04 容器內 `install-strace` 成功；install-trace 五項 PASS；installed `static --manifest` 十三項 PASS；`--verify` 成功；除 MANIFEST 外與 CMake tree 每個檔案 bytes 相同 |
| Oracle | fresh anchor 與 PR-2L `A_L_1` 的七個 sequence 相同；fresh serial oracle rows 完整有效；`build/` 既有 oracle extensions 不重建 |
| GPU parity | installed package 的 normal 與 `-yy` strace 容器 run 均 exit 0；`native_track_parity --native-from` 為 EXACT：detector 5316/5316、MOT 7/7、graph counts 7/7；normal run 與 PR-C4 package baseline 的 txt／trace hashes 與 graph counts 相同，traced run 與 normal run 相同 |
| Runtime boundary | `check_shipping_bundle runtime` 三項 PASS，trace 完整、無 Python 執行／open、第三方載入來源與 pin 正確 |
| Installed Completion | 完成 run 的第一行 run_id、v3 report、complete journal 與檔案 hash 相符；後段壞 JPEG 的失敗重跑沒有舊 report／本輪未提交的舊 txt，保留無關輸出；同 OUT 的真正第二次 invocation 被拒且不改檔案；真正 SIGKILL 留下 running journal，reader 拒絕當作 complete |
| Installed Gate A | CLI suite 的正控制通過 Gate A 並觀察到 `cuInit`，之後在壞 JPEG 失敗；五個拒絕控制留下 failed journal、沒有輸出、沒有 `cuInit`（以 `CUDA_INJECTION64_PATH` 與 `LD_DEBUG=files` 觀察） |
| Validity | 執行前後 pin、installed binary 與 attested operator hashes 相符；執行前後工作樹乾淨 |

CLI suites 的 `SACCADE_SHIPPING_TEST_PREFIX` 指定真正 installed launcher 與
`share/saccade` 的 config／lineage／attestation；指定錯誤 prefix 必須失敗，不得 skip。
package 只帶 shipping entrypoint，因此 package 模式不收集 measurement binary cases。
既有 CPU 協定、reader、bundle／package／signature tests 另行執行。

任一 validity 或 gate 不成立，逐項回報 FAIL／UNRESOLVED，不把部分結果當作整體 PASS。
installer／launcher／auditor 未改，歷史安裝器與 loader 負控制不重做；本切片的新增
負控制限 installed Completion／Gate A 與新 pin。沒有 FPS 或效能主張。

## 3. 重播入口與證據

本機 runner 保留在 `results/536_package_repin/run.sh`，Completion GPU controls 在
`results/536_package_repin/completion_controls.py`；正式 run 使用獨立的 `full_<commit>/`
目錄。run 前先檢查 runner，並把 SHA-256 留在 raw 證據中；runner 不在 CI 執行。
package／GPU run 尚未執行；結果與 retained manifest 將在本節回寫。
raw build logs 保留在 `results/536_package_repin/pin_{configure,build,install}.log`。
CI 的 CPU checks 與本機的 installed-package／GPU evidence 分開認定。

既有 [Completion known limits](../architecture/ship_export_contracts_536.md#cc-536-01-01-status)
與 [Gate A TOCTOU 邊界](../architecture/ship_export_contracts_536.md#cc-536-01-02-status)
不因此消失：identity level 仍為 null、報告與 trace 跨 OUT 共用不受 lock 保護、
power loss／network／Windows-mounted filesystem 未驗證、kill 留下的 temp files 不自動清理。
