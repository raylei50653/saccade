<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-11 -->
<!-- doc-module: cross -->

# #549 S2-1a：Legacy Native Identity Observability

本頁記錄既有 legacy native CLI 的 N-T3 `checksum_matched` 候選實作與驗證邊界。
Source baseline 是 `15e819502e4d5ea581792a3a16f97857435fc585`。
實作授權是 [#549 owner S2-1a authorization](https://github.com/raylei50653/saccade/issues/549#issuecomment-6098363082)，
nullable binding／expected source／status 與 journal v2／report v4 的釐清由
[#549 owner clarification acceptance](https://github.com/raylei50653/saccade/issues/549#issuecomment-6098552167)
接受，版本控制的裁決在 [CC-536-01-02](../architecture/ship_export_contracts_536.md#decision-536-s2-1a-legacy-binding)。
設計依據是 [ADR 028](../decisions/028-model-bundle-runtime-separation.md)。

**狀態：已由 [#572](https://github.com/raylei50653/saccade/pull/572) merged（merge `56540bf93a8988b546535f480e5123d8f8e97c52`，source-level）。**
runtime coordinate 的獨立兩檔 review surface 是 [#573](https://github.com/raylei50653/saccade/pull/573)（#572 祖先，未單獨 merge）。
實作狀態與 PR／main CI 結果的唯一寫入處是 [CC-536-01-02 N-T3 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-n-t3)；
本頁只保留下方逐列證據。GPU 的 frozen V5 qualification 仍為 unresolved，package 未驗證，不代表 CC-536-01-02 整體完成。

## 1. Implemented：候選分支的 legacy 觀測

[Gate A](../../shipping/src/preflight.cpp) 沿用 CUDA-free preflight 與既有拒絕規則，
用 Gate A 實際讀取的 config／lineage／attestation bytes 計算觀測 SHA，並把同一份
metadata bytes 交給既有 parser。metadata schema／欄位一致性、所有 sequence 輸入、
輸出位置及三個載入檔的檢查全部完成後，才一次提升至 `checksum_matched`。
任一檢查失敗維持 `level=null`，保留已取得的 binding 診斷與實際 failure message。

`identity.bindings` 固定只有 `config`、`lineage`、`attestation`、`op_library`、
`head`、`engine` 六項，每項包含 `{path, expected_sha256, observed_sha256, status,
expected_source}`。per-binding `expected_source` 為 nullable `{path, json_pointer}`，
只是 caller metadata 中 expected hash 的實際位置，不是獨立 trust anchor。

| Binding | 實際 expected SHA 來源 |
|:--|:--|
| config | 無；expected hash／source 均為 null。config 的 preset SHA 不是 config bytes 的 SHA |
| lineage | 有提供且可讀取 attestation 時，取其 `/frozen_lineage/sha256`；沒有可用 expected 欄位時為 null |
| attestation | 無；expected hash／source 均為 null |
| op_library | 通過既有 attestation 一致性檢查後取其 `/op_library/sha256`；否則取 lineage `/op_library/sha256` |
| head | lineage `/torchscript/sha256` |
| engine | lineage `/companions/backbone_engine/sha256` |

SHA status 沿用 `matched`／`mismatch`／`missing`／`unchecked`。只有非 null 的
expected 與 observed 已比對且相等才能記 `matched`；已比對且不等為 `mismatch`。
指定的必需檔案缺失或不符合既有一般檔案要求為 `missing`；沒有 expected、無法讀取
或尚未檢查為 `unchecked`。選用 attestation 未提供記 null path／hashes／source 與
`unchecked`，不拒絕；caller 有指定但檔案缺失則拒絕。metadata 語意失敗仍使整輪
Gate A 失敗，個別已 `matched` 的 SHA 不表示 schema／欄位一致性已通過。

[RunCompletion](../../shipping/include/saccade_shipping/run_completion.hpp) 維持唯一
journal writer。Gate A 是 identity 的唯一更新者；初始化 `level=null`，完成或拒絕後
固定本輪 identity。後續 sequence、report、complete 或 fail 更新使用同一份紀錄，
不把成功核對重設成 null。可捕捉的後續錯誤記 run `failed`；SIGKILL、abort 或 auditor
直接 `_exit(127)` 可能留下 `running`，不能當成完成。

`identity.expected_source=null` 表示沒有認可的獨立來源 authority；
`publisher_authentication:"not_checked_by_runtime"` 固定不變。本切片永不產生
`expected_source_verified`，不宣稱來源認證、publisher authentication、成功載入模型、
`loaded_buffer` 或行為等價。Gate B 不改 identity，也沒有新增 `load_verification`。

## 2. Journal／report／reader 格式

新 writer 使用 `saccade.native_track_journal/v2` 與
`saccade.native_track_report/v4`。[成功 report](../../shipping/tools/track_driver.hpp)
直接取 RunCompletion 的 immutable identity；report 的 `model_root` 記錄這次 caller
使用的既有解析基準，供 reader 核對 plan binding 與 identity 路徑。它是證據上下文，
不是 trusted root、路徑限制或來源批准。

[native_track_parity reader](../../scripts/eval/diagnostics/native_track_parity.py)
保留 historical report v2、report v3／journal v1 的原始意思，後者仍只接受
`identity={"level":null}`。新格式需核對相符的 version pair、run_id、完整 identity，
以及既有 `state=complete`、sequence 提交狀態與 txt／trace／report hashes。
不完整 run、格式混用、identity 不一致、不合法或損壞的輸入不能得到 `EXACT`。
歷史證據不被補寫 bindings 或重新解讀成 checksum identity。

## 3. Verified／unverified：逐層證據

以下每列獨立驗收；pending 不是 PASS，也不能由另一列的成功代替。

| 層 | 本候選的實際結果 | 必須保留的證據 |
|:--|:--|:--|
| CPU protocol／reader | PASS：standalone CTest 8/8；真實 frozen model Gate A 3515 checks／0 failures；completion 267 checks／0 failures；13 個 actual writer／reader cases；reader 174 passed／0 skipped | `cpu_controls/{ctest,frozen_model,completion}.log`、`reader_crosscheck.json`、`reader_tests.log`；包含實際拒絕原因與三份 metadata buffer replacement 控制 |
| 本機 pre_push | PASS（exit 0），source／document snapshot `5b5e3b8d789c0d18845738f7660d155b5907f176`；pytest 5575 passed、52 skipped、105 deselected、5 xfailed | `pre_push.log`；`UV_NO_SYNC=1` 保留既有 DALI extra，各原有 checks 均執行；shipping 手動 build／CTest 另列；development coordinate check 不等於 publication current |
| PR CI | 以 [#572](https://github.com/raylei50653/saccade/pull/572) 描述的指定 final head、checks 與 workflow run links 為準 | CPU CI 不驗 GPU；coordinate review PR #573 無 checks 不能解讀為遠端 CI 通過 |
| 本機新 binary | PASS：兩入口 CLI 23 passed／0 skipped；既有 native GPU CTest 6/6 | `build_final.log`、`build_release_final.log`、`cli_controls.log`、`native_gpu.log`；正控制觀察到 CUDA initialization，Gate A 拒絕控制未觀察到；Gate B 載入失敗保留 checksum identity |
| GPU／七序列 parity | 新 binary 的 4 個正向模式與 7 個負控制均實際重播七序列／5316 frames；qualified parity 全部 **UNRESOLVED**（V5 driver `617.42 != 616.92`），負控制標準 `caught=false`；不宣稱正式 EXACT／CAUGHT | fresh anchor／oracle rows；正向 raw detector／MOT／graph 比較相等（無 trace 為 NOT_RUN、serial graph 為 NOT_RUN）；7 個 raw 負控制在預期 section 為 DIFFERS；native 正向皆 complete 且 journal／report checksum identity 相同；shipping surface／link 通過、measurement surface 如期拒絕 |
| Runtime coordinate | PASS：fresh complete 五軸候選；獨立審查 #573 head `fe4b83f1fd08842b361e918f5ac9fb5e23f2851c` 無發現；靜態 attested 通過；額外完整 source／host／inputs／probe 比較無 failures／warnings | `runtime_identity_capture.log`、`runtime_identity.candidate.json`、`coordinate_review_attested.log`；舊 canonical 原樣 archive；`equivalence.state=unproven` |
| Installed package | UNVERIFIED；本切片未授權 re-pin／更新 | 不以舊 installed package 或歷史 EXACT 當成新 binary／package 驗證 |
| 獨立 Codex review | source head `df475bb0b633577e26a3b8e39b6a2e47e50525de` 預審無剩餘發現；獨立 CPU 226 passed／0 skipped；published final-head review 結果保留於 #572 描述 | 最終審查另涵蓋 coordinate／證據；保持 unmerged |

CPU／CI 正負控制須涵蓋全部六項 binding 與真實 source location、合法 frozen model
的 Gate A 正控制、各 artifact 缺失／不一致、lineage／attestation 不一致、optional
omission 與 requested missing 的區別、中途失敗不提前提升、後續載入失敗／sequence
失敗／report 失敗／真正 process 中斷後 identity 不變，以及 reader 歷史兼容。
每個拒絕控制核對失敗原因，不能只檢查非零 exit code。真 binary 的拒絕控制另直接觀察
CUDA 初始化是否發生；CPU-only library 沒有 link CUDA 不能代替這項觀察。
serial、double buffer 與 measurement surface 的既有控制須保持通過。

Runtime coordinate 的 environment 軸未變，只描述所編碼的 Torch／CUDA／cuDNN／TensorRT／
device capability／recipe closure；它沒有 NVIDIA driver 欄位。fresh probe 通過不能解除
legacy V5 的 `617.42 != 616.92` qualification refusal，也不宣稱來源、載入或全域行為等價。

## 4. Source／產物與 frozen inputs

Raw 證據保留於 gitignored、本機限定的 `results/549_s2_1a/`。此頁保留可審查摘要；本機 raw paths 不承諾
在 clone 或 CI 上存在，也不是 portable GPU replay。

| Coordinate／artifact | 記錄 |
|:--|:--|
| Source baseline | `15e819502e4d5ea581792a3a16f97857435fc585` |
| GPU source head／tree | `df475bb0b633577e26a3b8e39b6a2e47e50525de`／`d09bee3f6850d81a05881f44e8f062a9dc8d8280`；clean；後續只納入 reviewed coordinate 與本頁證據，不改 runtime source |
| PR head／獨立審查 head | #572 描述與 exact-head checks／review 記錄；不以本頁自我參照產生 head |
| 重播 `build-release/shipping/saccade_track` | 8,881,392 bytes；`f3593c00d55723d10b6f3ac2ed1afe57260cd7c67cc7986ef96440ac1756ef6d` |
| 重播 `build-release/shipping/saccade_track_measurement` | 8,896,504 bytes；`16a7f9d85a46b7ae75dad82a7f11abb955175f653662da794554e148854270d9` |
| GPU `source_manifest.json` SHA256 | `adc1d36a65ab0d7bdf78eb785a232c6041ec139bdb7c72aea8c0df1839a8c9de`；before／after 相同 |
| 本機 raw GPU evidence／SHA256SUMS | `results/549_s2_1a/gpu_df475bb0b633/`，入口 `results/549_s2_1a/gpu_prepared/{replay,continue}.py`；SHA256SUMS 的 SHA256 是 `2b0ddce3fefcc53a3591d4485d74c0f80831c1ef8f7c1e373826e03514c92dfa`；local-only |
| 起始 frozen-input snapshot | `results/549_s2_1a/start_baseline.json`；reading 是起始 bytes 觀察，不是來源認證或 runtime 驗證 |

起始 snapshot 記錄以下 unchanged-model inputs；正式 run 必須比對前後 bytes，
不重寫模型、原始 lineage／attestation、thresholds 或 presets。

| Input | 起始 SHA-256 |
|:--|:--|
| `build/libsaccade_scan_torchop.so` | `aa84cccd5c5094b33d63217eeb377452ae19c69cd1382ec7de8ba96d0d8d31c4` |
| `configs/shipping/mamba_head_realization.attestation.json` | `33289884c3586cc062a1c6ddbf84d7a45993070514664734329d809c7bae351c` |
| `configs/shipping/mamba_whole_graph.resolved.json` | `6576956f37c1a3f85134febf8ab7be86507beddbe90ac525a5947c91bd55ddf9` |
| `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json` | `677bc82320657a9ae289e78699c7a6f86e890fc629d1bb8189c08d35b57a277a` |
| `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt` | `1663ec97022879e8078fb5b9dc3b65f4d2b76e4f8df6d5848300c2b3c999d879` |
| `models/yolo/yolo26s_backbone_640_best.engine` | `2ef3d4d40dfb670982cbbb98e6ed7d07e5b1a590cfa126d5ccf7342ee1579ce4` |

## 5. 重播入口（recipe，尚非驗證結果）

CPU-only 協定與 reader／surface 控制：

```bash
cmake -S shipping -B build-shipping
cmake --build build-shipping --parallel
ctest --test-dir build-shipping --output-on-failure
.venv/bin/python -m pytest -q tests/unit/test_native_track_parity.py \
  tests/unit/test_shipping_measurement_surface.py
bash scripts/pre_push.sh
```

真 binary 的 CLI suites 使用 `build/shipping/` 的兩個 entrypoint；缺 build 或模型導致
skip 時不得寫成已驗證。build 使用既有 native toolchain 與
[shipping configure 契約](native_runtime_resolved_config.md#166-重現)，只 build 具名
`saccade_track`／`saccade_track_measurement` targets，避免重建 frozen operator。
GPU 使用 `tools/resctl.py run --wait gpu0` lease 依序執行：

```bash
.venv/bin/python tools/resctl.py run --wait gpu0 -- \
  .venv/bin/python -m pytest -q tests/unit/test_saccade_track_preflight_cli.py \
  tests/unit/test_saccade_track_completion_cli.py tests/unit/test_saccade_track_schedule_cli.py
```

七序列 parity 另先用 `native_detector_parity.py anchor` 與 `oracle-rows` fresh capture，
再用 `native_track_parity.py parity --track-binary <本輪固定 binary> --out <本輪目錄>
--oracle-rows <fresh rows> --oracle-txt <fresh anchor>`；shipping double buffer 與
measurement none／serial 各自執行，保留 report／journal、trace、graph counts、MOT
hashes 及既有 parity 負控制。native GPU ctest 在具備 tests 的 build 上依 lease 跑
serial runtime、double-buffer runtime、detector S2、post-detector host、ingest host。
對照歷史 run 的相同 hashes 只能由本輪新 run 比對得到；未重播時仍是 UNVERIFIED。
上述路徑 placeholders 必須換成本輪固定產物，正式結果另補 exact commands 與 digest。

Shipping source 改變時先用 `h2_path_partition.py --classify <path>` 檢查影響；涉及
published axes 時依唯一支援的 [runtime identity republication runbook](runbooks/runtime_identity_republication.md)
fresh capture 完整候選到 scratch，保留 independent promotion review 與 append-only
archive，再以 `check_runtime_identity_staleness.py --mode attested` 核對。
promotion 可用 stacked review surface 與 implementation atomic co-land；probe 永不重用，
相同 probe 不構成 semantic equivalence，`equivalence.state` 維持 `unproven`。
這不自動授權 installed package re-pin。

## 6. Deferred／known limits

| 義務 | 本切片邊界 |
|:--|:--|
| S2-1 | `--model-bundle`、TR-1b allowlist、雙 root、`--require-identity` policy、完整模式 `load_verification` 未實作／未授權 |
| S2-2 | Gate B 載入強化、loaded-buffer 不變式、TOCTOU 修復及其 GPU qualification 延後 |
| S2-3 | runtime／model package 拆分、installed launcher 預設與 package 新版本延後 |
| #537 | 完整 state transition／failure semantics 仍由該 Issue 擁有 |
| #541 | requirement↔check 對應、as-built 與完整驗收仍未完成 |
| #547 | 模型權利、publisher／公開發行裁決獨立；`distribution.status=local-only` 不變 |

Gate A 的 hash 與 Gate B 的實際開檔仍分離（F4 TOCTOU）；frame 內容不 hash，
Gate A 不解碼 JPEG，也不把 sequence 清單固定給後續 runtime。auditor 比路徑／名稱，
不比載入 bytes；F3 `_exit(127)` 的發生時點可能留下 `running` journal，沒有承諾
完整 failure 診斷。不同 OUT 共用 report／trace 不受同一 lock 保護，後 rename 者可能
覆蓋檔案，依 journal hash 偵測；中斷留下的 temp 不自動清理。fsync／rename 的
斷電 durability、網路檔案系統與 WSL 掛載 Windows 磁碟語義仍 unverified，既有證據
僅限 controlled host／WSL2 ext4。checksum identity 沒有消除任何上述限制。

