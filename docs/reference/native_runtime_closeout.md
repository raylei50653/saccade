# Native runtime：#465 收尾驗收紀錄（#546）

> 狀態：**提交 owner review 的建議 verdict**，不是 owner 決定；#465 不自動關閉。
> 基準：`main`＝`24dab817`（PR-C4 #544＋#545 合併後）＋#546 的 release policy 變更（[resolved config 文件](native_runtime_resolved_config.md) §21）。
> 本文只彙整與對照既有的正式證據，不重述技術細節；細節以各節連結的文件為準。沒有新的 GPU 量測、benchmark 或平台宣稱。

**建議 verdict：`ENGINEERING_COMPLETE — PUBLIC_DISTRIBUTION_DEFERRED`**

- 工程：§3 的每一條宣稱都有已合併的 PR 與正式 run 支持（§4）。重跑昂貴 GPU 評估的條件（證據缺少、過期或失效）不成立（§5）。
- 散佈：維持 `local-only`。授權 Open items 還沒有 owner 結論，也沒有 release key（§6）。後續以 #547 追蹤（§6.3）。

---

## 1. 兩個完成狀態

兩者分開判定。工程完成**不**推出可以公開散佈。

| 狀態 | 成立條件（全部） |
|:--|:--|
| **`ENGINEERING_COMPLETE`** | E1 shipping tree 符合 G2：靜態 G2-1／G2-3，執行時 G2-2／G2-4（[boundary](native_runtime_shipping_boundary.md) §0）。<br>E2 最終 package 安裝到乾淨容器後，輸出對已接受的 oracle `A_L` 為 `EXACT`。<br>E3 runtime identity current；operator library 與 entrypoint 由 attestation／pin 綁定。<br>E4 完整性控制是強制的，負控制都被擋下。<br>E5 named limits 有文件。<br>E6 沒有未解的工程 blocker。 |
| **`PUBLIC_DISTRIBUTION_READY`** | `ENGINEERING_COMPLETE`，加上：<br>D1 授權 Open items（§6.1）每一項都有 owner 的結論與證據；`license_audit.json` 的 `owner_confirmation` 不是 null，`distribution.status` 不再是 `local-only`。<br>D2 owner 的 release 公鑰已 commit，公開的每一份 package 都已簽章（**公開散佈時簽章是強制的**；可選只適用 local-only）。<br>D3 libgomp 的 GPL-3.0／GCC Runtime Library Exception 文本與 source 提供方式已處理，或已改用其他做法。<br>D4 owner 決定公開。 |

`PUBLIC_DISTRIBUTION_READY` 不能從工程證據、授權檔 sha256 相同或「條款看起來准許」推出；每一項都要 owner 的結論（PR-C4 owner 指示，resolved config 文件 §20）。

---

## 2. 架構與信任邊界

```
 開發機（repository、venv；Python 只在這一側）             ┊  使用者的系統（Ubuntu 24.04 base，無 Python／編譯器）
                                                           ┊
 cmake --install --component shipping                      ┊
   tree：pin 的 entrypoint、launcher、auditor、             ┊
   lib/vendor（27 個 sha256 釘住的物件）、model root          ┊
        │  [M] static 檢查：G2-1／G2-3、layout、pin           ┊
        ▼                                                  ┊
 build_shipping_package.py                                 ┊
   [M] 工作樹乾淨、runtime identity current                  ┊
   MANIFEST.json（每檔 sha256／大小／mode、pin、座標）         ┊
   <name>.tar.gz ＋ <name>.install.sh ＋ <name>.sha256 ─────┼──▶ ① 取得 release 檔案
        │                                                  ┊        [O] minisign -V（簽過的 release；公鑰取自 repository）
        ▼  [O] sign_shipping_package.py sign（owner 的 key）  ┊        [M] sha256sum -c <name>.sha256
   <name>.sha256.minisig ──────────────────────────────────┼──▶     │
                                                           ┊        ▼
 check_shipping_package.py tarball [--pubkey]              ┊  ② sh <name>.install.sh <tarball> TARGET（只用 base system 工具）
   （開發端的稽核；使用者手上沒有）                            ┊     [M] digest → 同檔案系統的 staging → MANIFEST 逐檔
                                                           ┊         （恰好的檔案集合、sha256、大小、mode）
                                                           ┊     [M] 一次 renameat2(RENAME_NOREPLACE)；失敗時 TARGET 不建立
                                                           ┊        │
                                                           ┊        ▼
                                                           ┊  ③ <prefix>/bin/saccade_track（sh launcher）
                                                           ┊     [M] auditor 的 readiness probe，不成功就 exit 127
                                                           ┊     [M] exec ld.so --library-path lib/vendor --audit …
                                                           ┊         bundle 名字在 tree 外被載入就 exit 127
                                                           ┊        │
                                                           ┊        ▼
                                                           ┊  ④ libexec/saccade_track（native C++／CUDA／TensorRT）
                                                           ┊     [M] 載入前依 lineage／attestation 比對 operator library、
                                                           ┊         head、backbone engine 的 sha256，不符就失敗
                                                           ┊     NVIDIA driver（libcuda 等）由系統提供，不在 package 內
```

- `[M]`＝強制，不能以選項或環境變數關掉。`[O]`＝可選。只有 `[O]` 的簽章會認證發行者。
- **信任邊界 A（┊）**：release 檔案離開開發機。跨過 A 之後，完整性靠 digest 與 MANIFEST，發行者靠簽章。沒有簽章時，**沒有任何控制認證發行者**：能換掉 tarball 的人也能換掉 digest（§19.6）。
- **信任邊界 B**：launcher 與 loader 之間。launcher 的 `/bin/sh` 不受保護，呼叫者的 `LD_PRELOAD` 會影響它（§17.6）。auditor 依名字分類（真實 basename、SONAME family），不比對位元組；位元組由安裝時的 sha256 負責。
- **信任根**：開發端是 Git commit 加 runtime-identity publication；使用者端是 digest（完整性），以及有簽章時的 repository 公鑰。

---

## 3. 工程宣稱

| # | 宣稱 | 範圍 |
|:--|:--|:--|
| C1 | shipping runtime 本身 Python-free（G2）：NEEDED 閉包與 tree 沒有 Python；執行時沒有 exec 或映射 Python，也沒有 JIT | 安裝後的 package tree |
| C2 | package 在乾淨的 Ubuntu 24.04 容器裡執行（沒有 Python 與編譯器，只有 base system 加 package） | 同一台機器、sm_120、driver 616.92 |
| C3 | 輸出對已接受的 `A_L` 為 `EXACT`：detector 5316/5316、MOT txt 7/7、graph captures 7/7；PR-12 → C1 → C2 → C3 → C4 每一步都逐位元組相同 | `A_L`／native 組態。**不是** headline 的位元組 |
| C4 | runtime identity current；operator library（`aa84cccd…`）、entrypoint（`92f74ef4…`）、27 個物件由 attestation／pin 綁定，並記在 MANIFEST | 同上 |
| C5 | 第三方物件由 package 提供：loader audit 拒絕 tree 外的同名物件；CLI 只剩 shipping 介面 | 依名字，不依位元組 |
| C6 | 安裝是完整性檢查過的 atomic staging：損壞、缺檔、多檔、被改的檔案、已存在的 TARGET 都被拒絕，而且 TARGET 不被建立或改動 | 只在 ext4 上確認 `RENAME_NOREPLACE` |
| C7 | 簽章是可選的 opt-in，而且 fail-closed：要求驗證時，壞的簽章、錯的 key、被改的 digest 或 trusted comment 都失敗；有簽章檔卻沒有要求驗證時，檢查失敗，不會略過 | 只用 test key；沒有 release key |
| C8 | 每個出貨的第三方物件都逐一稽核了授權證據；散佈狀態記為 `local-only` | 是對授權文本的讀法，不是法律結論 |

---

## 4. 驗收矩陣

正式 run 都在同一台機器上（RTX 5070 Ti Laptop，WSL2）。結果目錄不納入版本控制（`results/` 被 ignore）。「§」指 [resolved config 文件](native_runtime_resolved_config.md) 的節。

| 宣稱 | 已合併的 PR | 正式證據（verdict） | 限制 | 測試 |
|:--|:--|:--|:--|:--|
| C1 G2 | PR-12 #524（+#525）；PR-C1／C2 #527 | PR-12 r2 `results/465_pr12_shipping/full_c44dd876_r2/`：`PASS`，靜態 5/5、N1–N8，§16.4。PR-C4 r2 `results/465_prc4_release/full_bc75dcab/`：`PASS`，gate 2 `static` 12/12；gate 6 `static --manifest` 13/13；gate 8 `runtime` 3/3，兩次的 `python_libraries_mapped` 都是空的，§20.5 | launcher 的 sh 與 `LD_PRELOAD`；probe 與 exec 之間的 race 不做宣稱（§17.6、§18.11） | `test_shipping_g2_checks.py`、`test_shipping_bundle_checks.py`、`test_shipping_link_surface.py` |
| C2 乾淨容器 | PR-12；PR-C3 #542（+#543）；PR-C4 #544（+#545） | PR-C4 r2 gate 6–7：安裝與執行容器是 Ubuntu 24.04.4、dash，沒有 Python 與編譯器；exit 0 | 只有 sm_120；一台 WSL2 機器與一個 driver；原生 Linux 沒有驗證（§19.6、§20.6） | — |
| C3 `A_L` parity | U1 PR-1L／PR-2L（#482–#487，owner `ACCEPT`）；PR-8 #515；PR-9 #518；PR-10 #520；PR-11 #522；PR-12 | PR-2L：`HEAD_PARITY_WITHIN_TOLERANCE`。PR-8：`EXACT` 5316/5316（§12.5）。PR-9：`EXACT`（§13.4）。PR-10：`EXACT`（§14.4）。PR-C4 r2 gate 7：從最終 tarball 安裝後 `EXACT`（detector 5316/5316、mot_txt 7/7、graph 7/7），`--against` PR-C3 的 `parity_bundle` 7/7 相同 | **`A_L` ≠ headline**（U1 named limit）：任何引用到 package 的數字都要來自 native 組態；headline 的量測不轉移 | `test_native_track_parity.py`、`test_native_*_oracle_pins.py`、`test_shipping_mot_output_oracle_pins.py` |
| C4 identity／attestation | PR-8（attestation）；PR-10（re-attest，owner 接受）；每個 PR 的 republish chore | PR-C4 r2 有效性：`--mode attested` exit 0；pin 在 run 前後相同；`anchor` 與 `A_L_1` 7/7 相同。本文 V1–V2（§5） | probe 的相等不宣稱等價；operator library 的絕對 RUNPATH 是列舉的例外（§16.5） | `test_runtime_identity_staleness.py`、`test_package_signature.py`（MANIFEST reading） |
| C5 bundle／CLI | PR-C1／C2 #527（修訂 A2–A4） | A4 `results/465_prc2_cli/full_1ae402c2/`：`PASS`，gate 1–6、M1–M6、N1–N18（§18.15） | auditor 依名字不依位元組；alias 控制只在 CPU 測試（§18.13、§18.15） | `test_shipping_bundle_checks.py`、`test_shipping_measurement_surface.py` |
| C6 完整性／atomic | PR-C3 #542（+#543）；#546 | PR-C3 `results/465_prc3_package/full_9217ed92/`：`PASS`，gate 1–6，負控制 P1–P15（§19.5）。#546 `results/546_closeout/full_c7581100/`：E2–E3、U1、U5（§21.4） | digest 不是簽章（P3）；`--verify` 信任 tree 裡的 MANIFEST（S9）；SIGKILL 會留下 staging；`RENAME_NOREPLACE` 只在 ext4 上確認（§19.6） | `test_shipping_package.py` |
| C7 簽章 opt-in | PR-C4；#546 | PR-C4 r2：gate 4–5、S1–S9（§20.5）。#546：E4、U2–U4（§21.4） | 沒有 release key；trusted comment 要由使用者比對（S8）；簽章不是決定性的（§20.4a） | `test_package_signature.py`、`test_shipping_package.py` |
| C8 授權稽核 | PR-C4 | PR-C4 r2 gate 3 四項 PASS；L1–L3 照預期失敗；review 修正重播 `results/465_prc4_release/review_fix_96bb41c0/`（§20.8） | `local-only`；snapshot 是 2026-10-08 的；不是法律意見（§20.6） | `test_license_audit.py` |

CI：`main`＝`24dab817` 上 `CI (Lint, Type Check, Test)` 與 `C++ Core Build` 都通過。#546 的變更跑了 pre_push；它的 PR CI 見 PR 頁面。#546 的證據 `results/546_closeout/full_c7581100/`＝`PASS`：V1–V3、E1–E4、U1–U5（§21.4）。

---

## 5. 證據的沿用

| 證據 | 沿用的理由 | 是否重跑 |
|:--|:--|:--|
| 7-seq parity、G2-2／G2-4（PR-C4 r2 gate 7–8） | 執行時的位元組沒有變。#546 的 tree 與 r2 的 tree 只差 `README.txt`（§21.4 E1）；安裝器逐位元組相同；runtime identity 不動（`--mode attested` exit 0） | 不重跑 GPU |
| 安裝器負控制 P1–P15（PR-C3） | `install.sh` 與 `9217ed92` 逐位元組相同（§21.4 V3） | 重做其中兩項當代表：U1 是 P1 的未簽版，U5 是 P6 |
| 簽章負控制 S1–S9（PR-C4） | `signature` 的檢查函式沒有改 | 重做三項：U2、U3、U4 |
| stale identity 的拒絕 | builder 與 `metadata` 的邏輯沒有改 | 由單元測試覆蓋（§19.1） |

PR-12 r1（`full_c44dd876/`，INVALID）、PR-C4 r1（`full_bfd086c7/`，FAIL）與 PR-9 的第一次 run 都已在各自的節照實記錄，本文不引用它們當通過的證據。

---

## 6. 公開散佈：未解的項目

### 6.1 授權（PR-C4 §20.1 的 Open items；`shipping/THIRD_PARTY.md`）

| # | 項目 | 物件 | 需要的結論 |
|:--|:--|:--|:--|
| L-1 | 出貨文本沒有提到、只有版本對應的官方條款准許的物件 | `libnvJitLink.so.13`、`libcufile.so.0`、`libnvshmem_host.so.3` | 以 wheel 取得的物件適用哪一份文本；若以官方條款為依據，是否把那份條款的文本也放進 `licenses/` |
| L-2 | 兩份文本不同 | cuDNN（5 個）、`libnvinfer.so.10` | 以哪一份為準；TensorRT wheel 文本的一年期與「下游同等限制」怎麼滿足 |
| L-3 | 出貨時沒有附授權文本 | `libgomp.so.1`（GPL-3.0-or-later WITH GCC-exception-3.1） | 附上 GPL-3.0 與 exception 的文本，並決定 source 的提供方式；或改用其他做法 |
| L-4 | NVIDIA 條款共通的條件 | 所有 NVIDIA 物件 | 「material additional functionality」「only accessed by your application」「不得使之受 open source license 約束」是否滿足 |

**不從任何一項推出可以再散佈**：授權檔 sha256 相同不代表授權相同，稽核也不是法律意見。

### 6.2 簽章

- release 公鑰 `shipping/package/minisign.pub` 還不存在。owner 產生，另以 PR 加入（§20.1 的 key 管理）。
- 公開散佈時簽章是強制的（§1 D2）。local-only package 可以不簽，但不得被描述成 authenticated（§21.1）。

### 6.3 後續 issue

**需要。** 公開散佈的 blocker（§6.1、§6.2，以及 owner 的發布決定）不屬於工程，而且 #465 的 Definition of done 不要求公開散佈。所以它們移到獨立的 issue [#547](https://github.com/raylei50653/saccade/issues/547) 追蹤，不阻擋 #465 的工程收尾。

---

## 7. 延後的工作（不重述細節）

- 公開散佈（§6）。
- 其他 GPU／SM、TensorRT 版本：要 rebuild 並重新 attest operator library 與 engine（Phase C scope C-D2）。
- 原生 Linux 主機（非 WSL2）上的驗收；其他 driver。
- 其他檔案系統上 `RENAME_NOREPLACE` 的行為（§19.6）。
- 安裝後自動以簽過的 `manifest_sha256` 驗證 MANIFEST（目前由使用者手動對照，§20.6）。
- MANIFEST 的 `reading` 句子跟著可選簽章改寫（§21.5）。
- Phase D：release／CI 自動化。
- G1（bundled-Python 過渡包）不做（Phase C scope §1）。
- 研究支線（compiled-head、block-scope Inductor、tracker block divergence）不在 runtime 路徑上，不影響 verdict。

---

## 8. 本文沒有做的事

- 沒有重跑 GPU parity，也沒有新的 benchmark、FPS 或平台宣稱。
- 沒有改 runtime、安裝器、launcher、auditor、第三方集合、授權稽核或散佈狀態。
- 沒有關閉 #465，也沒有做任何授權判斷。
