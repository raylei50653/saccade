# Native runtime packaging：Phase C 範圍與決策（#465 Phase C）

> 狀態：Phase C 範圍凍結。owner 決策 C-D1–C-D4 於 2026-10-04 定案（本文之前）；**不改** runtime 行為、preset、threshold、weights、benchmark claim，也沒有實作任何打包步驟。
> 基準：Phase B 終點＝PR-12（#524，`main`＝`856ccccb`），[native_runtime_resolved_config.md](native_runtime_resolved_config.md) §16；範圍詞彙與 G1／G2 定義沿用 [native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §0–§1。
> 本文只定義 Phase C 的**決策、邊界與順序**；沒有執行 inference、沒有發任何數字。§3 的大小與 §4 的授權讀法是對現有檔案的靜態讀取，不是法律意見。

> **更正（2026-10-04，PR-C1）**：PR-C1 的 probe（[native_runtime_resolved_config.md](native_runtime_resolved_config.md) §17.1）推翻了 §5 的兩處：(1) torch 系列、`libnvinfer`、`libnvshmem_host` 帶的是 DT_RPATH 不是 RUNPATH，而 DT_RPATH 在 loader 的搜尋順序中排在 `LD_LIBRARY_PATH`／`--library-path` 之前；(2) 修法改為 launcher 以 `ld.so --library-path <prefix>/lib/vendor --audit <auditor>` 執行未改動的 entrypoint，第三方物件放在 `lib/vendor/`（相對 RPATH 全部展開在 prefix 內），次要的 fail-closed 檢查是 rtld-audit library，不是在 `saccade_track` 裡讀 `/proc/self/maps`。另外，重新 build 的 `saccade_track` 與 PR-12 的位元組不同，所以 tree 帶 PR-12 的那一份（`shipping/entrypoint_pin.json`）。**C-D5＝minisign**（owner，2026-10-04，v1）。以下 §5、§6 PR-C1 已據此更正，其餘維持本文凍結時的內容。

本文回答：**PR-12 的 shipping tree 要變成一個終端使用者可以安裝的包，還差什麼、按什麼順序做。**

---

## 0. 起點：PR-12 留下的狀態

| 軸 | PR-12 的狀態（§16.4／§16.5） |
|:--|:--|
| tree | `bin/saccade_track`（RUNPATH `$ORIGIN/../lib`）＋`share/saccade/` 6 個檔案，約 75 MB；`lib/` 是空的 |
| 第三方函式庫 | 27 個物件，不在 tree 裡，經 `LD_LIBRARY_PATH` 提供；集合以 sha256 定義（＝開發版 binary 載入的那一組） |
| G2 | 定義 1–4 在乾淨 `ubuntu:24.04` 容器成立；parity EXACT（detector 5316/5316、MOT txt 7/7、graph captures 7/7） |
| GPU | `saccade_track` 帶 SASS sm_75/80/86/90/100/120＋PTX sm_120；operator library（`aa84cccd`）與 backbone engine 只支援 sm_120 ⇒ 整體只在 sm_120 上成立 |
| glibc | Ubuntu 24.04 baseline（GLIBC 2.39／GLIBCXX 3.4.33／CXXABI 1.3.15） |
| 已知缺口 | 安裝不是 atomic；沒有 package manifest／digest；CLI 仍有開發選項；驗收只在一台 WSL2 機器、同一個 driver（616.92）上做過 |

Phase C 不重做 Phase B 的任何 parity；它的每一個 PR 都以「與 PR-12 正式 run（`results/465_pr12_shipping/full_c44dd876_r2/`）逐位元組相同」為輸出驗收。

---

## 1. owner 決策（2026-10-04）

| # | 決策 | 定案 | 理由（一句） |
|:--|:--|:--|:--|
| **C-D1** | 第三方 runtime 函式庫是否 bundle（issue 的 decision 2） | **bundle PR-12 的 attested 集合**：27 個物件以 sha256 原樣放進 `lib/`，執行時不再需要 `LD_LIBRARY_PATH` | parity 證據綁在這 27 份位元組上；任何其他來源都是未驗證的組態 |
| **C-D2** | engine 分發與支援的 GPU | **只支援 sm_120，ship 既有的 prebuilt engine 與 attested operator library**；不在首次執行時建 engine | 本機只有 sm_120；使用者端建出的 engine 不保證位元組相同，parity 無法證明。擴到其他 SM 要 rebuild＋re-attest，另開 scope |
| **C-D3** | package 形式 | **relocatable tarball**：tree＋`lib/`＋`MANIFEST.json`＋`licenses/`，附 POSIX sh 安裝器（staging → 逐檔驗證 → atomic rename） | 最接近 PR-12 實際驗證過的形狀；不要求使用者有 Docker |
| **C-D4** | glibc baseline | **維持 Ubuntu 24.04** | 已驗證；frozen toolchain（#214）不變 |

C-D1 讓 G2 維持成立（bundle 的是 C/C++ 函式庫，不是 Python）。G1 的 bundled-Python 過渡包不做。

---

## 2. 本文的預設（可在 review 推翻，不需另開 owner 決策）

| 項目 | 預設 | 依據 |
|:--|:--|:--|
| configure 時需要 Python／pybind11／GStreamer | 維持，屬 `developer_build_debug` | G2 只約束 runtime；包是在開發機上 build 的 |
| model root 路徑 | 照抄 repository 相對路徑（含 `share/saccade/build/libsaccade_scan_torchop.so`） | lineage 與 attestation 以這些路徑命名而且凍結（§16.1） |
| operator library | 不 rebuild、不 re-attest，位元組不變 | C-D2；PR-12 owner 決策 1 |
| 平台 | Linux x86_64 only；原生 Windows、macOS、aarch64 不在範圍 | ADR 025 矩陣；audit §8 |
| CLI 開發選項 | PR-C2 決定（§5）；預設提案：`--measurement-mutation` 不進 release binary，`--trace`／`--report` 保留（驗收要用） | `--measurement-mutation` 是負控制注入，不該出現在使用者手上 |
| 簽章形式 | PR-C3 只產 package digest（sha256）；簽章在 PR-C4 落地，機制＝**minisign**（C-D5，owner 2026-10-04，v1；key 管理在 PR-C4 寫定） | Phase B 的 trust root＝Git commit＋runtime-identity publication，對 release 不夠 |

---

## 3. 要 bundle 的集合（C-D1 的內容）

來源：`results/465_pr12_shipping/full_c44dd876_r2/deps.json`（27 項，sha256 為準）。合計 **3.55 GiB**（未壓縮）；加上 tree，包約 3.6 GiB。

| 群組 | 物件（MiB） | 來源 wheel |
|:--|:--|:--|
| LibTorch | `libtorch_cuda.so` 435.0、`libtorch_cpu.so` 430.3、`libtorch_nvshmem.so` 3.9、`libc10.so` 1.4、`libc10_cuda.so` 0.6、`libtorch.so` 0.3、`libgomp.so.1` 0.2 | `torch==2.11.0`（cu130） |
| TensorRT | `libnvinfer.so.10` 638.9 | `tensorrt_cu12_libs==10.16.1.11` |
| CUDA math／runtime | `libcublasLt.so.13` 516.5、`libcufft.so.12` 273.3、`libcusparse.so.12` 155.0、`libcurand.so.10` 126.6、`libnvrtc.so.13` 104.3、`libnvJitLink.so.13` 94.2、`libcublas.so.13` 51.7、`libcupti.so.13` 4.0、`libcufile.so.0` 3.0、`libcudart.so.13` 0.7 | `nvidia-*` cu13 wheels |
| cuDNN | `libcudnn_engines_precompiled.so.9` 234.0、`libcudnn_heuristic.so.9` 58.0、`libcudnn_engines_runtime_compiled.so.9` 28.0、`libcudnn_graph.so.9` 4.2、`libcudnn.so.9` 0.1 | `nvidia-cudnn-cu13==9.19.0.56` |
| 其他 | `libcusparseLt.so.0` 222.8、`libnccl.so.2` 207.9、`libnvshmem_host.so.3` 39.3 | `nvidia-cusparselt-cu13`、`nvidia-nccl-cu13`、`nvidia-nvshmem-cu13` |
| nvJPEG | `libnvjpeg.so.13` 5.6 | `nvidia-nvjpeg==13.0.1.86`（`shipping/CMakeLists.txt` FetchContent 釘住；resolved config 文件 §11） |

NCCL、nvshmem、cusparseLt、cuFile、cuDNN 等是 `libtorch_cuda.so` 的 NEEDED；在不換 LibTorch 的前提下不能刪減。刪減集合會改變載入集合，等於換組態，不在 Phase C。

不 bundle 的：base system（glibc、GCC runtime 的 `libstdc++`／`libgcc_s`、zlib）由目標系統提供；driver 函式庫（`libcuda.so.1`、`libnvcuvid.so.1` 等）由 NVIDIA driver 提供，不可散佈也不該散佈。

---

## 4. 授權：讀到什麼、還缺什麼

以下是 venv 中各 wheel 附帶的授權檔原文的讀法，**不是**法律結論；PR-C4 之前要有 owner（或 owner 指定的人）確認。

| 物件 | 授權檔 | 讀到的散佈依據 |
|:--|:--|:--|
| cudart、cufft、cublas、cublasLt、curand、cusparse、nvrtc、nvjpeg、cupti | CUDA EULA（各 wheel 的 `License.txt`） | Attachment A 列名為 distributable；附條件：隨具有實質額外功能的應用散佈、只供該應用存取、不得使之受 open-source license 約束 |
| **nvJitLink、cuFile、nvshmem** | CUDA EULA 文本 | **Attachment A 裡沒有找到它們的檔名** ⇒ 待確認 |
| libnvinfer | TensorRT SLA §12.1 | 准許以 binary 形式散佈 `libnvinfer`／`libnvinfer_plugin`，限作為具額外實質功能的應用之元件；下游再散佈要求同等限制；授權期一年自動續約 |
| cuDNN | cuDNN SLA §1.1–1.2 | 同 CUDA EULA 的 distributable 條件 |
| cusparseLt | `nvidia/cusparselt/LICENSE.txt` §2 | 「`.so` 與 `.h` runtime 檔」可作為應用的一部分散佈 |
| NCCL | BSD-3-Clause | 保留版權聲明 |
| LibTorch | BSD-3-Clause＋NOTICE（`torch-2.11.0.dist-info/licenses/`） | 保留 LICENSE 與 NOTICE；`libgomp` 是 GCC runtime（runtime library exception） |

另外兩點要一起確認：

- Saccade 本身是 Apache-2.0。NVIDIA 各條款都禁止「使 SDK 受 open-source license 約束」：包要把 NVIDIA 物件標成各自條款下的第三方元件，不能讓 Apache-2.0 涵蓋它們。
- `licenses/` 要放每個 wheel 的原始授權檔（逐檔 sha256 記在 MANIFEST），加一份 `THIRD_PARTY.md` 對照物件 → 授權檔。

---

## 5. 已知的技術問題（PR-C1 必須處理）

1. **operator library 的 `libnvrtc.so.13`**。operator library 由 `detector_host.cpp:198` 以 `dlopen` 載入；它的 NEEDED 有 `libnvrtc.so.13`，而 `saccade_track` 與 `libtorch_cuda.so` 都不 NEED 它；operator library 自己的 RUNPATH 是 7 個凍結的絕對路徑。不能 `patchelf`（會改動 attested 位元組）。**更正後的修法**（§17.1 P1）：loader 的 `--library-path` 排在 DT_RUNPATH 之前，所以 launcher 以 `ld.so --library-path <prefix>/lib/vendor` 執行時由 `lib/vendor` 解析；`saccade_track` 不必改。（原提案「讓 `saccade_track` NEED `libnvrtc.so.13`」不採用：它要改 entrypoint 的位元組。）
2. **build host 會假通過**。operator library 的絕對 RUNPATH（`/home/ray/developer/ai/saccade/...`）在 build host 上存在，所以 host 上「沒有 `LD_LIBRARY_PATH` 也能跑」不能當證據。驗收只認乾淨容器，而且要在容器裡 `env -u LD_LIBRARY_PATH`。
3. **bundled 物件之間的解析**。27 個物件中，凡是有第三方 NEEDED 的，搜尋路徑都含 `$ORIGIN`，平放在同一個目錄即可互相解析；沒有搜尋路徑的（`libcudart`、`libnvrtc`、`libnvJitLink`、`libnccl`、`libcufile`、`libcupti`）都沒有第三方 NEEDED。PR-C1 的靜態檢查對 `lib/vendor` 每個物件逐一確認 NEEDED 閉合。
4. **wheel 的搜尋路徑會指到包外**（更正：這些是 **DT_RPATH**）。torch 系列（`libtorch*`、`libc10*`、`libgomp`）的 DT_RPATH 在 `$ORIGIN` 之前列了 `$ORIGIN/../../nvidia/{cudnn,nvshmem,nccl,cusparselt,cu13}/lib`；`libnvinfer` 的 DT_RPATH 有 `$ORIGIN/../nvidia/...`、`$ORIGIN/../tensorrt_*_libs`；cuDNN 的 RUNPATH 有 `$ORIGIN/../../{cublas,cuda_nvrtc,cu13}/lib`。DT_RPATH 排在 `--library-path` 之前，那些位置若有同 SONAME 的檔案會被靜默載入（§17.1 P4）。不能改寫（C-D1 要求位元組相同）。PR-C1 的處理：(a) 第三方物件放在 `<prefix>/lib/vendor/`，每個相對搜尋路徑都展開在 `<prefix>` 之內（唯一例外是 `libcusparseLt.so.0` RUNPATH 結尾的空項＝工作目錄），配合 `static` 的 `layout_exact`（tree 恰好是預期的檔案）；(b) 次要的 fail-closed 檢查：launcher 以 `--audit` 載入 rtld-audit library，bundle 名字在 `lib/vendor` 以外的候選若存在就 exit 127（§17.1 P6）。
5. **TensorRT 是 cu12 build、其餘是 cu13**。PR-12 已在這個混合組態下驗收；Phase C 不改，記為 named limit。

---

## 6. Phase C 拆分

每個 PR 只改表內列出的範圍。輸出驗收一律是同機器、對 PR-12 正式 run 逐位元組相同（txt 與 trace 的 sha256 7/7），不發新的 benchmark claim，不量 FPS。

| PR | 內容 | 驗收 | 依賴 |
|:--|:--|:--|:--|
| **PR-C1** | **bundle**（更正後，詳見 resolved config 文件 §17）：`shipping/third_party_set.json`（SONAME、sha256、來源 wheel 與 root、授權檔）；安裝時從清單複製到 `lib/vendor/` 並逐檔比對 sha256；`bin/saccade_track`＝launcher（`ld.so --library-path lib/vendor --audit`），ELF 在 `libexec/`，位元組＝PR-12 的 pin；rtld-audit provenance check；`check_shipping_bundle.py`；`licenses/` 與 `THIRD_PARTY.md` | §17.4：靜態 11 項；host 經 launcher EXACT 且與 PR-12 正式 run 7/7 相同、載入來源全在 tree；乾淨容器（無 `LD_LIBRARY_PATH`）EXACT 且相同；strace 的 exec chain（launcher → loader）與開啟集合＝bundle；負控制 N1–N9 | — |
| **PR-C2** | **CLI surface**（詳見 resolved config 文件 §18：`--report`／`--trace` 留在 shipping，`--measurement-mutation`、`--schedule serial`、`--max-frames` 移到不安裝的 `saccade_track_measurement`；量測 hook 只編進 `_measurement` library）：決定開發選項去留（§2 預設提案）；release build 的 `saccade_track` 不含被移除的選項；`--help` 與錯誤訊息整理；不改任何 stage 計算 | 被移除的選項在 release binary 上是 unknown argument；保留的選項行為不變；release binary EXACT 且與 PR-C1 7/7 相同。若 binary 位元組改變，G2 靜態檢查與容器驗收重跑 | PR-C1 |
| **PR-C3** | **package＋atomic staging**：tarball（命名含版本、`linux-x86_64`、`cu13.0`、`trt10.16`、`sm120`、`glibc2.39`）；`MANIFEST.json`（每檔 sha256、版本 pin、SM 清單、glibc baseline、source commit、attestation 與 lineage 引用、runtime-identity 座標）；package digest；POSIX sh 安裝器：解到同檔案系統的 staging → 依 MANIFEST 逐檔驗證 → `mv` 一次 rename；任何失敗都刪 staging、目標不動；只用 base system 工具（`sh`、`tar`、`sha256sum`） | 從 tarball 安裝到乾淨容器，EXACT 且與 PR-C2 7/7 相同；安裝器負控制：tarball 損壞、單檔被改、目標已存在、磁碟寫入中斷（模擬）都讓目標目錄維持原狀；PR-12 的 N2（非 atomic）不再成立 | PR-C2 |
| **PR-C4** | **release readiness**：C-D5 簽章機制落地；§4 授權確認的結論寫入 `THIRD_PARTY.md`；`README.txt`（GPU＝sm_120、driver 需求、glibc ≥ 2.39、named limits）；最終驗收紀錄 | owner 確認 §4；簽章可被驗證、竄改後驗證失敗；最終 tarball 從乾淨容器安裝到跑完一次完整 7-seq EXACT | PR-C3、C-D5 |

**仍待 owner 的決策**

- ~~**C-D5 簽章機制**~~：已定案＝**minisign**（owner，2026-10-04，v1）。key 的產生、保管與輪替在 PR-C4 寫定。
- **§4 授權確認**（PR-C4 之前）：nvJitLink、cuFile、nvshmem 的散佈依據，以及整體散佈條件是否可接受。在確認之前可以完成 PR-C1–C3 的工程與本機驗收，但**不得公開散佈**任何 tarball。
- 是否擴到其他 SM／TRT 版本（C-D2 的後續）、是否在原生 Linux（非 WSL2）主機上加一次驗收：不在本次 Phase C，需要時另開 scope。

Phase D（release／CI 自動化）不在本文範圍。

Phase C 各 PR 共同不得做的事：改任何 stage 計算、preset、threshold、weights、benchmark claim；rebuild 或修改 operator library、backbone engine、TorchScript head；更換或刪減第三方集合中的任何物件；改 eval harness 的語義或預設；公開散佈包（§4 未確認前）。

---

## 7. 首版 release 帶著的 named limits

- 只支援 **sm_120**（operator library 與 backbone engine）；其他 SM 的 SASS／PTX 在 `saccade_track` 裡有編進去，但沒有在其他 GPU 上執行過。
- 只在**一台 WSL2 機器**、同一個 driver 上驗證過；原生 Linux 主機沒有驗證。
- TensorRT（cu12 build）與 CUDA 13 runtime 的混合組態照 PR-12 原樣。
- 包約 3.6 GiB（未壓縮），主要是 LibTorch 拖進來的 CUDA 函式庫。
- 包的 MOT 輸出等於 `A_L`／native 組態，**不是** headline 的位元組（U1 named limit）；任何對這個包引用的數字都要來自 native 組態。

---

## 8. 本文刻意沒有做的事

- 沒有實作 PR-C1–C4；沒有改 CMake、安裝規則、檢查工具或 shipping 原始碼。
- 沒有做法律判斷；§4 只是對授權檔的讀法。
- 沒有決定其他 SM、其他平台（C-D5 後來於 2026-10-04 定為 minisign，見開頭的更正）。
- 沒有執行任何 run、沒有量 parity、FPS 或精度。
