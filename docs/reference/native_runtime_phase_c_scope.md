# Native runtime packaging：Phase C 範圍與決策（#465 Phase C）

> 狀態：Phase C 範圍凍結。owner 決策 C-D1–C-D4 於 2026-10-04 定案（本文之前）；**不改** runtime 行為、preset、threshold、weights、benchmark claim，也沒有實作任何打包步驟。
> 基準：Phase B 終點＝PR-12（#524，`main`＝`856ccccb`），[native_runtime_resolved_config.md](native_runtime_resolved_config.md) §16；範圍詞彙與 G1／G2 定義沿用 [native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §0–§1。
> 本文只定義 Phase C 的**決策、邊界與順序**；沒有執行 inference、沒有發任何數字。§3 的大小與 §4 的授權讀法是對現有檔案的靜態讀取，不是法律意見。

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
| 簽章形式 | PR-C3 只產 package digest（sha256）；簽章機制（key 管理、工具）在 PR-C4 前由 owner 定（C-D5，見 §6） | Phase B 的 trust root＝Git commit＋runtime-identity publication，對 release 不夠 |

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

1. **operator library 的 `libnvrtc.so.13`**。operator library 由 `detector_host.cpp:198` 以 `dlopen` 載入；它的 NEEDED 有 `libnvrtc.so.13`，而 `saccade_track` 與 `libtorch_cuda.so` 都不 NEED 它。dynamic loader 解析這個 NEEDED 時只看 operator library 自己的 RUNPATH（7 個凍結的絕對路徑）、`LD_LIBRARY_PATH` 與系統路徑，**不看** `saccade_track` 的 `$ORIGIN/../lib`。拿掉 `LD_LIBRARY_PATH` 後，在別的機器上會解析失敗。不能 `patchelf`（會改動 attested 位元組）。候選修法：
   - (a) 讓 `saccade_track` 在連結時 NEED `libnvrtc.so.13`，使它在啟動時從 `$ORIGIN/../lib` 載入，`dlopen` 時以 SONAME 命中已載入物件。operator library 不變；載入集合不變（PR-12 已載入 nvrtc）；`saccade_track` 的 NEEDED 從 13 變 14，POST_BUILD 檢查要跟著更新。**預設提案。**
   - (b) 以 `$ORIGIN`-relative RUNPATH rebuild operator library 並 re-attest：違反 §2 預設與 C-D2，需 owner 另外決定。
2. **build host 會假通過**。operator library 的絕對 RUNPATH（`/home/ray/developer/ai/saccade/...`）在 build host 上存在，所以 host 上「沒有 `LD_LIBRARY_PATH` 也能跑」不能當證據。驗收只認乾淨容器，而且要在容器裡 `env -u LD_LIBRARY_PATH`。
3. **bundled 物件之間的解析**。27 個物件中，凡是有第三方 NEEDED 的，RUNPATH 都含 `$ORIGIN`（例如 `libtorch_cuda.so`、`libcublas.so.13`、`libcusparse.so.12`、cuDNN 各檔），平放在 `lib/` 即可互相解析；沒有 RUNPATH 的（`libcudart`、`libnvrtc`、`libnvJitLink`、`libnccl`、`libcufile`、`libcupti`）都沒有第三方 NEEDED。PR-C1 的靜態檢查要對 `lib/` 每個物件逐一確認 NEEDED 在 `lib/`＋base system＋driver 內閉合，而不是只看 `saccade_track`。
4. **wheel RUNPATH 會指到包外**。torch 系列的 RUNPATH 在 `$ORIGIN` **之前**列了 `$ORIGIN/../../nvidia/{cudnn,nvshmem,nccl,cusparselt,cu13}/lib`；cuDNN 有 `$ORIGIN/../../{cublas,cuda_nvrtc,cu13}/lib`；`libnvinfer` 有 `$ORIGIN/../nvidia/...`、`$ORIGIN/../tensorrt_*_libs`。在 `<prefix>/lib/` 的配置下，這些路徑落在 `<prefix>/..` 或 `<prefix>/` 底下，也就是包外。若使用者機器上剛好存在這些目錄，loader 會先載入那裡的同 SONAME 物件。因為 C-D1 要求位元組相同，不能改寫這些 RUNPATH。PR-C1 的處理：(a) `saccade_track` 啟動後檢查 `/proc/self/maps`，每個第三方共享物件都必須來自 `$ORIGIN/../lib`，否則 fail-closed（沿用 `detector_host.cpp:137` 讀 `/proc/self/maps` 的既有機制；目前它只回報 `python_libraries_mapped`，不 fail）；(b) 負控制：在 `<prefix>/../nvidia/cu13/lib` 放一份不同的 `libcublas.so.13`，必須被 (a) 擋下。
5. **TensorRT 是 cu12 build、其餘是 cu13**。PR-12 已在這個混合組態下驗收；Phase C 不改，記為 named limit。

---

## 6. Phase C 拆分

每個 PR 只改表內列出的範圍。輸出驗收一律是同機器、對 PR-12 正式 run 逐位元組相同（txt 與 trace 的 sha256 7/7），不發新的 benchmark claim，不量 FPS。

| PR | 內容 | 驗收 | 依賴 |
|:--|:--|:--|:--|
| **PR-C1** | **bundle**：committed 的第三方集合清單（`shipping/third_party_set.json`：SONAME、sha256、來源 wheel＋版本、授權檔）；`cmake --install --component shipping` 從清單複製到 `lib/` 並逐檔比對 sha256；§5.1 的 nvrtc 修法；§5.4 的載入來源自檢；`check_shipping_tree.py static` 改成閉包必須在 tree＋base＋driver 內完成（不再接受外部第三方目錄）；`licenses/` 與 `THIRD_PARTY.md` | 靜態：G2-1 閉包對 tree 內每個 ELF 完整、沒有 Python；G2-3；RUNPATH 規則（operator library 仍是唯一例外）；`lib/` 的 sha256 集合＝PR-12 的 27 項。乾淨容器 `env -u LD_LIBRARY_PATH`：EXACT，且與 PR-12 正式 run 7/7 相同；strace 的開啟集合＝PR-12 集合，全部來自 tree。負控制至少：少一個 lib、lib 多一個位元組、設回 `LD_LIBRARY_PATH` 指向一份不同的 lib（必須不被使用或被偵測）、移除 nvrtc 修法、§5.4 的包外 `libcublas.so.13` | — |
| **PR-C2** | **CLI surface**：決定開發選項去留（§2 預設提案）；release build 的 `saccade_track` 不含被移除的選項；`--help` 與錯誤訊息整理；不改任何 stage 計算 | 被移除的選項在 release binary 上是 unknown argument；保留的選項行為不變；release binary EXACT 且與 PR-C1 7/7 相同。若 binary 位元組改變，G2 靜態檢查與容器驗收重跑 | PR-C1 |
| **PR-C3** | **package＋atomic staging**：tarball（命名含版本、`linux-x86_64`、`cu13.0`、`trt10.16`、`sm120`、`glibc2.39`）；`MANIFEST.json`（每檔 sha256、版本 pin、SM 清單、glibc baseline、source commit、attestation 與 lineage 引用、runtime-identity 座標）；package digest；POSIX sh 安裝器：解到同檔案系統的 staging → 依 MANIFEST 逐檔驗證 → `mv` 一次 rename；任何失敗都刪 staging、目標不動；只用 base system 工具（`sh`、`tar`、`sha256sum`） | 從 tarball 安裝到乾淨容器，EXACT 且與 PR-C2 7/7 相同；安裝器負控制：tarball 損壞、單檔被改、目標已存在、磁碟寫入中斷（模擬）都讓目標目錄維持原狀；PR-12 的 N2（非 atomic）不再成立 | PR-C2 |
| **PR-C4** | **release readiness**：C-D5 簽章機制落地；§4 授權確認的結論寫入 `THIRD_PARTY.md`；`README.txt`（GPU＝sm_120、driver 需求、glibc ≥ 2.39、named limits）；最終驗收紀錄 | owner 確認 §4；簽章可被驗證、竄改後驗證失敗；最終 tarball 從乾淨容器安裝到跑完一次完整 7-seq EXACT | PR-C3、C-D5 |

**仍待 owner 的決策**

- **C-D5 簽章機制**（PR-C4 之前）：例如 minisign／GPG detached signature，或 Sigstore（cosign keyless）。牽涉 key 管理，不是工程預設可以決定的。
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
- 沒有決定簽章機制（C-D5）、其他 SM、其他平台。
- 沒有執行任何 run、沒有量 parity、FPS 或精度。
