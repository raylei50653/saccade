<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-09 -->
<!-- doc-module: cross -->

# CI / pytest / build timing baseline (#509)

本文件記錄 [#509](https://github.com/raylei50653/saccade/issues/509) 前段的工程耗時基線，source 為 `83c9bef0029ab4b969a39f52ab592c47f74d34f6`。跨 Issue 排序、checkpoint 與下一主線只在 [#550](https://github.com/raylei50653/saccade/issues/550)；本文不承載第二張 roadmap。這次交付是 measurement snapshot，尚未變更任何 workflow、required check、pytest 選取、cache、runtime 或 evidence gate，也不提出加速百分比。

可重新計算的 job/step 資料、checkout 身分、cache 證據及本機 phase 結果在 [evidence JSON](ci_timing_baseline_509.json)。原始 API、logs、JUnit XML 與 harness 保留於 `results/509_ci_baseline/20261009/`，不納入版本控制；JSON 保留它們的 SHA-256 與重取方法。這是工程過程資料，`doc-promotion: none`；任何後續優化決策需引用這份 baseline 並另附可比較的前後證據。

## Sampling and definitions

- 遠端：2026-10-08–09 的 3 組成功 main runs 與 3 組不同的代表性 PR runs，每組各含 CI 與 C++（共 12 runs）；分開報告 main / PR cohort。
- 這是兩個 routine CI workflows 的成功 cohort；不涵蓋 Docker Build、手動 H0／runtime-identity GPU qualification，也不估計失敗、取消或重試成本。未觸發的工作沒有 PASS 宣稱。
- 四份 workflow/config/lock files 在實際 checkouts 相同：各 checkout 的 `ci.yml`、`cpp_build.yml`、`pyproject.toml`、`uv.lock` 身分見 JSON。PR 的 `run.head_sha` 與 checkout 的 synthetic merge SHA 分開記錄。
- 程式／測試仍有變化，遠端 passing tests 為 5120–5158；這是相同四份 workflow/config/lock files 下的 baseline，並非同一 head 的重跑，更不是受控優化前後實驗。runner image、網路與 cache 壓力未固定。
- Workflow wall = `max(job.completed_at) - run.created_at`，含排隊與編排；job = `completed_at - started_at`；step 同理。API 時間精度是秒，0 秒不表示沒有工作；workflow completion housekeeping 不包含在此定義。
- 各欄各自取中位數；**不能相加各 step/job 中位數來當 total 中位數**。critical branch 依最後完成的 job 與現行 `needs` 確定，queue/orchestration gap 不當成 test execution。

## Remote baseline

| cohort / source head | actual checkout | CI run（wall 秒） | C++ run（wall 秒） |
|:--|:--|:--|:--|
| main `83c9bef0` | `83c9bef0` | [37878975871](https://github.com/raylei50653/saccade/actions/runs/37878975871) · 786 | [37878975874](https://github.com/raylei50653/saccade/actions/runs/37878975874) · 486 |
| main `54be85ad` | `54be85ad` | [37773867908](https://github.com/raylei50653/saccade/actions/runs/37773867908) · 677 | [37773867912](https://github.com/raylei50653/saccade/actions/runs/37773867912) · 583 |
| main `24dab817` | `24dab817` | [37764791730](https://github.com/raylei50653/saccade/actions/runs/37764791730) · 892 | [37764791793](https://github.com/raylei50653/saccade/actions/runs/37764791793) · 491 |
| PR `3c83cb96` | `0c808139` | [37877523166](https://github.com/raylei50653/saccade/actions/runs/37877523166) · 801 | [37877523223](https://github.com/raylei50653/saccade/actions/runs/37877523223) · 606 |
| PR `2ba322a2` | `e3dd69a9` | [37769937398](https://github.com/raylei50653/saccade/actions/runs/37769937398) · 870 | [37769937438](https://github.com/raylei50653/saccade/actions/runs/37769937438) · 615 |
| PR `1f83895b` | `4c23dbd0` | [37763391614](https://github.com/raylei50653/saccade/actions/runs/37763391614) · 744 | [37763391668](https://github.com/raylei50653/saccade/actions/runs/37763391668) · 491 |

全數是 attempt 1、completed success。C++ checkout 以 REST archive 取得；log 的七碼 resolved version 另以 GitHub commit API 解析到完整 SHA，並比對它的四份輸入檔。

| 秒；中位數 [min, max] | main n=3 | PR n=3 |
|:--|--:|--:|
| CI workflow wall | 786 [677, 892] | 801 [744, 870] |
| C++ workflow wall | 491 [486, 583] | 606 [491, 615] |

| job 秒；中位數 | main n=3 | PR n=3 |
|:--|--:|--:|
| lockfile | 11 | 10 |
| shipping-config-loader | 28 | 21 |
| ruff | 217 | 176 |
| mypy | 275 | 270 |
| contracts | 661 | 669 |
| pytest | 770 | 787 |
| lint-typecheck | 3 | 2 |
| C++ build | 488 | 603 |

`lockfile → pytest` 是 5/6 CI runs 的 critical branch；main `54be85ad` 的最後完成者是 `contracts`（661 秒，pytest 592 秒）。C++ workflow 的最後完成者都是單一 `build` job。不能只優化最快一次或把 contracts 的成本當成可忽略。

| 主要 step 秒；中位數 | main n=3 | PR n=3 |
|:--|--:|--:|
| pytest execution（含 coverage、collection） | 636 | 612 |
| H2 Measurement Archive Corpus | 407 | 511 |
| pytest disk cleanup / dependency sync | 65 / 57 | 83 / 57 |
| contracts disk cleanup / dependency sync | 100 / 85 | 52 / 55 |
| C++ container initialization / APT | 58 / 58 | 94 / 55 |
| C++ dependency sync | 128 | 235 |
| C++ first configure + build | 95 | 88 |
| shipping no-OpenCV configure + build | 123 | 107 |

遠端 CMake configure/build 在同一 step；JSON 中依 `Build files have been written` log marker 分出的 configure estimate 只用於定位，不當成獨立、精確的 compiler timing。

## Cache and repeated work

- 30/30 CI `setup-uv` jobs 有 exact-key restore（每個 CI run 的 lockfile、ruff、mypy、contracts、pytest）。四個後續 jobs 仍各自完整 `uv sync --refresh-package tensorrt-cu12-libs`，並清除 TensorRT cache；cache restore 不等於零 dependency setup。
- 6/6 C++ runs 的 APT 與 ccache 都 exact-key restore；全部 logs 同時記錄 `/github/home/.cache/uv` 沒有 cache。`ccache -s` 的 restored cumulative counters 不是本次 run hit ratio，本文不宣稱該比率。
- ruff、mypy、contracts、pytest 各自做 APT、disk cleanup、uv setup/sync；這是可量到的重複 setup。是否能改成共享 producer、縮小 lint 依賴或改 cleanup，必須另驗證 tool/dependency/trust 邊界。
- C++ 在同一 checkout configure/build 兩次；第一組與 no-OpenCV 組的 OpenCV/SM coverage 不同。現有雙 build 的義務由 YAML 明示，不能把它直接當成可刪的重複。
- contracts 與 pytest 都觸及 H2 verifier；它們的輸入、負控制與語義尚未證明等價。保留 archive 歷史資料的獨立檢查。

## Local pytest baseline

WSL2、32 logical CPUs、RTX 5070 Ti Laptop，既有 `.venv` 與 native extension；精確 versions / source hashes 見 JSON。使用原 `pre_push` selection（`tests/ -q --ignore=tests/benchmarks`，configured `not research`），加上 durations/JUnit instrumentation，沒有 coverage。每次 fresh Python process，在 `machine-bench` lease 下序列量測，source/head/worktree 保持乾淨。

| 秒 | rep 1 | rep 2 | rep 3 | median [min, max] |
|:--|--:|--:|--:|--:|
| collection-only external wall | 8.130 | 7.959 | 7.882 | 7.959 [7.882, 8.130] |
| full pytest external wall | 300.201 | 299.640 | 298.988 | 299.640 [298.988, 300.201] |
| pytest reported elapsed | 295.46 | 294.86 | 294.00 | 294.86 [294.00, 295.46] |

三次 full 均 exit 0：**5362 passed、52 skipped、105 deselected、5 xfailed**；collection 均 5419/5524 selected，105 deselected。JUnit 的 57 skipped 包含 5 xfailed；不把它誤當 57 個一般 skip。

collection-only 是另一次 process，不能把兩欄 median 的差當成精確 execution phase。全套 external wall 含 Python 啟動／退出、collection、fixture/setup、call、teardown、report；pytest reported elapsed 的邊界較窄。前面的未取得 lease 首輪保留為 exploratory，排除於上述 median。重複出現在三輪 slowest-30 的 setup/call：resolved-config export setup 27.90 秒、preseal independent expansion call 21.82 秒、real-package licence comparison call 21.32 秒（各取 median；不是跨 run 可重用的 PASS）。既有 pyc/filesystem/GPU caches 已暖，Windows host 與 ambient services 未隔離。這些數字與 GitHub coverage/fallback runner 不同，不是它們的加速結果。

## Local build baseline

三個獨立 fresh build directories，source 同為 `83c9bef0`，整批持有 `machine-bench` lease。沿用 CI `shipping-config-loader` 的 warning-as-error configure、parallel build、CTest argv；另量 incremental no-op。toolchain／input inventory、command/log hashes 在 JSON 與原始 evidence。

| 秒 | rep 1 | rep 2 | rep 3 | median [min, max] |
|:--|--:|--:|--:|--:|
| configure | 1.601 | 1.596 | 1.591 | 1.596 [1.591, 1.601] |
| fresh build | 3.461 | 3.408 | 3.427 | 3.427 [3.408, 3.461] |
| CTest | 3.353 | 3.418 | 3.357 | 3.357 [3.353, 3.418] |
| incremental no-op build | 0.090 | 0.093 | 0.091 | 0.091 [0.090, 0.093] |

每個 phase exit 0；三次 CTest 都 6/6 PASS。no-op 沒有 compile/link，products hashes 全部保持相同。此結果只覆蓋 CUDA-free standalone loader/config/MOT targets，未建置現有 native runtime、修改 attestation 或安裝 package；完整 CUDA C++ cost 使用上面的 remote baseline。fresh build directory 不代表 filesystem/compiler cache 已清空。首個 attempt 的三個 phase 均 exit 0、CTest 6/6，但 harness 未識別本機 `100% tests passed out of 6` 摘要而停止；原始 attempt 與原 harness 保留為 `build-incomplete/`，排除於以上 n=3。修正僅在量測 harness，之後用全新目錄重跑。

## What this evidence supports

可優先調查 pytest execution 與 H2 archive corpus 的實際工作，其次是 repeated setup 和 C++ dependency sync。這次沒有修改其實作或取得速度改善。下一個有界提速 PR 必須具名一個瓶頸、維持原 selection/coverage/負控制，並附同條件前後 evidence。

#509 後段 manifest/planner、routing、derived result reuse 與 cache policy 仍依 [#541](https://github.com/raylei50653/saccade/issues/541) 的相關 contract→check handoff；WSLc substrate 仍由 [#512](https://github.com/raylei50653/saccade/issues/512) 決定。此次歷史 remote PASS 與本機 timing 都不替代新 commit 的 required execution。

## Reproduce

遠端每個 run 的精確 ID、attempt、checkout 與 blob hash 在 JSON。收集時使用：

```sh
gh api repos/raylei50653/saccade/actions/runs/<run-id>
gh api 'repos/raylei50653/saccade/actions/runs/<run-id>/attempts/<attempt>/jobs?per_page=100'
gh run view <run-id> --attempt <attempt> --log
git rev-parse '<checkout-sha>:<path>'
git show '<checkout-sha>:<path>'
```

本機量測的精確 argv 與已記錄的環境／工具清單在 JSON／原始 harness。每個測試或 build 使用 fresh process，正式 timing 經 [resctl machine-bench](../WORKTREE_RESOURCES.md) 序列執行；使用已有環境，沒有清除全機 filesystem／compiler／Python caches。重新執行須在該 source checkout；它不是任意未來 head 的 baseline。

```sh
.venv/bin/python tools/resctl.py run machine-bench -- \
  .venv/bin/python -m pytest tests/ --collect-only -q --ignore=tests/benchmarks
.venv/bin/python tools/resctl.py run machine-bench -- \
  .venv/bin/python -m pytest tests/ -q --ignore=tests/benchmarks --durations=30 --junitxml=/tmp/509-pytest.xml
.venv/bin/python tools/resctl.py run machine-bench -- \
  .venv/bin/python results/509_ci_baseline/20261009/build/measure.py --output-dir /tmp/509-replay-fresh
```

build harness 會序列量測三個 fresh build directories 的 configure、build、6 個 CTests 與 incremental no-op build；輸出位置必須是尚未量過的新目錄。pytest 兩個 argv 各重跑三次，分別記錄 wall、summary 與 JUnit，不能只執行一次當成完整 protocol。本機 standalone loader 不代表完整 CUDA C++ build 的數字；後者以遠端兩組 build 的 source/coverage 為準。
