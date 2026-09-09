# Runtime Coordinate 候選捕捉與出版 Runbook

Date: 2026-09-09（[issue #334](https://github.com/raylei50653/saccade/issues/334) /
[ADR 022](../../decisions/022-check-taxonomy-and-publication-lag.md) PR 3）

改到 `decision_relevant` 或 `identity_semantics` 路徑之後，
`docs/reference/runtime_identity.generated.json` 會落後，pre-push §4.11 與
`test_the_published_coordinate_is_complete_and_static_axes_are_current` 會擋下來。
本 runbook 是**唯一受支援的補救路徑**：怎麼在座標**還沒 current** 的情況下捕捉一份
完整候選、怎麼審查、怎麼 promote，以及 math-model 要 re-audit 時怎麼做。

三件事本 runbook **不**做：不繞過任何 hook、不自動刷新 digest、不把 probe 相等
當成 semantic equivalence（`equivalence.state` 永遠是 `unproven`）。

## 0. 這條路徑要解的 chicken-and-egg

`.github/workflows/runtime_identity.yml` 有一步叫 **"Coordinate must already be
current"**，排在產 probe 之前。它能*驗證*一份既有出版，但座標一旦落後，它就無法
收集「更新座標所需的證據」——要 current 才肯跑，跑了才會 current。

兩條出口：

| 路徑 | 檔案 | 前提 |
|---|---|---|
| 本機 controlled host（**目前實際使用的**） | 本 runbook §2 | 需下列 §1 資源 |
| CI sibling workflow | [`runtime_identity_candidate.yml`](../../../.github/workflows/runtime_identity_candidate.yml) | 需一台已註冊的 `[self-hosted, linux, x64, h0-qualification]` runner |

兩條都**沒有** "must already be current" 那一步。撰寫本文時 repo 上註冊的
self-hosted runner 數為 0（`runtime_identity.yml` 最後一次成功是 2026-07-30，之後
四次 dispatch 都排隊 24h 後自動取消），所以 CI 那條**現在跑不動**；workflow 先放著，
等 runner 回來即可用。本機那條是現行做法，見 commit `4b24db6f`。

## 1. Controlled host 需要什麼（資源，不是權限）

> Runtime coordinate **不授予任何 research authority**——`runtime_identity.yml`
> 自己的檔頭就寫明它 publishes nothing、seals nothing、grants no execution
> authority。因此一台跑不出候選的機器是**缺資源**，不是「沒有角色／未被授權」。
> 這兩件事在故障排除時不可混為一談。

需要的具體資源：

1. **十二份 runtime input**（`h2_runtime_inputs.py` 的 manifest 逐一 content-bind，
   缺一份就 fail-closed，不會給你一份局部 manifest）：
   - identity fixture：`datasets/MOT17/train/MOT17-09-SDP`（`IDENTITY_SEQUENCE`）
   - measurement fixtures：`MOT17-02/04/05/10/11/13-SDP`（`MEASUREMENT_SEQUENCES`）
   - `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt`
   - `runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt`
   - `models/yolo/yolo26m.pt`
   - `models/yolo/yolo26m_backbone_640_best.engine`
   - `models/yolo/mamba_head_26m.engine`
2. **能 build native extension 與 TensorRT scan plugin 的 CUDA/TensorRT toolchain**。
   `environment_axis()` 記錄的是**觀測到的** Torch / CUDA / cuDNN / TensorRT / device
   capability，所以 `environment` 這一軸只在本機（`--strict`）比對；一般 CPU runner
   會把它報成 unresolved warning，而不是製造假 drift。
3. build 目錄，透過 `--build-dir`（或 `SACCADE_BUILD_PATH`）指給 probe。

## 2. 捕捉一份候選（不需要座標 current）

```bash
cmake -S . -B build/runtime_identity -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime_identity --parallel

.venv/bin/python scripts/tools/build_runtime_identity.py \
  --run-probe --build-dir build/runtime_identity \
  --emit "$SCRATCH/runtime_identity.candidate.json" --require-complete
```

- `--run-probe` 會**當場重跑**那個 bounded MOT17-09 probe。ADR 022 §3 決定 2：
  **probe 永不重用**，每次 republication 都要 fresh capture。不要拿舊的
  `behavior.json` 餵 `--probe-from` 來省時間。
- 只給 `--build-dir`（不給 `--runtime-inputs-from`）時，builder 會自己建 runtime-input
  manifest，所以不需要另外跑一次 `h2_runtime_inputs.py`。
- 成功的候選會有 `publication_complete: true`、五軸皆非 null。

> ⚠️ **`--emit` 一律指向 scratch，絕不直接指向
> `docs/reference/runtime_identity.generated.json`。**
> `--require-complete` 目前排在 `--emit` 寫檔**之後**
> （`scripts/tools/build_runtime_identity.py`，由
> `tests/contract/test_adr_022_check_taxonomy.py::test_case4_require_complete_reports_after_it_has_already_written`
> 釘住）。指向 canonical 路徑時，它會**先把那份完整出版覆蓋掉**，然後才回非零——
> 你會同時失去舊出版與新出版。修這個順序屬於 ADR §8 的 PR 5；在那之前，「emit 到
> scratch」就是唯一的守衛。

## 3. 審查與 promote

Promote 是**另開一支 review PR**，不是捕捉步驟的延伸。

1. **Archive 現行 canonical**，再覆蓋：

   ```bash
   .venv/bin/python - <<'PY'
   import datetime, json, pathlib, shutil
   src = pathlib.Path("docs/reference/runtime_identity.generated.json")
   digest = json.loads(src.read_text())["coordinate"]["implementation"][:16]
   today = datetime.date.today().isoformat()
   dst = src.parent / "runtime_identity" / "archive" / f"runtime_identity.{today}.{digest}.json"
   shutil.copy2(src, dst)
   print(dst)
   PY
   ```

   ADR 022 §3 決定 5：archive 進
   [`docs/reference/runtime_identity/archive/`](../runtime_identity/archive/README.md)，
   **不只靠 git history**。

2. 把審過的候選複製到 canonical 路徑。

3. **逐軸交代 delta**，寫進 commit message。`4b24db6f` 是範本：哪幾軸動了、檔案數
   從幾變幾、每一項變動的來源是什麼、哪幾軸沒動、probe 是**重跑**而非沿用——並明說
   probe 相等**不**構成 equivalence 主張。摘要欄不得比細節欄寬。

4. 驗證：

   ```bash
   .venv/bin/python scripts/tools/check_runtime_identity_staleness.py   # 應 exit 0
   ```

### 什麼仍然必須 exit 1

Promote **不能**用來讓下列任何一項通過（ADR 022 §4）：

- 有 `current` binding 卻帶 source lag（假 current-attestation）
- 把 binding 改寫成對一份落後出版為 `current`
- 用不完整 publication 覆蓋完整 canonical
- `research_lock open`（量測＝把出版當 substrate，仍要求出版 current）
- `equivalence` 被改成不是 `unproven`

## 4. Math-model re-audit

`scripts/tools/check_math_model_source_attestation.py` 有兩臂（ADR 022 §6、PR #361）：

| Mode | 問什麼 | 何時失敗 |
|---|---|---|
| `development`（預設） | **歷史完整性**：被接受的紀錄還是當初那份嗎 | 動到 `math_model.md`、audit 紀錄、manifest、或 audited ref 的 bytes |
| `attested` | 再加上**current-HEAD 主張**：文件描述的是現在的 code | 任一 audited source anchor 的工作樹 bytes 變了 |

日常改 source anchor 只會弄壞 `attested`，而那**不是**預設——ordinary development
不欠一份 re-audit。要主張文件描述當前 HEAD 時才跑：

```bash
.venv/bin/python scripts/tools/check_math_model_source_attestation.py --mode attested
```

真的要 re-audit 時，**scoped、逐 anchor**，同一支 PR 內：

1. 就改動的那個 anchor，比對它支撐的 `docs/reference/math_model.md` 小節，確認
   transcription 仍然成立（八個 anchor 見 checker 內 `AUDITED_SOURCE_PATHS`）。
2. **append** 一份新的 audit 紀錄到 `docs/research/tracker-decision/audit/`；舊紀錄
   永不改寫、永不刪除。
3. 更新 manifest `docs/reference/math_model_source_attestation_v1.json`，**並且**更新
   `check_math_model_source_attestation.py` 內 hard-code 的 digest
   （`ATTESTED_MODEL_SHA256` / `ATTESTED_AUDIT_SHA256` / `AUDITED_SOURCE_REF`）。
   digest 之所以 code-owned，就是為了讓 manifest **無法自我重新簽名**；改 checker 會
   走同一道 code review gate。
4. 沒有 `--fix`，沒有靜默刷新。

> `docs/reference/math_model.md` 的 bytes 本身是 attested 的。**任何 PR 都不要順手
> 動它**——動了就欠一次完整 re-audit。

## 5. 分類提醒

動工前先問 `scripts/tools/h2_path_partition.py --classify <path>`：

- `decision_relevant` / `identity_semantics` → 這次改動會讓座標落後，需要本 runbook
- `plumbing_only` / `non_execution` → 不動任何已出版軸，不需要 republication

## 相關

- [ADR 022: 檢查分類法與出版落後預設](../../decisions/022-check-taxonomy-and-publication-lag.md)
- [Archive 目錄約定](../runtime_identity/archive/README.md)
- 候選 workflow：[`runtime_identity_candidate.yml`](../../../.github/workflows/runtime_identity_candidate.yml)
