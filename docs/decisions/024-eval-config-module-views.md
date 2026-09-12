<!-- doc-status: accepted -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-12 -->
<!-- doc-module: cross -->

# ADR 024: EvalConfig Phase 4 module-view disposition

## Status

**Accepted** (2026-09-12)

**Terminal verdict: `retain_current_views`**

這是 [issue #139](https://github.com/raylei50653/saccade/issues/139) 的處置紀錄。它決定現有 Phase 4B module views 的去留，並只授權該判決所證明的有限後續。本文**不**授權繼續 bulk 遷移、**不**授權 rollback、**不**授權復活或 cherry-pick 已退役的歷史分支。

對照時點 2026-09-12，`main` = `000d98c3`。

歷史證據（非授權來源）：

- `132752f2` — 把 parameter registry、frozen views、以及 Core/Detection/Geometry/Motion 的 consumer 遷移落到 `main`
- `e8aa9612` — 對 `132752f2` 的 revert；**不是** `HEAD` 的 ancestor，也沒有活分支包含它

---

## 1. Context

`EvalConfig` 仍是單一扁平 dataclass（366 欄）。`__post_init__()` 另外組出八個 frozen view：

`CoreView`、`DetectionView`、`GeometryView`、`MotionView`、`ReIDView`、`SemanticView`、`TriggerView`、`LifecycleView`

這些 view 是 construction-time **value copy**，不是第二個可寫入的 source，也不是 live proxy。runtime 仍然把整個 `EvalConfig` 傳進 evaluator / pipeline / stages / cpp_runner。

#139 要回答的問題不是「要不要重新引入已刪除的設計」，而是：現有 shape 是否縮小責任面；若否，剩餘扁平 consumer 該不該繼續遷。

---

## 2. Inventory（對照 `000d98c3`）

### 2.1 Views 與投影

| View | 欄位數 | EvalConfig 投影 | 備註 |
|---|---:|---|---|
| `CoreView` | 23 | 全部是 `EvalConfig` 欄 | I/O 與 tracker 門檻 |
| `DetectionView` | 56 | 全部是 `EvalConfig` 欄 | 前處理、NMS、tiling、FP filter |
| `GeometryView` | 56 | 全部是 `EvalConfig` 欄 | Kalman / OAO / association |
| `MotionView` | 10 | 全部是 `EvalConfig` 欄 | 與 `MotionConfig` 完全對齊 |
| `ReIDView` | 16 | 全部是 `EvalConfig` 欄 | backbone / crop / lazy ReID |
| `SemanticView` | 51 | 全部是 `EvalConfig` 欄 | bank / relink scoring |
| `TriggerView` | 0 | 無 | 型別存在；`__post_init__` **不**設定 `cfg.trigger`。trigger 參數走 `kwargs` |
| `LifecycleView` | 129 | 全部是 `EvalConfig` 欄 | birth / relink / interpolate / Cheb-GR |
| **projected** | **341** | 彼此不重疊 | |
| `EvalConfig` | 366 | | `_DEFAULTS` 也是 366 |
| **unprojected** | **25** | 見 §2.2 | |

View 欄位沒有跨 view 重複，也沒有「view 有、EvalConfig 沒有」的欄。

`scripts/eval/config/*.py` 的模組 dataclass（argparse / YAML owner）**不是**這些 view。兩者 overlap 但不相等：module dataclass 還包含 CLI-only / kwargs-only 欄（例如 `TriggerConfig` 21 欄、`SemanticConfig` 比 `SemanticView` 多 29 欄）。View 只投影已經升上 `EvalConfig` 的扁平欄。

### 2.2 未投影的 `EvalConfig` 根欄

`output_root`、`seqs`、`kwargs`、`preprocess_modes`、`crop_hw`、`gmc_enabled`、`duplicate_suppression`、`reid_enabled`、`reid_engine`、`reid_work_enabled`、`reid_budget_raw`、`use_semantic_mode`、`use_tracker_reid`、`need_reid_enabled`、`appearance_bank_enabled`、`id_stability_filter_enabled`、`lifecycle_merge_enabled`、`geometry_suspect_support_score`、`occ_audit_bank_reference`、`occ_audit_bank_n`、`occ_vel_weight`、`pose_box_expand`、`pose_expand_ankle_conf`、`pose_expand_margin`、`pose_expand_flat_aspect`

這些是 I/O、衍生旗標、或仍走 `kwargs` 的實驗/相容欄，不是「忘了放進 view」。

### 2.3 Live nested consumers

Runtime（`src/saccade/perception/eval/`）對 nested view 的使用集中在四組：

| View | Runtime nested 讀取 | 主要檔案 |
|---|---|---|
| `detection` | 熱路徑 | `stages.py`、`evaluator.py`、`pipeline.py`、`cpp_runner.py` |
| `geometry` | 熱路徑 | 同上 |
| `core` | 熱路徑 | 同上 |
| `motion` | 熱路徑 | `pipeline.py`（relink 組態） |
| `reid` | **無** runtime nested | 只在投影測試讀 `cfg.reid.*` |
| `semantic` | **無** runtime nested | 只在投影測試讀 `cfg.semantic.*` |
| `lifecycle` | **無** runtime nested | 測試 `cfg.lifecycle.occ_audit_chebgr_probe` |
| `trigger` | 無 | 型別為空，且未掛上 `cfg.trigger` |

Headline contract 已依賴 nested 名稱（`cfg.geometry.kalman_r_scale`、`cfg.geometry.occ_*`、`cfg.core.match_thresh`）。這四組不能當「還沒落地的設計」看待。

Core / Detection / Geometry / Motion 在 runtime 的扁平殘留實質為零：剩下的扁平讀取只出現在 pin `flat == view` 的測試，以及 `cpp_runner.py` 的 `getattr(cfg, ...)` 相容路徑。

---

## 3. Remaining flat-field consumers

分類規則與 #139 相同。

### 3.1 Intentional compatibility / root-field use

- `parse_eval_config`、`_DEFAULTS`、argparse、YAML merge：構造面本來就是扁平的。
- `cfg.kwargs`：trigger 與未升格實驗參數的袋子；`TriggerView` 刻意為空。
- §2.2 的 25 個未投影根欄。
- `cpp_runner.py` 的 `getattr(cfg, "<flat>", default)`：對部分物件 / 缺欄的相容讀取，不是第二套政策。
- `tests/unit/eval/test_eval_utils.py` 等 parser 測試直接 assert 扁平欄（例如 `cfg.tiling`）：測的是 `parse_eval_config` 的公開面。
- 投影測試同時讀扁平與 nested：這是 invariant，不是未完成遷移。

### 3.2 Unfinished migration candidates（存在，但不授權繼續做）

Runtime 仍用扁平名讀已經投影進 view 的欄，主要是：

- `pipeline.py`：semantic / lifecycle / 部分 reid（約 100 個 unique 欄）
- `evaluator.py`：`reid_mode` 家族、`async_reid`、`lazy_reid_*`、`stage2_quality_*`、post-lifecycle 外觀欄
- `stages.py`：`reid_mode`、`async_reid`、bank occlusion、duplicate-suppression 的投影欄

數量上這就是歷史 Phase 4C 沒做完的三組（ReID / Semantic / Lifecycle）。#139 禁止復活那次 bulk 遷移。

### 3.3 Not worth migrating

在 `retain_current_views` 下，§3.2 整組標成 **not worth migrating as a program**。繼續改名不會縮小 `EvalConfig` 的責任面，只會再複製一次 100+ 欄的存取路徑。Trigger 維持 `kwargs`。未投影的 25 欄維持根欄，不為了「齊套」而硬塞進 view。

---

## 4. Do the views reduce responsibility surface?

**沒有。** 它們沒有把 366 欄的 `EvalConfig` 拆成可獨立傳遞的模組契約。

證據：

1. 每個 runtime 入口仍接收完整 `EvalConfig`。沒有函式簽名把責任限制在單一 view。
2. 八個 view 合計 341 欄，是扁平欄的第二份 construction-time 副本。`LifecycleView` 自己就有 129 欄。
3. ReID / Semantic / Lifecycle view 在 runtime **幾乎沒有 nested 讀取**；它們目前是測試用投影，不是模組邊界。
4. View 在 `__post_init__` 複製值。之後若改扁平欄，view 不會更新。可寫入的 source 仍是扁平 `EvalConfig`（加上會被原地改的 `kwargs`）。

它們**有**做到的，只是給已經遷移的四組 consumer 一個穩定的 nested 名稱（且 headline contract 已依賴那些名稱）。那是命名空間，不是責任隔離。

因此「把剩下的 semantic / lifecycle / reid consumer 遷完」不能當成 #139 的完成條件。那會是另一次 bulk rename。

---

## 5. Alternatives

| 判決 | 為何不選 |
|---|---|
| `finish_bounded_migration` | 真正未完成的是 ReID / Semantic / Lifecycle 整組，不是單一可獨立驗證的 consumer group。Core/Detection/Geometry/Motion 在 runtime 已完成。再遷一組只是 rename，不改變 §4 的結論。 |
| `rollback_existing_views` | 熱路徑與 headline contract 已讀 `cfg.detection` / `cfg.geometry` / `cfg.core` / `cfg.motion`。Rollback 必須另開 focused removal，不是本 issue 的授權；也不值得為了清掉未使用的四個 view 去動 300+ 處已遷移存取。 |
| `prune_unconsumed_views`（只移除沒有 runtime consumer 的 view） | 可行，且**不必**動 hot path，因此不能靠上一列的「300+ 處」順帶排除；但仍不採用，理由見 §5.1。 |

### 5.1 被考慮並拒絕的第四方案：只移除沒有 runtime consumer 的 view

§2.3 的事實容許一個上表前兩列都沒涵蓋的做法：`ReIDView`(16) / `SemanticView`(51) /
`LifecycleView`(129) / `TriggerView`(0) 在 `src/` 是 **零** nested 讀取 —— 361 次 nested
存取（350 行，同一行可能多個）全部落在 core(82) / detection(158) / geometry(101) /
motion(20) —— 所以「保留四個熱路徑 view、移除其餘四個」**不需要**把任何已遷移的 consumer
改回扁平。它與 `rollback_existing_views` 不是同一件事，成本也不同級。（計數對照
`000d98c3`：`grep -roh "cfg\.\(reid\|semantic\|lifecycle\|trigger\)\." src/ --include=*.py | wc -l`
為 0；同法對 `core\|detection\|geometry\|motion` 為 361。）仍不採用：

1. **它讓 disposition 取決於 §3.2 那個本 ADR 拒絕當成完成條件的狀態。** 分界線是「該模組的
   runtime 是否已遷移」。§3.2 已宣告那次遷移未完成且不再繼續，所以「有沒有 runtime
   consumer」是未完成遷移的副產品，不是責任面的性質。用它當處置標準，等於讓同一個
   Phase-4 投影形狀同時受兩套政策管轄，而分界線是一份本 ADR 明說不再推進的工作進度。
2. **它把 totality invariant 從「意圖聲明」降級成「第二份盤點」。** Parity 測試的不變式是
   `EvalConfig`(366) = 已投影(341) ⊎ 未投影(25)，而未投影的 25 欄是一份可讀的意圖聲明
   （I/O、衍生旗標、`kwargs` 相容欄，見 §2.2）。移除三個 view 會把 196 欄由左搬到右
   （25 → 221，佔 366 的 60%）：檢查仍會對新欄位 fail，但那張 allowlist 不再是能一眼讀完的
   intent，而是複製了 §2.1 已有的 inventory。
3. **它不改變 §4 的結論，卻是 196 欄的建構契約變更。** view 是 construction-time value
   copy；兩種做法在 runtime 的 nested 讀取都是零，責任面完全一樣。要移除的是
   `__post_init__` 的建構契約、parity 測試的投影地圖與 §2.1 的 inventory —— 這需要自己的
   範圍與自己的驗證，與 rollback 同級，不是附掛在 `retain_current_views` 上的免費清理。
4. **`TriggerView` 不構成先例。** 它是 0 欄的空型別，移除只是改名；移除其餘三個是 196 欄的
   投影契約變更。兩者處置成本不同級，不能合成一個「移除未使用 view」的動作。

因此本 ADR 保留全部現有 view；任何選擇性移除（selective pruning）都需要新的 issue。

---

## 6. Decision

**Retain the current views.**

具體鎖定：

1. 現有 frozen view 與 `EvalConfig.__post_init__()` 投影留在 `main`。`TriggerView` 型別保留，但繼續不掛 `cfg.trigger`。
2. 已遷移的 Core / Detection / Geometry / Motion nested 讀取保持 nested。
3. 扁平 `EvalConfig` 仍是可寫入的 source of truth 與構造面。
4. 不再進行 ReID / Semantic / Lifecycle / Trigger 的 consumer 遷移計畫。
5. 不 rollback、不 cherry-pick `132752f2` / `e8aa9612`。
6. 新欄位要進 view，必須是該欄已經是 `EvalConfig` 欄，且有對應模組的責任理由。預設不加。未投影根欄的集合由投影 parity 測試釘住。
7. 不做 selective pruning：即使某個 view 在 `src/` 沒有 runtime consumer（目前是 ReID / Semantic / Lifecycle / Trigger），也不因此移除它（§5.1）。

### 6.1 本判決授權的唯一後續

補上缺失的 **flat ↔ view 全欄投影 parity**（既有測試只抽樣約 15 欄）。Golden snapshot 必須零語意 diff。不改 argparse / 模組 dataclass / `EvalConfig` 預設。

任何 rollback、選擇性移除或後續遷移都要新的 issue；本 ADR 不預授權。

---

## 7. Consequences

- Runtime 行為與設定預設不變。
- Consumer 會繼續混用 `cfg.detection.x` 與 `cfg.reid_mode` 這類扁平名。這是接受的狀態，不是債務看板。
- 構造後不要改扁平欄再期望 view 跟著變；`kwargs` 仍可原地寫入。
- `TriggerView` 維持空的，避免假裝 trigger 已經升上 `EvalConfig`。
