# Research studies（§20.11 tiered studies）

**規則 owner：** [experiment contract §20.11](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)。
**執行：** `scripts/tools/research_study.py`（fail-closed，經 `tests/contract/test_research_study_protocol.py` 跑在 pre-push hook 與 CI）。
本 README 只說目錄形狀，不複述規則。

v1.4 之前宣告的研究維持原本的位置與 sealed procedure，不搬進來。

## 一個 study 的形狀

```text
<study_id>/
  study.yaml        §20.2 裡決定 tier 的欄位（machine-readable；唯一被機器讀的檔）
  declaration.md    人讀的宣告；首段帶 <!-- evidence-tier: … -->，只被 pin，不被解析
  results.md        選填；只能在 attempt 之後出現
  attempts/NNN/     每次執行一個目錄；attempt.json + payload，merge 後 append-only
```

`study.yaml` 欄位：

| 欄位 | 必填 | 說明 |
|:--|:--|:--|
| `schema` | ✓ | `research_study_v1` |
| `study_id` | ✓ | 等於目錄名 |
| `evidence_tier` | ✓ | `formal` / `exploratory`；**checker 依 `section_20_2` 重算，不一致即 fail** |
| `hypothesis` | ✓ | 一行 |
| `section_20_2.output_class` | ✓ | §20.4 類別：`design_candidate` / `performance_upper_bound` / `diagnostic` / `unexplained_residual` |
| `section_20_2.mainline_transition` | ✓ | 每個 terminal → §20.7 轉移：`closes_core_unknown` / `adds_decision_capability` / `changes_production_behavior` / `none` |
| `declaration` / `results` | ✓ / 選填 | 本目錄內的 `.md` 檔名 |
| `runner` | formal 必填 | repo-relative `.py`；有 runner 就必須有下面三項 |
| `inputs` | 隨 runner | `name → repo-relative path`；runner 只能經 `FrozenStudy.input(name)` 取得 |
| `validity_criteria` | 隨 runner | `id → 定義`；invalid attempt 必須引用其一 |
| `attempt_policy` | 隨 runner | `adoption`（`first_valid` / `unanimous_valid`）、`max_valid_attempts`、`max_attempts` |

## Runner 骨架

```python
from research_study import StudyBinding, open_frozen_study

BINDING = StudyBinding(
    study_id="<study_id>",
    runner_file=__file__,
    freeze_tag="freeze/<study_id>/1",          # 每個 attempt 一個新 tag
    pinned_blobs={                             # git rev-parse HEAD:<path>
        "docs/research/studies/<study_id>/study.yaml": "<blob>",
        "docs/research/studies/<study_id>/declaration.md": "<blob>",
    },
)

study = open_frozen_study(BINDING)             # 先驗 freeze，失敗就沒有 data path
packet = study.input("packet")
(study.payload_dir() / "out.csv").write_text(...)
study.record("valid", terminal="LOCALIZED")    # 或 record("invalid", invalid_criterion="V1")
```

流程：同一個 draft PR 內 commit `study.yaml`＋declaration＋runner → review → `git tag -a freeze/<study_id>/<n>` 並 push → 在乾淨的 tree 上執行 runner → commit attempt 與 results → 以 merge commit 合併（squash 會讓 freeze commit 消失，attempt 驗證會 fail）。
