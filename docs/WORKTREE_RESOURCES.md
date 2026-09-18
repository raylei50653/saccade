# Worktree 資源租約（resctl）

> 多個 `git worktree` / 多個 agent 共用同一台機器時，誰在用 GPU、誰在跑 build、
> 現在能不能開始正式 benchmark。工具：[`tools/resctl.py`](../tools/resctl.py)。
> 純本機、無 daemon、無 tracked file；狀態全部放在共用的 git common dir。

---

## 1. 要回答的問題

不是硬體監控。`resctl` 存在是為了讓另一個 agent 在開工前能一眼看到：

- 現在有哪些 worktree（path / branch / HEAD / clean-dirty）
- 誰正在持有哪個資源、跑什麼 command、跑多久了
- 我現在可以安全開始嗎？（`resctl who gpu0` 的 exit code 就是答案）
- 上一個 agent 在這個 worktree 做到哪裡（handoff）

## 2. 資源與衝突

| 資源 | 用途 | 衝突 |
|---|---|---|
| `gpu0` | 一般 CUDA / GPU 工作（eval、訓練、engine build） | `machine-bench` |
| `cpu-heavy` | 大型 `cmake --build`、pytest 全跑、資料生成 | `machine-bench` |
| `machine-bench` | **正式效能量測 = 整台機器獨占** | `gpu0`、`cpu-heavy` |

每個資源同時只能有一個持有者。`gpu0` 與 `cpu-heavy` 互不衝突（GPU eval 與 CPU
build 可並行，但這不代表量測乾淨——**正式數字一律用 `machine-bench`**，只鎖 GPU
不算）。衝突表寫死在 `CONFLICTS`，且有測試保證它對稱。

## 3. 命令

```bash
P=.venv/bin/python            # 任何 worktree 的 venv 都可以；只用 stdlib

$P tools/resctl.py status                      # WORKTREES / LEASES / GPU / SYSTEM / HANDOFF
$P tools/resctl.py status --json               # 給 agent 讀
$P tools/resctl.py who gpu0                    # exit 0 = FREE, 1 = BUSY（印 owner）

$P tools/resctl.py run gpu0 -- CMD ...         # 持有 gpu0 期間執行 CMD
$P tools/resctl.py run cpu-heavy -- bash scripts/pre_push.sh
$P tools/resctl.py run machine-bench -- $P scripts/eval/mot17.py ...
$P tools/resctl.py run --wait --timeout 600 gpu0 -- CMD   # 最多等 10 分鐘

$P tools/resctl.py handoff "finished X; next run Y" \
    --done "X" --pending "Y" --last-result "IDF1 80.4" --next "run Y" [--safe-to-remove]
$P tools/resctl.py handoff-show                # 所有 worktree 的 handoff
$P tools/resctl.py handoff-show --here         # 只看目前 worktree

$P tools/resctl.py clean                       # 只清 stale lease metadata，不碰 active lock
```

`run` 的 exit code：command 的 exit code；被 signal 殺掉 = 128+signal；
**拿不到鎖 = 75**（stderr 印出目前 owner 的 worktree / branch / PID / command）；用法錯 = 2。
`--` 之後的東西原封不動交給 command。

## 4. 機制（audit 用）

狀態根目錄 = `git rev-parse --path-format=absolute --git-common-dir` + `/worktree-runtime/`
（本 repo 即 `.git/worktree-runtime/`），所有 worktree 共用，永遠不會被 git track：

```text
.git/worktree-runtime/
├── locks/      <resource>.lock   flock 目標；.acquire.lock = 取鎖的臨界區
├── leases/     <resource>.json   持有者 metadata
└── handoffs/   <worktree-key>.json
```

**鎖 = kernel `flock(2)`，不是 JSON flag。**

1. `run` 先拿 `.acquire.lock`（exclusive，只包住幾毫秒的取鎖動作）。
2. 對自己的 `<resource>.lock` 做 non-blocking exclusive flock；失敗 = 已有人持有。
3. 對每個衝突資源的 lock 做 non-blocking probe；任何一個 probe 失敗就放掉自己的鎖，
   回報那個資源的 owner。probe 成功即立刻放掉——衝突的另一方之後取鎖時會反向 probe 到我們。
4. 寫 lease JSON（atomic rename），放掉 `.acquire.lock`，執行 command。
5. command 結束（正常、非零、被 signal）後：**先刪 lease、再放 flock**，所以不會出現
   「鎖已釋放但 lease 還在」被新持有者誤刪的視窗。

Lease 內容：`resource / pid / host / worktree / branch / head / dirty / start_time /
command / command_str`。lock fd 是 `O_CLOEXEC`，command 繼承不到，所以 command 自己
fork 出去的 daemon 不會把鎖帶走。

**Fail-safe：**

| 情境 | 行為 |
|---|---|
| command crash / 非零退出 | `finally` 釋放；exit code 照傳 |
| resctl 被 SIGTERM/SIGINT/SIGHUP | 轉送給 command，等它結束後釋放 |
| resctl 被 SIGKILL | kernel 隨 process 死亡釋放 flock（無假鎖）；lease 留下 → `status` 標 **stale**（pid DEAD）；下一個 `run` 直接回收並印 `reclaimed stale lease`。command 端掛了 `PR_SET_PDEATHSIG`，parent 死亡時收到 SIGTERM（best-effort，Linux only，不涵蓋再 re-parent 的孫 process） |
| lease JSON 壞掉 / 不見，但 flock 被持有 | **BUSY, owner unknown** —— metadata 永遠不能讓資源看起來可用 |
| lease 在、flock 沒人持 | FREE + stale 註記；`resctl clean` 可清 |
| 取鎖失敗 | 列出 owner 的 worktree / branch / PID（含存活）/ command / elapsed；**不殺 process、不 override** |
| 無 NVIDIA GPU / 無 `nvidia-smi` | `status` 照常，GPU 區印 `unavailable (reason)`；`RESCTL_NVIDIA_SMI` 可指定 binary 路徑（測試用它指向不存在的路徑） |
| 不在 git repo 內 | exit 2 |

`status` 觀察鎖時拿 `.acquire.lock` 的 shared lock，所以不會看到取鎖過程中 probe 造成的
瞬時 BUSY。

## 5. Handoff：執行層 vs 研究層

`resctl handoff` 是**這個 worktree 的短期執行狀態**，寫在 `.git/worktree-runtime/handoffs/`，
任何 worktree 都讀得到。欄位：

```text
branch / HEAD / dirty（寫入當下自動抓）
note（必填自由文字）
done / pending / last_result / next_action / safe_to_remove
updated_at
```

重複呼叫是 **merge**：沒給的欄位沿用上次值，git 狀態每次刷新；`--clear` 從零開始。
`handoff-show` 會標出 worktree 已不存在的 handoff（`WORKTREE GONE`）。

它**不取代** repo 內的正式文件：研究結論、可引用數字、決策狀態仍走
[docs/README.md 的決策樹](README.md)（evidence ledger、module README、ADR）。
handoff 只回答「下一個接手這個 worktree 的人要從哪一步繼續」。

## 6. 第一版刻意不做

daemon、web UI、scheduler / queue、跨機器協調、自動建立/刪除 worktree、自動 PR、
priority。需要以上任何一項時先開 issue；工具保持單檔、stdlib-only、可從 CLI 直接 audit。

## 7. 測試

[`tests/unit/test_resctl.py`](../tests/unit/test_resctl.py)：用臨時 repo + 兩個 worktree
以 subprocess 驅動真實 CLI，覆蓋：同資源互斥、`machine-bench` 雙向衝突、正常/非零/signal/
SIGKILL 後釋放、corrupt / missing / orphan lease、`--wait` / `--timeout`、跨 worktree
`status` 與 handoff、runtime 狀態不進 `git status`。
