# Runtime Coordinate Archive

每一次 republication 之前，被換掉的那份
`docs/reference/runtime_identity.generated.json` 先原封不動複製到本目錄。

依 [ADR 022](../../../decisions/022-check-taxonomy-and-publication-lag.md) §3 決定 5：
歷史出版要能直接被讀到，**不只靠 git history**。

## 命名

```
runtime_identity.<YYYY-MM-DD>.<implementation digest 前 16 碼>.json
```

日期是 archive 當天；digest 取**被換掉那份**（outgoing）的
`coordinate.implementation`，因為那是實務上最常移動、也最能識別該份出版的一軸。

同一天 republish 兩次、而 `implementation` 沒動時（例如只動 `environment`
recipe 的兩支 PR 接連落地），上面那個名字會撞。此時**不覆蓋**已存在的檔案，改附上
outgoing 出版的 `witness.head` 前 12 碼當 tie-breaker：

```
runtime_identity.<YYYY-MM-DD>.<implementation digest 前 16 碼>.<witness.head 前 12 碼>.json
```

先落地的那份保留原本的短名字；只有後來撞名的才加後綴。首例是
`runtime_identity.2026-09-11.5218626ec3f503fc.48cf0b83737e.json`。

## 規則

- 檔案是**不可變的歷史紀錄**：只新增，永不就地編輯、永不刪除。
- 這裡的檔案**不是** canonical。唯一被 checker 讀的出版是
  `docs/reference/runtime_identity.generated.json`；archive 不參與任何 gate。
- 存進來的必須是一份 `publication_complete: true` 的完整出版。不完整的候選屬於
  scratch，不進 archive。
- Archive 一份舊出版**不**構成任何 claim：它既不是 equivalence 證明，也不繼承或
  轉移任何 research evidence。

流程見 [republication runbook](../../runbooks/runtime_identity_republication.md)。
